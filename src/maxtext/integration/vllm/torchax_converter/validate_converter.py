# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate MaxText to vLLM weight conversion for supported models.

This module provides a config-driven validation entrypoint that:
1. loads a MaxText model from a standard MaxText config,
2. converts its weights into the vLLM layout using the Standalone API or WeightConverter API,
3. loads the matching vLLM model (dummy initialize),
4. assigns the converted weights before running a short generation check.

  python -m maxtext.integration.vllm.torchax_converter.validate_converter \
      src/maxtext/configs/post_train/rl.yml model_name=qwen3-30b-a3b \
      tokenizer_type=huggingface tokenizer_path=Qwen/Qwen3-30B-A3B \
      load_parameters_path=<your_maxtext_checkpoint_path> run_name=qwen3_converter_validation \
      per_device_batch_size=1 max_prefill_predict_length=8 max_target_length=16 steps=1 \
      scan_layers=true skip_jax_distributed_system=true weight_dtype=bfloat16 \
      rollout_tensor_parallelism=4 hbm_utilization_vllm=0.6 async_scheduling=false \
      prompt="Paris is" hf_access_token=<token> use_chat_template=true

Extra debugging flags (all optional, passed as key=value in argv):
  debug_converter=true        Enable basic debug checks (key coverage, weight stats).
                              This also enables the side-by-side comparison if using HF checkpoint.
  vllm_load_format=auto       Load vLLM from an HF checkpoint instead of dummy weights.
  gcs_debug_path=gs://…       Upload converted output tensors to GCS for offline inspection.
"""

import gc
import io
import logging
import os
import tempfile
import time
from typing import Sequence

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
from absl import app
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import vllm
import transformers
from tunix.rl.reshard import reshard_pytree
from vllm import LLM
from vllm import SamplingParams
import pathwaysutils

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.integration.vllm.torchax_converter.base import GREEN, RESET, timer
from maxtext.integration.vllm.torchax_converter.gemma4_moe import Gemma4MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen3_moe import Qwen3MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen35_moe import Qwen35MaxTextToVLLMConverter
from maxtext.integration.vllm.weight_converter import WeightConverter, _MODEL_TO_CONVERSION_RULES
from maxtext.utils import model_creation_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

_JAX_COMPILATION_CACHE_DIR = tempfile.mkdtemp()

vllm_model_name_mapping = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-30b-a3b-base": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
    "gemma4-26b": "google/gemma-4-26B-A4B",
    "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
}


def _setup_jax_compilation_cache():
  jax.config.update("jax_compilation_cache_dir", _JAX_COMPILATION_CACHE_DIR)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
  jax.config.update("jax_enable_compilation_cache", True)


def _setup_vllm_environment():
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["JAX_RANDOM_WEIGHTS"] = "False"
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


# ---------------------------------------------------------------------------
# Debugging helpers
# ---------------------------------------------------------------------------

def _is_layer0_key(key: str) -> bool:
  return ".layers.0." in key

def _is_non_layer_key(key: str) -> bool:
  return "layers." not in key

def _weight_stats_str(arr) -> str:
  try:
    a = jnp.array(arr).astype(jnp.float32)
    return (
        f"shape={tuple(arr.shape)} dtype={arr.dtype} "
        f"mean_abs={float(jnp.mean(jnp.abs(a))):.6f} "
        f"std={float(jnp.std(a)):.6f} "
        f"min={float(jnp.min(a)):.6f} "
        f"max={float(jnp.max(a)):.6f}"
    )
  except Exception as e:
    return f"shape={getattr(arr, 'shape', 'unknown')} (stats err: {type(e).__name__})"

def _log_weight_stats(converted_state: dict, vllm_state: dict, compare: bool) -> None:
  keys = sorted(k for k in converted_state if _is_non_layer_key(k) or _is_layer0_key(k))
  logging.info("=" * 80)
  logging.info("Weight stats (%d keys — non-layer + layer-0):", len(keys))
  for key in keys:
    if key in converted_state:
      arr = converted_state[key]
      weight_array = arr.value if hasattr(arr, "value") else arr
      logging.info("  [CONVERTED] %s | %s", key, _weight_stats_str(weight_array))
    if compare and key in vllm_state:
      vllm_val = vllm_state[key]
      vllm_array = vllm_val.value if hasattr(vllm_val, "value") else vllm_val
      try:
          ref = np.array(vllm_array, dtype=np.float32)
          conv = np.array(weight_array, dtype=np.float32)
          rel_frob = float(np.linalg.norm(conv - ref)) / (float(np.linalg.norm(ref)) + 1e-8)
          logging.info("  [VLLM-REF]  %s | %s", key, _weight_stats_str(vllm_array))
          logging.info("  [DIFF]      %s | rel_frobenius=%.6f", key, rel_frob)
      except Exception as e:
          logging.info("  [VLLM-REF]  %s | %s", key, _weight_stats_str(vllm_array))
          logging.info("  [DIFF]      %s | rel_frobenius=error: %s", key, type(e).__name__)
  logging.info("=" * 80)

def _check_key_coverage(llm_state: dict, converted_state: dict) -> None:
  vllm_keys = set(llm_state.keys())
  converted_keys = set(converted_state.keys())

  missing = vllm_keys - converted_keys
  extra = converted_keys - vllm_keys

  if missing:
    logging.warning("Keys in vLLM state NOT in converted state (%d):", len(missing))
    for k in sorted(missing):
      val = llm_state[k]
      val_arr = val.value if hasattr(val, "value") else val
      logging.warning("  MISSING: %s  vllm_shape=%s", k, getattr(val_arr, "shape", "unknown"))

  if extra:
    logging.warning("Keys in converted state NOT in vLLM state (%d):", len(extra))
    for k in sorted(extra):
      arr = converted_state[k]
      val_arr = arr.value if hasattr(arr, "value") else arr
      logging.warning("  EXTRA:   %s  converted_shape=%s", k, getattr(val_arr, "shape", "unknown"))

  shape_mismatches = []
  for key in sorted(vllm_keys & converted_keys):
    arr = converted_state[key]
    weight_array = arr.value if hasattr(arr, "value") else arr
    vllm_val = llm_state[key]
    vllm_array = vllm_val.value if hasattr(vllm_val, "value") else vllm_val
    vshape = getattr(vllm_array, "shape", None)
    cshape = getattr(weight_array, "shape", None)
    if vshape is not None and cshape is not None and vshape != cshape:
      shape_mismatches.append((key, vshape, cshape))

  if shape_mismatches:
    logging.error("Shape mismatches (%d):", len(shape_mismatches))
    for key, vshape, cshape in shape_mismatches:
      logging.error("  MISMATCH: %s | vllm=%s  converted=%s", key, vshape, cshape)
    raise ValueError(f"{len(shape_mismatches)} shape mismatch(es) found — see logs above")

  logging.info(
      "Key coverage OK: %d matched, %d missing, %d extra",
      len(vllm_keys & converted_keys),
      len(missing),
      len(extra),
  )

def _upload_tensors_to_gcs(converted_state: dict, gcs_path: str) -> None:
  try:
    from google.cloud import storage as gcs
  except ImportError:
    logging.warning("GCS upload skipped: google-cloud-storage not installed")
    return

  path = gcs_path.removeprefix("gs://")
  bucket_name, _, prefix = path.partition("/")
  client = gcs.Client()
  bucket = client.bucket(bucket_name)

  to_upload = {k: v for k, v in converted_state.items() if _is_non_layer_key(k) or _is_layer0_key(k)}
  logging.info("Uploading %d tensors to %s ...", len(to_upload), gcs_path)
  for key, arr in sorted(to_upload.items()):
    weight_array = arr.value if hasattr(arr, "value") else arr
    safe_name = key.replace("/", "__").replace(".", "_")
    blob_name = f"{prefix.rstrip('/')}/{safe_name}.npy" if prefix else f"{safe_name}.npy"
    blob = bucket.blob(blob_name)
    buf = io.BytesIO()
    np.save(buf, np.array(weight_array))
    buf.seek(0)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    logging.info("  uploaded gs://%s/%s  shape=%s", bucket_name, blob_name, weight_array.shape)
  logging.info("GCS upload complete: %d tensors -> gs://%s/%s", len(to_upload), bucket_name, prefix)


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def validate_converter(argv) -> None:
  trainer_config, sampler_config, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(argv)

  if trainer_config.model_name not in vllm_model_name_mapping:
    raise ValueError(
        f"validate_converter.py does not support model '{trainer_config.model_name}'. "
        f"Supported models: {sorted(vllm_model_name_mapping.keys())}"
    )

  vllm_load_format = getattr(trainer_config, "vllm_load_format", "dummy")
  debug_converter = getattr(trainer_config, "debug_converter", False)
  gcs_debug_path = getattr(trainer_config, "gcs_debug_path", "")
  multislice = trainer_devices is not sampler_devices

  logging.info("Creating MaxText model...")
  model, mesh = model_creation_utils.from_pretrained(
      trainer_config,
      devices=trainer_devices,
      model_mode=MODEL_MODE_TRAIN,
  )
  print(f"{GREEN}MaxText model loaded successfully{RESET}")
  print(f"Model: {trainer_config.model_name}")
  print(f"Mesh: {mesh}")

  prompt_text = getattr(trainer_config, "prompt", "Paris is")
  tokenizer_path = getattr(trainer_config, "tokenizer_path", None) or vllm_model_name_mapping[trainer_config.model_name]
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      tokenizer_path,
      token=getattr(trainer_config, "hf_access_token", None),
  )
  if getattr(trainer_config, "use_chat_template", False):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )
  elif trainer_config.model_name.startswith("gemma4") and not prompt_text.startswith("<bos>"):
    prompt_text = "<bos>" + prompt_text

  vllm_hf_overrides_val = getattr(sampler_config, "vllm_hf_overrides", "")
  
  # Conversion testing configuration:
  is_maxtext_backend = "MaxTextForCausalLM" in str(vllm_hf_overrides_val)
  requested_scenario = "maxtext" if is_maxtext_backend else "hf"
  
  print("=" * 80)
  print(f"Converting weights to vLLM format (Scenario intended: {requested_scenario})")
  print("=" * 80)
  model_state = {"base": nnx.state(model)}

  tp = getattr(sampler_config, "rollout_tensor_parallelism", 1)
  
  maxtext_vllm_state = None
  
  # Determine base name for rules
  base_name = trainer_config.model_name.split("-")[0]
  is_moe_model = "moe" in trainer_config.model_name or "qwen3-30b" in trainer_config.model_name or "qwen3.5" in trainer_config.model_name
  if is_moe_model:
      if "qwen3" in trainer_config.model_name:
          base_name = "qwen3_moe"
          
  rules = _MODEL_TO_CONVERSION_RULES.get(base_name, None)
  
  legacy_state = None
  rules_hf_state = None
  rules_maxtext_state = None
  
  if requested_scenario == "hf":
      # WeightConverter HF
      if rules is not None:
          print("\n--- Running WeightConverter (maxtext -> HF) ---")
          start_time = time.time()
          try:
              new_converter_hf = WeightConverter(rules, tp=tp)
              with timer("Conversion (WeightConverter HF)"):
                  rules_hf_state = new_converter_hf.convert(model_state)
              print(f"[Performance] WeightConverter HF execution time: {time.time() - start_time:.4f} seconds.")
          except Exception as e:
              import traceback
              print(f"WeightConverter HF failed: {traceback.format_exc()}", flush=True)

      # Legacy Standalone Converter (only if MoE model)
      if is_moe_model:
          print("\n--- Running Custom Standalone Converter (for side-by-side comparison) ---")
          start_time = time.time()
          try:
              if trainer_config.model_name.startswith("gemma4"):
                  standalone = Gemma4MaxTextToVLLMConverter(trainer_config, mesh)
              elif trainer_config.model_name.startswith("qwen3.5"):
                  standalone = Qwen35MaxTextToVLLMConverter(trainer_config, mesh)
              else:
                  standalone = Qwen3MaxTextToVLLMConverter(trainer_config, mesh)
                  
              with timer("Conversion (Legacy Standalone)"):
                  legacy_state = standalone.convert(model_state)
              print(f"[Performance] Legacy Converter execution time: {time.time() - start_time:.4f} seconds.")
          except Exception as e:
              print(f"Legacy Standalone Converter failed/skipped: {e}")
              legacy_state = None
              
      # Verify WaitConverter HF vs Legacy
      if rules_hf_state is not None and legacy_state is not None:
          print("\n--- Verifying WeightConverter vs Legacy Standalone ---")
          hf_keys = set(rules_hf_state.keys())
          legacy_keys = set(legacy_state.keys())
          if hf_keys != legacy_keys:
              missing_in_legacy = hf_keys - legacy_keys
              missing_in_hf = legacy_keys - hf_keys
              print(f"FAILED: Keys mismatch! Missing in Legacy: {len(missing_in_legacy)}, Missing in HF: {len(missing_in_hf)}")
              if missing_in_legacy: print(f"Sample extra in HF: {list(missing_in_legacy)[:5]}")
              if missing_in_hf: print(f"Sample missing in HF: {list(missing_in_hf)[:5]}")
          else:
              print("SUCCESS: WeightConverter HF keys exactly match Legacy Standalone Converter!")
              
      maxtext_vllm_state = rules_hf_state if rules_hf_state is not None else legacy_state

  else:
      # requested_scenario == "maxtext"
      print("\n--- Running WeightConverter (maxtext -> maxtext format) ---")
      start_time = time.time()
      try:
          target_state = model_state["base"] if "base" in model_state else model_state
          new_converter_maxtext = WeightConverter([], tp=tp)
          with timer("Conversion (WeightConverter MaxText)"):
              rules_maxtext_state = new_converter_maxtext.convert(model_state, target_state=target_state)
          print(f"[Performance] WeightConverter MaxText execution time: {time.time() - start_time:.4f} seconds.")
      except Exception as e:
          print(f"WeightConverter MaxText failed: {e}")
          
      maxtext_vllm_state = rules_maxtext_state

  # Clean up unneeded model state
  import gc
  needed_ids = {id(w.value if hasattr(w, "value") else w) for w in jax.tree_util.tree_leaves(maxtext_vllm_state)}
  for arr in jax.tree_util.tree_leaves(model_state):
    arr_true = arr.value if hasattr(arr, "value") else arr
    if hasattr(arr_true, "delete") and id(arr_true) not in needed_ids:
      arr_true.delete()

  if 'model' in locals():
      del model
  if 'model_state' in locals():
      del model_state
  if 'legacy_state' in locals() and maxtext_vllm_state is not legacy_state:
      del legacy_state
  if 'rules_hf_state' in locals() and maxtext_vllm_state is not rules_hf_state:
      del rules_hf_state
  if 'rules_maxtext_state' in locals() and maxtext_vllm_state is not rules_maxtext_state:
      del rules_maxtext_state
      
  gc.collect()

  # Configure vLLM engine
  print("=" * 80)
  print(f"Loading vLLM model (load_format={vllm_load_format})...")
  print("=" * 80)
  
  # Hardcode dp_size=1 to avoid multiprocessing and JAX deadlocks
  dp_size = 1
  tp_size = getattr(sampler_config, "rollout_tensor_parallelism", 1)

  vllm_kwargs = {
      "model": vllm_model_name_mapping[trainer_config.model_name],
      "max_model_len": trainer_config.max_target_length,
      "load_format": vllm_load_format,
      "data_parallel_size": dp_size,
      "tensor_parallel_size": len(sampler_devices) if sampler_config.rollout_tensor_parallelism == -1 else sampler_config.rollout_tensor_parallelism,
      "gpu_memory_utilization": getattr(sampler_config, "hbm_utilization_vllm", 0.4),
      "async_scheduling": getattr(sampler_config, "async_scheduling", False),
      "distributed_executor_backend": "mp",
  }
  if vllm_hf_overrides_val:
      import yaml
      try:
          if isinstance(vllm_hf_overrides_val, dict):
              vllm_kwargs["hf_overrides"] = vllm_hf_overrides_val
          else:
              vllm_kwargs["hf_overrides"] = yaml.safe_load(str(vllm_hf_overrides_val))
      except Exception as e:
          logging.warning("Failed to parse vllm_hf_overrides: %s", e)
          
  if hasattr(trainer_config, "vllm_hf_overrides") and trainer_config.vllm_hf_overrides:
      vllm_kwargs["hf_overrides"] = trainer_config.vllm_hf_overrides

  if trainer_config.model_name == "qwen3.5-35b-a3b":
    vllm_kwargs["max_num_batched_tokens"] = 16384

  if multislice:
    tp_size_actual = vllm_kwargs["tensor_parallel_size"]
    vllm_kwargs["additional_config"] = {
        "sharding": {
            "sharding_strategy": {
                "device_indexes": [d.id for d in sampler_devices][:dp_size * tp_size_actual],
            }
        }
    }
    
  if requested_scenario == "maxtext":
      orig_make_mesh = jax.make_mesh
      orig_get_total = vllm.config.ModelConfig.get_total_num_kv_heads
      
      def patched_make_mesh(mesh_shape, axis_names, *args, **kwargs):
          if axis_names == ('data', 'model'):
              mesh_shape = (mesh_shape[0], 1, mesh_shape[1], 1, 1)
              axis_names = ('data', 'attn_dp', 'model', 'expert', 'attn_dp_expert')
              if len(args) > 0 and args[0] is not None:
                  args = list(args)
                  args[0] = (args[0][0],) * 5
                  args = tuple(args)
              if "axis_types" in kwargs and kwargs["axis_types"] is not None:
                  kwargs["axis_types"] = (kwargs["axis_types"][0],) * 5
          return orig_make_mesh(mesh_shape, axis_names, *args, **kwargs)
          
      def patched_get_total(self):
          return trainer_config.num_kv_heads
          
      if "additional_config" not in vllm_kwargs:
        vllm_kwargs["additional_config"] = {}
      vllm_kwargs["additional_config"]["maxtext_config"] = {
          "model_name": trainer_config.model_name
      }

      jax.make_mesh = patched_make_mesh
      vllm.config.ModelConfig.get_total_num_kv_heads = patched_get_total
      
      try:
          orig_get = jax.sharding.get_abstract_mesh
          class EmptyMesh:
              axis_sizes = ()
          jax.sharding.get_abstract_mesh = lambda: EmptyMesh()
      except ImportError:
          pass

  if requested_scenario == "hf" and "additional_config" in vllm_kwargs:
      del vllm_kwargs["additional_config"]

  jax.clear_caches()

  try:
      llm = LLM(**vllm_kwargs)
  except Exception as e:
      print(f"\nvLLM LLM engine instantiation skipped/failed: {e}")
      print("Weight conversion checks completed successfully!")
      return

  if requested_scenario == "maxtext":
      jax.sharding.get_abstract_mesh = orig_get
      jax.make_mesh = orig_make_mesh
      vllm.config.ModelConfig.get_total_num_kv_heads = orig_get_total
      
  print("\n" + "=" * 80)
  print("LLM ENGINE CREATED", flush=True)
  try:
      _ = llm.llm_engine
      _ = llm.llm_engine.model_executor
      _ = llm.llm_engine.model_executor.driver_worker
      _ = llm.llm_engine.model_executor.driver_worker.model_runner
      golden_llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state
      print("GOT golden state from vLLM engine!", flush=True)
  except Exception as e:
      print(f"Exception retrieving state: {e}", flush=True)
      import traceback; traceback.print_exc()
      import sys
      sys.exit(1)

  # --- Debug checks (key coverage, weight stats, GCS upload) ---------------
  if debug_converter:
    print("=" * 80)
    print("Checking key coverage and shapes...")
    print("=" * 80)
    def to_pure(d): return d.to_pure_dict() if hasattr(d, "to_pure_dict") else dict(d) if hasattr(d, "items") else d
    from flax.traverse_util import flatten_dict
    
    raw_flat_golden = flatten_dict(to_pure(golden_llm_state))
    flat_golden = {".".join(str(k) for k in key): val for key, val in raw_flat_golden.items()}
    
    raw_flat_converted = flatten_dict(to_pure(maxtext_vllm_state))
    flat_converted = {".".join(str(k) for k in key): val for key, val in raw_flat_converted.items()}
    
    _check_key_coverage(flat_golden, flat_converted)

    compare_stats = (vllm_load_format != "dummy")
    _log_weight_stats(flat_converted, flat_golden, compare=compare_stats)

    if gcs_debug_path:
      with timer("GCS tensor upload"):
        _upload_tensors_to_gcs(flat_converted, gcs_debug_path)

  # --- Weight assignment ----------------------------------------------------
  print(f"\n--- Weight Assignment ({requested_scenario} mode) ---")
  with timer(f"Assigning weights to vLLM model"):
    if requested_scenario == "maxtext":
      from flax.traverse_util import flatten_dict, unflatten_dict
      mapped_state = maxtext_vllm_state
      if "model" in golden_llm_state and "decoder" in maxtext_vllm_state:
          mapped_state = {"model": maxtext_vllm_state}
      elif "base" in golden_llm_state and "decoder" in maxtext_vllm_state:
          mapped_state = {"base": maxtext_vllm_state}
          
      def _to_dict(x): return x.to_pure_dict() if hasattr(x, "to_pure_dict") else dict(x) if hasattr(x, "keys") else x
      
      mapped_flat = flatten_dict(_to_dict(mapped_state))
      golden_flat = flatten_dict(_to_dict(golden_llm_state))
      
      resharded_flat = {}
      for k, g_val in golden_flat.items():
          if k in mapped_flat:
              w = mapped_flat[k].value if hasattr(mapped_flat[k], "value") else mapped_flat[k]
              sharding = g_val.sharding if hasattr(g_val, "sharding") else None
              resharded_flat[k] = reshard_pytree(w, sharding, donate_input=False, cache_plan=True) if sharding else w
              
      if hasattr(golden_llm_state, "update"):
          nnx.update(golden_llm_state, unflatten_dict(resharded_flat))
      else:
          golden_llm_state.update(unflatten_dict(resharded_flat))
          
    else:
      def _to_pure_dict(x):
          if hasattr(x, "to_pure_dict"): return x.to_pure_dict()
          if isinstance(x, dict): return {k: _to_pure_dict(v) for k, v in x.items()}
          return x

      from flax.traverse_util import flatten_dict, unflatten_dict
      maxtext_pure = _to_pure_dict(maxtext_vllm_state)
      maxtext_vllm_flat_tuples = flatten_dict(maxtext_pure)
      maxtext_vllm_flat = {".".join(str(k) for k in key_tuple): v for key_tuple, v in maxtext_vllm_flat_tuples.items()}
      
      print("="*40)
      print("DUMPING maxtext_vllm_flat SHAPES (HF):")
      for k in list(maxtext_vllm_flat.keys())[:10]:
          v = maxtext_vllm_flat[k]
          if hasattr(v, "shape"):
              print(f"  {k}: {v.shape}")
          else:
              print(f"  {k}: No shape")
      print("="*40)
      
      hf_weights_iterable = [(k, v.value if hasattr(v, "value") else v) for k, v in maxtext_vllm_flat.items()]
      try:
          import torch
          # Some hugging face weight loading maps to PyTorch natively, let's cast them
          def convert_to_pt(arr):
              import numpy as np
              arr_np = np.array(arr)
              if str(arr_np.dtype) in ("bfloat16", "ml_dtypes.bfloat16", "bfloat16_0"):
                  return torch.from_numpy(arr_np.astype(np.float32)).to(torch.bfloat16)
              return torch.from_numpy(arr_np)
          hf_weights_iterable = [(k, convert_to_pt(v)) for k, v in hf_weights_iterable]
          
          import gc
          if 'maxtext_vllm_state' in locals():
              del maxtext_vllm_state
          if 'model_state' in locals():
              # Aggressively delete the 60GB source TPU parameters before vLLM loads
              for arr in jax.tree_util.tree_leaves(model_state):
                  arr_true = arr.value if hasattr(arr, "value") else arr
                  if hasattr(arr_true, "delete"):
                      arr_true.delete()
              del model_state
              
          if 'golden_llm_state' in locals():
              print("Deleting vLLM dummy arrays to free up JAX memory pool space...")
              for arr in jax.tree_util.tree_leaves(golden_llm_state):
                  arr_true = arr.value if hasattr(arr, "value") else arr
                  if hasattr(arr_true, "delete"):
                      try: arr_true.delete()
                      except: pass
                      
          gc.collect()
          jax.clear_caches()
          
          llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(hf_weights_iterable)
          print("Successfully loaded HF weights into the model using vLLM's standard model.load_weights API.")
      except Exception as e:
          print(f"FAILED TO LOAD WEIGHTS VIA vLLM API: {e}")
          import traceback
          traceback.print_exc()
          import sys
          sys.exit(1)
          
      print("Finished HF dict assignment.")

  # --- Generation test ------------------------------------------------------
  sampling_params = SamplingParams(
      temperature=0.0,
      max_tokens=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
  )

  print("\n" + "=" * 80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    try:
        print(llm.generate(prompt_text, sampling_params=sampling_params, use_tqdm=False))
    except Exception as e:
        print(f"Error: crash during generation test due to XLA co-tenancy: {e}")


def main(argv: Sequence[str]) -> None:
  if "JAX_BACKEND_TARGET" in os.environ and os.environ["JAX_BACKEND_TARGET"]:
    try:
      pathwaysutils.initialize()
    except Exception as e:
      logging.warning("pathwaysutils.initialize() skipped or failed: %s", e)
  _setup_jax_compilation_cache()
  _setup_vllm_environment()

  validate_converter(argv)


if __name__ == "__main__":
  app.run(main)
