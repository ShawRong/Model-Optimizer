# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron core distributed checkpointing plugin for sharded ``modelopt_state``."""

# TODO: Add unit tests for this plugin
import copy
import os
from pathlib import Path
from typing import Any

import torch
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.transformer.module import Float16Module

import modelopt
import modelopt.torch.opt as mto
import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import safe_load
from modelopt.torch.utils.network import SUPPORTED_WRAPPERS

SUPPORTED_WRAPPERS[Float16Module] = "module"


_RESTORE_DEBUG_ENV_VARS = ("MODELOPT_MCORE_RESTORE_DEBUG", "MODELOPT_NVFP4_STATIC_DEBUG")
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _restore_debug(message: str) -> None:
    """Print opt-in restore diagnostics for MCore distributed checkpoints."""
    if not any(
        os.environ.get(env, "").lower() in _TRUE_ENV_VALUES for env in _RESTORE_DEBUG_ENV_VARS
    ):
        return
    rank = "?"
    world = "?"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = str(torch.distributed.get_rank())
        world = str(torch.distributed.get_world_size())
    print(f"[modelopt-mcore-restore-debug rank {rank}/{world}] {message}", flush=True)


def remove_per_module_state(
    modelopt_state: dict[str, Any],
) -> None:
    """Remove metadata from the modelopt_state.

    The metadata of the modelopt_state contains keys which may change with different pipeline
    and expert parallelism. As a result, the metadata must be stored as several ShardedObject with
    global and local layer offset mapping.

    Args:
        modelopt_state: the state_dict that contains all algorithms that have been applied
            to the given model.
    """
    if "modelopt_state_dict" not in modelopt_state:
        return

    for mode, config in modelopt_state["modelopt_state_dict"]:
        metadata = config.get("metadata", None)
        if metadata is not None:
            _ = metadata.pop("quantizer_state", None)
            _ = metadata.pop("subnet_config", None)
            _ = metadata.pop("real_quantizer_state", None)
            _ = metadata.pop("q_tensor_state", None)
        else:
            config["metadata"] = {}


def save_modelopt_state(model: list[torch.nn.Module], state_dict: dict[str, Any]) -> None:
    """Save modelopt_state as a part of the per rank state_dict.

    NOTE: Only used for Megatron-LM.

    Args:
        model: the modelopt optimized model
        state_dict: the current modelopt optimized model state_dict to store
    """
    if not mto.ModeloptStateManager.is_converted(model[0]):
        return
    if len(model) == 1:
        state_dict["modelopt_state"] = mto.modelopt_state(model[0])
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict[f"modelopt_state_{i}"] = mto.modelopt_state(model[i])


def restore_modelopt_state(model: list[torch.nn.Module], state_dict: dict[str, Any]) -> None:
    """Restore modelopt_state from the per rank state_dict.

    NOTE: Only used for Megatron-LM.

    Args:
        model: the model to restore the modelopt optimization
        state_dict: the loaded state_dict to extract
    """
    if (
        len(model) == 1
        and "modelopt_state" in state_dict
        and not mto.ModeloptStateManager.is_converted(model[0])
    ):
        model[0] = mto.restore_from_modelopt_state(model[0], state_dict["modelopt_state"])
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            if f"modelopt_state_{i}" in state_dict and not mto.ModeloptStateManager.is_converted(
                model[i]
            ):
                model[i] = mto.restore_from_modelopt_state(
                    model[i], state_dict[f"modelopt_state_{i}"]
                )


def save_sharded_modelopt_state(
    model: list[torch.nn.Module],
    checkpoint_name: str | Path,
    sharded_strategy: tuple[str, int] | None = None,
    prefix: str = "",
) -> None:
    """Save modelopt_state in the sharded state_dict format.

    Args:
        model: the model to restore the modelopt optimization
        checkpoint_name: the checkpoint folder path
        sharded_strategy: configures sharded tensors saving behavior and backend
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
    """
    if not mto.ModeloptStateManager.is_converted(model[0]):
        return
    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")
    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"
    if dist.is_master():
        os.makedirs(modelopt_checkpoint_name, exist_ok=True)
    modelopt_state = copy.deepcopy(mto.modelopt_state(model[0]))
    remove_per_module_state(modelopt_state)
    dist_checkpointing.save(modelopt_state, modelopt_checkpoint_name, sharded_strategy)


def _load_extra_state_from_sharded_checkpoint(
    model: torch.nn.Module,
    checkpoint_name: str | Path,
    prefix: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Load extra state from sharded checkpoint.

    Note: since extra_state is a subset of full the sharded_state_dict, we use
        strict=StrictHandling.LOG_UNEXPECTED instead of LOG_ALL.

    Args:
        model: the model to load extra state into
        checkpoint_name: the checkpoint folder path
        prefix: the prefix to add to the modelopt_state keys
        metadata: the metadata for distributed checkpointing

    Note:
        The metadata includes several breaking changes. For example, `singleton_local_shards`
        is set to `True` (was not set before) in megatron-core-0.15.0. This flag affects the
        sharded state_dict format and must be consistent between saving and loading.
    """
    sharded_state_dict = model.sharded_state_dict(prefix=prefix)
    extra_sharded_state_dict = {k: v for k, v in sharded_state_dict.items() if "_extra_state" in k}
    _restore_debug(
        f"load_extra_state start checkpoint={checkpoint_name} num_extra_keys={len(extra_sharded_state_dict)}"
    )
    extra_state_dict = dist_checkpointing.load(
        extra_sharded_state_dict,
        checkpoint_name,
        get_default_load_sharded_strategy(checkpoint_name),
        strict=StrictHandling.LOG_UNEXPECTED,
    )
    _restore_debug(
        f"load_extra_state dist_checkpointing.load done num_loaded={len(extra_state_dict)}"
    )
    extra_state_dict_no_prefix = {}

    for k, v in extra_state_dict.items():
        if k.startswith(prefix):
            extra_state_dict_no_prefix[k[len(prefix) :]] = v
    incompatible = model.load_state_dict(extra_state_dict_no_prefix, strict=False)
    _restore_debug(
        "load_extra_state load_state_dict done "
        f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
    )


def restore_sharded_modelopt_state(
    model: list[torch.nn.Module],
    checkpoint_name: str | Path,
    prefix: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Restore modelopt_state from the sharded state_dict format.

    Args:
        model: the model to restore the modelopt optimization
        checkpoint_name: the checkpoint folder path
        prefix: the prefix to add to the modelopt_state keys ("model." for NeMo)
        metadata: the metadata for distributed checkpointing

    Note:
        The metadata includes several breaking changes. For example, `singleton_local_shards`
        is set to `True` (was not set before) in megatron-core-0.15.0. This flag affects the
        sharded state_dict format and must be consistent between saving and loading.
    """
    if len(model) > 1:
        raise ValueError("sharded_modelopt_state does not support virtual pipeline parallel!")

    modelopt_checkpoint_name = f"{checkpoint_name}/modelopt_state"
    _restore_debug(f"restore_sharded_modelopt_state enter checkpoint={checkpoint_name}")

    # Early return if the model already has a modelopt_state or the checkpoint does not exist.
    if not os.path.exists(modelopt_checkpoint_name) or mto.ModeloptStateManager.is_converted(
        model[0]
    ):
        _restore_debug(
            "restore_sharded_modelopt_state skip: no modelopt_state or already converted"
        )
        return

    # Loading the common modelopt_state (replicated on all ranks)
    _restore_debug(f"safe_load common modelopt_state start path={modelopt_checkpoint_name}")
    common_modelopt_state = safe_load(modelopt_checkpoint_name + "/" + COMMON_STATE_FNAME)
    modes = [mode for mode, _ in common_modelopt_state.get("modelopt_state_dict", [])]
    _restore_debug(f"safe_load common modelopt_state done modes={modes}")

    modelopt_load_version = common_modelopt_state["modelopt_version"]

    print(f"nvidia-modelopt ckpt/inst version: {modelopt_load_version}/{modelopt.__version__}")

    # After 0.29, we no longer store (or shard) any quantizer_state in the modelopt_state.
    # quantizer_state (or other per-module state) is stored with the main distributed
    # checkpoint as extra_state at the QuantModule level.
    #
    # The process of resuming modelopt_state becomes 2-phase:
    # 1. Load the global modelopt_state and call mto.restore_from_modelopt_state.
    #    Modes are restored in order. Modes with per-module state stored as
    #    extra_state are partially restored (stop at DynamicModule replacement)
    #
    _restore_debug("restore_from_modelopt_state start")
    model[0] = mto.restore_from_modelopt_state(model[0], common_modelopt_state)
    _restore_debug("restore_from_modelopt_state done")

    _load_extra_state_from_sharded_checkpoint(model[0], checkpoint_name, prefix, metadata=metadata)


def _quantizer_buffer_value_is_invalid(t: "torch.Tensor | None") -> bool:
    """Return True if a quantizer ``_amax`` / ``_global_amax`` buffer is invalid.

    Detects torch.empty(...) leakage that the phase-2 distcp load failed to fill:
      - None
      - meta-device tensor
      - any non-finite (NaN / Inf) entry
      - any negative entry
    Zero is treated as valid (legitimate dead-block amax for NVFP4 static).
    """
    if t is None:
        return True
    if hasattr(t, "device") and getattr(t.device, "type", None) == "meta":
        return True
    if not torch.is_floating_point(t):
        return False
    t = t.detach()
    return bool(torch.any(~torch.isfinite(t)).item() or torch.any(t < 0).item())


def repair_sharded_modelopt_state(
    model: list[torch.nn.Module],
    checkpoint_name: str | Path,
    prefix: str = "",
    metadata: dict[str, Any] | None = None,
) -> int:
    """Re-fill quantizer buffers that the phase-2 distcp load left as ``torch.empty``.

    On certain MCore PP/EP topology changes (observed: PP=1 EP=16 save reshard to
    PP=12 EP=1 export), the second-phase ``dist_checkpointing.load`` silently
    fails to fill the per-rank-first-MoE-local-layer expert ``_amax`` /
    ``_global_amax`` buffers. The runtime buffers stay at uninitialized memory
    from ``register_buffer(..., torch.empty(...))``.

    This helper detects those buffers (NaN / Inf / negative content) and
    re-issues a direct ``dist_checkpointing.load`` for just those keys, using a
    synthetic ``ShardedTensor`` request that bypasses whatever descriptor mismatch
    is upstream of the original phase-2 failure. It then copies the loaded values
    into the runtime buffers, preserving the saved (e.g. MSE-calibrated) state.

    Call this AFTER the main checkpoint phase-2 load completes (e.g. immediately
    after ``dist_checkpointing.load(...)`` and ``model.load_state_dict(...)``
    inside Megatron's checkpoint loader). It is a no-op if no buffers are
    invalid.

    Args:
        model: the model wrapped in a list (matching restore_sharded_modelopt_state).
        checkpoint_name: the checkpoint folder path (parent of "modelopt_state").
        prefix: the prefix added to modelopt_state keys ("model." for NeMo).
        metadata: distcp metadata (currently unused by this helper but kept for API
            symmetry; future versions may need it for sharded layout decisions).

    Returns:
        Number of quantizer buffers repaired (across this rank only).
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from torch.distributed.checkpoint import FileSystemReader

    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    print(
        f"[modelopt][repair_sharded_modelopt_state rank {rank}] enter "
        f"checkpoint={checkpoint_name} prefix={prefix!r} model_len={len(model)}",
        flush=True,
    )

    if len(model) != 1:
        print(
            f"[modelopt][repair_sharded_modelopt_state rank {rank}] "
            f"early-return: len(model)={len(model)} != 1",
            flush=True,
        )
        return 0
    if not mto.ModeloptStateManager.is_converted(model[0]):
        print(
            f"[modelopt][repair_sharded_modelopt_state rank {rank}] "
            "early-return: model not modelopt-converted",
            flush=True,
        )
        return 0

    ckpt_str = str(checkpoint_name)

    # Read saved metadata to know which keys exist and their shapes.
    try:
        reader = FileSystemReader(ckpt_str)
        saved_md = reader.read_metadata().state_dict_metadata
    except Exception as exc:
        _restore_debug(f"repair_sharded_modelopt_state: cannot read .metadata: {exc}")
        return 0

    # Build the runtime sharded_state_dict so we can look up the global key
    # for each runtime quantizer buffer (TransformerBlock + SequentialMLP rewrites).
    try:
        runtime_sd = model[0].sharded_state_dict(prefix=prefix, metadata=metadata)
    except Exception as exc:
        _restore_debug(f"repair_sharded_modelopt_state: sharded_state_dict() failed: {exc}")
        return 0

    # Identify invalid quantizer buffers. NOTE: dict keys in
    # ``model.sharded_state_dict()`` are the *runtime* names (e.g.
    # ``...local_experts.{local_idx}.linear_fc1.weight_quantizer._amax``), but
    # ``ShardedTensor.key`` is the *saved* / sharded name after
    # ``replace_prefix_for_sharding`` rewrites
    # ``local_experts.{local}`` -> ``experts.{global}``. The saved checkpoint
    # ``.metadata`` is keyed by the latter, so use ``v.key`` for both filter
    # and metadata lookup; otherwise we never match anything.
    sharded_key_to_runtime_buffer: dict[str, torch.Tensor] = {}
    n_sharded_tensor = 0
    n_amax_keys = 0
    sample_first_moe_amax_dump: list[str] = []
    for v in runtime_sd.values():
        if not isinstance(v, ShardedTensor):
            continue
        n_sharded_tensor += 1
        sharded_key = v.key
        if not sharded_key.endswith(("._amax", "._global_amax")):
            continue
        n_amax_keys += 1
        runtime_buf = v.data
        # Stash a sample of the buffer content for the FIRST MoE local layer's
        # expert 0 fc1 _amax — which we know empirically gets dropped by phase-2.
        if (
            len(sample_first_moe_amax_dump) < 4
            and ".mlp.experts.experts.0.linear_fc1.weight_quantizer." in sharded_key
        ):
            try:
                t = runtime_buf.detach()
                if t is None:
                    sample_first_moe_amax_dump.append(f"{sharded_key} -> None")
                else:
                    flat = t.float().reshape(-1)
                    n_nan = int(torch.isnan(flat).sum().item())
                    n_inf = int(torch.isinf(flat).sum().item())
                    n_neg = int((flat < 0).sum().item())
                    n_zero = int((flat == 0).sum().item())
                    finite = flat[torch.isfinite(flat)]
                    fmin = float(finite.min().item()) if finite.numel() else float("nan")
                    fmax = float(finite.max().item()) if finite.numel() else float("nan")
                    sample_first_moe_amax_dump.append(
                        f"{sharded_key} dtype={t.dtype} shape={tuple(t.shape)} "
                        f"nan={n_nan} inf={n_inf} neg={n_neg} zero={n_zero}/{flat.numel()} "
                        f"fmin={fmin:.4g} fmax={fmax:.4g}"
                    )
            except Exception as exc:
                sample_first_moe_amax_dump.append(f"{sharded_key} ERR: {exc!r}")
        if _quantizer_buffer_value_is_invalid(runtime_buf):
            sharded_key_to_runtime_buffer[sharded_key] = runtime_buf

    print(
        f"[modelopt][repair_sharded_modelopt_state rank {rank}] "
        f"runtime_sd_size={len(runtime_sd)} sharded_tensor_count={n_sharded_tensor} "
        f"amax_keys={n_amax_keys} invalid={len(sharded_key_to_runtime_buffer)}",
        flush=True,
    )
    for line in sample_first_moe_amax_dump:
        print(
            f"[modelopt][repair_sharded_modelopt_state rank {rank}] sample: {line}",
            flush=True,
        )

    if not sharded_key_to_runtime_buffer:
        return 0

    _restore_debug(
        f"repair_sharded_modelopt_state: detected {len(sharded_key_to_runtime_buffer)} invalid "
        f"quantizer buffer(s); re-loading directly from {ckpt_str}"
    )

    # Synthesize clean ShardedTensors with shape from saved metadata. This avoids
    # whatever descriptor mismatch caused the original phase-2 load to drop these.
    repair_sharded_sd: dict[str, ShardedTensor] = {}
    skipped_missing: list[str] = []
    for sharded_key in sharded_key_to_runtime_buffer:
        meta = saved_md.get(sharded_key)
        if meta is None:
            skipped_missing.append(sharded_key)
            continue
        shape = tuple(meta.size)
        dtype = (
            meta.properties.dtype
            if hasattr(meta, "properties") and meta.properties is not None
            else sharded_key_to_runtime_buffer[sharded_key].dtype
        )
        placeholder = torch.zeros(shape, dtype=dtype)
        repair_sharded_sd[sharded_key] = ShardedTensor.from_rank_offsets(
            sharded_key, placeholder, replica_id=(0, 0, 0)
        )

    if skipped_missing:
        _restore_debug(
            "repair_sharded_modelopt_state: "
            f"{len(skipped_missing)} keys are missing from saved metadata; "
            f"first 3: {skipped_missing[:3]}"
        )

    if not repair_sharded_sd:
        return 0

    loaded = dist_checkpointing.load(
        repair_sharded_sd,
        ckpt_str,
        get_default_load_sharded_strategy(ckpt_str),
    )

    # Copy loaded values into the model's runtime buffers.
    n_repaired = 0
    for sharded_key, runtime_buf in sharded_key_to_runtime_buffer.items():
        new_val = loaded.get(sharded_key)
        if new_val is None:
            continue
        new_val = new_val.to(device=runtime_buf.device, dtype=runtime_buf.dtype)
        if new_val.shape != runtime_buf.shape:
            try:
                new_val = new_val.view_as(runtime_buf)
            except RuntimeError:
                _restore_debug(
                    f"repair_sharded_modelopt_state: shape mismatch on {sharded_key}: "
                    f"loaded={tuple(new_val.shape)} runtime={tuple(runtime_buf.shape)}"
                )
                continue
        with torch.no_grad():
            runtime_buf.copy_(new_val)
        n_repaired += 1

    if n_repaired:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        print(
            f"[modelopt][repair_sharded_modelopt_state rank {rank}] "
            f"re-filled {n_repaired} quantizer buffer(s) from {ckpt_str} "
            "(workaround for distcp phase-2 first-MoE-local-layer drop)",
            flush=True,
        )

    return n_repaired
