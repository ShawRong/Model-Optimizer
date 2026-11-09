# Adapted from ResServe (export.py / kernel_metadata.py):
# https://github.com/ShawRong/resserve/tree/51b906836a69fa43915040314df57bc9f23030ad/quant/modelopt_integration
# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Native FP4 weight export with the ResServe residue metadata contract."""

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.quantization.nn import ResidueQuantLinear, ResidueTensorQuantizer
from modelopt.torch.quantization.utils.core_utils import has_accelerate_offload
from modelopt.torch.utils.distributed import is_fsdp2_model

from .quantized_weight_export import (
    capture_quantized_weight_export_state,
    export_quantized_weight_tensors,
)

__all__: list[str] = []

_FUSED_PROJECTIONS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z"),
    ("in_proj_b", "in_proj_a"),
)


def _residue_layer_metadata(module):
    weight = module.weight
    if weight.is_meta or hasattr(weight, "to_local") or not weight.is_floating_point():
        raise NotImplementedError("Residue export requires resident, uncompressed weights.")
    width = getattr(module, "_original_in_features", module.in_features)
    indices = module.input_quantizer.salient_indices
    if indices is None:
        raise ValueError("Calibrate residue channels before export.")
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Residue channels must be a one-dimensional integer tensor.")
    selected = indices.tolist()
    if selected != sorted(set(selected)) or any(i < 0 or i >= width for i in selected):
        raise ValueError("Residue channels must be sorted, unique and in range.")
    full = selected == list(range(width))
    mode = "mext_r1" if full else "extended_k" if selected else "standard"
    formats = []
    for quantizer in (module.input_quantizer, module.weight_quantizer):
        blocks = quantizer.block_sizes or {}
        bits = blocks.get("scale_bits")
        if (
            not isinstance(quantizer, ResidueTensorQuantizer)
            or not quantizer.is_enabled
            or not quantizer.fake_quant
            or quantizer.num_bits != (2, 1)
            or blocks.get("type") != "dynamic"
            or bits not in ((4, 3), (8, 0))
            or blocks.get(-1) != (16 if bits == (4, 3) else 32)
            or quantizer.backend is not None
        ):
            raise NotImplementedError("Residue export requires paired dynamic NVFP4 or MXFP4.")
        if quantizer.pre_quant_scale is not None:
            raise NotImplementedError("Fuse pre_quant_scale before residue deployment export.")
        if bits == (4, 3) and (quantizer.amax is None or quantizer.amax.numel() != 1):
            raise ValueError("NVFP4 residue export requires calibrated scalar amax values.")
        formats.append(bits)
    if formats[0] != formats[1]:
        raise ValueError("Residue activation and weight formats must match.")
    is_mx = formats[0] == (8, 0)
    if width % 128 or module.out_features % 32:
        raise ValueError(
            "Residue deployment requires input width divisible by 128 and output by 32."
        )
    if selected and not full:
        per_block = len(selected) * 8 // width
        if (
            per_block not in (1, 2, 4)
            or len(selected) != width // 8 * per_block
            or any(value // 8 != position // per_block for position, value in enumerate(selected))
        ):
            raise ValueError(
                "Residue deployment requires 1, 2 or 4 selected channels per block of 8."
            )
    expected = width
    if hasattr(module, "_original_in_features"):
        expected += len(selected)
        if is_mx:
            expected += -expected % 32
    if weight.shape != (module.out_features, expected):
        raise ValueError("Residue weight dimensions do not match the calibrated channels.")
    if (
        full
        and hasattr(module, "_original_in_features")
        and not torch.equal(weight[..., :width], weight[..., width : 2 * width])
    ):
        raise ValueError("Full-channel residue export requires identical copied weight columns.")
    if getattr(module, "output_quantizer", None) is not None and module.output_quantizer.is_enabled:
        raise NotImplementedError("Residue export does not support output quantization.")
    return {"salient_indices": selected, "num_salient": len(selected), "runtime_mode": mode}


def collect_residue_export_metadata(model):
    modules = dict(model.named_modules())
    residue = {name: m for name, m in modules.items() if isinstance(m, ResidueQuantLinear)}
    if not residue:
        return {}
    if (
        is_fsdp2_model(model)
        or has_accelerate_offload(model)
        or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)
    ):
        raise NotImplementedError("Residue export requires a resident, single-process model.")
    uses = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    if any(uses[id(m.weight)] > 1 for m in residue.values()):
        raise NotImplementedError("Residue export does not support tied weights.")
    layers = {name: _residue_layer_metadata(m) for name, m in residue.items()}
    for name in layers:
        parent, _, leaf = name.rpartition(".")
        for family in _FUSED_PROJECTIONS:
            if leaf not in family:
                continue
            names = [f"{parent}.{suffix}" if parent else suffix for suffix in family]
            present = [n for n in names if n in modules]
            if any(n not in layers or layers[n] != layers[name] for n in present):
                raise ValueError(
                    f"Fused projections must share residue channels and layout: {present}"
                )
    extended = {}
    for name, m in residue.items():
        layer = layers[name]
        if layer["runtime_mode"] != "extended_k":
            continue
        width = getattr(m, "_original_in_features", m.in_features) + layer["num_salient"]
        if m.weight_quantizer.block_sizes["scale_bits"] == (8, 0):
            width += -width % 128
        extended[name] = width
    return {"layers": layers, "extended_linear_dims": extended}


@torch.no_grad()
def export_residue_weight(module, dtype):
    metadata = _residue_layer_metadata(module)
    width = getattr(module, "_original_in_features", module.in_features)
    weight = module.weight.detach()[..., :width]
    if metadata["runtime_mode"] == "extended_k":
        if hasattr(module, "_original_in_features"):
            weight = module.weight.detach()[..., : width + metadata["num_salient"]]
        else:
            weight = torch.cat(
                [weight, weight.index_select(-1, module.input_quantizer.salient_indices)], dim=-1
            )
    is_mx = module.weight_quantizer.block_sizes["scale_bits"] == (8, 0)
    padding = -weight.shape[-1] % 128 if is_mx else 0
    if padding:
        weight = F.pad(weight, (0, padding))
    # A tensor-only view reuses native packing without exporting residue as an ordinary Linear.
    view = nn.Module()
    view.register_buffer("weight", weight)
    view.input_quantizer = module.input_quantizer
    view.weight_quantizer = module.weight_quantizer
    # Bound native packing temporaries while retaining the calibrated global scales.
    rows = max(1, 2**22 // weight.shape[-1])
    parts = []
    for start in range(0, weight.shape[0], rows):
        view.weight = weight[start : start + rows]
        state = capture_quantized_weight_export_state(view, cpu=False)
        assert state is not None
        parts.append(export_quantized_weight_tensors(view.weight, state, dtype))
    result = parts[0]
    if len(parts) > 1:
        for key in ("weight", "weight_scale"):
            result[key] = torch.cat([part[key] for part in parts], dim=0)
    return result
