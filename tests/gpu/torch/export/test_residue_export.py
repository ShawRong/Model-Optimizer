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

import json
from contextlib import nullcontext
from copy import deepcopy
from itertools import product

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.export.residue_export import (
    collect_residue_export_metadata,
    export_residue_weight,
)
from modelopt.torch.export.unified_export_hf import requantize_resmooth_fused_llm_layers
from modelopt.torch.quantization.nn import ResidueQuantLinear
from modelopt.torch.quantization.qtensor import MXFP4QTensor, NVFP4QTensor
from modelopt.torch.quantization.tensor_quant import dynamic_block_quant


def _loaded_linear(inputs, tensors, metadata, fmt):
    packed = tensors["weight"]
    shape = (packed.shape[0], packed.shape[1] * 2)
    is_mx = fmt == "MXFP4"
    block, bits = (32, (8, 0)) if is_mx else (16, (4, 3))
    quantized = (MXFP4QTensor if is_mx else NVFP4QTensor)(shape, inputs.dtype, packed)
    weight = quantized.dequantize(
        scale=tensors["weight_scale"],
        double_scale=tensors.get("weight_scale_2"),
        block_sizes={-1: block},
    )
    amax = None if is_mx else tensors["input_scale"] * (6 * 448)
    source = inputs.float() if is_mx else inputs
    base = dynamic_block_quant(source, block, amax, None, (2, 1), bits)
    indices = torch.tensor(metadata["salient_indices"], device=inputs.device, dtype=torch.long)
    if indices.numel():
        residual = (source - base).index_select(-1, indices)
        residual = dynamic_block_quant(residual, block, amax, None, (2, 1), bits)
        if metadata["runtime_mode"] == "mext_r1":
            return F.linear(base.to(inputs.dtype), weight, tensors.get("bias")) + F.linear(
                residual.to(inputs.dtype), weight
            )
        base = torch.cat([base, residual], dim=-1)
    base = F.pad(base, (0, shape[1] - base.shape[-1]))
    return F.linear(base.to(inputs.dtype), weight, tensors.get("bias"))


@pytest.mark.parametrize(
    ("fmt", "ratio", "materialized", "conversion"),
    [(*case, None) for case in product(["NVFP4", "MXFP4"], [0.0, 0.125, 0.25, 1.0], [False, True])]
    + [("NVFP4", 0.25, False, "split"), ("NVFP4", 0.25, False, "rename")],
)
@torch.no_grad()
def test_residue_hf_export_roundtrip(tmp_path, fmt, ratio, materialized, conversion):
    torch.manual_seed(31)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=32,
                tie_word_embeddings=False,
            )
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    config = deepcopy(getattr(mtq, f"{fmt}_DEFAULT_CFG"))
    config["algorithm"] = {"method": "residue_max", "residue_ratio": ratio}
    tokens = torch.randint(0, 64, (2, 8), device="cuda")
    mtq.quantize(model, config, lambda m: m(tokens))
    # Native export shares scales across fused projections; compare against that same QDQ state.
    requantize_resmooth_fused_llm_layers(model)
    references = {}
    for name, module in model.named_modules():
        if isinstance(module, ResidueQuantLinear):
            inputs = torch.randn(3, module.in_features, device="cuda", dtype=torch.bfloat16)
            if materialized:
                module.materialize_residue_weight()
            expected = module(inputs)
            if fmt == "MXFP4" and ratio == 0:
                # Plain MXFP4 export uses native packed-weight rounding, unlike fake quantization.
                packed, scale = MXFP4QTensor.quantize(module.weight, 32)
                weight = packed.dequantize(scale=scale, block_sizes={-1: 32})
                base = dynamic_block_quant(inputs, 32, None, None, (2, 1), (8, 0))
                expected = F.linear(base, weight, module.bias)
            references[name] = inputs, expected
    assert references
    if conversion:
        from transformers.core_model_loading import Concatenate, WeightConverter, WeightRenaming

        if conversion == "split":
            model._weight_conversions = [
                WeightConverter(
                    source_patterns=["mlp.gate_left_proj", "mlp.gate_right_proj"],
                    target_patterns="mlp.gate_proj",
                    operations=[Concatenate(dim=0)],
                )
            ]
        else:
            model._weight_conversions = [
                WeightRenaming(
                    source_patterns="mlp.feed_forward_gate", target_patterns="mlp.gate_proj"
                )
            ]
            references = {
                name.replace("mlp.gate_proj", "mlp.feed_forward_gate"): value
                for name, value in references.items()
            }
    context = (
        pytest.warns(UserWarning, match="Residue metadata cannot follow this weight conversion")
        if conversion == "split"
        else nullcontext()
    )
    with context:
        export_hf_checkpoint(
            model, export_dir=tmp_path, dtype=torch.bfloat16, max_shard_size="20KB"
        )
    metadata = json.loads((tmp_path / "residue_kernel_metadata.json").read_text())
    tensors = {}
    for shard in tmp_path.glob("*.safetensors"):
        tensors.update(load_file(shard, device="cuda"))
    assert not any("quantizer" in name or "salient_indices" in name for name in tensors)
    assert set(metadata["layers"]) == set(references)
    for name, (inputs, expected) in references.items():
        layer = metadata["layers"][name]
        assert layer["runtime_mode"] == (
            "mext_r1" if ratio == 1 else "extended_k" if ratio else "standard"
        )
        weights = {
            key.removeprefix(name + "."): value
            for key, value in tensors.items()
            if key.startswith(name + ".")
        }
        width = weights["weight"].shape[1] * 2
        if ratio in (0, 1):
            assert width == inputs.shape[-1]
            assert name not in metadata["extended_linear_dims"]
        else:
            assert width == metadata["extended_linear_dims"][name]
        actual = _loaded_linear(inputs, weights, layer, fmt)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.03)


@pytest.mark.parametrize("fmt", ["NVFP4", "MXFP4"])
@torch.no_grad()
def test_residue_export_across_packing_chunks(fmt):
    torch.manual_seed(31)
    config = deepcopy(getattr(mtq, f"{fmt}_DEFAULT_CFG"))
    config["algorithm"] = {"method": "residue_max", "residue_ratio": 0.25}
    inputs = torch.randn(3, 1024, device="cuda", dtype=torch.bfloat16)
    model = mtq.quantize(
        torch.nn.Linear(1024, 4096, device="cuda", dtype=torch.bfloat16),
        config,
        lambda m: m(inputs),
    )
    expected = model(inputs)
    before = model.weight.detach().clone()
    parameter = model.weight
    metadata = collect_residue_export_metadata(model)["layers"][""]
    tensors = export_residue_weight(model, torch.bfloat16)
    tensors["bias"] = model.bias
    actual = _loaded_linear(inputs, tensors, metadata, fmt)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    assert model.weight is parameter
    torch.testing.assert_close(model.weight, before, atol=0, rtol=0)
