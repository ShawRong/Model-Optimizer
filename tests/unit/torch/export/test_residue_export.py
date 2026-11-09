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

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


@pytest.mark.parametrize("invalid", ["indices", "pre_scale", "copied_weight"])
def test_residue_export_rejects_invalid_state_before_writing(tmp_path, invalid):
    config = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    config["algorithm"] = {"method": "residue_max", "residue_ratio": 1.0}
    model = mtq.quantize(nn.Linear(128, 32), config, lambda m: m(torch.ones(2, 128)))
    if invalid == "indices":
        model.input_quantizer.salient_indices = torch.arange(127, -1, -1)
    elif invalid == "pre_scale":
        model.input_quantizer.pre_quant_scale = torch.ones(128)
    else:
        model.materialize_residue_weight()
        with torch.no_grad():
            model.weight[0, 128] += 1
    before = deepcopy(model.state_dict())
    parameter = model.weight
    target = tmp_path / "checkpoint"
    with pytest.raises((ValueError, NotImplementedError)):
        export_hf_checkpoint(model, export_dir=target)
    assert not target.exists()
    assert model.weight is parameter
    assert model.state_dict().keys() == before.keys()
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_residue_export_rejects_incompatible_fused_channels(tmp_path):
    config = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    config["algorithm"] = {"method": "residue_max", "residue_ratio": 0.25}
    linear = mtq.quantize(nn.Linear(128, 32), config, lambda m: m(torch.ones(2, 128)))
    model = nn.ModuleDict({name: deepcopy(linear) for name in ("q_proj", "k_proj", "v_proj")})
    indices = model["v_proj"].input_quantizer.salient_indices
    model["v_proj"].input_quantizer.salient_indices = (
        (indices // 8 * 8 + (indices % 8 + 1) % 8).sort().values
    )
    before = deepcopy(model.state_dict())
    target = tmp_path / "checkpoint"
    with pytest.raises(ValueError):
        export_hf_checkpoint(model, export_dir=target)
    assert not target.exists()
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
