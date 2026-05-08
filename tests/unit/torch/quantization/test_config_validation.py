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

"""Test of quantization config validations."""

import copy
from collections.abc import MutableMapping

import pytest
from pydantic import ValidationError

from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.quantization.config import (
    FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    FP8_DEFAULT_CFG,
    FP8_PER_CHANNEL_PER_TOKEN_CFG,
    INT4_AWQ_CFG,
    NVFP4_DEFAULT_CFG,
    W4A8_AWQ_BETA_CFG,
    QuantizeConfig,
    QuantizerAttributeConfig,
    QuantizerCfgEntry,
    _base_disable_all,
    _default_disabled_quantizer_cfg,
    find_quant_cfg_entry_by_path,
    need_calibration,
    normalize_quant_cfg_list,
)


def _cfg_to_dict(cfg):
    if isinstance(cfg, QuantizerAttributeConfig):
        return cfg.model_dump(exclude_unset=True)
    if isinstance(cfg, list):
        return [_cfg_to_dict(item) for item in cfg]
    return cfg


def test_need_calibration():
    assert need_calibration(FP8_DEFAULT_CFG)
    assert not need_calibration(FP8_PER_CHANNEL_PER_TOKEN_CFG)
    assert not need_calibration(FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG)
    assert need_calibration(INT4_AWQ_CFG)
    assert need_calibration(W4A8_AWQ_BETA_CFG)
    assert need_calibration(NVFP4_DEFAULT_CFG)


def test_need_calibration_with_quantize_config_type():
    """need_calibration accepts schema-backed QuantizeConfig objects."""
    assert need_calibration(QuantizeConfig())
    assert need_calibration(QuantizeConfig.model_validate(FP8_DEFAULT_CFG))
    assert not need_calibration(QuantizeConfig.model_validate(FP8_PER_CHANNEL_PER_TOKEN_CFG))


def test_need_calibration_with_list_cfg():
    """need_calibration must handle sequential (list) cfg entries without crashing."""
    # Static list-cfg on a non-weight quantizer → needs calibration
    cfg_static = {
        "quant_cfg": [
            {
                "quantizer_name": "*input_quantizer",
                "cfg": [
                    {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                    {"num_bits": (4, 3)},
                ],
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    assert need_calibration(cfg_static)

    # Dynamic list-cfg on a non-weight quantizer → no calibration needed
    cfg_dynamic = {
        "quant_cfg": [
            {
                "quantizer_name": "*input_quantizer",
                "cfg": [{"num_bits": (4, 3), "type": "dynamic"}],
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    assert not need_calibration(cfg_dynamic)


def test_quantizer_cfg_entry_is_pydantic_and_dict_like():
    """QuantizerCfgEntry is typed but keeps the dict-style access used by callers."""
    entry = QuantizerCfgEntry(quantizer_name="*", enable=False)
    assert isinstance(entry, ModeloptBaseConfig)
    assert entry["quantizer_name"] == "*"
    assert entry["cfg"] is None
    assert "cfg" in entry
    assert list(entry) == ["quantizer_name", "parent_class", "cfg", "enable"]
    assert dict(entry.items()) == {
        "quantizer_name": "*",
        "parent_class": None,
        "cfg": None,
        "enable": False,
    }
    assert dict(entry.explicit_items()) == {"quantizer_name": "*", "enable": False}
    with pytest.raises(KeyError):
        entry["unknown"] = 1
    assert entry.model_dump(exclude_unset=True) == {"quantizer_name": "*", "enable": False}

    cfg_entry = QuantizerCfgEntry(quantizer_name="*weight_quantizer", cfg={"num_bits": 8})
    assert isinstance(cfg_entry["cfg"], QuantizerAttributeConfig)
    assert _cfg_to_dict(cfg_entry["cfg"]) == {"num_bits": 8}


def test_quantizer_cfg_entry_mutable_mapping_rejects_key_deletion():
    """ModeloptBaseConfig mappings have a fixed key set and reject deletion."""
    entry = QuantizerCfgEntry(quantizer_name="*weight_quantizer", cfg={"num_bits": 8}, enable=True)
    assert isinstance(entry, MutableMapping)
    assert entry.model_dump(exclude_unset=True) == {
        "quantizer_name": "*weight_quantizer",
        "cfg": {"num_bits": 8},
        "enable": True,
    }

    with pytest.raises(TypeError):
        del entry["cfg"]

    assert "cfg" in entry
    assert entry["cfg"] is not None
    assert entry.model_dump(exclude_unset=True) == {
        "quantizer_name": "*weight_quantizer",
        "cfg": {"num_bits": 8},
        "enable": True,
    }

    with pytest.raises(KeyError):
        del entry["missing"]


def test_public_preset_quant_cfg_entries_are_typed_and_dict_like():
    """Public preset constants are typed but keep dict-style entry access."""
    assert isinstance(FP8_DEFAULT_CFG, QuantizeConfig)
    assert isinstance(NVFP4_DEFAULT_CFG, QuantizeConfig)
    for preset in (FP8_DEFAULT_CFG, NVFP4_DEFAULT_CFG):
        assert all(isinstance(entry, QuantizerCfgEntry) for entry in preset["quant_cfg"])
        for entry in preset["quant_cfg"]:
            assert entry["quantizer_name"] == entry.quantizer_name
            assert dict(entry.items())["quantizer_name"] == entry.quantizer_name


def test_mixed_raw_dict_and_modelopt_config_entries_normalize_after_mutation():
    """Mixed raw dict and ModeloptBaseConfig entries normalize after mutation."""
    config = {
        "quant_cfg": [
            *copy.deepcopy(_base_disable_all),
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {
                    "backend": "custom_backend",
                    "num_bits": "e5m2",
                    "pass_through_bwd": True,
                    "backend_extra_args": {
                        "format": "scalar",
                        "block_sizes": 16,
                    },
                },
            },
            {"quantizer_name": "*input_quantizer", "enable": False},
            *copy.deepcopy(_default_disabled_quantizer_cfg),
        ],
        "algorithm": "max",
    }

    normalized = normalize_quant_cfg_list(config["quant_cfg"])
    weight_entry = find_quant_cfg_entry_by_path(normalized, "*weight_quantizer")
    assert weight_entry["cfg"]["num_bits"] == "e5m2"

    raw_weight_entry = find_quant_cfg_entry_by_path(config["quant_cfg"], "*weight_quantizer")
    raw_weight_entry["cfg"]["num_bits"] = "e2m1"
    normalized = normalize_quant_cfg_list(config["quant_cfg"])
    weight_entry = find_quant_cfg_entry_by_path(normalized, "*weight_quantizer")
    assert weight_entry["cfg"]["num_bits"] == "e2m1"

    weight_entry["cfg"]["num_bits"] = "e4m3"
    renormalized = normalize_quant_cfg_list(normalized)
    weight_entry = find_quant_cfg_entry_by_path(renormalized, "*weight_quantizer")
    assert weight_entry["cfg"]["num_bits"] == "e4m3"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"quantizer_name": "*", "cfg": None}, "'?cfg'? must be omitted"),
        ({"quantizer_name": "*", "enable": None}, "'?enable'? must be a boolean"),
    ],
)
def test_quantizer_cfg_entry_rejects_explicit_null_values(raw, match):
    """Explicit null cfg/enable values are rejected instead of treated as omitted."""
    with pytest.raises(ValidationError, match=match):
        QuantizerCfgEntry.model_validate(raw)

    with pytest.raises(ValueError, match=match):
        normalize_quant_cfg_list([raw])


def test_quantizer_cfg_entry_defaults_enable_true():
    """Direct QuantizerCfgEntry construction uses enable=True when omitted."""
    entry = QuantizerCfgEntry(quantizer_name="*")
    assert entry["enable"] is True
    assert entry["cfg"] is None
    assert dict(entry.explicit_items()) == {"quantizer_name": "*"}
    assert entry.model_dump(exclude_unset=True) == {"quantizer_name": "*"}


def test_quantizer_cfg_entry_rejects_empty_name():
    """Direct QuantizerCfgEntry construction rejects empty quantizer names."""
    with pytest.raises(ValidationError, match="non-empty string"):
        QuantizerCfgEntry(quantizer_name="", enable=False)


def test_quantizer_cfg_entry_rejects_empty_cfg_when_enabled():
    """Direct QuantizerCfgEntry construction rejects empty enabled cfg values."""
    with pytest.raises(ValidationError, match="non-empty dict"):
        QuantizerCfgEntry(quantizer_name="*weight_quantizer", cfg={})


def test_quantizer_cfg_entry_treats_empty_disabled_cfg_as_disable_only():
    """Empty cfg with enable=False remains a disable-only entry."""
    entry = QuantizerCfgEntry(quantizer_name="*input_quantizer", cfg={}, enable=False)
    assert entry["cfg"] is None
    assert entry["enable"] is False


class TestNormalizeQuantCfgList:
    def test_new_format_passthrough(self):
        """New-format dict entries are normalized into QuantizerCfgEntry objects."""
        raw = [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}}]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert isinstance(result[0], QuantizerCfgEntry)
        assert result[0]["quantizer_name"] == "*weight_quantizer"
        assert isinstance(result[0]["cfg"], QuantizerAttributeConfig)
        assert _cfg_to_dict(result[0]["cfg"]) == {"num_bits": 8, "axis": 0}
        assert result[0]["enable"] is True  # schema default
        assert "enable" not in dict(result[0].explicit_items())

    def test_typed_entry_list_passthrough(self):
        """Already-parsed QuantizerCfgEntry lists are returned unchanged."""
        raw = [
            QuantizerCfgEntry(
                quantizer_name="*weight_quantizer",
                cfg=QuantizerAttributeConfig(num_bits=8, axis=0),
                enable=True,
            )
        ]
        result = normalize_quant_cfg_list(raw)
        assert result is raw
        assert result[0] is raw[0]

    def test_mixed_typed_and_dict_entries_normalize_to_typed_entries(self):
        """Mixed QuantizerCfgEntry/dict input lists normalize dicts and preserve typed entries."""
        typed_entry = QuantizerCfgEntry(quantizer_name="*", enable=False)
        result = normalize_quant_cfg_list(
            [typed_entry, {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        assert result[0] is typed_entry
        assert isinstance(result[1], QuantizerCfgEntry)
        assert _cfg_to_dict(result[1]["cfg"]) == {"num_bits": 8}
        assert result[1]["enable"] is True
        assert "enable" not in dict(result[1].explicit_items())

    def test_new_format_enable_false(self):
        """Explicit enable=False is preserved."""
        raw = [{"quantizer_name": "*", "enable": False}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None  # defaulted

    def test_new_format_explicit_enable_true_no_cfg(self):
        """Explicit enable=True with no cfg is valid and cfg defaults to None."""
        raw = [{"quantizer_name": "*", "enable": True}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["enable"] is True
        assert result[0]["cfg"] is None

    def test_legacy_single_key_dict(self):
        """Legacy {'*path': {attrs}} is converted to new format."""
        raw = [{"*weight_quantizer": {"num_bits": 8, "axis": 0}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*weight_quantizer"
        assert _cfg_to_dict(result[0]["cfg"]) == {"num_bits": 8, "axis": 0}
        assert result[0]["enable"] is True  # schema default
        assert "enable" not in dict(result[0].explicit_items())

    def test_legacy_single_key_dict_with_enable(self):
        """Legacy {'*path': {'enable': False}} splits enable out from cfg."""
        raw = [{"*input_quantizer": {"enable": False}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*input_quantizer"
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None

    def test_legacy_nn_class_scoped(self):
        """Legacy {'nn.Linear': {'*': {attrs}}} is converted with parent_class."""
        raw = [{"nn.Linear": {"*": {"enable": False}}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["parent_class"] == "nn.Linear"
        assert result[0]["quantizer_name"] == "*"
        assert result[0]["enable"] is False

    def test_normalization_cfg_defaults_to_none(self):
        """Entries without cfg expose the default mapping key but keep it unset."""
        raw = [{"quantizer_name": "*lm_head*", "enable": False}]
        result = normalize_quant_cfg_list(raw)
        assert "cfg" in result[0]
        assert result[0]["cfg"] is None
        assert "cfg" not in dict(result[0].explicit_items())

    def test_normalization_enable_defaults_to_true(self):
        """Entries with cfg but no enable read as enable=True without marking it explicit."""
        raw = [{"quantizer_name": "*", "cfg": {"num_bits": 4}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["enable"] is True
        assert "enable" not in dict(result[0].explicit_items())

    def test_empty_list(self):
        """Empty list is returned unchanged."""
        assert normalize_quant_cfg_list([]) == []

    def test_multiple_entries_order_preserved(self):
        """The order of entries is preserved."""
        raw = [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}},
        ]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*"
        assert result[1]["quantizer_name"] == "*weight_quantizer"

    def test_quantizer_name_only_defaults_enable_true(self):
        """Entry with only quantizer_name uses enable=True from the schema default."""
        result = normalize_quant_cfg_list([{"quantizer_name": "*"}])
        assert result[0]["enable"] is True
        assert result[0]["cfg"] is None
        assert dict(result[0].explicit_items()) == {"quantizer_name": "*"}
        assert result[0].model_dump(exclude_unset=True) == {"quantizer_name": "*"}

    def test_error_on_empty_dict(self):
        """An empty dict entry is rejected."""
        with pytest.raises(ValueError):
            normalize_quant_cfg_list([{}])

    def test_error_on_multi_key_legacy_dict(self):
        """A multi-key legacy dict (no quantizer_name, no nn.* keys) is rejected."""
        with pytest.raises(ValueError):
            normalize_quant_cfg_list([{"*weight_quantizer": {}, "*input_quantizer": {}}])

    def test_error_on_empty_cfg_dict_implicit_enable(self):
        """Entry with cfg={} and implicit enable=True is rejected."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list([{"quantizer_name": "*weight_quantizer", "cfg": {}}])

    def test_error_on_empty_cfg_dict_explicit_enable_true(self):
        """Entry with cfg={} and explicit enable=True is rejected."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": {}, "enable": True}]
            )

    def test_error_on_empty_cfg_list_enable_true(self):
        """Entry with cfg=[] and enable=True is rejected."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": [], "enable": True}]
            )

    def test_error_on_non_dict_non_list_cfg_enable_true(self):
        """Entry with cfg of invalid type (e.g. int) and enable=True is rejected."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": 42, "enable": True}]
            )

    def test_error_on_cfg_list_with_empty_dict_enable_true(self):
        """Entry with cfg=[{}] and enable=True is rejected (empty dict element)."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": [{}], "enable": True}]
            )

    def test_error_on_cfg_list_with_non_dict_element_enable_true(self):
        """Entry with cfg=[42] and enable=True is rejected (non-dict element)."""
        with pytest.raises(ValueError, match="non-empty dict"):
            normalize_quant_cfg_list(
                [{"quantizer_name": "*weight_quantizer", "cfg": [42], "enable": True}]
            )

    def test_empty_cfg_dict_enable_false_accepted(self):
        """Entry with cfg={} and enable=False is allowed (disable-only entry)."""
        result = normalize_quant_cfg_list(
            [{"quantizer_name": "*input_quantizer", "cfg": {}, "enable": False}]
        )
        assert result[0]["enable"] is False

    def test_empty_cfg_list_enable_false_accepted(self):
        """Entry with cfg=[] and enable=False is allowed (disable-only entry)."""
        result = normalize_quant_cfg_list(
            [{"quantizer_name": "*input_quantizer", "cfg": [], "enable": False}]
        )
        assert result[0]["enable"] is False

    def test_new_format_with_list_cfg(self):
        """cfg can be a list of dicts for SequentialQuantizer."""
        raw = [
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": [
                    {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                    {"num_bits": (4, 3)},
                ],
            }
        ]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert isinstance(result[0], QuantizerCfgEntry)
        assert isinstance(result[0]["cfg"], list)
        assert all(isinstance(cfg, QuantizerAttributeConfig) for cfg in result[0]["cfg"])
        assert _cfg_to_dict(result[0]["cfg"]) == raw[0]["cfg"]
        assert result[0]["enable"] is True

    def test_legacy_flat_dict_conversion(self):
        """Legacy flat dict {'*': {...}, '*weight_quantizer': {...}} is converted to list."""
        raw = {"*": {"enable": False}, "*weight_quantizer": {"num_bits": 8, "axis": 0}}
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 2
        assert result[0]["quantizer_name"] == "*"
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None
        assert result[1]["quantizer_name"] == "*weight_quantizer"
        assert _cfg_to_dict(result[1]["cfg"]) == {"num_bits": 8, "axis": 0}
        assert result[1]["enable"] is True
        assert "enable" not in dict(result[1].explicit_items())

    def test_legacy_enable_only_produces_cfg_none(self):
        """Legacy {'*': {'enable': False}} should produce cfg=None, not cfg={}."""
        raw = [{"*": {"enable": False}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["cfg"] is None
        assert result[0]["enable"] is False

    def test_legacy_nn_class_enable_only_produces_cfg_none(self):
        """Legacy nn.* scoped format with only enable produces cfg=None."""
        raw = [{"nn.Linear": {"*": {"enable": False}}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["cfg"] is None
        assert result[0]["enable"] is False
        assert result[0]["parent_class"] == "nn.Linear"

    def test_legacy_default_key(self):
        """Legacy 'default' key is converted to quantizer_name='*'."""
        raw = [{"default": {"enable": False}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*"
        assert result[0]["enable"] is False
        assert result[0]["cfg"] is None

    def test_legacy_default_key_with_cfg(self):
        """Legacy 'default' key with cfg attributes maps to '*'."""
        raw = [{"default": {"num_bits": 8, "axis": None}}]
        result = normalize_quant_cfg_list(raw)
        assert result[0]["quantizer_name"] == "*"
        assert _cfg_to_dict(result[0]["cfg"]) == {"num_bits": 8, "axis": None}
        assert result[0]["enable"] is True

    def test_legacy_flat_dict_with_default_key(self):
        """Legacy flat dict containing 'default' key converts it to '*'."""
        raw = {"default": {"enable": False}, "*weight_quantizer": {"num_bits": 8}}
        result = normalize_quant_cfg_list(raw)
        default_entries = [e for e in result if e["quantizer_name"] == "*"]
        assert len(default_entries) == 1
        assert default_entries[0]["enable"] is False

    def test_legacy_nn_class_multi_key(self):
        """Legacy nn.* scoped format with multiple sub-keys produces multiple entries."""
        raw = [
            {
                "nn.Linear": {
                    "*input_quantizer": {"enable": False},
                    "*weight_quantizer": {"num_bits": 4},
                }
            }
        ]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 2
        paths = {e["quantizer_name"] for e in result}
        assert paths == {"*input_quantizer", "*weight_quantizer"}
        for e in result:
            assert e["parent_class"] == "nn.Linear"

    def test_legacy_nn_class_with_cfg(self):
        """Legacy nn.* scoped format with actual quantizer attributes (not just enable)."""
        raw = [{"nn.Linear": {"*weight_quantizer": {"num_bits": 4, "axis": 0}}}]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert result[0]["parent_class"] == "nn.Linear"
        assert result[0]["quantizer_name"] == "*weight_quantizer"
        assert _cfg_to_dict(result[0]["cfg"]) == {"num_bits": 4, "axis": 0}
        assert result[0]["enable"] is True

    def test_legacy_nn_class_with_list_valued_cfg(self):
        """Legacy nn.* scoped format preserves list-valued SequentialQuantizer cfg."""
        raw = [
            {
                "nn.Linear": {
                    "*weight_quantizer": [
                        {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                        {"num_bits": 8, "axis": 0},
                    ]
                }
            }
        ]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert result[0]["parent_class"] == "nn.Linear"
        assert result[0]["quantizer_name"] == "*weight_quantizer"
        assert isinstance(result[0]["cfg"], list)
        assert _cfg_to_dict(result[0]["cfg"]) == raw[0]["nn.Linear"]["*weight_quantizer"]
        assert result[0]["enable"] is True

    def test_legacy_list_valued_cfg(self):
        """Legacy dict format with list-valued cfg (SequentialQuantizer) normalizes correctly."""
        raw = [
            {
                "*weight_quantizer": [
                    {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                    {"num_bits": 8, "axis": 0},
                ]
            }
        ]
        result = normalize_quant_cfg_list(raw)
        assert len(result) == 1
        assert result[0]["quantizer_name"] == "*weight_quantizer"
        assert isinstance(result[0]["cfg"], list)
        assert len(result[0]["cfg"]) == 2
        assert result[0]["cfg"][0]["num_bits"] == 4
        assert result[0]["cfg"][1]["num_bits"] == 8
        assert result[0]["enable"] is True


class TestFindQuantCfgEntry:
    def test_finds_last_match(self):
        """When multiple entries share the same quantizer_name, returns the last one."""
        entries = normalize_quant_cfg_list(
            [
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}},
                {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 4}},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4}},
            ]
        )
        result = find_quant_cfg_entry_by_path(entries, "*weight_quantizer")
        assert _cfg_to_dict(result["cfg"]) == {"num_bits": 4}

    def test_exact_match_only(self):
        """Does not do fnmatch — only exact string equality on quantizer_name."""
        entries = normalize_quant_cfg_list(
            [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        with pytest.raises(KeyError):
            find_quant_cfg_entry_by_path(entries, "model.layer.weight_quantizer")

    def test_raises_on_missing(self):
        """Raises KeyError when no entry matches."""
        entries = normalize_quant_cfg_list(
            [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        with pytest.raises(KeyError):
            find_quant_cfg_entry_by_path(entries, "*input_quantizer")

    def test_single_entry(self):
        entries = normalize_quant_cfg_list([{"quantizer_name": "*", "enable": False}])
        result = find_quant_cfg_entry_by_path(entries, "*")
        assert result["enable"] is False

    def test_empty_list(self):
        with pytest.raises(KeyError):
            find_quant_cfg_entry_by_path([], "*")


def test_need_calibration_with_legacy_dict_format():
    """need_calibration should accept legacy dict-format quant_cfg without crashing."""
    legacy_config = {
        "quant_cfg": {"*input_quantizer": {"num_bits": 8, "axis": None}},
        "algorithm": "max",
    }
    assert need_calibration(legacy_config)


def test_need_calibration_with_legacy_list_of_single_key_dicts():
    """need_calibration should accept legacy list-of-single-key-dicts format."""
    legacy_config = {
        "quant_cfg": [{"*input_quantizer": {"num_bits": 8, "axis": None}}],
        "algorithm": "max",
    }
    assert need_calibration(legacy_config)


class TestMatchQuantizerCfg:
    """Tests for _match_quantizer_cfg in algorithms.py."""

    def test_wildcard_matches_bare_name(self):
        """'*weight_quantizer' matches bare 'weight_quantizer'."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        matched, enable = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert _cfg_to_dict(matched) == {"num_bits": 8}
        assert enable is True

    def test_star_matches_any_bare_name(self):
        """'*' matches any bare quantizer name."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list([{"quantizer_name": "*", "enable": False}])
        matched, enable = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert matched is None  # enable-only entry has cfg=None
        assert enable is False

    def test_path_scoped_pattern_matches_matching_suffix(self):
        """'*mlp*weight_quantizer' matches bare 'weight_quantizer' (suffix match)."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*mlp*weight_quantizer", "cfg": {"num_bits": 4}}]
        )
        matched, enable = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert _cfg_to_dict(matched) == {"num_bits": 4}

    def test_path_scoped_pattern_does_not_match_different_suffix(self):
        """'*mlp*weight_quantizer' does NOT match bare 'input_quantizer'."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*mlp*weight_quantizer", "cfg": {"num_bits": 4}}]
        )
        matched, enable = _match_quantizer_cfg(quant_cfg, "input_quantizer")
        assert matched is None
        assert enable is None

    def test_last_match_wins(self):
        """Later entries override earlier ones."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 4}},
            ]
        )
        matched, _ = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert _cfg_to_dict(matched) == {"num_bits": 4}

    def test_no_match_returns_none(self):
        """No matching entry returns (None, None)."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8}}]
        )
        matched, enable = _match_quantizer_cfg(quant_cfg, "output_quantizer")
        assert matched is None
        assert enable is None

    def test_bracket_pattern_matches_correctly(self):
        """'*[kv]_bmm_quantizer' matches 'k_bmm_quantizer' and 'v_bmm_quantizer'."""
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [{"quantizer_name": "*[kv]_bmm_quantizer", "cfg": {"num_bits": (4, 3)}}]
        )
        matched_k, _ = _match_quantizer_cfg(quant_cfg, "k_bmm_quantizer")
        matched_v, _ = _match_quantizer_cfg(quant_cfg, "v_bmm_quantizer")
        matched_w, _ = _match_quantizer_cfg(quant_cfg, "weight_quantizer")
        assert matched_k is not None
        assert matched_v is not None
        assert matched_w is None

    def test_path_scoped_does_not_overmatch(self):
        """'*mixer*weight_quantizer' should NOT match 'input_quantizer'.

        Regression test: the old rsplit('*') logic would strip to 'weight_quantizer' and
        overmatch any quantizer ending in 'weight_quantizer', but should not match unrelated names.
        """
        from modelopt.torch.quantization.algorithms import _match_quantizer_cfg

        quant_cfg = normalize_quant_cfg_list(
            [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*mixer*weight_quantizer", "cfg": {"num_bits": 4}},
            ]
        )
        # input_quantizer should only match the disable-all, not the mixer pattern
        matched, enable = _match_quantizer_cfg(quant_cfg, "input_quantizer")
        assert matched is None  # cfg is None (enable-only entry)
        assert enable is False


class TestQuantizeConfigValidators:
    """Tests for QuantizeConfig Pydantic field validators."""

    def test_normalize_validator_converts_legacy_dict(self):
        """The 'before' validator auto-normalizes legacy dict format."""
        cfg = QuantizeConfig(
            quant_cfg={"*": {"enable": False}, "*weight_quantizer": {"num_bits": 8}},
            algorithm="max",
        )
        assert isinstance(cfg.quant_cfg, list)
        assert all("quantizer_name" in e for e in cfg.quant_cfg)

    def test_validate_quant_cfg_entries_catches_invalid_cfg(self):
        """The 'after' validator surfaces QuantizerAttributeConfig errors early."""
        with pytest.raises(ValidationError):
            QuantizeConfig(
                quant_cfg=[
                    {
                        "quantizer_name": "*weight_quantizer",
                        "cfg": {"num_bits": 8, "axis": 0, "block_sizes": {-1: 128}},
                    }
                ],
                algorithm="max",
            )

    def test_validate_quant_cfg_entries_accepts_valid_cfg(self):
        """The 'after' validator passes for valid configs."""
        cfg = QuantizeConfig(
            quant_cfg=[
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
                {"quantizer_name": "*input_quantizer", "enable": False},
            ],
            algorithm="max",
        )
        assert len(cfg.quant_cfg) == 2

    def test_quant_cfg_parses_dict_cfg_to_pydantic_type(self):
        """Python dict cfg input is accepted and parsed to QuantizerAttributeConfig."""
        cfg = QuantizeConfig(
            quant_cfg=[
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            ],
            algorithm="max",
        )
        attr_cfg = cfg.quant_cfg[0]["cfg"]
        assert isinstance(attr_cfg, QuantizerAttributeConfig)
        assert attr_cfg.model_dump(exclude_unset=True) == {"num_bits": 8, "axis": 0}

    def test_quant_cfg_parses_list_of_dict_cfg_to_pydantic_type(self):
        """Python list-of-dict cfg input is accepted and parsed to QuantizerAttributeConfig."""
        cfg = QuantizeConfig(
            quant_cfg=[
                {
                    "quantizer_name": "*weight_quantizer",
                    "cfg": [
                        {"num_bits": 4, "block_sizes": {-1: 128, "type": "static"}},
                        {"num_bits": 8, "axis": 0},
                    ],
                },
            ],
            algorithm="max",
        )
        attr_cfgs = cfg.quant_cfg[0]["cfg"]
        assert isinstance(attr_cfgs, list)
        assert all(isinstance(attr_cfg, QuantizerAttributeConfig) for attr_cfg in attr_cfgs)
        assert attr_cfgs[0].num_bits == 4
        assert attr_cfgs[1].axis == 0

    def test_quant_cfg_accepts_pydantic_cfg_instances(self):
        """Already-parsed QuantizerAttributeConfig input remains valid."""
        attr_cfg = QuantizerAttributeConfig(num_bits=8, axis=0)
        cfg = QuantizeConfig(
            quant_cfg=[{"quantizer_name": "*weight_quantizer", "cfg": attr_cfg}],
            algorithm="max",
        )
        assert cfg.quant_cfg[0]["cfg"] == attr_cfg
