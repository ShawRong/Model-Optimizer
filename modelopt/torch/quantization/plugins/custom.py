# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom plugin base modules and utilities for quantization."""

import os
import warnings
from collections.abc import Callable, Iterator
from functools import partial
from types import ModuleType

import torch

from modelopt.torch.utils.distributed import ParallelState

from ..nn import NVFP4StaticQuantizer, QuantModule, SequentialQuantizer, TensorQuantizer
from ..nn.modules.quant_linear import _QuantLinear
from ..utils import multi_context, replace_function

CUSTOM_MODEL_PLUGINS = set()
CUSTOM_POST_CONVERSION_PLUGINS = set()


# TODO: This is a temporary solution
# In future implement a decorator to register methods updating QUANT_MODULE on the fly
def register_custom_model_plugins_on_the_fly(model):
    """Registers custom modules as QUANT_MODULE on the fly."""
    for callback in CUSTOM_MODEL_PLUGINS:
        callback(model)


def register_custom_post_conversion_plugins(model):
    """Registers custom modules as QUANT_MODULE after conversion."""
    for callback in CUSTOM_POST_CONVERSION_PLUGINS:
        callback(model)


class _QuantFunctionalMixin(QuantModule):
    """Mixin class for quantized functionals.

    Often we need to replace a functional with a quantized version. This class provides a way to do that.
    """

    # List of functionals to replace with quantized versions, e.g. [(package, func_name, quantized_func), ...]
    _functionals_to_replace: list[tuple[ModuleType, str, Callable]] = []

    @property
    def functionals_to_replace(self) -> Iterator[tuple[ModuleType, str, Callable]]:
        return (
            (package, func_name, quantized_func)
            for package, func_name, quantized_func in getattr(self, "_functionals_to_replace", [])
            if hasattr(package, func_name)
        )

    def forward(self, *args, **kwargs):
        with multi_context(
            *(
                replace_function(package, func_name, quantized_func)
                for package, func_name, quantized_func in self.functionals_to_replace
            )
        ):
            return super().forward(*args, **kwargs)


class _ParallelLinear(_QuantFunctionalMixin, QuantModule):
    """Quantized base class for ParallelLinear type classes.

    For this type of modules, we need to quantize the inputs and weights just before calling the actual linear
    functional. This is accomplished by replacing the linear functional with a custom one that quantizes the inputs
    and weights before calling the original functional.
    """

    # List of functionals to replace [(package, func_name), ...]
    _functionals_to_replace: list[tuple[ModuleType, str]] = []

    _parallel_state: ParallelState
    _is_column_parallel = False
    _is_row_parallel = False
    _quantized_linear_fn: Callable = _QuantLinear.quantized_linear_fn

    @property
    def functionals_to_replace(self) -> Iterator[tuple[ModuleType, str, Callable]]:
        for package, func_name in self._functionals_to_replace:
            if not hasattr(package, func_name):
                continue
            quantized_func = partial(
                self.__class__._quantized_linear_fn,
                package,
                "_" + func_name,
                self,
            )
            if hasattr(getattr(package, func_name), "__dict__"):
                quantized_func.__dict__.update(getattr(package, func_name).__dict__)
            yield package, func_name, quantized_func

    def _setup(self):
        self.input_quantizer = TensorQuantizer(_QuantLinear.default_quant_desc_input)
        self.weight_quantizer = TensorQuantizer(_QuantLinear.default_quant_desc_weight)
        self.output_quantizer = TensorQuantizer(_QuantLinear.default_quant_desc_output)
        self.output_quantizer.disable()

        # Memorize the original weight.dtype for modelopt_post_restore given that
        # the dtype can change later.
        self.original_weight_dtype = None if self.weight is None else self.weight.dtype

    def modelopt_post_restore(self, prefix: str = ""):
        """Post restore to correctly configure the TensorQuantizer states for MCore/distributed frameworks.

        ModelOpt restores the TensorQuantizer states such as `_amax` and `_pre_quant_scale` to their
        shape before saving. However this is not enough for MCore/distributed frameworks since the tensor parallelism
        could change between saving and restoring. If the tensor parallelism changes, the shape of the quantizer
        states also changes. So we need to re-calculate the quantizer states.
        """
        from modelopt.torch.quantization.model_calib import max_calibrate

        def _check_unsupported_states(quantizer: TensorQuantizer):
            for k in quantizer.state_dict():
                if k not in ["_amax", "_pre_quant_scale", "_global_amax"]:
                    warnings.warn(
                        f"Restore of {k} for {prefix} is not supported. The restore of this layer might be "
                        f"incorrect. Please implement a custom restore for {k}."
                    )

        def _has_state(quantizer, name):
            # Handling for SequentialQuantizer
            quantizer = quantizer[0] if isinstance(quantizer, SequentialQuantizer) else quantizer
            return hasattr(quantizer, name)

        def _has_complete_static_nvfp4_weight_state(quantizer, weight):
            quantizer = quantizer[0] if isinstance(quantizer, SequentialQuantizer) else quantizer
            if not isinstance(quantizer, NVFP4StaticQuantizer):
                return False
            amax = getattr(quantizer, "_amax", None)
            global_amax = getattr(quantizer, "global_amax", None)
            if amax is None or global_amax is None:
                return False
            block_sizes = getattr(quantizer, "block_sizes", None)
            block_size = block_sizes.get(-1) if isinstance(block_sizes, dict) else None
            if block_size is None or weight.shape[-1] % block_size != 0:
                return False
            expected_blocks = weight.numel() // block_size
            return amax.numel() == expected_blocks and global_amax.numel() == 1

        def _rank_for_restore_log():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank()
            return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))

        def _static_nvfp4_weight_state_details(quantizer, weight):
            quantizer = quantizer[0] if isinstance(quantizer, SequentialQuantizer) else quantizer
            if not isinstance(quantizer, NVFP4StaticQuantizer):
                return None
            amax = getattr(quantizer, "_amax", None)
            global_amax = getattr(quantizer, "global_amax", None)
            block_sizes = getattr(quantizer, "block_sizes", None)
            block_size = block_sizes.get(-1) if isinstance(block_sizes, dict) else None
            expected_blocks = (
                weight.numel() // block_size
                if block_size and weight.shape[-1] % block_size == 0
                else None
            )
            return {
                "block_size": block_size,
                "amax_numel": None if amax is None else amax.numel(),
                "expected_blocks": expected_blocks,
                "global_amax_numel": None if global_amax is None else global_amax.numel(),
                "weight_shape": tuple(weight.shape),
            }

        def _log_static_nvfp4_restore_decision(action, details, *, always=False):
            if details is None:
                return
            rank = _rank_for_restore_log()
            count = getattr(_ParallelLinear.modelopt_post_restore, "_static_nvfp4_log_count", 0)
            if not always and count >= int(
                os.environ.get("MODELOPT_STATIC_NVFP4_RESTORE_LOG_LIMIT", "8")
            ):
                return
            _ParallelLinear.modelopt_post_restore._static_nvfp4_log_count = count + 1
            print(
                "[ModelOpt][static-nvfp4-restore] "
                f"rank={rank} prefix={prefix} action={action} "
                f"weight_shape={details['weight_shape']} block_size={details['block_size']} "
                f"amax_numel={details['amax_numel']} expected_blocks={details['expected_blocks']} "
                f"global_amax_numel={details['global_amax_numel']}",
                flush=True,
            )

        if self.weight is None:
            return

        for quantizer in [self.weight_quantizer, self.input_quantizer, self.output_quantizer]:
            _check_unsupported_states(
                quantizer if isinstance(quantizer, TensorQuantizer) else quantizer[0]
            )
        # Skip max_calibrate when saved static NVFP4 state is intact; else MSE scales get overwritten.
        static_nvfp4_details = _static_nvfp4_weight_state_details(
            self.weight_quantizer, self.weight
        )
        has_complete_static_nvfp4_weight_state = _has_complete_static_nvfp4_weight_state(
            self.weight_quantizer, self.weight
        )
        if static_nvfp4_details is not None and has_complete_static_nvfp4_weight_state:
            _log_static_nvfp4_restore_decision("skip_weight_max_calibrate", static_nvfp4_details)
        if (
            _has_state(self.weight_quantizer, "_amax")
            and not has_complete_static_nvfp4_weight_state
        ):
            if static_nvfp4_details is not None:
                _log_static_nvfp4_restore_decision(
                    "run_weight_max_calibrate", static_nvfp4_details, always=True
                )
                warnings.warn(
                    f"Static NVFP4 weight quantizer for {prefix} has incomplete restore state; "
                    "running max_calibrate on weight_quantizer."
                )
            self.weight_quantizer.reset_amax()
            max_calibrate(self.weight_quantizer, lambda wq: wq(self.weight), distributed_sync=False)
        if _has_state(self.input_quantizer, "_pre_quant_scale"):
            if hasattr(self.input_quantizer, "_pre_quant_scale"):
                delattr(self.input_quantizer, "_pre_quant_scale")
            pqs = torch.zeros(
                (self.weight.shape[1]), device=self.weight.device, dtype=self.original_weight_dtype
            )
            self.input_quantizer.register_buffer("_pre_quant_scale", pqs)
        if _has_state(self.input_quantizer, "_amax"):
            self.input_quantizer.reset_amax()
            dummy_input = torch.ones(
                (1, 1, self.weight.shape[1]),
                device=self.weight.device,
                dtype=self.original_weight_dtype,
            )
            max_calibrate(self.input_quantizer, lambda iq: iq(dummy_input), distributed_sync=False)
        if _has_state(self.output_quantizer, "_amax"):
            self.output_quantizer.reset_amax()
            dummy_input = torch.ones(
                (1, 1, self.weight.shape[0]),
                device=self.weight.device,
                dtype=self.original_weight_dtype,
            )
            max_calibrate(self.output_quantizer, lambda oq: oq(dummy_input), distributed_sync=False)
        # If there are any other states, lets move them to the correct device
        super().modelopt_post_restore(prefix=prefix)
