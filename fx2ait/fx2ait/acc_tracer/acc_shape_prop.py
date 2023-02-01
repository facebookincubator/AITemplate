#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import sys
from typing import Any

import torch.fx
from torch.fx.passes import shape_prop

from . import acc_ops


class SuppressStderrPrints:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr


class AccShapeProp(shape_prop.ShapeProp):
    """
    Similar to standard shape prop, but if any node that is run with standard shape prop
    fails then it tries to upconvert any fp16 inputs to fp32, rerun shape prop, and then
    downconvert fp32 results back to fp16.

    Note that we currently mostly only look for/support up/down conversion for nodes
    with tensor outputs, but this is likely fine for most cases. Additionally the base
    shape_prop works for many ops with fp16, such as tensor.cat, tensor slice, tensor.to
    dtype conversion, etc.

    """

    def _run_node(self, n: torch.fx.Node) -> Any:
        # Run ops with XL weights by clamping their inputs, see
        # docstring for self.run_node_with_xl_weights for more details
        if any(
            isinstance(kwarg, torch.fx.Node) and kwarg.target == acc_ops.xl_weight
            for kwarg in n.kwargs.values()
        ):
            return self.run_node_with_xl_weights(n)
        else:
            return super().run_node(n)

    def run_node(self, n: torch.fx.Node) -> Any:
        # First try running shape_prop with the original inputs.
        with SuppressStderrPrints():
            try:
                return self._run_node(n)
            except Exception:
                pass

        # Base shape_prop failed, so temporarily upconvert the node's fp16 inputs in env
        # and retry. For now just support upconverting Tensor outputs.
        orig_dtype_env = []
        for in_node in n.all_input_nodes:
            in_ten = self.env[in_node]
            if isinstance(in_ten, torch.Tensor) and in_ten.dtype == torch.float16:
                orig_dtype_env.append((in_node, in_ten))
                self.env[in_node] = in_ten.clone().to(dtype=torch.float)

        # Now try running again with upconverted fp32 input tensor in env.
        result = self._run_node(n)

        # Now that we succeeded, assume it's thanks to upconverting. Therefore we
        # downconvert fp32 tensor results to fp16.
        if isinstance(result, torch.Tensor) and result.dtype == torch.float:
            result = result.to(dtype=torch.float16)
            self.env[n] = result
            n.meta["tensor_meta"] = n.meta["tensor_meta"]._replace(dtype=torch.float16)

        # Finally, restore the original env back to fp16 for any upconverted tensors.
        for in_node, in_ten in orig_dtype_env:
            self.env[in_node] = in_ten

        return result

    def run_node_with_xl_weights(self, n: torch.fx.Node) -> Any:
        """
        EmbeddingBag with XL Weights of shape (num_embeddings, embedding_dim)
        are replaced with smaller proxies of shape
        (acc_ops.PROXY_EMBEDDING_SIZE, embedding_dim) during tracing. This can
        cause index out of bounds issues when sample inputs lead to the
        embedding bag op indexing into the first dimension of the weight tensor
        which it expects to be bigger than it is during tracing.

        For these ops, return a zeros tensor of the correct shape and dtype.
        """

        op = n.target.__module__ + "." + n.target.__name__

        if op.endswith("acc_ops.int_nbit_split_embedding_codegen_lookup_function"):
            output_dtype_int = n.kwargs["output_dtype"]
            assert output_dtype_int < 2, "only support float16 and float32"
            output_dtype = torch.float if output_dtype_int == 0 else torch.float16
            total_D = n.kwargs["total_D"]

            D_offsets_shape = self.env[n.kwargs["D_offsets"]].shape
            offsets_shape = self.env[n.kwargs["offsets"]].shape
            batches = (offsets_shape[0] - 1) // (D_offsets_shape[0] - 1)
            result = torch.zeros((batches, total_D), dtype=output_dtype)

        elif op.find("acc_ops.embedding_bag"):
            weight = self.env[n.kwargs["weight"]]
            offsets_shape = self.env[n.kwargs["offsets"]].shape
            batches = offsets_shape[0] - int(n.kwargs["include_last_offset"])
            output_dtype = weight.dtype

            embedding_size = weight.shape[1]
            if op.endswith("acc_ops.embedding_bag_byte_rowwise_offsets"):
                embedding_size -= 8
                output_dtype = torch.float32
            elif op.endswith("acc_ops.embedding_bag_4bit_rowwise_offsets"):
                embedding_size = (embedding_size - 4) * 2
                output_dtype = torch.float32

            result = torch.zeros((batches, embedding_size), dtype=output_dtype)

        else:
            raise NotImplementedError(
                f"The op {op} cannot be run with xl_weight(s) inputs"
            )

        return result
