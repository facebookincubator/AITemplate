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
# import torch
# import numpy as np


# from aitemplate.frontend import IntVar, Tensor
# from aitemplate.compiler import ops
# from aitemplate.frontend import nn
# from aitemplate.testing import compile_model, detect_target


# def test_fp16(batch_size=[4, 32, 48]):
#     target = detect_target()
#     X = Tensor(
#         shape=[
#             IntVar(values=batch_size, name="input_batch"),
#             28,
#             28,
#             128
#         ],
#         dtype="float16",
#         name="input_0"
#     )
#     W = Tensor(
#         shape=[
#             256,
#             3,
#             3,
#             128
#         ],
#         dtype="float16",
#         name="input_1"
#     )
#     OP = ops.conv2d(stride=1, pad=1, dilate=1)
#     Y = OP(X, W)
#     Y._attrs["name"] = "output_0"
#     Y._attrs["is_output"] = True
#     module = compile_model(Y, target, "./tmp", "dynamic_conv", dynamic_batch=True)
#     for batch in range(batch_size[0], batch_size[-1] + 1):
#         print("Test batch: %d" % batch)
#         X_pt = torch.randn(batch, 128, 28, 28).cuda().half()
#         W_pt = torch.randn(256, 128, 3, 3).cuda().half()
#         Y_pt = torch.nn.functional.conv2d(X_pt, W_pt,  padding=1)
#         Y_np = Y_pt.cpu().numpy()
#         module.SetDim("input_batch", batch)
#         x = np.transpose(X_pt.cpu().numpy(), (0, 2, 3, 1)).copy()
#         w = np.transpose(W_pt.cpu().numpy(), (0, 2, 3, 1)).copy()
#         module.SetInput("input_0", x)
#         module.SetInput("input_1", w)
#         module.benchmark()
#         y = module.GetOutput("output_0", [batch, 28, 28, 256])
#         np.testing.assert_allclose(Y_np,
#                                    np.transpose(y, (0, 3, 1, 2)),
#                                    atol=1e-2, rtol=1e-2)


# test_fp16()
