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
import torch
from fx2ait.acc_tracer import acc_tracer
from fx2ait.ait_splitter import (  # @manual=//aitemplate/AITemplate/fx2ait/fx2ait:fx2ait
    AITSplitter,
    AITSplitterSettings,
    create_ait_operator_support,
)
from fx2ait.tools.common_fx2ait import AITTestCase
from torch.fx.passes import operator_support as op_support


class TestSplit(AITTestCase):
    def test_exclude_support_node_by_name(self):
        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.sin(a)
                c = torch.relu(b)
                d = torch.cos(c)
                e = torch.sigmoid(d)
                f = torch.tanh(e)
                return f

        # Support all ops
        _support_dict = {
            "acc_ops.sin": None,
            "acc_ops.cos": None,
            "acc_ops.relu": None,
            "acc_ops.sigmoid": None,
            "acc_ops.tanh": None,
        }
        custom_op_support = op_support.OperatorSupport(_support_dict)

        # With no ops excluded, the entire module should be lowered
        # into one acc graph
        mod = acc_tracer.trace(TestModule(), [torch.randn(2, 3)])
        settings = AITSplitterSettings(min_acc_module_size=0)
        splitter = AITSplitter(
            mod,
            (torch.randn(2, 3),),
            custom_op_support,
            settings,
        )

        res_no_exclusion = splitter.generate_split_results()
        split_named_mods = dict(res_no_exclusion.split_module.named_children())
        self.assertEqual(len(split_named_mods), 1)
        self.assertIn("_run_on_acc_0", split_named_mods)

        # Add "relu" to exclude_support_node_name
        # The graph should be split into 3 parts now(_run_on_acc_0, _run_on_gpu_1, _run_on_acc_2)
        mod = acc_tracer.trace(TestModule(), [torch.randn(2, 3)])
        settings.exclude_support_node_name.add("relu_1")
        splitter = AITSplitter(
            mod,
            (torch.randn(2, 3),),
            custom_op_support,
            settings,
        )
        res_post_exclusion = splitter.generate_split_results()
        split_named_mods = dict(res_post_exclusion.split_module.named_children())
        self.assertEqual(len(split_named_mods), 3)
        self.assertIn("_run_on_acc_0", split_named_mods)
        self.assertIn("_run_on_gpu_1", split_named_mods)
        self.assertIn("_run_on_acc_2", split_named_mods)

        run_on_acc_0_nodes = [
            n
            for n in split_named_mods["_run_on_acc_0"].graph.nodes
            if n.op == "call_function"
        ]
        self.assertEqual(len(run_on_acc_0_nodes), 1)
        self.assertEqual(acc_tracer.acc_ops.sin, run_on_acc_0_nodes[0].target)

        run_on_gpu_1_nodes = [
            n
            for n in split_named_mods["_run_on_gpu_1"].graph.nodes
            if n.op == "call_function"
        ]
        self.assertEqual(len(run_on_gpu_1_nodes), 1)
        self.assertEqual(acc_tracer.acc_ops.relu, run_on_gpu_1_nodes[0].target)

        run_on_acc_2_nodes = [
            n
            for n in split_named_mods["_run_on_acc_2"].graph.nodes
            if n.op == "call_function"
        ]
        self.assertEqual(len(run_on_acc_2_nodes), 3)
        self.assertEqual(acc_tracer.acc_ops.cos, run_on_acc_2_nodes[0].target)
        self.assertEqual(acc_tracer.acc_ops.sigmoid, run_on_acc_2_nodes[1].target)
        self.assertEqual(acc_tracer.acc_ops.tanh, run_on_acc_2_nodes[2].target)

    def test_decline_if_input_dtype(self):
        operator_support = create_ait_operator_support()

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.relu(a)
                return b

        test_mod = TestModule().cuda().eval()
        x = torch.randn(2, 3)
        mod = acc_tracer.trace(test_mod, [x])
        settings = AITSplitterSettings()
        settings.min_acc_module_size = 0
        # nodes w/ float16 input should be lowered
        splitter = AITSplitter(
            mod,
            (x.half().cuda(),),
            operator_support,
            settings,
        )
        split_results_half = splitter.generate_split_results()
        self.assertTrue(len(split_results_half), 1)
        self.assertEqual(
            dict(split_results_half.split_module.named_children()).keys(),
            {"_run_on_acc_0"},
        )

        # nodes w/ float64 input should not be lowered
        mod = acc_tracer.trace(test_mod, [x])
        splitter = AITSplitter(
            mod,
            (x.double().cuda(),),
            operator_support,
            settings,
        )

        split_results_double = splitter.generate_split_results()

        self.assertTrue(len(split_results_double), 1)
        self.assertEqual(
            dict(split_results_double.split_module.named_children()).keys(),
            {"_run_on_gpu_0"},
        )
