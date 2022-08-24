# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
[summary] a python module to run compiled model
"""
import ctypes
import logging
import math
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union

import numpy as np

from aitemplate.utils.torch_utils import torch_dtype_to_string

# pylint: disable=C0103

DTYPE_TO_BYTES: Dict[str, str] = {
    "float16": 2,
    "float32": 4,
    "float": 4,
    "int": 4,
    "int32": 4,
    "int64": 8,
}


# Stand-in for torch.Tensor. Use a TypeVar for some APIs since we can't introduce
# a torch dependency.
TorchTensor = TypeVar("TorchTensor")


def _dlclose(dll: ctypes.CDLL):
    syms = ctypes.CDLL(None)
    if hasattr(syms, "dlclose"):
        f_dlclose = syms.dlclose
        f_dlclose.argtypes = [ctypes.c_void_p]
        f_dlclose(dll._handle)
    else:
        logging.warning("dlclose() not found, library may not be unloaded properly!")


def _check_tensors(
    tensor_list: Union[Dict[str, TorchTensor], List[TorchTensor]],
    is_error_fn: Callable[[TorchTensor], bool],
    list_name: str,
    condition_description: str,
):
    """
    Helper for various input/output sanity checks.
    """
    if isinstance(tensor_list, dict):
        tensor_list = tensor_list.values()

    for i, tensor in enumerate(tensor_list):
        if is_error_fn(tensor):
            raise ValueError(f"{list_name}[{i}] failed check: {condition_description}")


def _check_tensors_contiguous_and_on_gpu(
    tensors: Union[Dict[str, TorchTensor], List[TorchTensor]], name: str
):
    def is_bad_tensor(tensor: TorchTensor) -> bool:
        return not tensor.is_contiguous() or not tensor.is_cuda

    _check_tensors(tensors, is_bad_tensor, name, "contiguous and on GPU")


def _check_tensors_contiguous_and_on_host(
    tensors: Union[Dict[str, TorchTensor], List[TorchTensor]], name: str
):
    def is_bad_tensor(tensor: TorchTensor) -> bool:
        return not tensor.is_contiguous() or tensor.is_cuda

    _check_tensors(tensors, is_bad_tensor, name, "contiguous and on host")


def torch_to_tensor_info(tensor):
    """
    Convert a torch Tensor to a AITemplateTensor.
    """
    return AITemplateTensor(
        tensor.data_ptr(), list(tensor.size()), torch_dtype_to_string(tensor.dtype)
    )


def _convert_tensor_args(params):
    """
    Helper function for the WithTensors APIs.
    """
    if isinstance(params, dict):
        result = {name: torch_to_tensor_info(x) for name, x in params.items()}
    else:
        result = [torch_to_tensor_info(x) for x in params]
    return result


def _reshape_tensor(tensor: TorchTensor, shape: List[int]) -> TorchTensor:
    """
    Reinterpret a blob of contiguous memory as some shape. Used to convert
    outputs in RunWithTensors.
    """
    assert tensor.ndim == len(
        shape
    ), f"Expected output tensor's ndim to match the length of Run()'s return value: {tensor.ndim=} != {len(shape)=}"
    numel = math.prod(shape)
    new_tensor = tensor.flatten()[:numel]
    return new_tensor.reshape(shape)


class AITemplateTensor(NamedTuple):
    """
    Input or output tensor for Model.Run. We require the extra data for safety
    checks inside the runtime.
    """

    data_ptr: int
    shape: List[int]
    dtype: str


class Model(object):
    class _DLLWrapper:
        def __init__(self, lib_path: str, num_runtimes: int):
            self.lib_path = lib_path
            self.DLL = ctypes.cdll.LoadLibrary(lib_path)

            self.handle = ctypes.c_void_p()
            self.DLL.AITemplateModelContainerCreate(
                ctypes.pointer(self.handle), ctypes.c_size_t(num_runtimes)
            )
            self.is_open = True

        def close(self):
            if self.is_open:
                self.DLL.AITemplateModelContainerDelete(self.handle)
                _dlclose(self.DLL)
                self.is_open = False

        def __getattr__(self, name):
            if not self.is_open:
                raise RuntimeError(f"Cannot use closed AIT library: {self.lib_path}")

            method = getattr(self.DLL, name)

            def _wrapped_func(*args):
                err = method(*args)
                if err:
                    raise RuntimeError(f"Error in function: {method.__name__}")

            return _wrapped_func

    def __init__(self, lib_path: str, num_runtimes: int = 2):
        """
        Instantiates a wrapper around the C++ model_interface.

        Parameters
        ----------
        lib_path : str
            The path to the compiled .so
        num_runtimes : int, optional
            How many runtimes should be stored in the internal pool. This
            determines how many inferences can happen concurrently. By
            default, set to 2. Must be positive.
        """
        if num_runtimes <= 0:
            raise ValueError(f"num_runtimes must be positive, but got {num_runtimes}")

        self.DLL = self._DLLWrapper(lib_path, num_runtimes)
        self.handle = self.DLL.handle
        self.lib_path = self.DLL.lib_path

        # The corresponding sorted_graph. Optional. For debugging purpose.
        self.debug_sorted_graph = None

        # Maps dtype strings to AITemplateDtype enum in model_interface.h.
        # Must be kept in sync!
        # We can consider defining an AITemplateDtype enum to use on the Python
        # side at some point, but stick to strings for now to keep things consistent
        # with other Python APIs.
        self._DTYPE_TO_ENUM = {
            "float16": 1,
            "float32": 2,
            "float": 2,
            "int": 3,
            "int32": 3,
            "int64": 4,
        }
        self._output_name_to_index = self._construct_output_name_to_index_map()
        self._input_name_to_index = self._construct_input_name_to_index_map()
        self._output_ndims = [
            len(self.GetOutputMaximumShape(i))
            for i in range(len(self._output_name_to_index))
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.DLL.close()

    def __getstate__(self):
        return {"lib_path": self.DLL.lib_path}

    def __setstate__(self, d):
        if "lib_path" not in d:
            raise RuntimeError(f"Didn't find 'lib_path' property in {d}")
        self.__init__(d["lib_path"])

    def _dtype_str_to_enum(self, dtype: str) -> int:
        if dtype not in self._DTYPE_TO_ENUM:
            raise ValueError(
                f"Got unsupported input dtype {dtype}! Supported dtypes are: {list(self._DTYPE_TO_ENUM.keys())}"
            )
        return self._DTYPE_TO_ENUM[dtype]

    def _convert_params_to_c_format(self, params: List[AITemplateTensor]):
        class Shape(ctypes.Structure):
            _fields_ = [
                ("shape_data", ctypes.POINTER(ctypes.c_longlong)),
                ("size", ctypes.c_size_t),
            ]

        class AITemplateTensor(ctypes.Structure):
            _fields_ = [
                ("pointer", ctypes.c_void_p),
                ("shape", Shape),
                ("dtype", ctypes.c_int),
            ]

        c_params = (AITemplateTensor * len(params))()
        for i, (pointer, shape, dtype) in enumerate(params):
            c_pointer = ctypes.c_void_p(pointer)
            c_shape_data = (ctypes.c_longlong * len(shape))()
            for j, dim in enumerate(shape):
                c_shape_data[j] = ctypes.c_longlong(dim)
            c_shape = Shape(c_shape_data, ctypes.c_size_t(len(shape)))
            c_dtype = self._dtype_str_to_enum(dtype)
            c_params[i] = AITemplateTensor(c_pointer, c_shape, c_dtype)

        return c_params

    def _prepare_run(
        self,
        inputs,
        outputs,
        stream_ptr,
    ):
        c_inputs = self._convert_params_to_c_format(inputs)
        c_outputs = self._convert_params_to_c_format(outputs)
        c_stream = (
            ctypes.c_void_p() if stream_ptr is None else ctypes.c_void_p(stream_ptr)
        )

        num_outputs = len(self._output_ndims)
        c_output_shapes_out = (ctypes.POINTER(ctypes.c_int64) * num_outputs)()
        for i in range(num_outputs):
            c_output_shapes_out[i] = ctypes.cast(
                (ctypes.c_int64 * self._output_ndims[i])(),
                ctypes.POINTER(ctypes.c_int64),
            )

        return (
            c_inputs,
            c_outputs,
            c_stream,
            c_output_shapes_out,
        )

    def _dict_to_ordered_list(self, params, is_inputs):
        if is_inputs:
            index_map = self._input_name_to_index
        else:
            index_map = self._output_name_to_index
        if len(params) != len(index_map):
            raise ValueError(
                f"Did not get correct number of {'inputs' if is_inputs else 'outputs'} expected {len(index_map)}, got {len(params)}"
            )

        result = [None for i in range(len(index_map))]
        for name, tensor in params.items():
            if name not in index_map:
                raise ValueError(
                    f"Got unexpected {'input' if is_inputs else 'output'}: {name}"
                )

            result[index_map[name]] = tensor

        return result

    def _c_output_shapes_to_python(self, c_output_shapes) -> Dict[str, List[int]]:
        output_shapes = []
        for i, c_shape in enumerate(c_output_shapes):
            shape = []
            for j in range(self._output_ndims[i]):
                shape.append(c_shape[j])
            output_shapes.append(shape)

        return {
            name: output_shapes[idx] for name, idx in self._output_name_to_index.items()
        }

    def _run_impl(
        self,
        inputs: Union[Dict[str, AITemplateTensor], List[AITemplateTensor]],
        outputs: Union[Dict[str, AITemplateTensor], List[AITemplateTensor]],
        stream_ptr: Optional[int] = None,
        sync: bool = True,
        graph_mode: bool = False,
        outputs_on_host: bool = False,
    ) -> Dict[str, List[int]]:
        if isinstance(inputs, dict):
            inputs = self._dict_to_ordered_list(inputs, is_inputs=True)
        if isinstance(outputs, dict):
            outputs = self._dict_to_ordered_list(outputs, is_inputs=False)
        (c_inputs, c_outputs, c_stream, c_output_shapes_out,) = self._prepare_run(
            inputs,
            outputs,
            stream_ptr,
        )

        if not outputs_on_host:
            self.DLL.AITemplateModelContainerRun(
                self.handle,
                c_inputs,
                ctypes.c_size_t(len(inputs)),
                c_outputs,
                ctypes.c_size_t(len(outputs)),
                c_stream,
                ctypes.c_bool(sync),
                ctypes.c_bool(graph_mode),
                c_output_shapes_out,
            )
        else:
            self.DLL.AITemplateModelContainerRunWithOutputsOnHost(
                self.handle,
                c_inputs,
                ctypes.c_size_t(len(inputs)),
                c_outputs,
                ctypes.c_size_t(len(outputs)),
                c_stream,
                ctypes.c_bool(graph_mode),
                c_output_shapes_out,
            )

        return self._c_output_shapes_to_python(c_output_shapes_out)

    def Run(
        self,
        inputs: Union[Dict[str, AITemplateTensor], List[AITemplateTensor]],
        outputs: Union[Dict[str, AITemplateTensor], List[AITemplateTensor]],
        stream_ptr: Optional[int] = None,
        sync: bool = True,
        graph_mode: bool = False,
    ) -> Dict[str, List[int]]:
        """
        Run the model.

        Parameters
        ----------
        inputs: Union[Dict[str, AITemplateTensor], List[AITemplateTensor]]
            The inputs to use. AITemplateTensor is a named tuple containing
            the tensor's data_ptr, size, and dtype. If inputs is a list,
            it must be ordered correctly (as specified by GetInputNameToIndexMap).
            This parameter can also be a dictionary (name -> AITemplateTensor).
        outputs: Union[Dict[str, AITemplateTensor], List[AITemplateTensor]]
            The outputs to use. Similar to inputs, can either be a list of ordered
            outputs, or a dictionary (output name -> AITemplateTensor).
            These should be allocated with enough memory to store their maximum
            size (which can be queried with GetOutputMaximumSize).
        stream_ptr: int
            A pointer to CUDA stream to run on. If None, use the legacy stream.
        sync: bool:
            If True, synchronize the stream at the end of the run
        graph_mode: bool
            If True, use a CUDA graph kernel (experimental)

        Returns
        -------
        The output shapes that are computed by shape inference. This may not be
        the maximum shape. The output memory blobs that are passed in to Run()
        should be interpreted and possibly truncated according to these sizes.
        """
        return self._run_impl(
            inputs, outputs, stream_ptr, sync, graph_mode, outputs_on_host=False
        )

    def _interpret_tensors_as_shapes(
        self,
        outputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        shapes: Dict[str, List[int]],
    ) -> Dict[str, TorchTensor]:
        if isinstance(outputs, dict):
            return {
                name: _reshape_tensor(tensor, shapes[name])
                for name, tensor in outputs.items()
            }
        else:
            return {
                name: _reshape_tensor(outputs[idx], shapes[name])
                for name, idx in self._output_name_to_index.items()
            }

    def RunWithTensors(
        self,
        inputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        outputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        stream_ptr: Optional[int] = None,
        sync: bool = True,
        graph_mode: bool = False,
    ) -> Dict[str, TorchTensor]:
        """
        Run the model with torch.Tensors. See Run() for information about the
        arguments.

        Inputs may either be a dictionary (name -> torch.Tensor), or a list
        of torch.Tensors ordered according to GetInputNameToIndexMap. Outputs
        can also be a dictionary, or a list ordered according to GetOutputNameToIndexMap.
        """

        _check_tensors_contiguous_and_on_gpu(
            inputs,
            name="inputs",
        )
        _check_tensors_contiguous_and_on_gpu(
            outputs,
            name="outputs",
        )
        output_shapes = self.Run(
            _convert_tensor_args(inputs),
            _convert_tensor_args(outputs),
            stream_ptr=stream_ptr,
            sync=sync,
            graph_mode=graph_mode,
        )

        return self._interpret_tensors_as_shapes(outputs, output_shapes)

    def _RunWithOutputsOnHost(
        self,
        inputs: Union[Dict[str, AITemplateTensor], List[AITemplateTensor]],
        outputs: Union[Dict[str, int], List[int]],
        stream_ptr: Optional[int] = None,
        graph_mode: bool = False,
    ) -> Dict[str, List[int]]:
        """
        Like Run(), but takes host memory outputs. Note that there is no sync parameter;
        the stream will always be synchronized after copying the outputs to the host.

        Warning: don't use this! It's not optimal with respect to performance.
        It's here for use by internal constant folding passes.
        """
        return self._run_impl(
            inputs, outputs, stream_ptr, graph_mode=graph_mode, outputs_on_host=True
        )

    def _RunWithTensorsOutputsOnHost(
        self,
        inputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        outputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        stream_ptr: Optional[int] = None,
        graph_mode: bool = False,
    ) -> Dict[str, TorchTensor]:
        """
        Like RunWithTensors(), but takes host memory tensors

        Warning: don't use this! It's not optimal with respect to performance.
        It's here for use by internal constant folding passes.
        """
        _check_tensors_contiguous_and_on_gpu(
            inputs,
            name="inputs",
        )
        _check_tensors_contiguous_and_on_host(
            outputs,
            name="outputs",
        )
        output_shapes = self._RunWithOutputsOnHost(
            _convert_tensor_args(inputs),
            _convert_tensor_args(outputs),
            stream_ptr=stream_ptr,
            graph_mode=graph_mode,
        )
        return self._interpret_tensors_as_shapes(outputs, output_shapes)

    def Benchmark(
        self,
        inputs: Union[Dict[str, AITemplateTensor], List[AITemplateTensor]],
        outputs: Union[Dict[str, int], List[int]],
        stream_ptr: Optional[int] = None,
        graph_mode: bool = False,
        count: int = 10,
        repeat: int = 1,
        num_threads: int = 1,
        use_unique_stream_per_thread: bool = False,
    ) -> Tuple[float, float, List[List[int]]]:
        """
        Benchmark the model. See Run() for information on most parameters.
        """
        if isinstance(inputs, dict):
            inputs = self._dict_to_ordered_list(inputs, is_inputs=True)
        if isinstance(outputs, dict):
            outputs = self._dict_to_ordered_list(outputs, is_inputs=False)
        (c_inputs, c_outputs, c_stream, c_output_shapes_out,) = self._prepare_run(
            inputs,
            outputs,
            stream_ptr,
        )
        time_ms = []
        runtime_ms = ctypes.c_float()
        for _ in range(repeat):
            self.DLL.AITemplateModelContainerBenchmark(
                self.handle,
                c_inputs,
                ctypes.c_size_t(len(inputs)),
                c_outputs,
                ctypes.c_size_t(len(outputs)),
                c_stream,
                ctypes.c_bool(graph_mode),
                ctypes.c_size_t(count),
                ctypes.c_size_t(num_threads),
                ctypes.c_bool(use_unique_stream_per_thread),
                ctypes.byref(runtime_ms),
                c_output_shapes_out,
            )
            time_ms.append(runtime_ms.value)
        mean = np.mean(time_ms)
        std = np.std(time_ms)
        return (mean, std, self._c_output_shapes_to_python(c_output_shapes_out))

    def BenchmarkWithTensors(
        self,
        inputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        outputs: Union[List[TorchTensor], Dict[str, TorchTensor]],
        stream_ptr: Optional[int] = None,
        graph_mode: bool = False,
        count: int = 10,
        repeat: int = 1,
        num_threads: int = 1,
        use_unique_stream_per_thread: bool = False,
    ) -> Tuple[float, float, Dict[str, TorchTensor]]:
        """
        Benchmark the model. See RunWithTensors() for information on most parameters.
        """

        _check_tensors_contiguous_and_on_gpu(
            inputs,
            name="inputs",
        )
        _check_tensors_contiguous_and_on_gpu(
            outputs,
            name="outputs",
        )

        mean, std, shapes = self.Benchmark(
            _convert_tensor_args(inputs),
            _convert_tensor_args(outputs),
            stream_ptr,
            graph_mode,
            count,
            repeat,
            num_threads,
            use_unique_stream_per_thread,
        )
        return (mean, std, self._interpret_tensors_as_shapes(outputs, shapes))

    def _get_map_helper(self, n: int, get_name_func) -> Dict[str, int]:
        result = {}
        for i in range(n):
            c_name = ctypes.c_char_p()
            c_idx = ctypes.c_size_t(i)
            get_name_func(c_idx, ctypes.byref(c_name))
            name = c_name.value.decode("utf-8")
            result[name] = i
        return result

    def _construct_input_name_to_index_map(self) -> Dict[str, int]:
        num_inputs = ctypes.c_size_t()
        self.DLL.AITemplateModelContainerGetNumInputs(
            self.handle, ctypes.byref(num_inputs)
        )
        get_input_name = (
            lambda idx, name: self.DLL.AITemplateModelContainerGetInputName(
                self.handle, idx, name
            )
        )
        return self._get_map_helper(num_inputs.value, get_input_name)

    def GetInputNameToIndexMap(self) -> Dict[str, int]:
        # Copy so people can't modify our version of the map
        return self._input_name_to_index.copy()

    def _construct_output_name_to_index_map(self) -> Dict[str, int]:
        num_outputs = ctypes.c_size_t()
        self.DLL.AITemplateModelContainerGetNumOutputs(
            self.handle, ctypes.byref(num_outputs)
        )
        get_output_name = (
            lambda idx, name: self.DLL.AITemplateModelContainerGetOutputName(
                self.handle, idx, name
            )
        )
        return self._get_map_helper(num_outputs.value, get_output_name)

    def GetOutputNameToIndexMap(self) -> Dict[str, int]:
        # Copy so people can't modify our version of the map
        return self._output_name_to_index.copy()

    def SetConstant(self, name: str, arr: np.ndarray, dtype="float16"):
        size = int(np.prod(arr.shape) * DTYPE_TO_BYTES[dtype])
        c_size = ctypes.c_size_t(size)
        b_name = name.encode("utf-8")
        c_name = ctypes.c_char_p(b_name)
        src = arr.ctypes.data_as(ctypes.c_void_p)
        self.DLL.AITemplateModelContainerSetConstant(self.handle, c_name, src, c_size)

    def GetOutputMaximumShape(self, output_idx_or_name: Union[int, str]) -> List[int]:
        """
        Get the maximum output shape. The input here can either be an output name
        or an index. The index is the runtime's internal index (as specified by
        GetOutputNameToIndexMap)
        """
        if isinstance(output_idx_or_name, int):
            output_idx = output_idx_or_name
        elif isinstance(output_idx_or_name, str):
            if output_idx_or_name not in self._output_name_to_index:
                raise ValueError(
                    f"Name {output_idx_or_name} not in OutputNameToIndexMap! Available names: {list(self._output_name_to_index.keys())}"
                )
            output_idx = self._output_name_to_index[output_idx_or_name]
        else:
            raise TypeError(
                f"output_idx_or_name must be str or int, but got {type(output_idx_or_name)}"
            )

        class Shape(ctypes.Structure):
            _fields_ = [
                ("shape_data", ctypes.POINTER(ctypes.c_longlong)),
                ("size", ctypes.c_size_t),
            ]

        raw_shape = Shape()
        self.DLL.AITemplateModelContainerGetMaximumOutputShape(
            self.handle, output_idx, ctypes.byref(raw_shape)
        )
        return [raw_shape.shape_data[idx] for idx in range(raw_shape.size)]

    def GetOutputDtype(self, index):
        output = ctypes.c_int()
        self.DLL.AITemplateModelContainerGetOutputDtype(
            self.handle, index, ctypes.byref(output)
        )
        return output.value
