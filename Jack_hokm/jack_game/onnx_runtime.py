"""ONNX Runtime inference on Android via the native Java API (pyjnius)."""

from __future__ import annotations

from time import perf_counter

import numpy as np

try:
    from jnius import autoclass
except ImportError:
    autoclass = None

ByteBuffer = None
ByteOrder = None
HashMap = None
OrtEnvironment = None
OrtSession = None
OrtSessionOptions = None
OnnxTensor = None


def _ensure_jni():
    global ByteBuffer, ByteOrder, HashMap, OrtEnvironment, OrtSession, OrtSessionOptions, OnnxTensor
    if OrtSession is not None:
        return
    if autoclass is None:
        raise RuntimeError("pyjnius is required for ONNX inference on Android")
    OrtSession = autoclass("ai.onnxruntime.OrtSession")
    OrtSessionOptions = autoclass("ai.onnxruntime.OrtSession$SessionOptions")
    OrtEnvironment = autoclass("ai.onnxruntime.OrtEnvironment")
    OnnxTensor = autoclass("ai.onnxruntime.OnnxTensor")
    ByteBuffer = autoclass("java.nio.ByteBuffer")
    ByteOrder = autoclass("java.nio.ByteOrder")
    HashMap = autoclass("java.util.HashMap")


class OnnxModel:
    def __init__(self, model_path: str, num_threads: int | None = 2):
        _ensure_jni()
        self.env = OrtEnvironment.getEnvironment()
        options = OrtSessionOptions()
        if num_threads is not None:
            options.setIntraOpNumThreads(num_threads)
            options.setInterOpNumThreads(1)
        self.session = self.env.createSession(model_path, options)

    def run(self, inputs: dict[str, np.ndarray], outputs: dict[str, list[int]]) -> dict[str, np.ndarray]:
        jmap = HashMap()
        for name, value in inputs.items():
            arr = np.ascontiguousarray(value, dtype=np.float32)
            shape = [int(dim) for dim in arr.shape]
            buffer_bytes = arr.ravel().tobytes()
            java_byte_buffer = ByteBuffer.wrap(buffer_bytes)
            java_byte_buffer.order(ByteOrder.nativeOrder())
            float_buffer = java_byte_buffer.asFloatBuffer()
            tensor = OnnxTensor.createTensor(self.env, float_buffer, shape)
            jmap.put(name, tensor)

        t0 = perf_counter()
        results = self.session.run(jmap)
        _ = perf_counter() - t0

        output_dict: dict[str, np.ndarray] = {}
        for out_name, shape in outputs.items():
            tensor_obj = results.get(out_name).get()
            bytebuffer = bytes(tensor_obj.getByteBuffer().array())
            n_elems = int(np.prod(shape))
            output_dict[out_name] = np.frombuffer(bytebuffer, dtype=np.float32, count=n_elems).reshape(shape)
            tensor_obj.close()
        results.close()
        return output_dict

    def close(self):
        self.session.close()
