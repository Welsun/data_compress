from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency.
    np = None

from .config import FieldStrategy

try:
    import zstandard as zstd
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency.
    zstd = None

try:
    import pysz as sz_backend
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency.
    try:
        import sz3 as sz_backend
    except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency.
        sz_backend = None

@dataclass
class EncodedSample:
    codec_id: str
    payload: bytes
    shape: tuple[int, ...]
    dtype: str
    params_digest: str


def infer_shape(values):
    if isinstance(values, list) and values:
        return (len(values),) + infer_shape(values[0])
    if isinstance(values, list):
        return (0,)
    return ()


def flatten(values):
    if isinstance(values, list):
        for v in values:
            yield from flatten(v)
    else:
        yield float(values)


def reshape(flat: list[float], shape: tuple[int, ...]):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return flat[: shape[0]]
    step = 1
    for s in shape[1:]:
        step *= s
    return [reshape(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def _digest(strategy: FieldStrategy) -> str:
    return (
        f"eps_abs={strategy.eps_abs};eps_rel={strategy.eps_rel};"
        f"quant_bits={strategy.quant_bits};codec={strategy.codec_family}"
    )


def _call_sz_with_fallbacks(op: str, payload: bytes) -> bytes:
    if sz_backend is None:
        raise RuntimeError(
            "SZ codec requested but no supported SZ Python package is installed. "
            "Install with: pip install pysz (preferred) or pip install sz3"
        )

    candidates: list[str]
    if op == "compress":
        candidates = ["compress", "encode", "sz_compress", "sz3_compress"]
    else:
        candidates = ["decompress", "decode", "sz_decompress", "sz3_decompress"]

    search_spaces = [sz_backend]
    nested_api = getattr(sz_backend, "sz", None)
    if nested_api is not None:
        search_spaces.append(nested_api)

    if nested_api is not None and hasattr(sz_backend, "szConfig"):
        if np is None:
            raise RuntimeError(
                "SZ codec requested but numpy is not installed. Install with: pip install numpy"
            )
        if op == "compress" and callable(getattr(nested_api, "compress", None)):
            arr = np.frombuffer(payload, dtype=np.uint8)
            config = sz_backend.szConfig()
            error_mode = getattr(sz_backend, "szErrorBoundMode", None)
            abs_mode = getattr(error_mode, "ABS", None)
            if abs_mode is not None and hasattr(config, "errorBoundMode"):
                config.errorBoundMode = abs_mode
            if hasattr(config, "absErrorBound"):
                config.absErrorBound = 0.0
            compressed = nested_api.compress(arr, config)
            if isinstance(compressed, tuple):
                compressed = compressed[0]
            if not isinstance(compressed, (bytes, bytearray)):
                compressed = bytes(compressed)
            return struct.pack("<I", len(arr)) + compressed

        if op == "decompress" and callable(getattr(nested_api, "decompress", None)):
            n = struct.unpack("<I", payload[:4])[0]
            compressed = payload[4:]
            restored = nested_api.decompress(compressed, np.uint8, (n,))
            if isinstance(restored, tuple):
                restored = restored[0]
            return np.asarray(restored, dtype=np.uint8).tobytes()

    for space in search_spaces:
        for name in candidates:
            fn = getattr(space, name, None)
            if callable(fn):
                return fn(payload)

    available = [n for n in dir(sz_backend) if not n.startswith("_")]
    raise RuntimeError(
        f"Installed SZ module does not provide a compatible {op} API for bytes. "
        f"Tried {candidates}; available symbols include: {available[:12]}"
    )


def _official_pysz_api():
    nested_api = getattr(sz_backend, "sz", None)
    if (
        sz_backend is not None
        and nested_api is not None
        and hasattr(sz_backend, "szConfig")
        and callable(getattr(nested_api, "compress", None))
        and callable(getattr(nested_api, "decompress", None))
    ):
        return nested_api
    return None


def _np_dtype(name: str):
    if np is None:
        raise RuntimeError("SZ codec requested but numpy is not installed. Install with: pip install numpy")
    return getattr(np, name, name)


def _compress_sz_array(arr, dtype, abs_error_bound: float = 0.0) -> bytes:
    if sz_backend is None:
        raise RuntimeError(
            "SZ codec requested but no supported SZ Python package is installed. "
            "Install with: pip install pysz (preferred) or pip install sz3"
        )
    _ = _np_dtype("uint8")

    arr_np = np.asarray(arr, dtype=dtype)
    nested_api = _official_pysz_api()
    if nested_api is not None:
        config = sz_backend.szConfig()
        error_mode = getattr(sz_backend, "szErrorBoundMode", None)
        abs_mode = getattr(error_mode, "ABS", None)
        if abs_mode is not None and hasattr(config, "errorBoundMode"):
            config.errorBoundMode = abs_mode
        if hasattr(config, "absErrorBound"):
            config.absErrorBound = float(abs_error_bound)
        compressed = nested_api.compress(arr_np, config)
        if isinstance(compressed, tuple):
            compressed = compressed[0]
        if not isinstance(compressed, (bytes, bytearray)):
            compressed = bytes(compressed)
        return compressed

    return _call_sz_with_fallbacks("compress", arr_np.tobytes())


def _decompress_sz_array(payload: bytes, dtype, shape):
    if sz_backend is None:
        raise RuntimeError(
            "SZ codec requested but no supported SZ Python package is installed. "
            "Install with: pip install pysz (preferred) or pip install sz3"
        )
    _ = _np_dtype("uint8")

    nested_api = _official_pysz_api()
    if nested_api is not None:
        restored = nested_api.decompress(payload, dtype, shape)
        if isinstance(restored, tuple):
            restored = restored[0]
        return np.asarray(restored, dtype=dtype)

    raw = _call_sz_with_fallbacks("decompress", payload)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _compress(raw: bytes, compression: str) -> bytes:
    if compression == "zstd":
        if zstd is None:
            raise RuntimeError(
                "zstd codec requested but optional dependency 'zstandard' is not installed. "
                "Install it with: pip install zstandard"
            )
        return zstd.ZstdCompressor(level=3).compress(raw)
    if compression == "sz":
        return _call_sz_with_fallbacks("compress", raw)
    return zlib.compress(raw, level=6)


def _decompress(payload: bytes, compression: str) -> bytes:
    if compression == "zstd":
        if zstd is None:
            raise RuntimeError(
                "zstd codec requested but optional dependency 'zstandard' is not installed. "
                "Install it with: pip install zstandard"
            )
        return zstd.ZstdDecompressor().decompress(payload)
    if compression == "sz":
        return _call_sz_with_fallbacks("decompress", payload)
    return zlib.decompress(payload)


def _compression_from_codec(codec_family: str) -> str:
    return codec_family.split("_")[-1]


def encode_timeseries_delta(arr, strategy: FieldStrategy) -> EncodedSample:
    data = list(flatten(arr))
    shape = infer_shape(arr)
    codec_id = strategy.codec_family
    compression = _compression_from_codec(codec_id)
    if not data:
        return EncodedSample(codec_id, b"", shape, "float32", _digest(strategy))

    delta = [0.0]
    for i in range(1, len(data)):
        delta.append(data[i] - data[i - 1])
    step = max(strategy.eps_abs, strategy.eps_rel * max(abs(x) for x in data))
    q_delta = [int(round(d / step)) for d in delta]

    header = struct.pack("<fI", float(data[0]), len(data))
    if compression == "sz":
        compressed_delta = _compress_sz_array(q_delta, "int32", abs_error_bound=0.0)
        payload = header + struct.pack("<f", step) + compressed_delta
    else:
        packed = header + struct.pack("<f", step) + struct.pack(f"<{len(q_delta)}i", *q_delta)
        payload = _compress(packed, compression)
    return EncodedSample(codec_id, payload, shape, "float32", _digest(strategy))


def decode_timeseries_delta(encoded: EncodedSample):
    if not encoded.payload:
        return []
    compression = _compression_from_codec(encoded.codec_id)
    if compression == "sz":
        first, n = struct.unpack("<fI", encoded.payload[:8])
        step = struct.unpack("<f", encoded.payload[8:12])[0]
        q_delta_arr = _decompress_sz_array(encoded.payload[12:], "int32", (n,))
        q_delta_list = q_delta_arr.tolist() if hasattr(q_delta_arr, "tolist") else list(q_delta_arr)
        q_delta = [int(x) for x in q_delta_list]
    else:
        raw = _decompress(encoded.payload, compression)
        first, n = struct.unpack("<fI", raw[:8])
        step = struct.unpack("<f", raw[8:12])[0]
        q_delta = struct.unpack(f"<{n}i", raw[12 : 12 + n * 4])
    delta = [d * step for d in q_delta]
    restored = [first]
    for i in range(1, n):
        restored.append(restored[-1] + delta[i])
    return reshape(restored, encoded.shape)


def encode_fp16(arr, strategy: FieldStrategy) -> EncodedSample:
    data = list(flatten(arr))
    shape = infer_shape(arr)
    q = [struct.unpack("<e", struct.pack("<e", x))[0] for x in data]
    compression = _compression_from_codec(strategy.codec_family)
    if compression == "sz":
        compressed = _compress_sz_array(q, "float32", abs_error_bound=strategy.eps_abs)
        payload = struct.pack("<I", len(q)) + compressed
        return EncodedSample(strategy.codec_family, payload, shape, "float32", _digest(strategy))
    payload = _compress(struct.pack(f"<{len(q)}e", *q), compression)
    return EncodedSample(strategy.codec_family, payload, shape, "float16", _digest(strategy))


def decode_fp16(encoded: EncodedSample):
    compression = _compression_from_codec(encoded.codec_id)
    if compression == "sz":
        n = struct.unpack("<I", encoded.payload[:4])[0]
        vals_arr = _decompress_sz_array(encoded.payload[4:], "float32", (n,))
        vals = vals_arr.tolist() if hasattr(vals_arr, "tolist") else list(vals_arr)
    else:
        raw = _decompress(encoded.payload, compression)
        n = len(raw) // 2
        vals = list(struct.unpack(f"<{n}e", raw))
    return reshape(vals, encoded.shape)


def encode_int8(arr, strategy: FieldStrategy) -> EncodedSample:
    data = list(flatten(arr))
    shape = infer_shape(arr)
    max_abs = max((abs(x) for x in data), default=1.0)
    scale = max(max_abs / 127.0, 1e-12)
    q = [max(-127, min(127, int(round(x / scale)))) for x in data]
    compression = _compression_from_codec(strategy.codec_family)
    if compression == "sz":
        compressed_q = _compress_sz_array(q, "int32", abs_error_bound=0.0)
        payload = struct.pack("<fI", scale, len(q)) + compressed_q
    else:
        compressed_q = _compress(struct.pack(f"<{len(q)}b", *q), compression)
        payload = struct.pack("<f", scale) + compressed_q
    return EncodedSample(strategy.codec_family, payload, shape, "int8", _digest(strategy))


def decode_int8(encoded: EncodedSample):
    compression = _compression_from_codec(encoded.codec_id)
    if compression == "sz":
        scale, n = struct.unpack("<fI", encoded.payload[:8])
        q_arr = _decompress_sz_array(encoded.payload[8:], "int32", (n,))
        q_list = q_arr.tolist() if hasattr(q_arr, "tolist") else list(q_arr)
        q = [int(x) for x in q_list]
    else:
        scale = struct.unpack("<f", encoded.payload[:4])[0]
        raw = _decompress(encoded.payload[4:], compression)
        n = len(raw)
        q = struct.unpack(f"<{n}b", raw)
    vals = [x * scale for x in q]
    return reshape(vals, encoded.shape)


def encode_sample(arr, strategy: FieldStrategy) -> EncodedSample:
    if strategy.codec_family.startswith("delta_"):
        return encode_timeseries_delta(arr, strategy)
    if strategy.codec_family.startswith("int8_"):
        return encode_int8(arr, strategy)
    return encode_fp16(arr, strategy)


def decode_sample(encoded: EncodedSample):
    if encoded.codec_id.startswith("delta_"):
        return decode_timeseries_delta(encoded)
    if encoded.codec_id.startswith("int8_"):
        return decode_int8(encoded)
    return decode_fp16(encoded)
