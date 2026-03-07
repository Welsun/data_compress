from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

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


def _compress(raw: bytes, compression: str) -> bytes:
    if compression == "zstd":
        if zstd is None:
            raise RuntimeError(
                "zstd codec requested but optional dependency 'zstandard' is not installed. "
                "Install it with: pip install zstandard"
            )
        return zstd.ZstdCompressor(level=3).compress(raw)
    if compression == "sz":
        if sz_backend is None:
            raise RuntimeError(
                "SZ codec requested but no supported SZ Python package is installed. "
                "Install with: pip install pysz (preferred) or pip install sz3"
            )
        if not hasattr(sz_backend, "compress"):
            raise RuntimeError("Installed SZ module does not provide compress(raw: bytes)")
        return sz_backend.compress(raw)
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
        if sz_backend is None:
            raise RuntimeError(
                "SZ codec requested but no supported SZ Python package is installed. "
                "Install with: pip install pysz (preferred) or pip install sz3"
            )
        if not hasattr(sz_backend, "decompress"):
            raise RuntimeError("Installed SZ module does not provide decompress(payload: bytes)")
        return sz_backend.decompress(payload)
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
    packed = header + struct.pack("<f", step) + struct.pack(f"<{len(q_delta)}i", *q_delta)
    payload = _compress(packed, compression)
    return EncodedSample(codec_id, payload, shape, "float32", _digest(strategy))


def decode_timeseries_delta(encoded: EncodedSample):
    if not encoded.payload:
        return []
    raw = _decompress(encoded.payload, _compression_from_codec(encoded.codec_id))
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
    payload = _compress(struct.pack(f"<{len(q)}e", *q), _compression_from_codec(strategy.codec_family))
    return EncodedSample(strategy.codec_family, payload, shape, "float16", _digest(strategy))


def decode_fp16(encoded: EncodedSample):
    raw = _decompress(encoded.payload, _compression_from_codec(encoded.codec_id))
    n = len(raw) // 2
    vals = list(struct.unpack(f"<{n}e", raw))
    return reshape(vals, encoded.shape)


def encode_int8(arr, strategy: FieldStrategy) -> EncodedSample:
    data = list(flatten(arr))
    shape = infer_shape(arr)
    max_abs = max((abs(x) for x in data), default=1.0)
    scale = max(max_abs / 127.0, 1e-12)
    q = [max(-127, min(127, int(round(x / scale)))) for x in data]
    compressed_q = _compress(struct.pack(f"<{len(q)}b", *q), _compression_from_codec(strategy.codec_family))
    payload = struct.pack("<f", scale) + compressed_q
    return EncodedSample(strategy.codec_family, payload, shape, "int8", _digest(strategy))


def decode_int8(encoded: EncodedSample):
    scale = struct.unpack("<f", encoded.payload[:4])[0]
    raw = _decompress(encoded.payload[4:], _compression_from_codec(encoded.codec_id))
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
