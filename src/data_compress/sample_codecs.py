from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from .config import FieldStrategy


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


def encode_timeseries_delta(arr, strategy: FieldStrategy) -> EncodedSample:
    data = list(flatten(arr))
    shape = infer_shape(arr)
    if not data:
        return EncodedSample("delta_zlib", b"", shape, "float32", _digest(strategy))

    delta = [0.0]
    for i in range(1, len(data)):
        delta.append(data[i] - data[i - 1])
    step = max(strategy.eps_abs, strategy.eps_rel * max(abs(x) for x in data))
    q_delta = [int(round(d / step)) for d in delta]

    header = struct.pack("<fI", float(data[0]), len(data))
    packed = header + struct.pack("<f", step) + struct.pack(f"<{len(q_delta)}i", *q_delta)
    payload = zlib.compress(packed, level=6)
    return EncodedSample("delta_zlib", payload, shape, "float32", _digest(strategy))


def decode_timeseries_delta(encoded: EncodedSample):
    if not encoded.payload:
        return []
    raw = zlib.decompress(encoded.payload)
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
    payload = zlib.compress(struct.pack(f"<{len(q)}e", *q), level=6)
    return EncodedSample("fp16_zlib", payload, shape, "float16", _digest(strategy))


def decode_fp16(encoded: EncodedSample):
    raw = zlib.decompress(encoded.payload)
    n = len(raw) // 2
    vals = list(struct.unpack(f"<{n}e", raw))
    return reshape(vals, encoded.shape)


def encode_int8(arr, strategy: FieldStrategy) -> EncodedSample:
    data = list(flatten(arr))
    shape = infer_shape(arr)
    max_abs = max((abs(x) for x in data), default=1.0)
    scale = max(max_abs / 127.0, 1e-12)
    q = [max(-127, min(127, int(round(x / scale)))) for x in data]
    payload = struct.pack("<f", scale) + zlib.compress(struct.pack(f"<{len(q)}b", *q), level=6)
    return EncodedSample("int8_zlib", payload, shape, "int8", _digest(strategy))


def decode_int8(encoded: EncodedSample):
    scale = struct.unpack("<f", encoded.payload[:4])[0]
    raw = zlib.decompress(encoded.payload[4:])
    n = len(raw)
    q = struct.unpack(f"<{n}b", raw)
    vals = [x * scale for x in q]
    return reshape(vals, encoded.shape)


def encode_sample(arr, strategy: FieldStrategy) -> EncodedSample:
    if strategy.codec_family == "delta_zlib":
        return encode_timeseries_delta(arr, strategy)
    if strategy.codec_family == "int8_zlib":
        return encode_int8(arr, strategy)
    return encode_fp16(arr, strategy)


def decode_sample(encoded: EncodedSample):
    if encoded.codec_id == "delta_zlib":
        return decode_timeseries_delta(encoded)
    if encoded.codec_id == "int8_zlib":
        return decode_int8(encoded)
    return decode_fp16(encoded)
