from __future__ import annotations

from dataclasses import dataclass

from .config import CompressionConfig, FieldStrategy
from .sample_codecs import EncodedSample, encode_sample, flatten


@dataclass
class BlockProfile:
    variance: float
    dynamic_range: float
    smoothness: float


def profile_block(arr) -> BlockProfile:
    data = list(flatten(arr))
    if not data:
        return BlockProfile(0.0, 0.0, 1.0)
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    dynamic_range = max(data) - min(data)
    if len(data) < 2:
        smoothness = 1.0
    else:
        diffs = [abs(data[i] - data[i - 1]) for i in range(1, len(data))]
        smoothness = 1.0 / (1.0 + sum(diffs) / len(diffs))
    return BlockProfile(variance, dynamic_range, smoothness)


def choose_strategy(profile: BlockProfile, base: FieldStrategy) -> FieldStrategy:
    strategy = FieldStrategy(
        field_name=base.field_name,
        loss_mode=base.loss_mode,
        eps_abs=base.eps_abs,
        eps_rel=base.eps_rel,
        quant_bits=base.quant_bits,
        codec_family=base.codec_family,
        conservative=base.conservative,
    )
    compression = base.codec_family.split("_")[-1]
    if profile.smoothness > 0.65:
        strategy.codec_family = f"delta_{compression}"
    elif profile.dynamic_range > 20:
        strategy.codec_family = f"fp16_{compression}"
    else:
        strategy.codec_family = f"int8_{compression}"
    if strategy.conservative:
        strategy.codec_family = f"fp16_{compression}"
        strategy.eps_abs = min(strategy.eps_abs, 1e-5)
        strategy.eps_rel = min(strategy.eps_rel, 1e-4)
    return strategy


def compress_shard(samples: list, field_name: str, config: CompressionConfig) -> list[EncodedSample]:
    encoded: list[EncodedSample] = []
    base = config.strategy_for(field_name)
    for sample in samples:
        profile = profile_block(sample)
        strategy = choose_strategy(profile, base)
        encoded.append(encode_sample(sample, strategy))
    return encoded
