from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


LossMode = Literal["bounded_lossy"]
CodecFamily = Literal["delta_zlib", "fp16_zlib", "int8_zlib"]


@dataclass
class FieldStrategy:
    field_name: str
    loss_mode: LossMode = "bounded_lossy"
    eps_abs: float = 1e-4
    eps_rel: float = 1e-3
    quant_bits: int = 16
    codec_family: CodecFamily = "fp16_zlib"
    conservative: bool = False


@dataclass
class CompressionConfig:
    shard_size_mb: int = 256
    block_size_mb: int = 1
    strategies: dict[str, FieldStrategy] = field(default_factory=dict)

    def strategy_for(self, field_name: str) -> FieldStrategy:
        if field_name in self.strategies:
            return self.strategies[field_name]
        return FieldStrategy(field_name=field_name)
