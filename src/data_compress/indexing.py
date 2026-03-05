from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import error_stats
from .sample_codecs import EncodedSample, decode_sample


@dataclass(slots=True)
class SampleIndex:
    sample_id: int
    shard_id: str
    block_id: int
    offset: int
    length: int


@dataclass(slots=True)
class QualityRecord:
    codec_id: str
    codec_params_digest: str
    error_stat: dict[str, float]
    task_guardrail: str


@dataclass(slots=True)
class ShardIndex:
    shard_id: str
    block_offsets: list[int] = field(default_factory=list)
    sample_index: list[SampleIndex] = field(default_factory=list)
    quality: list[QualityRecord] = field(default_factory=list)


def build_indexes(shard_id: str, originals, encoded: list[EncodedSample], guardrail: str) -> ShardIndex:
    index = ShardIndex(shard_id=shard_id)
    cursor = 0
    for i, (origin, enc) in enumerate(zip(originals, encoded)):
        index.block_offsets.append(cursor)
        length = len(enc.payload)
        index.sample_index.append(SampleIndex(i, shard_id, i, cursor, length))
        restored = decode_sample(enc)
        index.quality.append(
            QualityRecord(
                codec_id=enc.codec_id,
                codec_params_digest=enc.params_digest,
                error_stat=error_stats(origin, restored),
                task_guardrail=guardrail,
            )
        )
        cursor += length
    return index


def build_manifest(version: str, shard_indexes: list[ShardIndex]) -> dict:
    return {
        "version": version,
        "shard_count": len(shard_indexes),
        "shards": [
            {
                "shard_id": s.shard_id,
                "blocks": len(s.block_offsets),
                "samples": len(s.sample_index),
            }
            for s in shard_indexes
        ],
    }
