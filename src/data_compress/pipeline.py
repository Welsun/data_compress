from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .config import CompressionConfig
from .indexing import ShardIndex, build_indexes, build_manifest
from .sample_codecs import EncodedSample
from .shard import compress_shard


@dataclass
class PipelineResult:
    manifest: dict
    shard_indexes: list[ShardIndex]
    encoded_shards: dict[str, list[EncodedSample]]


class CompressionPipeline:
    def __init__(self, config: CompressionConfig):
        self.config = config

    def _sanitize(self, x: float) -> float:
        if math.isnan(x):
            return 0.0
        if math.isinf(x):
            return 1e6 if x > 0 else -1e6
        return float(x)

    def normalize(self, sample):
        if isinstance(sample, list):
            return [self.normalize(s) for s in sample]
        return self._sanitize(sample)

    def load_csv_samples(
        self,
        csv_path: str | Path,
        start_col: int = 2,
        end_col: int = 31,
        has_header: bool = True,
    ) -> list[list[float]]:
        """Load CSV rows and keep only [start_col, end_col] (1-based, inclusive)."""
        if start_col < 1 or end_col < start_col:
            raise ValueError("Invalid column range")

        selected_rows: list[list[float]] = []
        start_idx = start_col - 1
        end_idx = end_col - 1

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if has_header:
                next(reader, None)

            for row_no, row in enumerate(reader, start=1 if not has_header else 2):
                if len(row) <= end_idx:
                    raise ValueError(
                        f"Row {row_no} has {len(row)} columns, expected at least {end_col}."
                    )
                values = [self._sanitize(float(cell)) for cell in row[start_idx : end_idx + 1]]
                selected_rows.append(values)

        return selected_rows

    def pack_csv(
        self,
        csv_path: str | Path,
        field_name: str = "csv_features_2_31",
        start_col: int = 2,
        end_col: int = 31,
        has_header: bool = True,
        shard_id: str = "shard-0",
    ) -> PipelineResult:
        samples = self.load_csv_samples(
            csv_path=csv_path,
            start_col=start_col,
            end_col=end_col,
            has_header=has_header,
        )
        return self.pack_field(field_name=field_name, samples=samples, shard_id=shard_id)

    def pack_field(self, field_name: str, samples: list, shard_id: str = "shard-0") -> PipelineResult:
        normalized = [self.normalize(s) for s in samples]
        encoded = compress_shard(normalized, field_name, self.config)
        shard_index = build_indexes(
            shard_id=shard_id,
            originals=normalized,
            encoded=encoded,
            guardrail="delta_top1 <= 0.2%",
        )
        manifest = build_manifest("v1", [shard_index])
        return PipelineResult(manifest, [shard_index], {shard_id: encoded})

    def validate(self, result: PipelineResult, max_mae: float = 0.02, max_rel: float = 0.01) -> bool:
        for shard in result.shard_indexes:
            for q in shard.quality:
                if q.error_stat["mae"] > max_mae:
                    return False
                if q.error_stat["max_rel"] > max_rel:
                    return False
        return True
