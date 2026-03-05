from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

from .config import CompressionConfig, FieldStrategy
from .pipeline import CompressionPipeline
from .sample_codecs import decode_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress a CSV file with data_compress pipeline.")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--field-name", default="csv_features_2_31", help="Field name used in strategy")
    parser.add_argument(
        "--codec-family",
        choices=["delta_zlib", "fp16_zlib", "int8_zlib"],
        default="fp16_zlib",
        help="Codec family to use for compression",
    )
    parser.add_argument("--start-col", type=int, default=2, help="Start column index (1-based)")
    parser.add_argument("--end-col", type=int, default=31, help="End column index (1-based, inclusive)")
    parser.add_argument("--no-header", action="store_true", help="Set when CSV has no header row")
    parser.add_argument("--max-mae", type=float, default=0.1, help="Validation MAE threshold")
    parser.add_argument("--max-rel", type=float, default=0.1, help="Validation max relative error threshold")
    parser.add_argument(
        "--failed-topn",
        type=int,
        default=10,
        help="How many failed samples to include in failure statistics (0 to disable)",
    )
    parser.add_argument(
        "--compressed-out",
        default="compressed_output.bin",
        help="Path to write compressed payload file",
    )
    parser.add_argument(
        "--decompressed-out",
        default="decompressed_output.csv",
        help="Path to write decompressed CSV file",
    )
    return parser


def build_failed_sample_topn(result, max_mae: float, max_rel: float, topn: int) -> list[dict]:
    if topn <= 0:
        return []

    failed_items: list[dict] = []
    for shard in result.shard_indexes:
        for sample_idx, quality in zip(shard.sample_index, shard.quality):
            mae = quality.error_stat["mae"]
            max_relative = quality.error_stat["max_rel"]
            if mae > max_mae or max_relative > max_rel:
                failed_items.append(
                    {
                        "sample_id": sample_idx.sample_id,
                        "shard_id": sample_idx.shard_id,
                        "codec_id": quality.codec_id,
                        "mae": mae,
                        "max_rel": max_relative,
                    }
                )

    failed_items.sort(key=lambda item: (item["mae"], item["max_rel"]), reverse=True)
    return failed_items[:topn]


def write_compressed_file(result, output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(b"DCP1")
        for shard_id in sorted(result.encoded_shards.keys()):
            for encoded in result.encoded_shards[shard_id]:
                f.write(struct.pack("<I", len(encoded.payload)))
                f.write(encoded.payload)

    return path.stat().st_size


def write_decompressed_csv(result, output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for shard_id in sorted(result.encoded_shards.keys()):
        for encoded in result.encoded_shards[shard_id]:
            rows.append(decode_sample(encoded))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            if isinstance(row, list):
                writer.writerow(row)
            else:
                writer.writerow([row])

    return path.stat().st_size


def main() -> None:
    args = build_parser().parse_args()

    config = CompressionConfig(
        strategies={
            args.field_name: FieldStrategy(field_name=args.field_name, codec_family=args.codec_family),
        }
    )
    pipeline = CompressionPipeline(config)

    result = pipeline.pack_csv(
        csv_path=args.csv_path,
        field_name=args.field_name,
        start_col=args.start_col,
        end_col=args.end_col,
        has_header=not args.no_header,
    )

    is_valid = pipeline.validate(result, max_mae=args.max_mae, max_rel=args.max_rel)

    csv_size_bytes = Path(args.csv_path).stat().st_size
    compressed_size_bytes = write_compressed_file(result, args.compressed_out)
    decompressed_size_bytes = write_decompressed_csv(result, args.decompressed_out)
    compression_ratio = (csv_size_bytes / compressed_size_bytes) if compressed_size_bytes else None

    output = {
        "manifest": result.manifest,
        "is_valid": is_valid,
        "shard_count": len(result.shard_indexes),
        "sample_count": len(result.encoded_shards["shard-0"]),
        "failed_samples_topn": build_failed_sample_topn(
            result,
            max_mae=args.max_mae,
            max_rel=args.max_rel,
            topn=args.failed_topn,
        ),
        "compressed_file": str(Path(args.compressed_out)),
        "compressed_size_bytes": compressed_size_bytes,
        "source_csv_size_bytes": csv_size_bytes,
        "compression_ratio": compression_ratio,
        "decompressed_file": str(Path(args.decompressed_out)),
        "decompressed_size_bytes": decompressed_size_bytes,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
