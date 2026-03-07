from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

from .config import CompressionConfig, FieldStrategy
from .pipeline import CompressionPipeline
from .sample_codecs import EncodedSample, decode_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress a CSV file with data_compress pipeline.")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--field-name", default="csv_features_2_31", help="Field name used in strategy")
    parser.add_argument(
        "--codec-family",
        choices=["delta_zlib", "fp16_zlib", "int8_zlib", "delta_zstd", "fp16_zstd", "int8_zstd", "delta_sz", "fp16_sz", "int8_sz"],
        default="fp16_zlib",
        help="Codec family to use for compression",
    )
    parser.add_argument("--start-col", type=int, default=2, help="Start column index (1-based)")
    parser.add_argument("--end-col", type=int, default=31, help="End column index (1-based, inclusive)")
    parser.add_argument("--no-header", action="store_true", help="Set when CSV has no header row")
    parser.add_argument("--max-mae", type=float, default=0.1, help="Validation MAE threshold")
    parser.add_argument("--max-rel", type=float, default=0.1, help="Validation max relative error threshold")
    parser.add_argument(
        "--compressed-output",
        help="Output path for packed compressed binary; defaults to <csv_stem>.compressed.bin",
    )
    parser.add_argument(
        "--decompressed-output",
        help="Output path for decompressed CSV preview; defaults to <csv_stem>.decompressed.csv",
    )
    return parser


def _resolve_output_path(
    csv_path: Path,
    explicit_path: str | None,
    suffix: str,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return csv_path.with_name(f"{csv_path.stem}.{suffix}")


def _build_compressed_blob(encoded_samples: list[EncodedSample]) -> bytes:
    header = b"DCP1" + struct.pack("<I", len(encoded_samples))
    payloads = [header]
    for enc in encoded_samples:
        codec = enc.codec_id.encode("utf-8")
        payloads.append(struct.pack("<H", len(codec)))
        payloads.append(codec)
        payloads.append(struct.pack("<I", len(enc.payload)))
        payloads.append(enc.payload)
    return b"".join(payloads)


def _write_decompressed_csv(path: Path, restored_rows: list[list[float]]) -> None:
    max_len = max((len(row) for row in restored_rows), default=0)
    header = [f"f{i+1}" for i in range(max_len)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(restored_rows)


def main() -> None:
    args = build_parser().parse_args()
    csv_path = Path(args.csv_path)

    config = CompressionConfig(
        strategies={
            args.field_name: FieldStrategy(field_name=args.field_name, codec_family=args.codec_family),
        }
    )
    pipeline = CompressionPipeline(config)

    selected_rows = pipeline.load_csv_samples(
        csv_path=csv_path,
        start_col=args.start_col,
        end_col=args.end_col,
        has_header=not args.no_header,
    )

    result = pipeline.pack_field(
        field_name=args.field_name,
        samples=selected_rows,
    )

    is_valid = pipeline.validate(result, max_mae=args.max_mae, max_rel=args.max_rel)
    encoded_samples = result.encoded_shards["shard-0"]

    compressed_output = _resolve_output_path(csv_path, args.compressed_output, "compressed.bin")
    compressed_output.parent.mkdir(parents=True, exist_ok=True)
    compressed_blob = _build_compressed_blob(encoded_samples)
    compressed_output.write_bytes(compressed_blob)

    decompressed_output = _resolve_output_path(csv_path, args.decompressed_output, "decompressed.csv")
    decompressed_output.parent.mkdir(parents=True, exist_ok=True)
    restored_rows = [decode_sample(enc) for enc in encoded_samples]
    _write_decompressed_csv(decompressed_output, restored_rows)

    original_bytes = len(selected_rows) * (args.end_col - args.start_col + 1) * 4
    compressed_size = compressed_output.stat().st_size
    compression_ratio = (original_bytes / compressed_size) if compressed_size else 0.0

    output = {
        "manifest": result.manifest,
        "is_valid": is_valid,
        "shard_count": len(result.shard_indexes),
        "sample_count": len(encoded_samples),
        "compressed_output": str(compressed_output),
        "decompressed_output": str(decompressed_output),
        "original_bytes_estimate": original_bytes,
        "compressed_bytes": compressed_size,
        "compression_ratio": compression_ratio,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
