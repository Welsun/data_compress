from __future__ import annotations

import argparse
import json

from .config import CompressionConfig, FieldStrategy
from .pipeline import CompressionPipeline


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
    return parser


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
    output = {
        "manifest": result.manifest,
        "is_valid": is_valid,
        "shard_count": len(result.shard_indexes),
        "sample_count": len(result.encoded_shards["shard-0"]),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
