#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without installation: python scripts/run_compress.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_compress import CompressionConfig, CompressionPipeline, FieldStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress CSV columns with data_compress pipeline")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("--field-name", default="csv_features_2_31", help="Field strategy name")
    parser.add_argument("--start-col", type=int, default=2, help="1-based start column (inclusive)")
    parser.add_argument("--end-col", type=int, default=31, help="1-based end column (inclusive)")
    parser.add_argument("--no-header", action="store_true", help="Set if CSV does not have a header row")
    parser.add_argument("--codec", choices=["delta_zlib", "fp16_zlib", "int8_zlib"], default="fp16_zlib")
    parser.add_argument("--max-mae", type=float, default=0.1)
    parser.add_argument("--max-rel", type=float, default=0.1)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = CompressionConfig(
        strategies={
            args.field_name: FieldStrategy(field_name=args.field_name, codec_family=args.codec)
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
        "samples": len(result.shard_indexes[0].sample_index) if result.shard_indexes else 0,
        "valid": is_valid,
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
