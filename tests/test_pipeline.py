import csv
import math
import random

from data_compress import CompressionConfig, CompressionPipeline, FieldStrategy


def test_pipeline_pack_and_validate_timeseries():
    t = [i * (20 / 2047) for i in range(2048)]
    samples = [[math.sin(v + i * 0.01) for v in t] for i in range(4)]

    config = CompressionConfig(
        strategies={
            "sensor": FieldStrategy(field_name="sensor", codec_family="delta_zlib", eps_abs=1e-3, eps_rel=1e-3)
        }
    )
    pipeline = CompressionPipeline(config)
    result = pipeline.pack_field("sensor", samples)

    assert result.manifest["version"] == "v1"
    assert len(result.shard_indexes[0].quality) == 4
    assert pipeline.validate(result, max_mae=0.01, max_rel=5.0)


def test_pipeline_pack_generic_embeddings():
    random.seed(0)
    samples = [[[random.gauss(0, 1) for _ in range(32)] for _ in range(64)] for _ in range(3)]

    config = CompressionConfig(
        strategies={
            "embedding": FieldStrategy(field_name="embedding", codec_family="fp16_zlib", conservative=True)
        }
    )
    pipeline = CompressionPipeline(config)
    result = pipeline.pack_field("embedding", samples)

    assert len(result.encoded_shards["shard-0"]) == 3
    assert pipeline.validate(result, max_mae=0.05, max_rel=0.05)


def test_pipeline_supports_csv_columns_2_to_31(tmp_path):
    csv_path = tmp_path / "train.csv"
    header = [f"c{i}" for i in range(1, 35)]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row_id in range(20):
            row = [row_id] + [row_id * 0.1 + j for j in range(1, 34)]
            writer.writerow(row)

    config = CompressionConfig(
        strategies={
            "csv_features_2_31": FieldStrategy(field_name="csv_features_2_31", codec_family="fp16_zlib")
        }
    )
    pipeline = CompressionPipeline(config)

    samples = pipeline.load_csv_samples(csv_path)
    assert len(samples) == 20
    assert all(len(sample) == 30 for sample in samples)

    result = pipeline.pack_csv(csv_path)
    assert len(result.encoded_shards["shard-0"]) == 20
    assert pipeline.validate(result, max_mae=0.1, max_rel=0.1)


def test_run_script_accepts_csv_path_argument(tmp_path):
    import json
    import subprocess
    import sys

    csv_path = tmp_path / "cli.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"c{i}" for i in range(1, 33)])
        writer.writerow([i * 0.5 for i in range(1, 33)])
        writer.writerow([i * 1.5 for i in range(1, 33)])

    proc = subprocess.run(
        [sys.executable, "scripts/run_compress.py", str(csv_path)],
        cwd=".",
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["samples"] == 2
    assert payload["manifest"]["version"] == "v1"
