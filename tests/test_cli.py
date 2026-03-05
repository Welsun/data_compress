import csv
import json
import subprocess
import sys
from pathlib import Path

from data_compress.cli import main


def test_cli_accepts_csv_path_argument(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "train.csv"
    compressed_out = tmp_path / "compressed.bin"
    decompressed_out = tmp_path / "decoded.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"c{i}" for i in range(1, 35)])
        for row_id in range(5):
            writer.writerow([row_id] + [row_id * 0.1 + j for j in range(1, 34)])

    monkeypatch.setattr(
        "sys.argv",
        [
            "data-compress",
            str(csv_path),
            "--start-col",
            "2",
            "--end-col",
            "31",
            "--compressed-out",
            str(compressed_out),
            "--decompressed-out",
            str(decompressed_out),
        ],
    )

    main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["sample_count"] == 5
    assert "manifest" in payload
    assert "failed_samples_topn" in payload
    assert payload["compressed_size_bytes"] > 0
    assert payload["compression_ratio"] is not None
    assert Path(payload["compressed_file"]).exists()
    assert Path(payload["decompressed_file"]).exists()


def test_cli_outputs_failed_sample_topn(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"c{i}" for i in range(1, 35)])
        for row_id in range(10):
            row = [row_id] + [row_id * 1000.0 + j * 1.25 for j in range(1, 34)]
            writer.writerow(row)

    monkeypatch.setattr(
        "sys.argv",
        [
            "data-compress",
            str(csv_path),
            "--max-mae",
            "0",
            "--max-rel",
            "0",
            "--failed-topn",
            "3",
            "--compressed-out",
            str(tmp_path / "c.bin"),
            "--decompressed-out",
            str(tmp_path / "d.csv"),
        ],
    )

    main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["is_valid"] is False
    assert len(payload["failed_samples_topn"]) == 3
    first = payload["failed_samples_topn"][0]
    assert {"sample_id", "shard_id", "codec_id", "mae", "max_rel"} <= set(first)


def test_start_script_runs_without_install(tmp_path):
    csv_path = tmp_path / "train.csv"
    compressed_out = tmp_path / "start_compressed.bin"
    decompressed_out = tmp_path / "start_decoded.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"c{i}" for i in range(1, 35)])
        writer.writerow([0] + [j for j in range(1, 34)])

    repo_root = Path(__file__).resolve().parent.parent
    start_script = repo_root / "start_csv_compress.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(start_script),
            str(csv_path),
            "--compressed-out",
            str(compressed_out),
            "--decompressed-out",
            str(decompressed_out),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["sample_count"] == 1
    assert payload["is_valid"] is True
    assert Path(payload["compressed_file"]).exists()
    assert Path(payload["decompressed_file"]).exists()
