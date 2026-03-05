import csv
import json
import subprocess
import sys
from pathlib import Path

from data_compress.cli import main


def test_cli_accepts_csv_path_argument(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"c{i}" for i in range(1, 35)])
        for row_id in range(5):
            writer.writerow([row_id] + [row_id * 0.1 + j for j in range(1, 34)])

    monkeypatch.setattr(
        "sys.argv",
        ["data-compress", str(csv_path), "--start-col", "2", "--end-col", "31"],
    )

    main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["sample_count"] == 5
    assert "manifest" in payload
    assert payload["compression_ratio"] > 0
    assert Path(payload["compressed_output"]).exists()
    assert Path(payload["decompressed_output"]).exists()


def test_start_script_runs_without_install(tmp_path):
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"c{i}" for i in range(1, 35)])
        writer.writerow([0] + [j for j in range(1, 34)])

    repo_root = Path(__file__).resolve().parent.parent
    start_script = repo_root / "start_csv_compress.py"

    proc = subprocess.run(
        [sys.executable, str(start_script), str(csv_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["sample_count"] == 1
    assert payload["is_valid"] is True
    assert payload["compressed_bytes"] > 0
