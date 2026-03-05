import csv
import json

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
