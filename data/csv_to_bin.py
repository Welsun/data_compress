from __future__ import annotations

import argparse
import csv
from array import array
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从原始 CSV 中提取指定列的浮点数据，并写入连续的 .bin 文件（float32/float64），"
            "用于评估压缩前内存占用。"
        )
    )
    parser.add_argument("csv_path", type=Path, help="输入 CSV 文件路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 .bin 路径（默认与 CSV 同名，仅后缀改为 .bin）",
    )
    parser.add_argument(
        "--start-col",
        type=int,
        default=2,
        help="起始列（1-based，包含，默认 2）",
    )
    parser.add_argument(
        "--end-col",
        type=int,
        default=31,
        help="结束列（1-based，包含，默认 31）",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="输出二进制的浮点类型（默认 float32）",
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="强制跳过 CSV 首行（若未设置，也会自动识别并跳过非浮点表头）",
    )
    return parser.parse_args()


def _can_parse_selected_columns(row: list[str], start_col: int, end_col: int) -> bool:
    if len(row) < end_col:
        return False

    try:
        for value in row[start_col - 1 : end_col]:
            float(value)
    except ValueError:
        return False

    return True


def csv_to_bin(
    csv_path: Path,
    output_path: Path,
    start_col: int,
    end_col: int,
    dtype: str,
    skip_header: bool,
) -> tuple[int, int, bool]:
    if start_col < 1 or end_col < start_col:
        raise ValueError("列范围非法：请确保 1 <= start-col <= end-col")

    typecode = "f" if dtype == "float32" else "d"
    numbers = array(typecode)
    row_count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        first_row = next(reader, None)
        auto_header_skipped = False
        if first_row is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as fw:
                numbers.tofile(fw)
            return 0, 0, False

        if skip_header:
            auto_header_skipped = True
        elif _can_parse_selected_columns(first_row, start_col, end_col):
            numbers.extend(float(v) for v in first_row[start_col - 1 : end_col])
            row_count += 1
        else:
            auto_header_skipped = True

        for row_idx, row in enumerate(reader, start=2):
            if len(row) < end_col:
                raise ValueError(
                    f"第 {row_idx} 行列数不足：需要至少 {end_col} 列，实际 {len(row)} 列"
                )

            try:
                selected = row[start_col - 1 : end_col]
                numbers.extend(float(v) for v in selected)
            except ValueError as exc:
                raise ValueError(f"第 {row_idx} 行存在非浮点值") from exc

            row_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fw:
        numbers.tofile(fw)

    return row_count, len(numbers), auto_header_skipped


def format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def main() -> None:
    args = parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else csv_path.with_suffix(".bin")
    )

    row_count, value_count, header_skipped = csv_to_bin(
        csv_path=csv_path,
        output_path=output_path,
        start_col=args.start_col,
        end_col=args.end_col,
        dtype=args.dtype,
        skip_header=args.skip_header,
    )

    total_bytes = output_path.stat().st_size
    print(f"输入 CSV: {csv_path}")
    print(f"输出 BIN: {output_path}")
    print(f"总行数: {row_count}")
    print(f"提取列范围: {args.start_col}~{args.end_col} (共 {args.end_col - args.start_col + 1} 列)")
    print(f"首行处理: {'已跳过（表头）' if header_skipped else '作为数据读取'}")
    print(f"数据类型: {args.dtype}")
    print(f"浮点数量: {value_count}")
    print(f"原始浮点内存占用: {total_bytes} bytes ({format_size(total_bytes)})")


if __name__ == "__main__":
    main()
