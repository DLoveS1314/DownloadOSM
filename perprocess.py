import argparse
import csv
import os


DEFAULT_INPUT_CSV = os.path.join("input", "build.csv")
DEFAULT_OUTPUT_CSV = os.path.join("input", "areas.csv")


def parse_location(value: str, row_number: int, column_name: str) -> tuple[float, float]:
    """Parse a "lon,lat" string into two floats."""
    if value is None:
        raise ValueError(f"第 {row_number} 行 {column_name} 为空")

    parts = [part.strip() for part in str(value).strip().split(",")]
    if len(parts) != 2:
        raise ValueError(f"第 {row_number} 行 {column_name} 坐标格式错误：{value}")

    try:
        lon = float(parts[0])
        lat = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"第 {row_number} 行 {column_name} 坐标不是数字：{value}") from exc

    if not (-180 <= lon <= 180):
        raise ValueError(f"第 {row_number} 行 {column_name} 经度超出范围：{lon}")
    if not (-90 <= lat <= 90):
        raise ValueError(f"第 {row_number} 行 {column_name} 纬度超出范围：{lat}")

    return lon, lat


def get_required_column(fieldnames: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"CSV 缺少字段，候选字段：{', '.join(candidates)}")


def convert_build_csv(input_csv: str, output_csv: str) -> int:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV 为空：{input_csv}")

        fieldnames = [name.strip() for name in reader.fieldnames]
        reader.fieldnames = fieldnames

        name_col = get_required_column(fieldnames, ["map_name", "name"])
        start_col = get_required_column(fieldnames, ["start_location（左下）", "start_location", "left_bottom"])
        end_col = get_required_column(fieldnames, ["end_location（右上）", "end_location", "right_top"])

        rows = []
        for row_number, row in enumerate(reader, start=2):
            name = str(row.get(name_col, "")).strip()
            if not name:
                raise ValueError(f"第 {row_number} 行 {name_col} 为空")

            start_lon, start_lat = parse_location(row.get(start_col), row_number, start_col)
            end_lon, end_lat = parse_location(row.get(end_col), row_number, end_col)

            rows.append({
                "name": name,
                "min_lon": min(start_lon, end_lon),
                "min_lat": min(start_lat, end_lat),
                "max_lon": max(start_lon, end_lon),
                "max_lat": max(start_lat, end_lat),
            })

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "min_lon", "min_lat", "max_lon", "max_lat"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert build.csv coordinates to downloadOSM.py areas.csv format."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="输入 CSV，默认 input/build.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV, help="输出 CSV，默认 input/areas.csv")
    args = parser.parse_args()

    count = convert_build_csv(args.input, args.output)
    print(f"已转换 {count} 个区域：{args.output}")


if __name__ == "__main__":
    main()
