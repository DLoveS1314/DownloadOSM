import os
import time
import traceback
from typing import Any, Optional


def configure_proj_data_dir() -> None:
    """
    Fix pyproj/PROJ database lookup issues in some conda environments.

    On Windows conda installs, PROJ's database is usually under
    <conda-env>/Library/share/proj. If PROJ_LIB is missing, pyproj may fail
    with: "proj_create: no database context specified".
    """
    possible_dirs = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        possible_dirs.append(os.path.join(conda_prefix, "Library", "share", "proj"))
        possible_dirs.append(os.path.join(conda_prefix, "share", "proj"))

    possible_dirs.extend([
        os.path.join(os.sys.prefix, "Library", "share", "proj"),
        os.path.join(os.sys.prefix, "share", "proj"),
    ])

    current_proj_dir = os.environ.get("PROJ_LIB") or os.environ.get("PROJ_DATA")
    if current_proj_dir:
        possible_dirs.append(current_proj_dir)

    for proj_dir in possible_dirs:
        if os.path.exists(os.path.join(proj_dir, "proj.db")):
            os.environ["PROJ_LIB"] = proj_dir
            os.environ["PROJ_DATA"] = proj_dir
            try:
                from pyproj import datadir

                datadir.set_data_dir(proj_dir)
            except Exception:
                pass
            break


configure_proj_data_dir()

import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box


# =========================================================
# 1. OSM 六类映射规则
# =========================================================

ROAD_CLASS_MAP = {
    # 高速
    "motorway": "高速",
    "motorway_link": "高速",
    "trunk": "高速",
    "trunk_link": "高速",

    # 主要道路
    "primary": "主要道路",
    "primary_link": "主要道路",
    "secondary": "主要道路",
    "secondary_link": "主要道路",

    # 次要道路
    "tertiary": "次要道路",
    "tertiary_link": "次要道路",
    "residential": "次要道路",
    "unclassified": "次要道路",
    "service": "次要道路",
    "living_street": "次要道路",
    "road": "次要道路",
    "track": "次要道路",
    "path": "次要道路",
    "pedestrian": "次要道路",
    "footway": "次要道路",
    "cycleway": "次要道路",
    "steps": "次要道路",
}

RAIL_CLASS_MAP = {
    # 地铁
    "subway": "地铁",

    # 铁路
    "rail": "铁路",
    "narrow_gauge": "铁路",

    # 轻轨
    "light_rail": "轻轨",
    "tram": "轻轨",
    "monorail": "轻轨",
}


# =========================================================
# 2. 基础工具函数
# =========================================================

def normalize_osm_value(value: Any) -> Optional[str]:
    """
    OSM 字段有时是字符串，有时是 list。
    如果是 list，优先返回第一个值。
    """
    if value is None:
        return None

    if isinstance(value, list):
        if len(value) == 0:
            return None
        return str(value[0])

    if pd.isna(value):
        return None

    return str(value)


def classify_feature(row) -> str:
    """
    OSM 路网六类分类规则：

    1. 先判断 highway 是否属于高速；
    2. 再判断 highway 是否属于主要道路；
    3. 再判断 highway 是否属于次要道路；
    4. 如果道路没有明确标注或不在道路映射表中，再判断 railway；
    5. railway=subway 归为地铁；
    6. railway=rail/narrow_gauge 归为铁路；
    7. railway=light_rail/tram/monorail 归为轻轨；
    8. 如果都不是，归为次要道路。
    """

    highway = normalize_osm_value(row.get("highway"))
    railway = normalize_osm_value(row.get("railway"))

    # 1. 先判断道路等级
    if highway in ["motorway", "motorway_link", "trunk", "trunk_link"]:
        return "高速"

    if highway in ["primary", "primary_link", "secondary", "secondary_link"]:
        return "主要道路"

    if highway in [
        "tertiary",
        "tertiary_link",
        "residential",
        "unclassified",
        "service",
        "living_street",
        "road",
        "track",
        "path",
        "pedestrian",
        "footway",
        "cycleway",
        "steps",
    ]:
        return "次要道路"

    # 2. 如果没有明确道路等级，再判断轨道交通
    if railway == "subway":
        return "地铁"

    if railway in ["rail", "narrow_gauge"]:
        return "铁路"

    if railway in ["light_rail", "tram", "monorail"]:
        return "轻轨"

    # 3. 兜底：都没有识别到，归为次要道路
    return "次要道路"


def safe_filename(text: str) -> str:
    """
    清理文件名中的非法字符。
    """
    text = str(text)
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        text = text.replace(ch, "_")
    return text.strip()


def extract_linework(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep all linework returned by OSM.

    OSMnx may return LineString, MultiLineString, GeometryCollection, or area
    geometries for highway/railway tags. A strict LineString-only filter can
    drop visible road features, so this function extracts usable linework from
    every returned geometry.
    """

    def line_parts(geometry):
        if geometry is None or geometry.is_empty:
            return []

        geom_type = geometry.geom_type
        if geom_type == "LineString":
            return [geometry]
        if geom_type == "MultiLineString":
            return list(geometry.geoms)
        if geom_type == "GeometryCollection":
            parts = []
            for sub_geometry in geometry.geoms:
                parts.extend(line_parts(sub_geometry))
            return parts
        if geom_type in ["Polygon", "MultiPolygon"]:
            return line_parts(geometry.boundary)

        return []

    rows = []
    for _, row in gdf.iterrows():
        for geometry in line_parts(row.geometry):
            new_row = row.copy()
            new_row.geometry = geometry
            rows.append(new_row)

    if not rows:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry="geometry", crs=gdf.crs)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)


# =========================================================
# 3. 下载 OSM 路网
# =========================================================

def download_osm_network_by_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> gpd.GeoDataFrame:
    """
    按 bbox 下载 OSM 路网数据。
    只下载 highway 和 railway，不下载河流、边界、电力线等其他线要素。

    bbox 顺序：
    min_lon, min_lat, max_lon, max_lat
    """

    tags = {
        "highway": True,
        "railway": True,
    }

    # OSMnx bbox 顺序：left, bottom, right, top
    bbox = (min_lon, min_lat, max_lon, max_lat)

    gdf = ox.features.features_from_bbox(
        bbox=bbox,
        tags=tags,
    )

    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(columns=["osm_class", "geometry"], geometry="geometry", crs="EPSG:4326")

    # 确保 CRS 是 EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    original_count = len(gdf)
    original_geom_types = gdf.geometry.geom_type.value_counts().to_dict()

    # 提取所有可用线要素，避免 GeometryCollection/面状道路被误删。
    gdf = extract_linework(gdf)

    if gdf.empty:
        print(f"[诊断] bbox 原始返回 {original_count} 条，但没有可提取线要素。几何类型：{original_geom_types}")
        return gpd.GeoDataFrame(columns=["osm_class", "geometry"], geometry="geometry", crs="EPSG:4326")

    # 裁剪到 bbox 范围内，避免跨区域长道路整段导出影响判断。
    bbox_polygon = gpd.GeoDataFrame(
        geometry=[box(min_lon, min_lat, max_lon, max_lat)],
        crs="EPSG:4326",
    )
    gdf = gpd.clip(gdf, bbox_polygon)

    if gdf.empty:
        print(f"[诊断] bbox 原始返回 {original_count} 条，提取线要素后裁剪为空。几何类型：{original_geom_types}")
        return gpd.GeoDataFrame(columns=["osm_class", "geometry"], geometry="geometry", crs="EPSG:4326")

    # 重置索引，避免 element/osmid 多级索引影响导出
    gdf = gdf.reset_index(drop=True)

    print(f"[诊断] 原始返回 {original_count} 条，提取/裁剪后 {len(gdf)} 条，原始几何类型：{original_geom_types}")

    # 分类
    gdf["osm_class"] = gdf.apply(classify_feature, axis=1)

    return gdf


# =========================================================
# 4. 按类别导出 GeoJSON
# =========================================================

def export_by_class_to_geojson(
    gdf: gpd.GeoDataFrame,
    area_name: str,
    output_dir: str,
    class_field: str = "osm_class",
) -> None:
    """
    按类别分别导出 GeoJSON。

    输出示例：
    output/
    └─ beijing_test/
       ├─ beijing_test_高速.geojson
       ├─ beijing_test_主要道路.geojson
       ├─ beijing_test_次要道路.geojson
       ├─ beijing_test_铁路.geojson
       ├─ beijing_test_地铁.geojson
       └─ beijing_test_轻轨.geojson
    """

    if gdf is None or gdf.empty:
        print(f"[跳过] {area_name}：输入数据为空")
        return

    if class_field not in gdf.columns:
        raise ValueError(f"缺少分类字段：{class_field}")

    if "geometry" not in gdf.columns:
        raise ValueError("缺少 geometry 字段")

    area_name_safe = safe_filename(area_name)
    area_output_dir = os.path.join(output_dir, area_name_safe)
    os.makedirs(area_output_dir, exist_ok=True)

    # 去掉空几何和无效几何
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        print(f"[跳过] {area_name}：没有有效几何")
        return

    # GeoJSON 建议使用 EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    keep_cols = []
    for col in [
        "osm_class",
        "highway",
        "railway",
        "name",
        "ref",
        "oneway",
        "bridge",
        "tunnel",
        "maxspeed",
        "lanes",
        "geometry",
    ]:
        if col in gdf.columns:
            keep_cols.append(col)

    all_geojson_path = os.path.join(
        area_output_dir,
        f"{area_name_safe}_全部路网.geojson",
    )
    gdf[keep_cols].copy().to_file(
        all_geojson_path,
        driver="GeoJSON",
    )
    print(f"[已导出] {all_geojson_path}")

    classes_order = ["高速", "主要道路", "次要道路", "铁路", "地铁", "轻轨"]

    for cls in classes_order:
        sub = gdf[gdf[class_field] == cls].copy()

        if sub.empty:
            print(f"[无数据] {area_name} - {cls}")
            continue

        # GeoJSON 字段尽量保留必要字段，避免过多 OSM 字段导致文件过大
        sub = sub[keep_cols].copy()

        cls_safe = safe_filename(cls)
        geojson_path = os.path.join(
            area_output_dir,
            f"{area_name_safe}_{cls_safe}.geojson"
        )

        sub.to_file(
            geojson_path,
            driver="GeoJSON",
        )

        print(f"[已导出] {geojson_path}")


def export_bbox_to_geojson(
    area_name: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    output_dir: str,
) -> None:
    """
    导出当前区域的 bbox 网格范围 GeoJSON。
    """

    area_name_safe = safe_filename(area_name)
    area_output_dir = os.path.join(output_dir, area_name_safe)
    os.makedirs(area_output_dir, exist_ok=True)

    bbox_gdf = gpd.GeoDataFrame(
        [{
            "name": area_name,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "geometry": box(min_lon, min_lat, max_lon, max_lat),
        }],
        geometry="geometry",
        crs="EPSG:4326",
    )

    bbox_path = os.path.join(area_output_dir, f"{area_name_safe}_网格范围.geojson")
    bbox_gdf.to_file(bbox_path, driver="GeoJSON")
    print(f"[已导出] {bbox_path}")


# =========================================================
# 5. 处理单个区域
# =========================================================

def process_one_area(
    area_name: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    output_dir: str,
) -> None:
    """
    下载并导出单个区域的 OSM 路网。
    """

    print("=" * 80)
    print(f"[开始] {area_name}")
    print(f"范围：min_lon={min_lon}, min_lat={min_lat}, max_lon={max_lon}, max_lat={max_lat}")

    try:
        export_bbox_to_geojson(
            area_name=area_name,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            output_dir=output_dir,
        )

        gdf = download_osm_network_by_bbox(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
        )

        if gdf.empty:
            print(f"[警告] {area_name}：没有下载到路网数据")
            return

        print(f"[下载完成] {area_name}：共 {len(gdf)} 条线要素")

        # 打印分类统计
        class_count = gdf["osm_class"].value_counts()
        print("[分类统计]")
        for cls, count in class_count.items():
            print(f"  {cls}: {count}")

        export_by_class_to_geojson(
            gdf=gdf,
            area_name=area_name,
            output_dir=output_dir,
        )

        print(f"[完成] {area_name}")

    except Exception as e:
        print(f"[错误] {area_name} 处理失败：{e}")
        traceback.print_exc()


# =========================================================
# 6. 批量处理 CSV
# =========================================================

def batch_process_from_csv(
    csv_path: str = "input/areas.csv",
    output_dir: str = "output",
    sleep_seconds: float = 10.0,
) -> None:
    """
    从 CSV 批量读取区域并处理。

    CSV 字段要求：
    name,min_lon,min_lat,max_lon,max_lat
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")

    areas = pd.read_csv(csv_path)

    required_cols = ["name", "min_lon", "min_lat", "max_lon", "max_lat"]
    for col in required_cols:
        if col not in areas.columns:
            raise ValueError(f"CSV 缺少字段：{col}")

    os.makedirs(output_dir, exist_ok=True)

    total = len(areas)
    print(f"共读取 {total} 个区域")

    for index, row in areas.iterrows():
        current = index + 1
        percent = current / total if total else 1
        bar_width = 30
        filled_width = int(bar_width * percent)
        progress_bar = "#" * filled_width + "-" * (bar_width - filled_width)

        print(f"\n[进度] {current}/{total} [{progress_bar}] {percent:.1%}")
        area_name = str(row["name"])

        process_one_area(
            area_name=area_name,
            min_lon=float(row["min_lon"]),
            min_lat=float(row["min_lat"]),
            max_lon=float(row["max_lon"]),
            max_lat=float(row["max_lat"]),
            output_dir=output_dir,
        )

        # 避免连续请求 Overpass 太频繁
        if current < total and sleep_seconds > 0:
            print(f"[等待] {sleep_seconds:g} 秒后继续下一个区域")
            time.sleep(sleep_seconds)


# =========================================================
# 7. 程序入口
# =========================================================

if __name__ == "__main__":
    # 可按需修改
    CSV_PATH = "input/areas.csv"
    OUTPUT_DIR = "output1"

    # OSMnx 设置
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.timeout = 180

    batch_process_from_csv(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        sleep_seconds=10.0,
    )
