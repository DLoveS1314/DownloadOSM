import os
import time
import traceback
from collections import defaultdict
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
from shapely.geometry import LineString


# =========================================================
# 1. OSM 六类映射规则
# =========================================================

# Chinese class labels stored with unicode escapes to keep source encoding safe.
CLASS_EXPRESSWAY = "\u9ad8\u901f"  # expressway / trunk road
CLASS_MAIN_ROAD = "\u4e3b\u8981\u9053\u8def"  # main road
CLASS_MINOR_ROAD = "\u6b21\u8981\u9053\u8def"  # minor road by OSM highway category
CLASS_OTHER_ROAD = "\u5176\u4ed6\u9053\u8def"  # path / unknown or uncategorized road
CLASS_RAILWAY = "\u94c1\u8def"  # railway
CLASS_SUBWAY = "\u5730\u94c1"  # subway
CLASS_LIGHT_RAIL = "\u8f7b\u8f68"  # light rail / tram / monorail

EXPRESSWAY_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link"}
MAIN_ROAD_HIGHWAYS = {"primary", "primary_link", "secondary", "secondary_link"}
MINOR_ROAD_HIGHWAYS = {
    "tertiary",
    "tertiary_link",
    "residential",
    "unclassified",
    "service",
    "living_street",
}
OTHER_ROAD_HIGHWAYS = {
    "road",
    "track",
    "path",
    "pedestrian",
    "footway",
    "cycleway",
    "steps",
}

ROAD_CLASS_MAP = {
    **{highway: CLASS_EXPRESSWAY for highway in EXPRESSWAY_HIGHWAYS},
    **{highway: CLASS_MAIN_ROAD for highway in MAIN_ROAD_HIGHWAYS},
    **{highway: CLASS_MINOR_ROAD for highway in MINOR_ROAD_HIGHWAYS},
    **{highway: CLASS_OTHER_ROAD for highway in OTHER_ROAD_HIGHWAYS},
}

RAIL_CLASS_MAP = {
    "subway": CLASS_SUBWAY,
    "rail": CLASS_RAILWAY,
    "narrow_gauge": CLASS_RAILWAY,
    "light_rail": CLASS_LIGHT_RAIL,
    "tram": CLASS_LIGHT_RAIL,
    "monorail": CLASS_LIGHT_RAIL,
}

CLASS_FILENAME_SUFFIX = {
    CLASS_EXPRESSWAY: CLASS_EXPRESSWAY,
    CLASS_MAIN_ROAD: CLASS_MAIN_ROAD,
    CLASS_MINOR_ROAD: CLASS_MINOR_ROAD,
    CLASS_OTHER_ROAD: CLASS_OTHER_ROAD,
    CLASS_RAILWAY: CLASS_RAILWAY,
    CLASS_SUBWAY: CLASS_SUBWAY,
    CLASS_LIGHT_RAIL: CLASS_LIGHT_RAIL,
}

UNDERGROUND_ROUTE_CLASSES = {CLASS_RAILWAY, CLASS_SUBWAY, CLASS_LIGHT_RAIL}


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
    if highway in EXPRESSWAY_HIGHWAYS:
        return CLASS_EXPRESSWAY

    if highway in MAIN_ROAD_HIGHWAYS:
        return CLASS_MAIN_ROAD

    if highway in MINOR_ROAD_HIGHWAYS:
        return CLASS_MINOR_ROAD

    if highway in OTHER_ROAD_HIGHWAYS:
        return CLASS_OTHER_ROAD

    # 2. 如果没有明确道路等级，再判断轨道交通
    if railway == "subway":
        return CLASS_SUBWAY

    if railway in ["rail", "narrow_gauge"]:
        return CLASS_RAILWAY

    if railway in ["light_rail", "tram", "monorail"]:
        return CLASS_LIGHT_RAIL

    # 3. 兜底：都没有识别到，归为其他道路
    return CLASS_OTHER_ROAD


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


def _project_for_meter_operations(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    try:
        projected_crs = gdf.estimate_utm_crs()
    except Exception:
        projected_crs = None

    if projected_crs is None:
        projected_crs = "EPSG:3857"

    return gdf.to_crs(projected_crs)


def _node_key(coord, tolerance: float):
    if tolerance <= 0:
        return (coord[0], coord[1])
    return (round(coord[0] / tolerance), round(coord[1] / tolerance))


def _line_vertex_count(geometry) -> int:
    if geometry is None or geometry.is_empty:
        return 0
    if geometry.geom_type == "LineString":
        return len(geometry.coords)
    if geometry.geom_type == "MultiLineString":
        return sum(len(part.coords) for part in geometry.geoms)
    return 0


def remove_dead_end_segments(
    gdf: gpd.GeoDataFrame,
    node_snap_tolerance_m: float = 1.0,
    dead_end_max_length_m: Optional[float] = 80.0,
) -> gpd.GeoDataFrame:
    """
    Remove short dangling line segments from all route classes.

    A dangling segment is an edge whose one endpoint has only one neighbor in
    the snapped line graph. Set dead_end_max_length_m to None to remove every
    dangling edge regardless of length.
    """
    if gdf is None or gdf.empty:
        return gdf

    original_crs = gdf.crs
    projected = _project_for_meter_operations(extract_linework(gdf))

    segment_rows = []

    for _, row in projected.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty or geometry.geom_type != "LineString":
            continue

        coords = list(geometry.coords)
        for coord_index in range(len(coords) - 1):
            start = coords[coord_index]
            end = coords[coord_index + 1]
            if start == end:
                continue

            start_key = _node_key(start, node_snap_tolerance_m)
            end_key = _node_key(end, node_snap_tolerance_m)
            if start_key == end_key:
                continue

            segment = LineString([start, end])
            edge_id = len(segment_rows)
            segment_rows.append({
                "row": row,
                "geometry": segment,
                "start": start_key,
                "end": end_key,
                "length": segment.length,
            })

    if not segment_rows:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry="geometry", crs=original_crs)

    active = set(range(len(segment_rows)))

    while True:
        degree = defaultdict(int)
        for edge_id in active:
            segment = segment_rows[edge_id]
            degree[segment["start"]] += 1
            degree[segment["end"]] += 1

        to_remove = set()
        for edge_id in active:
            segment = segment_rows[edge_id]
            is_dangling = degree[segment["start"]] <= 1 or degree[segment["end"]] <= 1
            is_short_enough = (
                dead_end_max_length_m is None
                or segment["length"] <= dead_end_max_length_m
            )
            if is_dangling and is_short_enough:
                to_remove.add(edge_id)

        if not to_remove:
            break

        active -= to_remove

    cleaned_rows = []
    for edge_id in sorted(active):
        original_row = segment_rows[edge_id]["row"].copy()
        original_row.geometry = segment_rows[edge_id]["geometry"]
        cleaned_rows.append(original_row)

    if not cleaned_rows:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry="geometry", crs=original_crs)

    cleaned = gpd.GeoDataFrame(cleaned_rows, geometry="geometry", crs=projected.crs)
    if original_crs is not None:
        cleaned = cleaned.to_crs(original_crs)
    return cleaned.reset_index(drop=True)


def _sample_line_distances(line_a, line_b, sample_count: int = 9) -> list[float]:
    if line_a.length == 0:
        return [line_a.distance(line_b)]

    distances = []
    for index in range(sample_count):
        fraction = index / (sample_count - 1) if sample_count > 1 else 0
        point = line_a.interpolate(fraction * line_a.length)
        distances.append(point.distance(line_b))
    return distances


def _are_near_duplicate_lines(
    line_a,
    line_b,
    parallel_tolerance_m: float,
    length_ratio_threshold: float,
) -> bool:
    if line_a.is_empty or line_b.is_empty or line_a.length == 0 or line_b.length == 0:
        return False

    length_ratio = min(line_a.length, line_b.length) / max(line_a.length, line_b.length)
    if length_ratio < length_ratio_threshold:
        return False

    if line_a.distance(line_b) > parallel_tolerance_m:
        return False

    distances = (
        _sample_line_distances(line_a, line_b)
        + _sample_line_distances(line_b, line_a)
    )
    mean_distance = sum(distances) / len(distances)
    max_distance = max(distances)

    return (
        mean_distance <= parallel_tolerance_m
        and max_distance <= parallel_tolerance_m * 3
    )


def _route_quality_score(row) -> float:
    geometry = row.geometry
    length_score = geometry.length if geometry is not None and not geometry.is_empty else 0.0
    vertex_score = _line_vertex_count(geometry) * 5.0
    attr_score = 0.0
    for col in ["name", "ref", "railway", "highway", "tunnel", "bridge", "oneway"]:
        if col in row.index and normalize_osm_value(row.get(col)) is not None:
            attr_score += 100.0
    return length_score + vertex_score + attr_score


def merge_nearby_underground_return_lines(
    gdf: gpd.GeoDataFrame,
    parallel_tolerance_m: float = 12.0,
    length_ratio_threshold: float = 0.75,
) -> gpd.GeoDataFrame:
    """
    Keep the best route from near-duplicate underground railway line pairs.

    Only features classified as subway, light rail, or railway are considered.
    Surface roads are never merged by this function.
    """
    if gdf is None or gdf.empty or "osm_class" not in gdf.columns:
        return gdf

    original_crs = gdf.crs
    projected = _project_for_meter_operations(extract_linework(gdf)).reset_index(drop=True)
    candidate_indexes = [
        index
        for index, row in projected.iterrows()
        if row.get("osm_class") in UNDERGROUND_ROUTE_CLASSES
        and row.geometry is not None
        and not row.geometry.is_empty
        and row.geometry.geom_type == "LineString"
    ]

    if len(candidate_indexes) < 2:
        return gdf.reset_index(drop=True)

    parent = {index: index for index in candidate_indexes}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_pos, left_index in enumerate(candidate_indexes):
        left_line = projected.at[left_index, "geometry"]
        for right_index in candidate_indexes[left_pos + 1:]:
            right_line = projected.at[right_index, "geometry"]
            if _are_near_duplicate_lines(
                left_line,
                right_line,
                parallel_tolerance_m=parallel_tolerance_m,
                length_ratio_threshold=length_ratio_threshold,
            ):
                union(left_index, right_index)

    clusters = defaultdict(list)
    for index in candidate_indexes:
        clusters[find(index)].append(index)

    indexes_to_drop = set()
    for indexes in clusters.values():
        if len(indexes) < 2:
            continue
        best_index = max(indexes, key=lambda idx: _route_quality_score(projected.loc[idx]))
        indexes_to_drop.update(index for index in indexes if index != best_index)

    if not indexes_to_drop:
        return gdf.reset_index(drop=True)

    cleaned = projected.drop(index=list(indexes_to_drop)).copy()
    if original_crs is not None:
        cleaned = cleaned.to_crs(original_crs)
    return cleaned.reset_index(drop=True)


def clean_route_network(
    gdf: gpd.GeoDataFrame,
    node_snap_tolerance_m: float = 1.0,
    dead_end_max_length_m: Optional[float] = 80.0,
    underground_parallel_tolerance_m: float = 12.0,
) -> gpd.GeoDataFrame:
    """
    Clean downloaded route linework.

    1. Remove dangling segments for every road/rail class.
    2. Merge near duplicate subway/light-rail/railway return lines by keeping
       the best-quality representative.
    """
    without_dead_ends = remove_dead_end_segments(
        gdf,
        node_snap_tolerance_m=node_snap_tolerance_m,
        dead_end_max_length_m=dead_end_max_length_m,
    )
    return merge_nearby_underground_return_lines(
        without_dead_ends,
        parallel_tolerance_m=underground_parallel_tolerance_m,
    )


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
    before_clean_count = len(gdf)
    gdf = clean_route_network(gdf)
    print(f"[cleaned] route features {before_clean_count} -> {len(gdf)}")

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
        f"{area_name_safe}_\u5168\u90e8\u8def\u7f51.geojson",
    )
    gdf[keep_cols].copy().to_file(
        all_geojson_path,
        driver="GeoJSON",
    )
    print(f"[已导出] {all_geojson_path}")

    classes_order = [
        CLASS_EXPRESSWAY,
        CLASS_MAIN_ROAD,
        CLASS_MINOR_ROAD,
        CLASS_OTHER_ROAD,
        CLASS_RAILWAY,
        CLASS_SUBWAY,
        CLASS_LIGHT_RAIL,
    ]

    for cls in classes_order:
        sub = gdf[gdf[class_field] == cls].copy()

        if sub.empty:
            print(f"[无数据] {area_name} - {cls}")
            continue

        # GeoJSON 字段尽量保留必要字段，避免过多 OSM 字段导致文件过大
        sub = sub[keep_cols].copy()

        cls_safe = CLASS_FILENAME_SUFFIX.get(cls, safe_filename(cls))
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

    bbox_path = os.path.join(area_output_dir, f"{area_name_safe}_\u7f51\u683c\u8303\u56f4.geojson")
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

[116.2475,39.97704]
[116.295524,40.016667]
# =========================================================
# 7. 程序入口
# =========================================================

if __name__ == "__main__":
    # 可按需修改
    CSV_PATH = "input/areas.csv"
    OUTPUT_DIR = "output1_20260515"

    # OSMnx 设置
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.timeout = 180

    batch_process_from_csv(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        sleep_seconds=10.0,
    )
