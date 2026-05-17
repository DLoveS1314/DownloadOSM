import os
import unittest
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import LineString

import downloadOSM


class TestNormalizeOsmValue(unittest.TestCase):
    """测试 normalize_osm_value 函数"""

    def test_none_value(self):
        self.assertIsNone(downloadOSM.normalize_osm_value(None))

    def test_nan_value(self):
        import pandas as pd
        self.assertIsNone(downloadOSM.normalize_osm_value(pd.NA))

    def test_empty_list(self):
        self.assertIsNone(downloadOSM.normalize_osm_value([]))

    def test_list_with_single_value(self):
        self.assertEqual(downloadOSM.normalize_osm_value(["motorway"]), "motorway")

    def test_list_with_multiple_values(self):
        self.assertEqual(downloadOSM.normalize_osm_value(["motorway", "trunk"]), "motorway")

    def test_string_value(self):
        self.assertEqual(downloadOSM.normalize_osm_value("primary"), "primary")

    def test_numeric_value(self):
        self.assertEqual(downloadOSM.normalize_osm_value(123), "123")


class TestClassifyFeature(unittest.TestCase):
    """测试 classify_feature 函数"""

    def test_motorway(self):
        row = {"highway": "motorway", "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_EXPRESSWAY)

    def test_trunk_link(self):
        row = {"highway": "trunk_link", "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_EXPRESSWAY)

    def test_primary(self):
        row = {"highway": "primary", "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_MAIN_ROAD)

    def test_secondary_link(self):
        row = {"highway": "secondary_link", "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_MAIN_ROAD)

    def test_residential(self):
        row = {"highway": "residential", "railway": None, "name": "Test Road"}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_MINOR_ROAD)

    def test_unnamed_residential_goes_to_minor_roads(self):
        row = {"highway": "residential", "railway": None, "name": None}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_MINOR_ROAD)

    def test_named_service_goes_to_minor_roads(self):
        row = {"highway": "service", "railway": None, "name": "Service Road"}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_MINOR_ROAD)

    def test_unnamed_service_goes_to_minor_roads(self):
        row = {"highway": "service", "railway": None, "name": None}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_MINOR_ROAD)

    def test_path_like_roads_go_to_other_roads(self):
        for highway in ["footway", "path", "track", "road"]:
            with self.subTest(highway=highway):
                row = {"highway": highway, "railway": None, "name": "Park path"}
                self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_OTHER_ROAD)

    def test_subway(self):
        row = {"highway": None, "railway": "subway"}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_SUBWAY)

    def test_rail(self):
        row = {"highway": None, "railway": "rail"}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_RAILWAY)

    def test_light_rail(self):
        row = {"highway": None, "railway": "light_rail"}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_LIGHT_RAIL)

    def test_tram(self):
        row = {"highway": None, "railway": "tram"}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_LIGHT_RAIL)

    def test_unknown_road_fallback(self):
        row = {"highway": None, "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_OTHER_ROAD)

    def test_list_highway_value(self):
        row = {"highway": ["motorway"], "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), downloadOSM.CLASS_EXPRESSWAY)


class TestSafeFilename(unittest.TestCase):
    """测试 safe_filename 函数"""

    def test_normal_filename(self):
        self.assertEqual(downloadOSM.safe_filename("beijing"), "beijing")

    def test_filename_with_slash(self):
        self.assertEqual(downloadOSM.safe_filename("bei/jing"), "bei_jing")

    def test_filename_with_backslash(self):
        self.assertEqual(downloadOSM.safe_filename("bei\\jing"), "bei_jing")

    def test_filename_with_colon(self):
        self.assertEqual(downloadOSM.safe_filename("bei:jing"), "bei_jing")

    def test_filename_with_multiple_special_chars(self):
        self.assertEqual(downloadOSM.safe_filename("bei:jing/test"), "bei_jing_test")


class TestDownloadOsmNetworkByBBox(unittest.TestCase):
    """测试 download_osm_network_by_bbox 函数"""

    @patch('downloadOSM.ox.features.features_from_bbox')
    def test_download_success(self, mock_features):
        mock_gdf = gpd.GeoDataFrame({
            "highway": ["motorway", "residential"],
            "railway": [None, None],
            "name": [None, "Test Road"],
            "geometry": [
                LineString([(116.1, 39.1), (116.2, 39.2)]),
                LineString([(116.3, 39.3), (116.4, 39.4)]),
            ]
        }, geometry="geometry", crs="EPSG:4326")
        mock_features.return_value = mock_gdf

        result = downloadOSM.download_osm_network_by_bbox(
            min_lon=116.0, min_lat=39.0, max_lon=117.0, max_lat=40.0
        )

        self.assertFalse(result.empty)
        self.assertIn("osm_class", result.columns)

    @patch('downloadOSM.ox.features.features_from_bbox')
    def test_download_empty_result(self, mock_features):
        mock_features.return_value = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

        result = downloadOSM.download_osm_network_by_bbox(
            min_lon=116.0, min_lat=39.0, max_lon=117.0, max_lat=40.0
        )

        self.assertTrue(result.empty)


class TestExportByClassToGeojson(unittest.TestCase):
    """测试 export_by_class_to_class_to_geojson 函数"""

    @patch('downloadOSM.os.makedirs')
    @patch('downloadOSM.gpd.GeoDataFrame.to_file')
    def test_export_success(self, mock_to_file, mock_makedirs):
        gdf = gpd.GeoDataFrame({
            "osm_class": [
                downloadOSM.CLASS_EXPRESSWAY,
                downloadOSM.CLASS_EXPRESSWAY,
                downloadOSM.CLASS_RAILWAY,
            ],
            "geometry": [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)]), LineString([(4, 4), (5, 5)])]
        }, geometry="geometry", crs="EPSG:4326")

        downloadOSM.export_by_class_to_geojson(gdf, "test_area", "output")

        self.assertEqual(mock_makedirs.call_count, 1)
        self.assertEqual(mock_to_file.call_count, 3)
        output_paths = [call.args[0] for call in mock_to_file.call_args_list]
        self.assertIn(os.path.join("output", "test_area", "test_area_\u5168\u90e8\u8def\u7f51.geojson"), output_paths)
        self.assertIn(os.path.join("output", "test_area", "test_area_\u9ad8\u901f.geojson"), output_paths)
        self.assertIn(os.path.join("output", "test_area", "test_area_\u94c1\u8def.geojson"), output_paths)

    @patch('downloadOSM.os.makedirs')
    @patch('downloadOSM.gpd.GeoDataFrame.to_file')
    def test_export_other_roads_with_chinese_filename(self, mock_to_file, mock_makedirs):
        gdf = gpd.GeoDataFrame({
            "osm_class": [downloadOSM.CLASS_OTHER_ROAD],
            "geometry": [LineString([(0, 0), (1, 1)])]
        }, geometry="geometry", crs="EPSG:4326")

        downloadOSM.export_by_class_to_geojson(gdf, "test_area", "output")

        output_paths = [call.args[0] for call in mock_to_file.call_args_list]
        self.assertIn(os.path.join("output", "test_area", "test_area_\u5176\u4ed6\u9053\u8def.geojson"), output_paths)

    def test_export_empty_gdf(self):
        gdf = gpd.GeoDataFrame(columns=["osm_class", "geometry"], geometry="geometry", crs="EPSG:4326")

        with patch('downloadOSM.print') as mock_print:
            downloadOSM.export_by_class_to_geojson(gdf, "test_area", "output")
            mock_print.assert_called()

    def test_missing_class_field(self):
        gdf = gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1, 1)])]}, geometry="geometry", crs="EPSG:4326")

        with self.assertRaises(ValueError):
            downloadOSM.export_by_class_to_geojson(gdf, "test_area", "output")


class TestExportBboxToGeojson(unittest.TestCase):
    @patch('downloadOSM.os.makedirs')
    @patch('downloadOSM.gpd.GeoDataFrame.to_file')
    def test_export_bbox_uses_chinese_filename(self, mock_to_file, mock_makedirs):
        downloadOSM.export_bbox_to_geojson(
            area_name="test_area",
            min_lon=116.0,
            min_lat=39.0,
            max_lon=117.0,
            max_lat=40.0,
            output_dir="output",
        )

        self.assertEqual(
            mock_to_file.call_args.args[0],
            os.path.join("output", "test_area", "test_area_\u7f51\u683c\u8303\u56f4.geojson"),
        )


class TestCleanRouteNetwork(unittest.TestCase):
    def test_remove_dead_end_segments_applies_to_ordinary_road_classes(self):
        gdf = gpd.GeoDataFrame({
            "osm_class": [
                downloadOSM.CLASS_MAIN_ROAD,
                downloadOSM.CLASS_MAIN_ROAD,
                downloadOSM.CLASS_EXPRESSWAY,
            ],
            "geometry": [
                LineString([(0, 0), (100, 0)]),
                LineString([(100, 0), (200, 0)]),
                LineString([(100, 0), (100, 10)]),
            ],
        }, geometry="geometry", crs="EPSG:3857")

        result = downloadOSM.remove_dead_end_segments(
            gdf,
            node_snap_tolerance_m=0.01,
            dead_end_max_length_m=20,
        )

        self.assertEqual(len(result), 2)
        self.assertNotIn(downloadOSM.CLASS_EXPRESSWAY, set(result["osm_class"]))

    def test_remove_dead_end_segments_ignores_rail_classes(self):
        gdf = gpd.GeoDataFrame({
            "osm_class": [
                downloadOSM.CLASS_MAIN_ROAD,
                downloadOSM.CLASS_SUBWAY,
                downloadOSM.CLASS_RAILWAY,
                downloadOSM.CLASS_LIGHT_RAIL,
            ],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(-10, 0), (0, 0)]),
                LineString([(10, 0), (20, 0)]),
                LineString([(0, 5), (10, 5)]),
            ],
        }, geometry="geometry", crs="EPSG:3857")

        result = downloadOSM.remove_dead_end_segments(
            gdf,
            node_snap_tolerance_m=0.01,
            dead_end_max_length_m=20,
        )

        self.assertNotIn(downloadOSM.CLASS_MAIN_ROAD, set(result["osm_class"]))
        self.assertIn(downloadOSM.CLASS_SUBWAY, set(result["osm_class"]))
        self.assertIn(downloadOSM.CLASS_RAILWAY, set(result["osm_class"]))
        self.assertIn(downloadOSM.CLASS_LIGHT_RAIL, set(result["osm_class"]))

    def test_merge_nearby_underground_return_lines_keeps_best_route(self):
        gdf = gpd.GeoDataFrame({
            "osm_class": [
                downloadOSM.CLASS_SUBWAY,
                downloadOSM.CLASS_SUBWAY,
                downloadOSM.CLASS_MAIN_ROAD,
            ],
            "name": ["Line A", None, "Surface road"],
            "railway": ["subway", "subway", None],
            "geometry": [
                LineString([(0, 0), (50, 0), (100, 0)]),
                LineString([(0, 5), (100, 5)]),
                LineString([(0, 2), (100, 2)]),
            ],
        }, geometry="geometry", crs="EPSG:3857")

        result = downloadOSM.merge_nearby_underground_return_lines(
            gdf,
            parallel_tolerance_m=8,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual((result["osm_class"] == downloadOSM.CLASS_SUBWAY).sum(), 1)
        self.assertEqual((result["osm_class"] == downloadOSM.CLASS_MAIN_ROAD).sum(), 1)
        kept_subway = result[result["osm_class"] == downloadOSM.CLASS_SUBWAY].iloc[0]
        self.assertEqual(kept_subway["name"], "Line A")


class TestMap10Download(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("RUN_OSM_DOWNLOAD_TESTS") == "1",
        "set RUN_OSM_DOWNLOAD_TESTS=1 to run live OSM download tests",
    )
    def test_download_only_map_10_to_test_directory(self):
        output_dir = os.path.join(".", "test")

        downloadOSM.process_one_area(
            area_name="map_10",
            min_lon=116.2475,
            min_lat=39.97704,
            max_lon=116.295524,
            max_lat=40.016667,
            output_dir=output_dir,
        )

        map_10_dir = os.path.join(output_dir, "map_10")
        all_roads_path = os.path.join(map_10_dir, "map_10_\u5168\u90e8\u8def\u7f51.geojson")

        self.assertTrue(os.path.isdir(map_10_dir))
        self.assertTrue(os.path.exists(all_roads_path))
        self.assertFalse(gpd.read_file(all_roads_path).empty)


if __name__ == "__main__":
    TestMap10Download()
    # unittest.main()
