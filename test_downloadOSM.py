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
        self.assertEqual(downloadOSM.classify_feature(row), "高速")

    def test_trunk_link(self):
        row = {"highway": "trunk_link", "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), "高速")

    def test_primary(self):
        row = {"highway": "primary", "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), "主要道路")

    def test_secondary_link(self):
        row = {"highway": "secondary_link", "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), "主要道路")

    def test_residential(self):
        row = {"highway": "residential", "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), "次要道路")

    def test_subway(self):
        row = {"highway": None, "railway": "subway"}
        self.assertEqual(downloadOSM.classify_feature(row), "地铁")

    def test_rail(self):
        row = {"highway": None, "railway": "rail"}
        self.assertEqual(downloadOSM.classify_feature(row), "铁路")

    def test_light_rail(self):
        row = {"highway": None, "railway": "light_rail"}
        self.assertEqual(downloadOSM.classify_feature(row), "轻轨")

    def test_tram(self):
        row = {"highway": None, "railway": "tram"}
        self.assertEqual(downloadOSM.classify_feature(row), "轻轨")

    def test_unknown_road_fallback(self):
        row = {"highway": None, "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), "次要道路")

    def test_list_highway_value(self):
        row = {"highway": ["motorway"], "railway": None}
        self.assertEqual(downloadOSM.classify_feature(row), "高速")


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
            "osm_class": ["高速", "高速", "铁路"],
            "geometry": [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 3)]), LineString([(4, 4), (5, 5)])]
        }, geometry="geometry", crs="EPSG:4326")

        downloadOSM.export_by_class_to_geojson(gdf, "test_area", "output")

        self.assertEqual(mock_makedirs.call_count, 1)
        self.assertEqual(mock_to_file.call_count, 3)

    def test_export_empty_gdf(self):
        gdf = gpd.GeoDataFrame(columns=["osm_class", "geometry"], geometry="geometry", crs="EPSG:4326")

        with patch('downloadOSM.print') as mock_print:
            downloadOSM.export_by_class_to_geojson(gdf, "test_area", "output")
            mock_print.assert_called()

    def test_missing_class_field(self):
        gdf = gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1, 1)])]}, geometry="geometry", crs="EPSG:4326")

        with self.assertRaises(ValueError):
            downloadOSM.export_by_class_to_geojson(gdf, "test_area", "output")


if __name__ == "__main__":
    unittest.main()
