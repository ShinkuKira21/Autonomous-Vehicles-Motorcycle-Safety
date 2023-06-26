import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import os

from modules.pre_processing.CSV_PreProcessor import (
    combine_csv,
    load_csv_from_path,
    load_and_preprocess_image,
    load_and_preprocess_from_path_label,
)

class TestConversionDataPreparation(unittest.TestCase):
    prefix_dir = "./tests/component_tests/preparation/csv/mapping/"
    output_dir = os.path.join(prefix_dir, "output.csv")

    def setUp(self) -> None:
        self.df = pd.DataFrame()

        output_file: list[str] = [
            os.path.join(self.prefix_dir, "test1.csv"),
            os.path.join(self.prefix_dir, "test2.csv"),
        ]

        combine_csv(output_file, self.output_dir)

        self.assertTrue(os.path.exists(self.output_dir))

        train_df, test_df = load_csv_from_path(self.output_dir)

        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)

        self.xY = (train_df, test_df)

    # Test 1: Failed (Not Implemented)
    def test_csv_to_(self) :
        self.assertIsNone(True)
        
    # Test 1: Failed (Not Implemented)
    def test_csv_to_rcnn(self) :
        self.assertIsNone(True)

    # Test 1: Failed (Not Implemented)
    def test_yolo_to_rcnn_bbox(self) :
        self.assertIsNone(True)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.output_dir):
            os.remove(cls.output_dir)


if __name__ == "__main__":
    unittest.main()