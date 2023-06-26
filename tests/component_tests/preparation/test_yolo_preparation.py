import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import csv

from modules.pre_processing.CSV_PreProcessor import (
    combine_csv,
    load_csv_from_path,
    load_and_preprocess_image,
    load_and_preprocess_from_path_label,
)


class TestDataPreparation(unittest.TestCase):
    # Class Variables #
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
    # Test 2: Passed (Implemented)
    def test_combine_csv(self) -> None:
        with open(self.output_dir, "r") as f:
            reader: csv._reader = csv.reader(f)
            lines: list[csv._reader] = list(reader)

        self.assertEqual(len(lines) - 1, 4)

    # Test 1: Failed (Not Implemented)
    # Test 2: Passed (Implemented)
    def test_load_csv_from_path(self):
        self.assertIsNotNone(self.xY[0])
        self.assertIsNotNone(self.xY[1])
        self.assertTrue(self.xY[0]["class"].notna().all())
        self.assertTrue(self.xY[0]["class"].apply(np.isreal).all())

        self.assertTrue(self.xY[1]["class"].notna().all())
        self.assertTrue(self.xY[1]["class"].apply(np.isreal).all())

    # Test 1: Failed (Not Implemented)
    # Test 2: Passed (Implemented)
    def test_load_and_preprocess_image(self):
        test_image_path: str = self.xY[0]["image_path"][0]
        processed_image: tf.Tensor = load_and_preprocess_image(test_image_path)

        self.assertIsInstance(processed_image, tf.Tensor)
        self.assertEqual(processed_image.shape, (416, 416, 3))
        self.assertTrue((processed_image.numpy() <= 1).all())
        self.assertTrue((processed_image.numpy() >= 0).all())

    # Test 1: Failed (Not Implemented)
    # Test 2: Passed (Implemented)
    def test_load_and_preprocess_from_path_label(self):
        test_image_path = self.xY[0]["image_path"][0]
        test_label = self.xY[0].iloc[0].drop("image_path").to_dict()

        image, label = load_and_preprocess_from_path_label(test_image_path, test_label)

        self.assertIsInstance(image, tf.Tensor)
        self.assertEqual(image.shape, (416, 416, 3))
        self.assertEqual(label, test_label)
        self.assertTrue((image.numpy() <= 1).all())
        self.assertTrue((image.numpy() >= 0).all())

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.output_dir):
            os.remove(cls.output_dir)


if __name__ == "__main__":
    unittest.main()
