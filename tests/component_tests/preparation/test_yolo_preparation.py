import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import csv

from modules.pre_processing.CSV_PreProcessor import combine_csv


class TestDataPreparation(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame()
        self.prefix_dir = "./tests/component_tests/preparation/csv/mapping/"
        print(os.getcwd())
        self.output_dir = os.path.join(self.prefix_dir, "output.csv")

    # Test 1: Failed (Not Implemented)
    # Test 2: Passed (Implemented)
    def test_combine_csv(self) -> None:
        output_file: list[str] = [
            os.path.join(self.prefix_dir, "test1.csv"),
            os.path.join(self.prefix_dir, "test2.csv"),
        ]

        combine_csv(output_file, self.output_dir)

        self.assertTrue(os.path.exists(self.output_dir))

        with open(self.output_dir, "r") as f:
            reader: csv._reader = csv.reader(f)
            lines: list[csv._reader] = list(reader)

        self.assertEqual(len(lines) - 1, 4)

    def tearDown(self) -> None:
        if os.path.exists(self.output_dir):
            os.remove(self.output_dir)


if __name__ == "__main__":
    unittest.main()
