import unittest
import pandas as pd
import os
import shutil

from modules.pre_processing.CSV_PreProcessor import (
    load_csv_from_path,
    combine_csv
)

from modules.pre_processing.CSV_Conversion import (
    csv_to_model_format,
    yolo_to_rcnn_bbox
)

class TestConversionDataPreparation(unittest.TestCase):
    prefix_dir: str = "./tests/component_tests/preparation"
    prefix_dir_csv: str = f'{prefix_dir}/csv/mapping/'
    prefix_dir_output: str = f'{prefix_dir}/model_conversion/'

    output_dir_csv: str = os.path.join(prefix_dir_csv, "output.csv")
    output_dir_yolo: str = os.path.join(prefix_dir_output, "output.yolo")
    output_dir_rcnn: str = os.path.join(prefix_dir_output, "output.rcnn")

    def setUp(self) -> None:
        self.df = pd.DataFrame()

        output_file: list[str] = [
            os.path.join(self.prefix_dir_csv, "test1.csv"),
            os.path.join(self.prefix_dir_csv, "test2.csv"),
        ]

        combine_csv(output_file, self.output_dir_csv)

        self.assertTrue(os.path.exists(self.output_dir_csv))

        train_df, test_df = load_csv_from_path(self.output_dir_csv)

        self.assertIsInstance(train_df, pd.DataFrame)
        self.assertIsInstance(test_df, pd.DataFrame)

        self.xY = (train_df, test_df)

    # Test 1: Failed (Not Implemented)
    # Test 2: Passed (Implemented)
    def test_csv_to_yolo(self) :
        csv_to_model_format(self.output_dir_csv, self.output_dir_yolo)

    # Test 1: Failed (Not Implemented)
    # Test 2: Passed (Implemented)
    def test_yolo_to_rcnn_bbox(self) :
        result = yolo_to_rcnn_bbox(['0.5', '0.5', '0.2', '0.2'])
        self.assertListEqual(result, [0.4, 0.4, 0.6, 0.6])
        
    # Test 1: Failed (Not Implemented)
    # Test 2: Passed (Implemented)
    def test_csv_to_rcnn(self) :
        csv_to_model_format(self.output_dir_csv, self.output_dir_rcnn, 'RCNN')

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isdir(cls.output_dir_yolo):
                shutil.rmtree(cls.output_dir_yolo)
        if os.path.isdir(cls.output_dir_rcnn):
            shutil.rmtree(cls.output_dir_rcnn)
        if os.path.exists(cls.output_dir_csv):
            os.remove(cls.output_dir_csv)
        
if __name__ == "__main__":
    unittest.main()