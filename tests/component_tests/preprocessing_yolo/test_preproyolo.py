import unittest
import os
import csv
from PIL import Image
import tempfile
from modules.pre_processing.CSV_Creation_YOLO import (
    read_image,
    read_labels,
    normalise_labels,
    get_config,
    get_data,
)

from modules.pre_processing.CSV_PreProcessor import preprocess_to_csv


# YOLO uses <object-class> <x> <y> <width> <height> classification labels to identify the frames.
# The test case ensures that the methods implemented will work properly for YOLO classification.
class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary directory
        self.test_dir: str = tempfile.mkdtemp()

        # Create images and labels subdirectories
        self.images_dir: str = os.path.join(self.test_dir, "images")
        self.labels_dir: str = os.path.join(self.test_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Create a dummy config file
        self.config_path: str = os.path.join(self.test_dir, "data.yml")
        with open(self.config_path, "w") as f:
            f.write("names: ['car', 'bus', 'motorcycle']\nnc: 3")

        # Create a dummy image file in the images subdirectory
        self.image_path: str = os.path.join(self.images_dir, "image.jpg")
        image: Image = Image.new("RGB", (100, 100))
        image.save(self.image_path)

        # Create a dummy label file in the labels subdirectory
        self.label_path: str = os.path.join(self.labels_dir, "image.txt")
        with open(self.label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n2 0.7 0.7 0.3 0.3")

        # Define output_path as a temporary CSV file
        self.output_dir: str = os.path.join(self.test_dir, "output.csv")

    # Test 1: Failed (No Implementation)
    # Test 2: Passed (Implemented)
    def test_read_labels(self) -> None:
        labels: list = read_labels(self.label_path)
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0], (0, 0.5, 0.5, 0.2, 0.2))

    # Test 1: Failed (No Implementation)
    # Test 2: Passed (Implemented)
    def test_read_image(self) -> None:
        image: list = read_image(self.image_path)
        self.assertEqual(image.size, (100, 100))

    # Test 1: Failed (No Implementation)
    # Test 2: Passed (Implemented)
    def test_normalise_labels(self) -> None:
        labels: list[tuple] = [(0, 50, 50, 20, 20)]
        normalised_labels = normalise_labels(labels, (100, 100))
        self.assertEqual(normalised_labels[0], (0, 0.5, 0.5, 0.2, 0.2))

    # Test 1: Failed (No Implementation)
    # Test 2: Failed Notes:
    # -> Could not convert float to 'car'
    # -> get_config is not defined
    def test_get_config(self) -> None:
        config: dict = get_config(self.test_dir)
        self.assertIsNotNone(config)
        self.assertEqual(config["names"], ["car", "bus", "motorcycle"])

    # Test 1: Failed (No Implementation)
    # Test 2: Passed (Implemented)
    # Test 3: Failed (Changed setup to support sub-directories images/ & labels/)
    # Test 4: Passed
    # Test 5: Failed (Updated test configuration to address labelling issues)
    # Test 6: Passed
    def test_get_data(self) -> None:
        config: dict = get_config(self.test_dir)
        data: list[tuple] = get_data(self.test_dir)
        self.assertEqual(len(data), 1)
        self.assertEqual(len(data[0][1]), 2)
        self.assertEqual(data[0][1][0][0], config["names"][0])
        self.assertEqual(data[0][1][1][0], config["names"][2])

    # Test 1: Failed (No Implementation)
    # Test 2: Passed (Implemented)
    # Test 3: Failed (Changed setup to support sub-directories images/ & labels/)
    # Test 4: Passed
    # Test 5: Failed (Updated test configuration to address labelling issues)
    # Test 6: Passed
    def test_preprocess_to_csv(self) -> None:
        preprocess_to_csv(self.test_dir, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))

        with open(self.output_dir, "r") as f:
            reader: csv._reader = csv.reader(f)
            lines: list[csv._reader] = list(reader)

        self.assertEqual(lines[0], ["Image Path", "Label"])
        self.assertEqual(lines[1], [self.image_path, "car 0.5 0.5 0.2 0.2"])
        self.assertEqual(lines[2], [self.image_path, "motorcycle 0.7 0.7 0.3 0.3"])

    def tearDown(self) -> None:
        # Remove temporary directory and all its contents
        import shutil

        shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    unittest.main()
