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
        self.image_dir: str = os.path.join(
            self.test_dir, "images"
        )  # Use self.image_dir
        self.label_dir: str = os.path.join(
            self.test_dir, "labels"
        )  # Use self.label_dir
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        # Create a dummy config file
        self.config_path: str = os.path.join(self.test_dir, "data.yaml")
        with open(self.config_path, "w") as f:
            f.write("names: ['car', 'bus', 'bike', 'motorcycle', 'hgv']\nnc: 5")

        # Create dummy image files and corresponding label files
        self.image_paths = []
        self.label_paths = []
        for i in range(5):
            # Create a dummy image file
            image_path: str = os.path.join(self.image_dir, f"image{i}.jpg")
            image: Image = Image.new("RGB", (100, 100))
            image.save(image_path)
            self.image_paths.append(image_path)

            # Create a dummy label file
            label_path: str = os.path.join(self.label_dir, f"image{i}.txt")
            with open(label_path, "w") as f:
                f.write(f"{i} 0.{i+1} 0.{i+1} 0.{(i+2)%10} 0.{(i+2)%10}")
            self.label_paths.append(label_path)

        # Define output_path as a temporary CSV file
        self.output_dir: str = os.path.join(self.test_dir, "output.csv")

    # Test 1: Failed (Required Modification)
    # Test 2: Passed
    def test_read_image(self) -> None:
        for i, image_path in enumerate(self.image_paths):
            image: list = read_image(image_path)
            self.assertEqual(image.size, (100, 100))

    # Test 1: Failed (Required Modification)
    # Test 2: Passed
    def test_read_labels(self) -> None:
        config: dict = get_config(self.test_dir)
        for i, label_path in enumerate(self.label_paths):
            labels: list = read_labels(label_path)
            self.assertEqual(len(labels), 1)
            self.assertAlmostEqual(labels[0][0], i, places=7)
            self.assertAlmostEqual(labels[0][1], 0.1 * (i + 1), places=7)
            self.assertAlmostEqual(labels[0][2], 0.1 * (i + 1), places=7)
            self.assertAlmostEqual(labels[0][3], 0.1 * ((i + 2) % 10), places=7)
            self.assertAlmostEqual(labels[0][4], 0.1 * ((i + 2) % 10), places=7)

    # Test 1: Failed (Required Modification)
    # Test 2: Passed
    def test_get_config(self) -> None:
        config: dict = get_config(self.test_dir)
        self.assertIsNotNone(config)
        self.assertEqual(config["names"], ["car", "bus", "bike", "motorcycle", "hgv"])

    # Test 1: Failed (Required Modification)
    # Test 2: Passed
    # Test 3: Failed (Updated test configuration to address labelling issues)
    # Test 4: Passed
    def test_preprocess_to_csv(self) -> None:
        preprocess_to_csv(self.test_dir, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))

        with open(self.output_dir, "r") as f:
            reader: csv._reader = csv.reader(f)
            lines: list[csv._reader] = list(reader)

        self.assertEqual(lines[0], ["Image Path", "Label"])
        config: dict = get_config(self.test_dir)

        for i in range(1, 6):
            image_path = self.image_paths[i - 1]
            label_name = config["names"][i - 1]
            label_values = " ".join(
                map(
                    str,
                    [
                        round(0.1 * i, 1),
                        round(0.1 * i, 1),
                        round(0.1 * ((i + 1) % 10), 1),
                        round(0.1 * ((i + 1) % 10), 1),
                    ],
                )
            )
            expected_line = [image_path, f"{label_name} {label_values}"]
            self.assertEqual(lines[i], expected_line)

    def tearDown(self) -> None:
        # Remove temporary directory and all its contents
        import shutil

        shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    unittest.main()
