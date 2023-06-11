import unittest
import os
from PIL import Image
import tempfile

# YOLO uses <object-class> <x> <y> <width> <height> classification labels to identify the frames.
# The test case ensures that the methods implemented will work properly for YOLO classification.
    
class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir: str = tempfile.mkdtemp()

        self.image_path: str = os.path.join(self.test_dir, 'image.jpg')
        image: Image = Image.new('RGB', (100, 100))
        image.save(self.image_path)

        self.label_path: str = os.path.join(self.test_dir, 'image.txt')
        with open(self.label_path, 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n1 0.7 0.7 0.3 0.3")

    # Test 1: Failed (No Implementation)
    def test_read_labels(self):
        labels: list = read_labels(self.label_path)
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0], (0, 0.5, 0.5, 0.2, 0.2))

    # Test 1: Failed (No Implementation)
    def test_read_image(self):
        image: list = read_image(self.image_path)
        self.assertEqual(image.size, (100, 100))

    # Test 1: Failed (No Implementation)
    def test_normalise_labels(self):
        labels: list[tuple] = [(0, 50, 50, 20, 20)]
        normalised_labels = normalise_labels(labels, (100, 100))
        self.assertEqual(normalised_labels[0], (0, 0.5, 0.5, 0.2, 0.2))

    # Test 1: Failed (No Implementation)
    def test_get_data(self):
        data: list[tuple] = get_data(self.test_dir)
        self.assertEqual(len(data), 1)
        self.assertEqual(len(data[0][1]), 2)

    def tearDown(self):
        # Remove temporary directory and all its contents
        import shutil
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()