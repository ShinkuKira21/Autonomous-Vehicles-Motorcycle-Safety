import csv
from PIL import Image
import os
import keras
from modules.pre_processing.CSV_Creation_YOLO import (
    get_data
)

import csv


def preprocess_to_csv(input_dir: str, output_file: str):
    try:
        data: list[tuple] = get_data(input_dir)

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Path", "Label"])

            for img_path, labels in data:
                for label in labels:
                    str_labels: str = " ".join(f"{float(val):.6g}" for val in label)
                    writer.writerow([img_path, str_labels])
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        raise
