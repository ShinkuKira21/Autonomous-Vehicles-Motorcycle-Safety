import csv
import pandas as pd
from modules.pre_processing.CSV_Creation_YOLO import get_data

import csv


def preprocess_to_csv(
    input_dir: str, output_file: str, allowed_classes: list = None
) -> None:
    try:
        data: list[tuple] = get_data(input_dir)

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Path", "Label"])

            for img_path, labels in data:
                for label in labels:
                    class_name, *values = label
                    # If allowed_classes is None or the class name is in the allowed classes
                    if allowed_classes is None or class_name in allowed_classes:
                        str_labels: str = f"{class_name} " + " ".join(
                            f"{float(val):.6g}" for val in values
                        )
                        writer.writerow([img_path, str_labels])
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        raise


def combine_csv(fDir: list[str], output_file: str) -> None:
    dfs: list[pd.DataFrame] = []
    for dir in fDir:
        dfs.append(pd.read_csv(dir))

    cols = dfs[0].columns
    for i in range(len(dfs)):
        dfs[i] = dfs[i][cols]

    combined_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
