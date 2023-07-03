import csv
from os.path import dirname
import pandas as pd
from modules.pre_processing.CSV_Creation_YOLO import (
    get_data,
    get_class_id_from_class_name,
)
import tensorflow as tf
from sklearn.model_selection import train_test_split


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
                    class_name = class_name.replace(
                        " ", "_"
                    )  # remove any spaces in class name
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


def load_csv_from_path(
    path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    headers: bool = True,
    mode: bool = True,
) -> tuple:
    data: list = []

    with open(path, "r") as f:
        if headers:
            next(f)

        config_directory = dirname(path)

        for line in f:
            image_path, label = line.strip().split(",", 1)
            label_parts = label.split()
            if len(label_parts) >= 5:
                class_, x_center, y_center, width, height = label_parts[:5]

                if mode:
                    class_ = get_class_id_from_class_name(class_, config_directory)

                data.append(
                    {
                        "image_path": image_path,
                        "class": class_,
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "width": float(width),
                        "height": float(height),
                    }
                )

    df = pd.DataFrame(data)

    if df.empty:
        print("Dataframe is empty. Please check the CSV file.")
        return None, None

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    return (train_df, test_df)


def load_and_preprocess_image(
    image_path: str, img_dimensions: list[int] = [416, 416]
) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_dimensions)
    image /= 255.0
    return image


def load_and_preprocess_from_path_label(image_path: str, label: dict) -> tuple:
    return load_and_preprocess_image(image_path), label
