import os
from PIL import Image
import glob
import yaml


def read_image(file_path: str) -> Image:
    return Image.open(file_path)


def read_labels(file_path: str) -> list[tuple]:
    with open(file_path, "r") as file:
        lines: list[str] = file.readlines()

    labels: list[tuple] = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        labels.append((class_id, x_center, y_center, width, height))

    return labels


def normalise_labels(labels: list[tuple], image_size: list):
    normalised_labels: list[tuple] = []
    for label in labels:
        class_id, x_center, y_center, width, height = label
        normalised_labels.append(
            (
                class_id,
                x_center / image_size[0],
                y_center / image_size[1],
                width / image_size[0],
                height / image_size[1],
            )
        )
    return normalised_labels


def get_config(directory: str) -> dict:
    conf_file: list[str] = glob.glob(os.path.join(directory, "*.yml"))

    if conf_file:
        with open(conf_file[0], "r") as file:
            config = yaml.safe_load(file)
        return config

    return None


def get_data(directory: str, bNormalise: bool = False) -> list[tuple]:
    image_files: list[str] = sorted(
        [f for f in os.listdir(directory) if f.endswith(".jpg")]
    )

    data: list = []
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"

        if label_file in os.listdir(directory):
            labels: list[tuple] = read_labels(os.path.join(directory, label_file))

            # If a config is provided, only keep the labels that are in the config
            config: dict = get_config(directory)
            if config is not None:
                id_to_class_name: dict[int, any] = {
                    idx: name for idx, name in enumerate(config["names"])
                }
                labels: list[tuple] = [
                    (class_id, x_center, y_center, width, height)
                    for class_id, x_center, y_center, width, height in labels
                    if class_id in id_to_class_name
                ]

            image_path: str = os.path.join(directory, image_file)

            if bNormalise:
                labels = normalise_labels(labels, Image.open(image_path).size)

            data.append((image_path, labels))

    return data
