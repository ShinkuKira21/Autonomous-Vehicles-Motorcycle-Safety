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
    # Error: only checks for .yml and not yaml files
    # Fixed, now checks for yml and
    conf_file: list[str] = glob.glob(os.path.join(directory, "*.y*ml"))

    if conf_file:
        with open(conf_file[0], "r") as file:
            config = yaml.safe_load(file)
        return config

    return None


def get_class_id_from_class_name(class_name: str, directory: str) -> int:
    config: dict = get_config(directory)

    if config is not None:
        class_names: list[str] = config["names"]

        if class_name in class_names:
            return class_names.index(class_name)

    return None


def get_data(directory: str, bNormalise: bool = False) -> list[tuple]:
    image_dir = os.path.join(directory, "images")
    label_dir = os.path.join(directory, "labels")

    image_files: list[str] = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ]
    )
    label_files: list[str] = sorted(
        [
            os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.endswith(".txt")
        ]
    )

    data: list = []
    for image_file, label_file in zip(image_files, label_files):
        labels: list[tuple] = read_labels(label_file)

        # If a config is provided, only keep the labels that are in the config
        config: dict = get_config(directory)
        if config is not None:
            id_to_class_name: dict[int, any] = {
                idx: name for idx, name in enumerate(config["names"])
            }
            labels: list[tuple] = [
                (id_to_class_name[class_id], x_center, y_center, width, height)
                for class_id, x_center, y_center, width, height in labels
                if class_id in id_to_class_name
            ]

        if bNormalise:
            labels = normalise_labels(labels, Image.open(image_file).size)

        data.append((image_file, labels))

    return data
