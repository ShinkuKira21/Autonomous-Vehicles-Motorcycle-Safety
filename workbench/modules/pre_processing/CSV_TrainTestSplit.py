import os
import shutil
from sklearn.model_selection import train_test_split


def move_files(file_list: list[str], output_dir: str, target_sdir: str):
    image_ext = ".jpg"
    label_ext = ".txt"

    for file in file_list:
        orig_image_dir = os.path.join(output_dir, "images")
        orig_label_dir = os.path.join(output_dir, "labels")

        new_image_dir = os.path.join(output_dir, target_sdir, "images")
        new_label_dir = os.path.join(output_dir, target_sdir, "labels")

        os.makedirs(new_image_dir, exist_ok=True)
        os.makedirs(new_label_dir, exist_ok=True)

        # Define the full paths for the original and new files
        orig_image_file = os.path.join(orig_image_dir, file + image_ext)
        orig_label_file = os.path.join(orig_label_dir, file + label_ext)

        new_image_file = os.path.join(new_image_dir, file + image_ext)
        new_label_file = os.path.join(new_label_dir, file + label_ext)

        print(f"Moving {orig_image_file} to {new_image_file}")
        print(f"Moving {orig_label_file} to {new_label_file}")

        shutil.move(orig_image_file, new_image_file)
        shutil.move(orig_label_file, new_label_file)


def yolo_train_test_split(
    input_dir: str,
    test_size: list[float] = [0.2, 0.5],
    random_state: list[int] = [42, 42],
    train_proportion: float = 1
) -> tuple:
    filenames: list[str] = [
        os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".jpg")
    ]

    if not filenames:
        raise ValueError("No .jpg files found in the provided directory.")

    train_files, test_files = train_test_split(
        filenames, test_size=test_size[0], random_state=random_state[0]
    )

    train_size = int(len(train_files) * train_proportion)
    train_files = train_files[:train_size]

    val_files, test_files = train_test_split(
        test_files, test_size=test_size[1], random_state=random_state[1]
    )

    return train_files, val_files, test_files
