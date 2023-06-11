import os
from PIL import Image

def read_image(file_path: str) -> Image:
    return Image.open(file_path)

def read_labels(file_path: str) -> list[tuple]:
    with open(file_path, 'r') as file:
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
        normalised_labels.append((
            class_id,
            x_center / image_size[0],
            y_center / image_size[1],
            width / image_size[0],
            height / image_size[1]
        ))
    return normalised_labels

def get_data(directory: str) -> list[tuple]:
    image_files: list[str] = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
    
    data: list = []
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'

        if label_file in os.listdir(directory):
            labels: list[tuple] = read_labels(os.path.join(directory, label_file))
            image: Image = read_image(os.path.join(directory, image_file))
            labels = normalise_labels(labels, image.size)
            
            data.append((image, labels))

    return data