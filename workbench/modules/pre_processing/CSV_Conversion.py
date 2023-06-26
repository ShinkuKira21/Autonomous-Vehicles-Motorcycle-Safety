import csv
import os
from collections import defaultdict
from modules.pre_processing.CSV_Creation_YOLO import ( get_config, get_class_id_from_class_name )

def csv_to_model_format(csv_path: str, output_folder: str, format: str = 'YOLO') -> None:
    config_directory: str = os.path.dirname(csv_path)
    data_dict: dict = defaultdict(list)
    output_data: list = []

    with open(csv_path, 'r') as f:
        reader: csv.reader = csv.reader(f)
        next(reader)

        for row in reader:
            img_path, labels = row[0], row[1].split()
            class_name: str = labels[0]
            class_id: int = get_class_id_from_class_name(class_name, config_directory)
            bbox: list[float] = labels[1:]

            if format == 'RCNN' :
                bbox = yolo_to_rcnn_bbox(bbox)
                output_data.append([img_path, class_id] + bbox)
            else:
                data_dict[img_path].append([class_id] + bbox)
            
    if format == 'RCNN':
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = os.path.join(output_folder, 'annotations.csv')
        with open(output_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            for data in output_data:
                writer.writerow(data)

    elif format == 'YOLO':
        for img_path, bboxes in data_dict.items():
            txt_filename: str = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            txt_filepath: str = os.path.join(output_folder, txt_filename)

            os.makedirs(os.path.dirname(txt_filepath), exist_ok=True)

            with open(txt_filepath, 'w') as f:
                for bbox in bboxes:
                    # Write each bounding box on a separate line
                    f.write(' '.join(map(str, bbox)) + '\n')

def yolo_to_rcnn_bbox(yolo_bbox: list):
    x_center, y_center, width, height = map(float, yolo_bbox)
    xmin: float = x_center - width / 2
    xmax: float = x_center + width / 2
    ymin: float = y_center - height / 2
    ymax: float = y_center + height / 2
    return [xmin, ymin, xmax, ymax]