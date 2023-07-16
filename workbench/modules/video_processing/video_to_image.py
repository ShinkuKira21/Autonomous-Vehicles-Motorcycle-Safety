import cv2
import os


def video_to_frames(video_path: str, output_dir: str, sample_name: str) -> None:
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count: int = 0
    while success:
        cv2.imwrite(os.path.join(output_dir, f"frame{count}.jpg"), image)
        success, image = vidcap.read()
        print(f"{sample_name}-{count}.jpg")
        count += 1

    print("Done extracting frames.")
