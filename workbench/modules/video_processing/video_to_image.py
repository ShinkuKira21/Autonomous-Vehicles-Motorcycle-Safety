import cv2
import os
import random


def select_frames(frame_total: int, limiter: float = None) -> list:
    if limiter is not None:
        frames_to_sample = min(frame_total, limiter)
        frames = sorted(random.sample(range(frame_total), frames_to_sample))
    else:
        frames = list(range(frame_total))

    return frames


def video_to_frames(
    video_path: str,
    output_dir: str,
    sample_name: str,
    start_time: float = 0,
    end_time: float = None,
    limiter: float = None,
) -> None:
    # updates the parameters
    start_time = start_time * 60 * 1000
    end_time = float("inf") if end_time is None else end_time * 60 * 1000

    # ensure that the video output directory exists
    os.makedirs(output_dir, exist_ok=True)

    vidcap: cv2.VideoCapture = cv2.VideoCapture(video_path)

    vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time)

    # Limiter so we don't end up with too many test files
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    start_frame = int((start_time / 1000) * frame_rate)
    end_frame = int((end_time / 1000) * frame_rate)
    frame_total = end_frame - start_frame
    frames = select_frames(frame_total, limiter=limiter)

    success, image = vidcap.read()

    frames_idx: int = 0
    while success and frames_idx < len(frames):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frames[frames_idx])
        success, image = vidcap.read()
        if success:
            output_file_path = os.path.join(
                output_dir, f"{sample_name}-{frames[frames_idx]}.jpg"
            )
            cv2.imwrite(output_file_path, image)
            print(f"Saved frame as {os.path.abspath(output_file_path)}")
        frames_idx += 1

    print("Done extracting frames.")
