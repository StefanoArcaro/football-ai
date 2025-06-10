import subprocess

import numpy as np
import supervision as sv


def get_crops(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        list[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def reencode_for_web(input_mp4: str, output_mp4: str):
    """
    Re-encode video to baseline H.264 with moov atom at start for web compatibility.

    Args:
        input_mp4 (str): Path to the input MP4 file.
        output_mp4 (str): Path to the output MP4 file.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_mp4,
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-movflags",
        "+faststart",
        output_mp4,
    ]
    subprocess.run(cmd, check=True)
