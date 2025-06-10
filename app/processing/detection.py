from typing import Iterator

import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.configs.annotators import (
    BOX_ANNOTATOR,
    BOX_LABEL_ANNOTATOR,
    VERTEX_LABEL_ANNOTATOR,
)
from app.configs.model_paths import (
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
)
from app.configs.pitch import SoccerPitchConfiguration

# Soccer pitch configuration
CONFIG = SoccerPitchConfiguration()


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels
        )
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


# def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
#     """
#     Run ball detection on a video and yield annotated frames.

#     Args:
#         source_video_path (str): Path to the source video.
#         device (str): Device to run the model on (e.g., 'cpu', 'cuda').

#     Yields:
#         Iterator[np.ndarray]: Iterator over annotated frames.
#     """
#     ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
#     frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
#     ball_tracker = BallTracker(buffer_size=20)
#     ball_annotator = BallAnnotator(radius=6, buffer_size=10)

#     def callback(image_slice: np.ndarray) -> sv.Detections:
#         result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
#         return sv.Detections.from_ultralytics(result)

#     slicer = sv.InferenceSlicer(
#         callback=callback,
#         overlap_filter_strategy=sv.OverlapFilter.NONE,
#         slice_wh=(640, 640),
#     )

#     for frame in frame_generator:
#         detections = slicer(frame).with_nms(threshold=0.1)
#         detections = ball_tracker.update(detections)
#         annotated_frame = frame.copy()
#         annotated_frame = ball_annotator.annotate(annotated_frame, detections)
#         yield annotated_frame
