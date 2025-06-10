import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLORS = ["#00BFFF", "#FF6347", "#FFD700"]
BALL_ID = 0


def run_detection_pipeline(source_video_path: str, output_video_path: str):
    model_path = "models/best.pt"
    model = YOLO(model_path)

    # Load model to device and convert to half precision for performance
    model.to(DEVICE).half()

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frame_generator = sv.get_video_frames_generator(source_video_path)
    video_sink = sv.VideoSink(target_path=output_video_path, video_info=video_info)

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(COLORS), thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        text_color=sv.Color.from_hex("#000"),
        text_position=sv.Position.BOTTOM_CENTER,
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("FFD700"), base=20, height=17
    )

    tracker = sv.ByteTrack()
    tracker.reset()

    with video_sink:
        for frame in tqdm(
            frame_generator, total=video_info.total_frames, desc="Processing frames"
        ):
            original_h, original_w = frame.shape[:2]
            target_size = 1280  # Match the size used during training

            # Resize to 1280x1280 for inference
            resized_frame = cv2.resize(
                frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )

            # Convert to float16 tensor on GPU, normalize
            frame_tensor = (
                torch.from_numpy(resized_frame)
                .permute(2, 0, 1)  # HWC to CHW
                .contiguous()
                .to(DEVICE)
                .half()
                / 255.0
            ).unsqueeze(0)  # Add batch dimension

            # Inference
            result = model(frame_tensor)[0]

            # Convert result to Detections (note: they are in 1280x1280 space)
            detections = sv.Detections.from_ultralytics(result)

            # Scale detections back to original resolution
            scale_x = original_w / target_size
            scale_y = original_h / target_size
            detections.xyxy *= np.array([scale_x, scale_y, scale_x, scale_y])

            # Ball detection and padding
            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            # Other detections, tracking and NMS
            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections.class_id = all_detections.class_id - 1
            all_detections = tracker.update_with_detections(all_detections)

            # Labels with tracker IDs
            labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=all_detections, labels=labels
            )
            annotated_frame = triangle_annotator.annotate(
                scene=annotated_frame, detections=ball_detections
            )

            # Write annotated frame
            video_sink.write_frame(annotated_frame)
