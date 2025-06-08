import os
import shutil
import subprocess

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.processing.video_pipeline import run_detection_pipeline

router = APIRouter()
UPLOAD_PATH = "videos/input.mp4"
RAW_OUTPUT_PATH = "videos/analyzed_raw.mp4"
FINAL_OUTPUT_PATH = "videos/analyzed.mp4"


def reencode_for_web(input_mp4: str, output_mp4: str):
    """Re-encode video to baseline H.264 with moov atom at start for web compatibility."""
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


os.makedirs("videos", exist_ok=True)


@router.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    if not video.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Save input
    with open(UPLOAD_PATH, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Run analysis
    run_detection_pipeline(UPLOAD_PATH, RAW_OUTPUT_PATH)

    # Re-encode for browser compatibility
    try:
        reencode_for_web(RAW_OUTPUT_PATH, FINAL_OUTPUT_PATH)

        # Clean up raw output
        os.remove(RAW_OUTPUT_PATH)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Video encoding failed")

    return {"status": "analysis complete"}


@router.get("/result")
def get_result_video():
    if not os.path.exists(FINAL_OUTPUT_PATH):
        raise HTTPException(status_code=404, detail="No result yet")

    return FileResponse(FINAL_OUTPUT_PATH, media_type="video/mp4")
