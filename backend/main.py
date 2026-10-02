from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gradio_client import Client

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)

FRONTEND_DIR = BASE_DIR.parent / "frontend"

HF_SPACE = "numanajmal0/wan-video-api"

app = FastAPI(title="AI Video Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoRequest(BaseModel):
    prompt: str
    duration_seconds: float = 3.0
    aspect_ratio: str = "16:9"

    negative_prompt: str = (
        "blurry, low quality, distorted body, deformed hands, deformed legs, "
        "extra limbs, duplicate character, frozen motion, static scene, flickering, "
        "camera shake, malformed anatomy"
    )

    steps: int = 4
    guidance_scale: float = 5.0
    seed: int = -1


def get_dimensions(aspect_ratio: str):
    dimensions = {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1": (640, 640),
        "4:5": (640, 800),
    }

    if aspect_ratio not in dimensions:
        raise HTTPException(status_code=400, detail="Unsupported aspect ratio.")

    return dimensions[aspect_ratio]


def get_num_frames(duration_seconds: float):
    # Wan API expects 4n+1 frames and supports 21-81 frames.
    # Approximate 16 fps while staying within the model limits.
    raw_frames = round(duration_seconds * 16)
    raw_frames = max(21, min(81, raw_frames))

    frames = raw_frames
    while (frames - 1) % 4 != 0:
        frames -= 1

    return max(21, frames)


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "AI Video Studio API",
        "provider": HF_SPACE,
        "model": "wan-base",
    }


@app.post("/api/generate-video")
def generate_video(request: VideoRequest):
    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    if not 0.5 <= request.duration_seconds <= 5.0:
        raise HTTPException(
            status_code=400,
            detail="Duration must be between 0.5 and 5 seconds."
        )

    if not 1 <= request.steps <= 50:
        raise HTTPException(
            status_code=400,
            detail="Steps must be between 1 and 50."
        )

    width, height = get_dimensions(request.aspect_ratio)
    num_frames = get_num_frames(request.duration_seconds)

    try:
        client = Client(HF_SPACE)

        result = client.predict(
            model_key="wan-base",
            prompt=prompt,
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            lora_scale=1.0,
            custom_ckpt="",
            api_name="/generate",
        )

        video_result = result[0]

        if isinstance(video_result, dict):
            video_path = video_result.get("path") or video_result.get("url")
        else:
            video_path = str(video_result)

        if not video_path:
            raise RuntimeError("The video provider returned no video file.")

        source = Path(video_path)

        if not source.exists():
            raise RuntimeError(
                f"Generated video file was not found: {video_path}"
            )

        output_name = f"{uuid.uuid4().hex}.mp4"
        destination = MEDIA_DIR / output_name
        shutil.copy2(source, destination)

        seed_used = result[1] if len(result) > 1 else request.seed

        return {
            "status": "completed",
            "video_url": f"/media/{output_name}",
            "seed": seed_used,
            "model": "wan-base",
            "width": width,
            "height": height,
            "frames": num_frames,
            "duration_seconds": request.duration_seconds,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Video generation failed: {exc}"
        )


app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
