from pathlib import Path
import os
import shutil
import subprocess
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gradio_client import Client, handle_file
from tts_service import generate_tts

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)

FRONTEND_DIR = BASE_DIR.parent / "frontend"

QWEN_SPACE = "assembledchaos/qwen-image-2-1-studio"
WAN_SPACE = "zerogpu-aoti/wan2-2-fp8da-aoti-faster"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not available in the environment.")

app = FastAPI(title="AI Video Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "16:9"
    steps: int = 40
    seed: int = 42
    randomize_seed: bool = True


class VideoRequest(BaseModel):
    prompt: str
    voice_over_text: str = ""
    voice: str = "Hindi Female"
    aspect_ratio: str = "16:9"
    duration_seconds: float = 3.5
    steps: int = 6
    seed: int = 42
    randomize_seed: bool = True


ASPECT_RATIOS = {
    "16:9": "Landscape · 16:9",
    "9:16": "Portrait · 9:16",
    "1:1": "Square · 1:1",
    "4:5": "Portrait · 4:5",
}


def save_gradio_file(file_result, extension):
    if isinstance(file_result, dict):
        local_path = file_result.get("path")
        url = file_result.get("url")
    else:
        local_path = str(file_result)
        url = None

    output_name = f"{uuid.uuid4().hex}.{extension}"
    destination = MEDIA_DIR / output_name

    if local_path and Path(local_path).exists():
        shutil.copy2(local_path, destination)

    elif url:
        import urllib.request
        urllib.request.urlretrieve(url, destination)

    else:
        raise RuntimeError("Provider did not return a usable file.")

    return output_name


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "AI Video Studio API",
        "image_provider": QWEN_SPACE,
        "video_provider": WAN_SPACE,
    }


@app.post("/api/generate-image")
def generate_image(request: ImageRequest):
    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    if request.aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail="Unsupported aspect ratio.")

    try:
        client = Client(QWEN_SPACE)

        result = client.predict(
            prompt=prompt,
            mode="Create an image",
            reference=None,
            aspect_ratio=ASPECT_RATIOS[request.aspect_ratio],
            steps=request.steps,
            seed=request.seed,
            randomize_seed=request.randomize_seed,
            api_name="/generate",
        )

        image_result = result[0] if isinstance(result, (list, tuple)) else result
        filename = save_gradio_file(image_result, "png")

        return {
            "status": "completed",
            "image_url": f"/media/{filename}",
            "aspect_ratio": request.aspect_ratio,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Image generation failed: {exc}",
        )


def merge_video_audio(video_file: str, audio_file: str) -> str:
    input_video = MEDIA_DIR / video_file
    input_audio = MEDIA_DIR / audio_file
    output_file = MEDIA_DIR / f"{uuid.uuid4().hex}_final.mp4"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-i", str(input_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_file),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio merge failed: {result.stderr[-2000:]}"
        )

    return output_file.name


@app.post("/api/create-video")
def create_video(request: VideoRequest):
    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    if request.aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail="Unsupported aspect ratio.")

    if not 0.5 <= request.duration_seconds <= 5:
        raise HTTPException(
            status_code=400,
            detail="Duration must be between 0.5 and 5 seconds.",
        )

    try:
        # 1. Generate a high-quality reference image.
        qwen = Client(QWEN_SPACE, token=HF_TOKEN)

        image_result = qwen.predict(
            prompt=prompt,
            mode="Create an image",
            reference=None,
            aspect_ratio=ASPECT_RATIOS[request.aspect_ratio],
            steps=40,
            seed=request.seed,
            randomize_seed=request.randomize_seed,
            api_name="/generate",
        )

        image_data = image_result[0] if isinstance(image_result, (list, tuple)) else image_result

        if isinstance(image_data, dict):
            image_path = image_data.get("path")
        else:
            image_path = str(image_data)

        if not image_path or not Path(image_path).exists():
            raise RuntimeError("Qwen returned no usable image file.")

        # 2. Animate that exact image with Wan2.2 I2V.
        wan = Client(WAN_SPACE, token=HF_TOKEN)

        video_result = wan.predict(
            input_image=handle_file(image_path),
            prompt=(
                f"{prompt}. Smooth natural motion, stable character identity, "
                "stable anatomy, properly connected body parts, consistent face, "
                "clean detailed motion, cinematic movement."
            ),
            steps=request.steps,
            negative_prompt=(
                "blurry, low quality, distorted body, deformed hands, deformed legs, "
                "extra limbs, duplicate character, disappearing body parts, "
                "flickering, jitter, unstable face, malformed anatomy"
            ),
            duration_seconds=request.duration_seconds,
            guidance_scale=1,
            guidance_scale_2=1,
            seed=request.seed,
            randomize_seed=request.randomize_seed,
            api_name="/generate_video",
        )

        video_data = video_result[0] if isinstance(video_result, (list, tuple)) else video_result

        video_filename = save_gradio_file(video_data, "mp4")

        narration = request.voice_over_text.strip() or request.prompt.strip()
        audio_filename = generate_tts(
            narration,
            request.voice,
            MEDIA_DIR,
        )

        final_filename = merge_video_audio(
            video_filename,
            audio_filename,
        )

        return {
            "status": "completed",
            "video_url": f"/media/{final_filename}",
            "audio_url": f"/media/{audio_filename}",
            "voice": request.voice,
            "duration_seconds": request.duration_seconds,
            "aspect_ratio": request.aspect_ratio,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Video creation failed: {exc}",
        )


app.mount(
    "/media",
    StaticFiles(directory=MEDIA_DIR),
    name="media",
)


class AudioRequest(BaseModel):
    text: str
    voice: str = "Hindi Female"


@app.post("/api/generate-audio")
def generate_audio(request: AudioRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    if len(text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Text is too long. Maximum is 5000 characters."
        )

    try:
        filename = generate_tts(text, request.voice, MEDIA_DIR)

        return {
            "status": "completed",
            "audio_url": f"/media/{filename}",
            "voice": request.voice,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Audio generation failed: {exc}",
        )
