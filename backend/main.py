from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)

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
    negative_prompt: str = (
        "blurry, low quality, distorted body, deformed hands, deformed legs, "
        "extra limbs, duplicate character, frozen motion, static scene, flickering"
    )
    width: int = 832
    height: int = 480
    num_frames: int = 49
    steps: int = 30
    guidance_scale: float = 5.0
    seed: int = -1
    lora_scale: float = 1.0
    custom_ckpt: str = ""

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

    if request.width < 320 or request.width > 832:
        raise HTTPException(status_code=400, detail="Width must be 320-832.")

    if request.height < 320 or request.height > 832:
        raise HTTPException(status_code=400, detail="Height must be 320-832.")

    if request.num_frames < 21 or request.num_frames > 81:
        raise HTTPException(status_code=400, detail="Frames must be 21-81.")

    if request.steps < 1 or request.steps > 50:
        raise HTTPException(status_code=400, detail="Steps must be 1-50.")

    try:
        client = Client(HF_SPACE)

        result = client.predict(
            model_key="wan-base",
            prompt=prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            lora_scale=request.lora_scale,
            custom_ckpt=request.custom_ckpt,
            api_name="/generate",
        )

        # The first returned value is the generated video file.
        video_result = result[0]

        if isinstance(video_result, dict):
            video_path = video_result.get("path") or video_result.get("url")
        else:
            video_path = str(video_result)

        if not video_path:
            raise RuntimeError("The video provider returned no video file.")

        source = Path(video_path)

        if not source.exists():
            raise RuntimeError(f"Generated video file was not found: {video_path}")

        output_name = f"{uuid.uuid4().hex}.mp4"
        destination = MEDIA_DIR / output_name

        shutil.copy2(source, destination)

        seed_used = result[1] if len(result) > 1 else request.seed

        return {
            "status": "completed",
            "video_url": f"/media/{output_name}",
            "seed": seed_used,
            "model": "wan-base",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Video generation failed: {exc}",
        )

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")


FRONTEND_DIR = BASE_DIR.parent / "frontend"

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
