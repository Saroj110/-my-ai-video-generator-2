from pathlib import Path
import shutil
import uuid
import urllib.request

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

QWEN_SPACE = "assembledchaos/qwen-image-2-1-studio"

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


ASPECT_RATIOS = {
    "16:9": "Landscape · 16:9",
    "9:16": "Portrait · 9:16",
    "1:1": "Square · 1:1",
    "4:5": "Portrait · 4:5",
}


def save_gradio_file(file_result):
    if isinstance(file_result, dict):
        local_path = file_result.get("path")
        url = file_result.get("url")
    else:
        local_path = str(file_result)
        url = None

    output_name = f"{uuid.uuid4().hex}.png"
    destination = MEDIA_DIR / output_name

    if local_path and Path(local_path).exists():
        shutil.copy2(local_path, destination)

    elif url:
        urllib.request.urlretrieve(url, destination)

    else:
        raise RuntimeError("Qwen did not return a usable image file.")

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
    }


@app.post("/api/generate-image")
def generate_image(request: ImageRequest):
    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    if request.aspect_ratio not in ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail="Unsupported aspect ratio.")

    if not 1 <= request.steps <= 50:
        raise HTTPException(status_code=400, detail="Steps must be between 1 and 50.")

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
        filename = save_gradio_file(image_result)

        return {
            "status": "completed",
            "image_url": f"/media/{filename}",
            "aspect_ratio": request.aspect_ratio,
            "steps": request.steps,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Image generation failed: {exc}",
        )


app.mount(
    "/media",
    StaticFiles(directory=MEDIA_DIR),
    name="media",
)
