import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from gradio_client import Client, handle_file
from tts_service import generate_tts

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
STORYBOARD_DIR = BASE_DIR / "storyboards"

QWEN = "assembledchaos/qwen-image-2-1-studio"
WAN = "zerogpu-aoti/wan2-2-fp8da-aoti-faster"

NEGATIVE = (
    "blurry, low quality, flickering, jitter, distorted anatomy, "
    "deformed hands, deformed legs, extra limbs, disappearing body parts, "
    "duplicate characters, unstable face, malformed anatomy"
)


def save_file(result, ext):
    if isinstance(result, dict):
        path = result.get("path")
        url = result.get("url")
    else:
        path = str(result)
        url = None

    name = f"{uuid.uuid4().hex}.{ext}"
    dest = MEDIA_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if path and Path(path).exists():
        shutil.copy2(path, dest)
    elif url:
        import requests
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        dest.write_bytes(r.content)
    else:
        raise RuntimeError("Provider returned no usable file.")

    return name


def make_clip(video, audio, output):
    if audio:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(audio),
            "-t", "5",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-t", "5",
            "-c:v", "libx264",
            "-an",
            str(output),
        ]

    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2500:])


def concatenate(files, output):
    listing = MEDIA_DIR / f"{uuid.uuid4().hex}.txt"

    with listing.open("w", encoding="utf-8") as f:
        for item in files:
            f.write(f"file '{Path(item).resolve()}'\n")

    r = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(listing),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    listing.unlink(missing_ok=True)

    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2500:])


def generate_full_video(job_id, storyboard_name, voice, jobs):
    job_dir = MEDIA_DIR / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        storyboard = json.loads(
            (STORYBOARD_DIR / f"{storyboard_name}.json").read_text(
                encoding="utf-8"
            )
        )

        scenes = storyboard["scenes"]

        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is missing.")

        qwen = Client(QWEN, token=token)
        wan = Client(WAN, token=token)

        clips = []
        total = len(scenes)

        jobs[job_id]["status"] = "running"
        jobs[job_id]["total"] = total

        for number, scene in enumerate(scenes, 1):
            jobs[job_id]["current_scene"] = number
            jobs[job_id]["message"] = f"Generating scene {number} of {total}"

            final_clip = job_dir / f"scene_{number:02d}.mp4"

            if final_clip.exists():
                clips.append(final_clip)
                jobs[job_id]["completed"] = number
                continue

            prompt = (
                "High quality 3D children's animation, consistent visual style, "
                "clear face, stable character, correct anatomy, sharp details, "
                "clean edges, vibrant colors, cinematic lighting. "
                + scene["visual"]
            )

            image_result = qwen.predict(
                prompt=prompt,
                mode="Create an image",
                reference=None,
                aspect_ratio="Landscape · 16:9",
                steps=40,
                seed=1000 + scene["id"],
                randomize_seed=False,
                api_name="/generate",
            )

            image_data = (
                image_result[0]
                if isinstance(image_result, (list, tuple))
                else image_result
            )

            image_path = (
                image_data.get("path")
                if isinstance(image_data, dict)
                else str(image_data)
            )

            if not image_path or not Path(image_path).exists():
                raise RuntimeError(
                    f"Scene {number}: image generation returned no file."
                )

            video_result = wan.predict(
                input_image=handle_file(image_path),
                prompt=(
                    "Smooth natural motion, stable character identity, "
                    "stable face, natural movement, clean animation. "
                    + scene["visual"]
                ),
                steps=6,
                negative_prompt=NEGATIVE,
                duration_seconds=5,
                guidance_scale=1,
                guidance_scale_2=1,
                seed=2000 + scene["id"],
                randomize_seed=False,
                api_name="/generate_video",
            )

            video_data = (
                video_result[0]
                if isinstance(video_result, (list, tuple))
                else video_result
            )

            video_name = save_file(video_data, "mp4")
            video_path = MEDIA_DIR / video_name

            audio_path = None
            audio_text = scene.get("audio", "").strip()

            skip_audio = any(
                x in audio_text.lower()
                for x in ["music", "lullaby", "chime", "instrumental"]
            )

            if audio_text and not skip_audio:
                audio_name = generate_tts(
                    audio_text,
                    voice,
                    MEDIA_DIR,
                )
                audio_path = MEDIA_DIR / audio_name

            make_clip(
                video_path,
                audio_path,
                final_clip,
            )

            clips.append(final_clip)
            jobs[job_id]["completed"] = number

        final_video = job_dir / "final_video.mp4"
        concatenate(clips, final_video)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Full video completed."
        jobs[job_id]["video_url"] = (
            f"/media/jobs/{job_id}/final_video.mp4"
        )

    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(exc)
