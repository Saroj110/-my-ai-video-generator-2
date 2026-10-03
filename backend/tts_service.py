import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import requests


TTS_BASE = "https://ai4bharat-indic-parler-tts.hf.space"
TTS_FN_INDEX = 1

VOICES = {
    "Hindi Female": "Divya speaks in Hindi in a warm, clear and natural female voice, with a moderate speaking rate and clear pronunciation. The recording is very high quality with no background noise.",
    "Hindi Male": "Rohit speaks in Hindi in a warm, clear and natural male voice, with a moderate speaking rate and clear pronunciation. The recording is very high quality with no background noise.",
    "English Female": "Mary speaks in English in a warm, clear and natural female voice, with a moderate speaking rate and clear pronunciation. The recording is very high quality with no background noise.",
    "English Male": "Thoma speaks in English in a warm, clear and natural male voice, with a moderate speaking rate and clear pronunciation. The recording is very high quality with no background noise.",
}


def generate_tts(text: str, voice: str, media_dir: Path) -> str:
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError("HF_TOKEN is not available.")

    if voice not in VOICES:
        raise RuntimeError(f"Unsupported voice: {voice}")

    session_hash = uuid.uuid4().hex

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "data": [text, VOICES[voice]],
        "fn_index": TTS_FN_INDEX,
        "session_hash": session_hash,
    }

    join = requests.post(
        f"{TTS_BASE}/gradio_api/queue/join",
        headers=headers,
        json=payload,
        timeout=60,
    )
    join.raise_for_status()

    deadline = time.time() + 600

    with requests.get(
        f"{TTS_BASE}/gradio_api/queue/data",
        params={"session_hash": session_hash},
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=(60, 660),
    ) as stream:

        for raw in stream.iter_lines(decode_unicode=True):
            if time.time() > deadline:
                raise RuntimeError("TTS generation timed out.")

            if not raw:
                continue

            line = raw.strip()

            if not line.startswith("data:"):
                continue

            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue

            if event.get("msg") == "queue_full":
                raise RuntimeError("TTS queue is full.")

            if event.get("msg") == "unexpected_error":
                raise RuntimeError(event.get("error", "TTS service error."))

            if event.get("msg") == "process_completed":
                data = event.get("output", {}).get("data", [])
                if not data:
                    raise RuntimeError("TTS returned no audio.")

                result = data[0]

                if isinstance(result, dict):
                    audio_url = result.get("url")
                    audio_path = result.get("path")
                else:
                    audio_url = None
                    audio_path = str(result)

                if not audio_url and audio_path:
                    audio_url = (
                        f"{TTS_BASE}/gradio_api/file="
                        f"{quote(audio_path, safe='/:')}"
                    )

                if not audio_url:
                    raise RuntimeError("TTS returned an invalid audio file.")

                audio = requests.get(
                    audio_url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=120,
                )
                audio.raise_for_status()

                media_dir.mkdir(exist_ok=True)
                filename = f"{uuid.uuid4().hex}.mp3"
                destination = media_dir / filename
                destination.write_bytes(audio.content)

                return filename

    raise RuntimeError("TTS generation did not complete.")


AVAILABLE_VOICES = list(VOICES.keys())
