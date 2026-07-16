"""
Audio conversion for Eve's voice replies — V5.

Gemini Live returns raw PCM audio chunks. Instagram's direct_send_voice()
requires AAC audio in an M4A/MP4 container. This module bridges the two:

  raw PCM bytes -> temp WAV (correct header via stdlib wave) -> ffmpeg -> temp M4A

ffmpeg is invoked as an argument array (never shell=True). Temp files are
always cleaned up. If ffmpeg is not on PATH, voice mode must be disabled at
startup — see voice/health.py and main.py — never assumed present.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger("yap.voice.audio")

DEFAULT_SAMPLE_RATE = 24000  # documented Gemini Live native-audio output rate
FFMPEG_TIMEOUT_SECONDS = 30


def check_ffmpeg_available() -> bool:
    """Zero-cost presence check — does not spawn a process, consumes no quota."""
    return shutil.which(config.FFMPEG_PATH) is not None


def parse_sample_rate(mime_type: Optional[str]) -> Optional[int]:
    """Parse 'rate=24000' out of a mime type like 'audio/pcm;rate=24000'."""
    if not mime_type:
        return None
    match = re.search(r"rate=(\d+)", mime_type)
    if match:
        return int(match.group(1))
    return None


def convert_pcm_to_m4a(
    pcm_bytes: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> Path:
    """
    Convert raw PCM bytes to an AAC-in-M4A file suitable for
    instagrapi's direct_send_voice(). Returns the path to the M4A file —
    caller is responsible for deleting it (see cleanup_audio_file) once sent.

    Raises on empty input or ffmpeg failure; never raises on cleanup of the
    intermediate WAV (best-effort).
    """
    if not pcm_bytes:
        raise ValueError("empty PCM data — nothing to convert")

    wav_fd, wav_path_str = tempfile.mkstemp(suffix=".wav", prefix="eve_voice_")
    m4a_fd, m4a_path_str = tempfile.mkstemp(suffix=".m4a", prefix="eve_voice_")
    os.close(m4a_fd)  # ffmpeg writes to this path directly; just reserving the name
    wav_path = Path(wav_path_str)
    m4a_path = Path(m4a_path_str)

    try:
        with os.fdopen(wav_fd, "wb") as raw_f:
            with wave.open(raw_f, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)

        start = time.perf_counter()
        result = subprocess.run(
            [
                config.FFMPEG_PATH, "-y",
                "-i", str(wav_path),
                "-c:a", "aac", "-b:a", "64k",
                str(m4a_path),
            ],
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
        duration = time.perf_counter() - start

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:500]
            raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {stderr}")

        size = m4a_path.stat().st_size
        logger.info("[VOICE] audio converted duration=%.3fs size=%d bytes", duration, size)
        return m4a_path

    except Exception:
        # Conversion failed — don't leave a broken/empty m4a behind.
        cleanup_audio_file(m4a_path)
        raise
    finally:
        cleanup_audio_file(wav_path)


def cleanup_audio_file(path: Optional[Path]) -> None:
    """Best-effort temp file cleanup — never raises."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning("[VOICE] failed to clean up temp audio file %s: %s", path, e)
