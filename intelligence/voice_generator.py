"""
Voice reply generation via Gemini Live native audio — V5.

CRITICAL ARCHITECTURAL RULE (see CLAUDE.md / V5 spec): there is NO text
intermediary here. This module never calls intelligence/response_generator.py
and never asks Live to "say" or "read" a pre-generated line. It sends Eve's
raw world context (identity + voice-mode addition + the SAME canonical
ResponseContext used by text) directly to Gemini Live as a single
"turn", and Live reasons and speaks as Eve itself.

Uses ONE dedicated API key (config.GEMINI_LIVE_API_KEY, falling back to the
first text-pool key only if unset) — never the text round-robin pool, and
never round-robins itself.

A short-lived Live session is created per voice reply (connect -> send ->
collect -> close) rather than one long-running session, so each reply's
context is isolated and always current — see PART 13.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

import config
from models.context import ResponseContext
from intelligence.prompts import EVE_CORE_INSTRUCTION, EVE_VOICE_MODE_ADDITION, format_voice_context
from intelligence import gemini_pool
from voice import audio

logger = logging.getLogger("yap.intelligence.voice_generator")

_voice_client: Optional[genai.Client] = None


@dataclass
class VoiceResult:
    success: bool
    audio_path: Optional[Path] = None
    failure_reason: Optional[str] = None
    connect_duration: float = 0.0
    first_chunk_duration: float = 0.0
    generation_duration: float = 0.0
    conversion_duration: float = 0.0
    transcript: Optional[str] = None


def resolve_live_api_key() -> Optional[str]:
    if config.GEMINI_LIVE_API_KEY:
        return config.GEMINI_LIVE_API_KEY
    pool = gemini_pool.get_pool()
    if pool._keys:
        logger.warning("[VOICE] GEMINI_LIVE_API_KEY not set — falling back to text pool key #1 for Live")
        return pool._keys[0]._api_key
    return None


def _get_voice_client() -> genai.Client:
    global _voice_client
    if _voice_client is None:
        key = resolve_live_api_key()
        if not key:
            raise RuntimeError("no Gemini API key available for Live voice generation")
        _voice_client = genai.Client(api_key=key)
    return _voice_client


async def _generate_voice_async(ctx: ResponseContext, plan: Optional[Any] = None) -> VoiceResult:
    client = _get_voice_client()

    system_instruction = types.Content(
        parts=[types.Part(text=EVE_CORE_INSTRUCTION + "\n\n" + EVE_VOICE_MODE_ADDITION)]
    )
    live_config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=system_instruction,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.GEMINI_LIVE_VOICE)
            )
        ),
    )

    # This is Eve's own world/context, NOT a script to read and NOT
    # generated text — see format_voice_context() and EVE_VOICE_MODE_ADDITION.
    context_text = format_voice_context(ctx, plan=plan)

    audio_chunks: list[bytes] = []
    transcript_chunks: list[str] = []
    sample_rate = audio.DEFAULT_SAMPLE_RATE
    connect_duration = 0.0
    first_chunk_duration = 0.0
    generation_duration = 0.0

    t_connect_start = time.perf_counter()
    try:
        async with client.aio.live.connect(model=config.GEMINI_LIVE_MODEL, config=live_config) as session:
            connect_duration = time.perf_counter() - t_connect_start

            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=context_text)]),
                turn_complete=True,
            )

            first_chunk_time: Optional[float] = None
            t_gen_start = time.perf_counter()

            async for message in session.receive():
                server_content = getattr(message, "server_content", None)
                if server_content is None:
                    continue

                model_turn = getattr(server_content, "model_turn", None)
                if model_turn is not None and getattr(model_turn, "parts", None):
                    for part in model_turn.parts:
                        inline = getattr(part, "inline_data", None)
                        if inline is not None and inline.data:
                            if first_chunk_time is None:
                                first_chunk_time = time.perf_counter() - t_gen_start
                            audio_chunks.append(inline.data)
                            parsed_rate = audio.parse_sample_rate(getattr(inline, "mime_type", None))
                            if parsed_rate:
                                sample_rate = parsed_rate
                        
                        text_val = getattr(part, "text", None)
                        if text_val:
                            transcript_chunks.append(text_val)

                if getattr(server_content, "turn_complete", False):
                    break

            generation_duration = time.perf_counter() - t_gen_start
            first_chunk_duration = first_chunk_time or 0.0

    except Exception as e:
        logger.warning("[VOICE] Live session error: %s", e)
        return VoiceResult(success=False, failure_reason=f"live_error: {e}", connect_duration=connect_duration)

    if not audio_chunks:
        return VoiceResult(
            success=False, failure_reason="empty_audio",
            connect_duration=connect_duration, generation_duration=generation_duration,
        )

    pcm_bytes = b"".join(audio_chunks)
    t_conv_start = time.perf_counter()
    try:
        m4a_path = audio.convert_pcm_to_m4a(pcm_bytes, sample_rate=sample_rate, channels=1, sample_width=2)
    except Exception as e:
        logger.warning("[VOICE] audio conversion failed: %s", e)
        return VoiceResult(
            success=False, failure_reason=f"conversion_error: {e}",
            connect_duration=connect_duration, generation_duration=generation_duration,
        )
    conversion_duration = time.perf_counter() - t_conv_start
    voice_transcript = "".join(transcript_chunks).strip() or None

    return VoiceResult(
        success=True,
        audio_path=m4a_path,
        connect_duration=connect_duration,
        first_chunk_duration=first_chunk_duration,
        generation_duration=generation_duration,
        conversion_duration=conversion_duration,
        transcript=voice_transcript,
    )


def generate_voice(ctx: ResponseContext, plan: Optional[Any] = None) -> VoiceResult:
    """
    Synchronous entry point used by workers/message_worker.py (which is
    plain-threaded, not asyncio). Runs a bounded, self-contained event loop
    per call — safe because each burst is processed sequentially on the
    single message-worker thread.
    """
    try:
        from typing import Any
        return asyncio.run(asyncio.wait_for(_generate_voice_async(ctx, plan=plan), timeout=config.VOICE_TIMEOUT_SECONDS))
    except asyncio.TimeoutError:
        logger.warning("[VOICE] generation timed out after %.0fs", config.VOICE_TIMEOUT_SECONDS)
        return VoiceResult(success=False, failure_reason="timeout")
    except Exception as e:
        logger.warning("[VOICE] unexpected error: %s", e)
        return VoiceResult(success=False, failure_reason=f"unexpected_error: {e}")
