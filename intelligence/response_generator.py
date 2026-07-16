"""
Text response generator — V5.

Consumes a pre-built models.context.ResponseContext (see
intelligence/context_builder.py) so TEXT and VOICE (intelligence/voice_generator.py)
share identical state. This module no longer fetches profile/memory data
itself — that happens once, upstream, in context_builder.build_response_context().

Retry/failover ownership: intelligence/gemini_pool.py is the SINGLE owner of
key selection, cooldown, and bounded failover across the whole codebase. This
module makes exactly one generate_content() call — it does not wrap the pool
in its own retry loop, which used to multiply failed attempts (pool retries
* an outer retry loop here) into as many as ~9 attempts with sleeps between
each on a bad key. On any failure the pool has already exhausted its own
bounded failover, so we fail fast and let the caller fall back to text-only
degradation instead of stacking more retries.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from google.genai import types

from models.context import ResponseContext
from intelligence.prompts import EVE_CORE_INSTRUCTION, EVE_TEXT_MODE_ADDITION, format_text_context
from intelligence import gemini_pool

logger = logging.getLogger("yap.intelligence.response_generator")


def generate_from_context(ctx: ResponseContext, plan: Optional[Any] = None) -> Tuple[Optional[str], float]:
    """
    Generate a TEXT reply from a pre-built ResponseContext.
    Returns (reply_text_or_None, gen_time_seconds).
    """
    prompt = format_text_context(ctx, plan=plan)

    start_gen = time.perf_counter()
    try:
        response = gemini_pool.generate_content(
            contents=prompt,
            config_opts=types.GenerateContentConfig(
                system_instruction=EVE_CORE_INSTRUCTION + "\n\n" + EVE_TEXT_MODE_ADDITION,
                temperature=1.0,
                top_p=0.95,
                max_output_tokens=150,
                thinking_config=types.ThinkingConfig(
                    thinking_level="MINIMAL"
                ),
            ),
        )
        gen_time = time.perf_counter() - start_gen
        text = (response.text or "").strip()
        if text:
            return text, gen_time
        logger.warning("[REPLY_GEN] empty response from Gemini")
        return None, gen_time
    except Exception as e:
        gen_time = time.perf_counter() - start_gen
        logger.error("[REPLY_GEN] generation failed: %s", e)
        return None, gen_time
