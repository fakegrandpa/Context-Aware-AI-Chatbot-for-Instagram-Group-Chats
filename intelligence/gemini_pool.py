import logging
import os
import threading
import time
from typing import List, Optional, Any

from google import genai
from google.genai import types

import config

logger = logging.getLogger("yap.intelligence.gemini_pool")


class GeminiKey:
    def __init__(self, key_index: int, api_key: str):
        self.key_index = key_index
        # Store api_key in a private attribute so it doesn't get serialized or printed in logs/reprs
        self._api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.healthy = True
        self.cooldown_until = 0.0
        self.consecutive_failures = 0
        self.total_requests = 0
        self.total_successes = 0
        self.total_failures = 0
        self.last_used = 0.0

    def __repr__(self) -> str:
        return f"GeminiKey(key={self.key_index}, healthy={self.healthy}, cooldown={self.is_cooling_down()})"

    def is_cooling_down(self) -> bool:
        return time.time() < self.cooldown_until

    def mark_cooldown(self, seconds: float = 60.0):
        self.cooldown_until = time.time() + seconds

    def mark_unhealthy(self):
        self.healthy = False


class GeminiPool:
    def __init__(self):
        self._lock = threading.Lock()
        self._keys: List[GeminiKey] = []
        self._rr_index = 0
        self.initialize_pool()

    def initialize_pool(self):
        """Load, deduplicate, and build clients from environment variables."""
        raw_keys = []
        for i in range(1, 6):
            val = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
            if val:
                raw_keys.append(val)

        legacy_key = os.getenv("GEMINI_API_KEY", "").strip()
        if legacy_key and legacy_key not in raw_keys:
            raw_keys.append(legacy_key)

        # Deduplicate keys
        unique_keys = []
        for k in raw_keys:
            if k not in unique_keys:
                unique_keys.append(k)

        if not unique_keys:
            logger.error("[GEMINI-POOL] Initialized with 0 valid keys!")
            return

        for idx, key in enumerate(unique_keys, 1):
            self._keys.append(GeminiKey(key_index=idx, api_key=key))

        logger.info("[GEMINI-POOL] initialized configured_keys=%d", len(self._keys))

    def get_keys_status(self) -> dict:
        with self._lock:
            available = sum(1 for k in self._keys if k.healthy and not k.is_cooling_down())
            cooldown = sum(1 for k in self._keys if k.healthy and k.is_cooling_down())
            unhealthy = sum(1 for k in self._keys if not k.healthy)
            return {"available": available, "cooldown": cooldown, "unhealthy": unhealthy}

    def _select_key(self) -> Optional[GeminiKey]:
        """Thread-safe round-robin selection of an available healthy, non-cooldown key."""
        with self._lock:
            active_keys = [k for k in self._keys if k.healthy and not k.is_cooling_down()]
            if not active_keys:
                # If all are cooling down, fallback to any healthy key to avoid total silence
                active_keys = [k for k in self._keys if k.healthy]
                if not active_keys:
                    return None

            # Sort active keys by key_index to maintain deterministic round robin ordering
            active_keys.sort(key=lambda k: k.key_index)

            selected = None
            for k in active_keys:
                if (k.key_index - 1) >= self._rr_index:
                    selected = k
                    break
            if not selected:
                selected = active_keys[0]

            # Move round-robin index forward
            self._rr_index = (selected.key_index) % len(self._keys) if len(self._keys) > 0 else 0
            return selected

    def generate_content(
        self,
        contents: Any,
        config_opts: Optional[types.GenerateContentConfig] = None,
        model: Optional[str] = None,
    ) -> Any:
        """
        Thread-safe wrapper around content generation with automatic failover and status logging.
        """
        if not self._keys:
            raise ValueError("[GEMINI-POOL] Configuration Error: Zero valid API keys loaded in pool.")

        max_attempts = max(3, len(self._keys))
        attempt = 1
        model_name = model or config.GEMINI_MODEL

        while attempt <= max_attempts:
            key_obj = self._select_key()
            if not key_obj:
                raise RuntimeError("[GEMINI-POOL] All configured Gemini API keys are unhealthy.")

            with self._lock:
                key_obj.total_requests += 1
                key_obj.last_used = time.time()
                idx = key_obj.key_index

            logger.info("[GEMINI-POOL] operation=generate_content model=%s key=%d attempt=%d",
                        model_name, idx, attempt)

            try:
                # API request is made OUTSIDE the lock
                response = key_obj.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config_opts,
                )

                with self._lock:
                    key_obj.total_successes += 1
                    key_obj.consecutive_failures = 0

                logger.info("[GEMINI-POOL] success key=%d", idx)
                return response

            except Exception as e:
                err_str = str(e)
                logger.warning("[GEMINI-POOL] failure key=%d attempt=%d error: %s", idx, attempt, err_str)

                with self._lock:
                    key_obj.total_failures += 1
                    key_obj.consecutive_failures += 1

                # Quota limit (429 or RESOURCE_EXHAUSTED)
                is_quota = "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower()

                # Auth / Invalid key
                is_auth = "API_KEY_INVALID" in err_str or "invalid" in err_str.lower() or "400" in err_str or "403" in err_str

                if is_auth:
                    with self._lock:
                        key_obj.mark_unhealthy()
                    logger.error("[GEMINI-POOL] key=%d marked UNHEALTHY (auth/invalid key)", idx)
                elif is_quota:
                    with self._lock:
                        key_obj.mark_cooldown(60.0)  # 60s cooldown
                    status = self.get_keys_status()
                    logger.warning("[GEMINI-POOL] key=%d cooldown activated reason=rate_limit available=%d cooldown=%d unhealthy=%d",
                                   idx, status["available"], status["cooldown"], status["unhealthy"])
                else:
                    # Network / Timeout transient error
                    if key_obj.consecutive_failures >= 3:
                        with self._lock:
                            key_obj.mark_cooldown(30.0)  # 30s cooldown
                        status = self.get_keys_status()
                        logger.warning("[GEMINI-POOL] key=%d cooldown activated reason=transient_failures available=%d cooldown=%d unhealthy=%d",
                                       idx, status["available"], status["cooldown"], status["unhealthy"])

                # Fails over immediately to next index
                next_idx = (idx % len(self._keys)) + 1
                logger.info("[GEMINI-POOL] failover key=%d -> key=%d", idx, next_idx)

                attempt += 1

        raise RuntimeError(f"[GEMINI-POOL] Generation failed after trying all {max_attempts} keys in the pool.")


# Global singleton instance
_pool_instance: Optional[GeminiPool] = None
_pool_lock = threading.Lock()


def get_pool() -> GeminiPool:
    global _pool_instance
    with _pool_lock:
        if _pool_instance is None:
            _pool_instance = GeminiPool()
        return _pool_instance


def generate_content(
    contents: Any,
    config_opts: Optional[types.GenerateContentConfig] = None,
    model: Optional[str] = None,
) -> Any:
    return get_pool().generate_content(contents, config_opts, model)
