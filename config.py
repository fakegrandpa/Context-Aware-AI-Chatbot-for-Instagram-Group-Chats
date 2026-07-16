import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SESSION_PATH = DATA_DIR / "session.json"
STATE_PATH = DATA_DIR / "state.json"

# V4: SQLite database path
DB_PATH = DATA_DIR / "yap.db"

IG_USERNAME = os.getenv("IG_USERNAME", "")
IG_PASSWORD = os.getenv("IG_PASSWORD", "")
IG_SESSIONID = os.getenv("IG_SESSIONID", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

TARGET_THREAD_ID = os.getenv("TARGET_THREAD_ID", "").strip()

POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "2"))

CONTEXT_SIZE = 20
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY_SECONDS = 2

# V5: active character name. Historical DB rows may still say "Yap" — that's
# fine, they're historical. Runtime identity is Eve everywhere.
BOT_NAME = "Eve"

# V4: Burst coalescing window (milliseconds)
BURST_WINDOW_MS = int(os.getenv("BURST_WINDOW_MS", "600"))

# V4: Memory extraction
MEMORY_BATCH_SIZE = int(os.getenv("MEMORY_BATCH_SIZE", "15"))
MEMORY_WORKER_POLL_SECONDS = int(os.getenv("MEMORY_WORKER_POLL_SECONDS", "30"))

# V4: Lane context window (messages per lane for Gemini context)
CONTEXT_LANE_SIZE = int(os.getenv("CONTEXT_LANE_SIZE", "15"))

# V5: Raw GC scene window for social targeting (independent of lanes — see
# conversation/attention.py and intelligence/social_judge.py). PART 7 of the
# V5 spec requires at least the last 15 raw messages be visible to the router.
SOCIAL_SCENE_SIZE = int(os.getenv("SOCIAL_SCENE_SIZE", "15"))

# V4: Social fatigue thresholds
FATIGUE_MAX_REPLIES_60S = int(os.getenv("FATIGUE_MAX_REPLIES_60S", "4"))
FATIGUE_MAX_REPLIES_5MIN = int(os.getenv("FATIGUE_MAX_REPLIES_5MIN", "10"))
FATIGUE_MAX_CONSECUTIVE = int(os.getenv("FATIGUE_MAX_CONSECUTIVE", "5"))

# V4: Bot user_id — populated at runtime from cl.user_id after login
# Do NOT hardcode here; set via config.BOT_USER_ID = str(cl.user_id) in main.py
BOT_USER_ID: str = ""

# ======================================================================
# V5: Voice mode (Gemini Live native audio)
# ======================================================================

def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


VOICE_ENABLED = _env_bool("VOICE_ENABLED", "true")

# Target long-run fraction of replies that become voice (~3/7). This is a
# behavioral target for conversation/mode_selector.py, not a strict counter.
VOICE_TARGET_RATIO = float(os.getenv("VOICE_TARGET_RATIO", "0.43"))

# Voice uses ONE dedicated key — never the text round-robin pool.
GEMINI_LIVE_API_KEY = os.getenv("GEMINI_LIVE_API_KEY", "").strip()
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")

# Must be one of the prebuilt voices guaranteed available on Live native-audio
# models (Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr). Leda reads
# as a youthful female voice, matching Eve's persona.
GEMINI_LIVE_VOICE = os.getenv("GEMINI_LIVE_VOICE", "Leda")

VOICE_TIMEOUT_SECONDS = float(os.getenv("VOICE_TIMEOUT_SECONDS", "20"))
VOICE_FAILURE_THRESHOLD = int(os.getenv("VOICE_FAILURE_THRESHOLD", "3"))
VOICE_COOLDOWN_SECONDS = float(os.getenv("VOICE_COOLDOWN_SECONDS", "300"))

# How many of Eve's most recent replies (text or voice) to consider when the
# mode selector decides whether a voice reply is "due" or "too recent".
VOICE_HISTORY_WINDOW = int(os.getenv("VOICE_HISTORY_WINDOW", "7"))

FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg").strip()
