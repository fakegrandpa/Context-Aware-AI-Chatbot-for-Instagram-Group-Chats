# Eve

Eve is a persona-driven AI that participates in an **Instagram group chat** as if it were
one of the members. It watches a target group thread in real time, decides when and how to
respond, and posts back text or voice replies in a consistent character voice.

> **Version:** V6

## What it does

- **Persona-driven replies.** Eve reads the group conversation and generates in-character
  responses instead of generic assistant answers.
- **Gemini API key pool.** Text generation runs through a pool of up to five Gemini API
  keys with automatic **round-robin selection** and **failover** — if a key hits a rate
  limit or goes unhealthy, Eve fails over to another key so the bot keeps talking.
- **Optional voice replies.** When enabled, Eve can send native-audio voice replies via
  the **Gemini Live** API (using a separate, dedicated key from the text pool).
- **Memory extraction.** A background worker distills chat history into stored memories so
  Eve can remember facts about the group over time.
- **Conversation "lanes" & fatigue tracking.** Eve tracks separate conversation lanes and
  applies social-fatigue thresholds so it doesn't reply too often or dominate the chat.
- **SQLite persistence.** Messages, state, profiles, and extracted memories are stored in a
  local SQLite database.
- **Realtime + fallback.** Eve subscribes to Instagram's realtime (MQTT) stream and falls
  back to HTTP polling when the stream is unavailable.

## Requirements

- Python 3.10+
- An Instagram account (the one Eve will post from)
- One or more [Gemini API keys](https://aistudio.google.com/)
- (Optional, for voice replies) an `ffmpeg` binary and a dedicated Gemini Live API key

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/fakegrandpa/eve.git
cd eve

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your own .env from the template
cp .env.example .env
```

Then open `.env` and fill in **your own** values:

- `IG_USERNAME` / `IG_PASSWORD` — your Instagram login credentials.
- `GEMINI_API_KEY_1` … `GEMINI_API_KEY_5` — one or more Gemini API keys (empty slots are ignored). `GEMINI_API_KEY` is a legacy single-key fallback.
- `TARGET_THREAD_ID` — the group chat Eve should join (see below).
- (Optional) Voice settings such as `VOICE_ENABLED` and `GEMINI_LIVE_API_KEY`.

### Automatic Folders & Dependencies Setup (First Run)

On first run or startup:
- The application automatically creates the `./bin` and `./data` folders.
- A local SQLite database is automatically initialized at `./data/yap.db` to persist conversation history and extracted memories.
- **Voice Reply Setup (FFmpeg)**: To enable voice replies, Eve requires `ffmpeg`. You do not need to manually configure PATH variables or edit `FFMPEG_PATH` in `.env`. Simply download the `ffmpeg` binary for your platform and place the executable (`ffmpeg.exe` on Windows, or `ffmpeg` on macOS/Linux) directly inside the automatically created `./bin/` folder. The application automatically searches `./bin/` and uses it.

### Finding your TARGET_THREAD_ID

With your Instagram credentials filled into `.env`, run:

```bash
python list_threads.py
```

This prints your DM/group threads and their IDs. Copy the ID of the target group chat into `TARGET_THREAD_ID` in your `.env`.


## Running

Start the bot:

```bash
python main.py
```

Eve will log in (reusing a saved session when possible), bootstrap safely without replying
to historical messages, and begin participating in the target group chat. Stop it with
`Ctrl+C`.

## Tests

```bash
pytest tests/
```

## Disclaimer

⚠️ **This project automates a personal Instagram account.** Using automation with Instagram
may violate [Instagram's Terms of Service](https://help.instagram.com/581066165581870) and
could result in restrictions or a ban on your account. This software is provided for
**personal and educational use only**. You assume all risk for how you use it.

## Privacy — no data or credentials are shipped

This repository ships **no chat data and no credentials**. There are no real usernames,
API keys, passwords, session cookies, or thread ids in this repo. The live SQLite database,
Instagram session, runtime state, and `.env` are git-ignored and never committed. Every user
must supply their **own** Instagram account, Gemini API key(s), and target thread id.

## License

Released under the [MIT License](LICENSE).
