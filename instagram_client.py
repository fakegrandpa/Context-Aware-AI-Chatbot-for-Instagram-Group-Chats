import logging
import datetime
import instagrapi.extractors as ex

# Monkeypatch instagrapi to handle invalid/negative timestamps on Windows safely
_orig_direct_timestamp = ex._direct_timestamp_from_microseconds


def _safe_direct_timestamp_from_microseconds(timestamp):
    try:
        val = int(timestamp)
    except (ValueError, TypeError):
        val = 0

    # Normalize clearly invalid negative/placeholder timestamps (like -1000)
    # Valid microsecond timestamps are positive numbers > 0.
    if val <= 0:
        val = 0

    try:
        return _orig_direct_timestamp(val)
    except OSError:
        return datetime.datetime.fromtimestamp(0)


ex._direct_timestamp_from_microseconds = _safe_direct_timestamp_from_microseconds

from instagrapi import Client

import config

logger = logging.getLogger("n1.instagram")


def _new_client() -> Client:
    """
    Create a Client and immediately give it a persisted, stable device
    fingerprint (UUIDs/device/user-agent).

    Without this, every fresh Client() may use new device identifiers,
    making repeated login attempts look like a new device to Instagram.
    """
    cl = Client()
    cl.delay_range = [1, 3]

    if config.SESSION_PATH.exists():
        session = cl.load_settings(config.SESSION_PATH)
        if session:
            cl.set_settings(session)
    else:
        # Persist the initial stable device fingerprint.
        cl.dump_settings(config.SESSION_PATH)

    return cl


def build_client() -> Client:
    cl = _new_client()

    # ---------------------------------------------------------
    # 1. Try the existing saved session first.
    #
    # IMPORTANT:
    # We validate using Instagram Direct because that is what Eve
    # actually needs, and our standalone test proved Direct works.
    # ---------------------------------------------------------
    if cl.user_id:
        try:
            cl.direct_threads(amount=1)

            if not cl.username and config.IG_USERNAME:
                cl.username = config.IG_USERNAME

            logger.info(
                "Reusing existing authenticated session for %s",
                cl.username,
            )

            return cl

        except Exception as e:
            logger.info(
                "Saved session is no longer valid (%s), trying authentication recovery",
                e,
            )

            # instagrapi login() can short-circuit if user_id is still set.
            # Clear it before attempting a real authentication recovery.
            cl.user_id = None

    # ---------------------------------------------------------
    # 2. Prefer imported browser sessionid if configured.
    #
    # This is the authentication path that succeeded in
    # test_session.py.
    # ---------------------------------------------------------
    if config.IG_SESSIONID:
        logger.info("Attempting Instagram login via imported sessionid")

        cl.login_by_sessionid(config.IG_SESSIONID)

        logger.info(
            "Logged in via imported sessionid as %s",
            cl.username,
        )

    # ---------------------------------------------------------
    # 3. Fall back to username/password only if no sessionid exists.
    # ---------------------------------------------------------
    else:
        logger.info("No IG_SESSIONID configured; attempting username/password login")

        cl.login(
            config.IG_USERNAME,
            config.IG_PASSWORD,
            relogin=True,
        )

        logger.info(
            "Logged in using username/password as %s",
            cl.username,
        )

    # ---------------------------------------------------------
    # 4. Basic authentication sanity check.
    # ---------------------------------------------------------
    if not cl.user_id:
        raise RuntimeError(
            "Instagram login did not produce a valid session (no user_id)"
        )

    # ---------------------------------------------------------
    # 5. Verify the exact API family Eve depends on.
    #
    # Do NOT use get_timeline_feed() here.
    # The standalone test proved Direct API access works.
    # ---------------------------------------------------------
    logger.info("Verifying Instagram Direct API access")

    cl.direct_threads(amount=1)

    logger.info(
        "Instagram Direct API verified for %s (user_id=%s)",
        cl.username,
        cl.user_id,
    )

    # ---------------------------------------------------------
    # 6. Persist the now-verified working session.
    # ---------------------------------------------------------
    cl.dump_settings(config.SESSION_PATH)

    logger.info(
        "Saved verified Instagram session to %s",
        config.SESSION_PATH,
    )

    return cl


def list_threads(cl: Client, amount: int = 20):
    threads = cl.direct_threads(amount=amount)

    result = []

    for t in threads:
        result.append(
            {
                "id": t.id,
                "title": t.thread_title,
                "is_group": t.is_group,
                "user_count": len(t.users),
            }
        )

    return result


def fetch_thread(cl: Client, thread_id: str, amount: int = 40):
    """
    Returns:
        (messages_sorted_ascending, username_map)
    """

    thread = cl.direct_thread(
        int(thread_id),
        amount=amount,
    )

    username_map = {}

    for u in thread.users:
        username_map[str(u.pk)] = u.username

    username_map[str(cl.user_id)] = cl.username

    messages = sorted(
        thread.messages,
        key=lambda m: m.timestamp,
    )

    return messages, username_map


def send_text(cl: Client, thread_id: str, text: str):
    return cl.direct_answer(
        int(thread_id),
        text,
    )