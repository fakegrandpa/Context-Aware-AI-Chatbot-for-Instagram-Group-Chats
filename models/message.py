"""
NormalizedMessage — clean internal message model consumed by all V4 systems.

All timestamps are timezone-aware UTC datetime objects.
Identity is always by user_id (str), never by username.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("yap.models.message")


@dataclass
class NormalizedMessage:
    """Clean internal message model. Both realtime (MQTT) and HTTP paths produce this."""

    # Core identity
    message_id: str
    thread_id: str
    sender_id: str          # Instagram user_id as str — permanent identity key
    sender_username: str    # May change; never used as identity key

    # Content
    text: Optional[str]
    timestamp: datetime     # Always timezone-aware UTC
    item_type: str          # "text", "reaction", "media", etc.
    is_sent_by_viewer: bool

    # Reply graph — populated when available, None otherwise
    reply_to_message_id: Optional[str] = None
    reply_to_user_id: Optional[str] = None    # user_id of the original message author
 
    # Internal tracking
    is_historical: bool = False   # True = from bootstrap; never reply to these
    
    # Raw message object reference to avoid HTTP fetching on replies
    raw_dm: Optional[any] = None

    # V5.5 thread disentanglement
    conversation_id: Optional[str] = None


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware UTC. Handles both naive and aware inputs."""
    if dt.tzinfo is None:
        # Assume local time, convert to UTC
        return dt.astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_realtime(msg_dict: dict, viewer_id: str, username_cache: dict) -> Optional[NormalizedMessage]:
    """
    Build a NormalizedMessage from the MQTT realtime payload dict.

    The msg_dict is the `message` field from the realtime event wrapper:
    {
        path: str,
        op: str,             # "add" = new message
        thread_id: str,
        item_id: str,
        item_type: str,
        text: str | None,
        user_id: int | str,
        timestamp: int,      # microseconds since epoch
        # May contain reply fields:
        replied_to_item_id: str | None,
        replied_to_user_id: int | str | None,
    }

    Returns None if the payload is not a processable new message.
    """
    if not isinstance(msg_dict, dict):
        return None

    op = msg_dict.get("op")
    if op != "add":
        return None

    item_type = msg_dict.get("item_type", "")
    item_id = msg_dict.get("item_id") or msg_dict.get("id")
    thread_id = str(msg_dict.get("thread_id", ""))
    user_id = str(msg_dict.get("user_id", ""))
    text = msg_dict.get("text")

    if not item_id or not thread_id or not user_id:
        logger.debug("[NORM] realtime: missing required fields item_id=%s thread_id=%s user_id=%s",
                     item_id, thread_id, user_id)
        return None

    # Parse microsecond timestamp
    ts_raw = msg_dict.get("timestamp", 0)
    try:
        ts = datetime.fromtimestamp(int(ts_raw) / 1_000_000.0, tz=timezone.utc)
    except (ValueError, OSError, TypeError):
        ts = datetime.now(timezone.utc)
        logger.warning("[NORM] realtime: bad timestamp %r, using now", ts_raw)

    is_viewer = (user_id == str(viewer_id))
    username = username_cache.get(user_id, user_id)

    # Extract reply metadata — may be present in MQTT payload
    reply_to_msg_id = msg_dict.get("replied_to_item_id") or None
    reply_to_uid = msg_dict.get("replied_to_user_id")
    if reply_to_uid is not None:
        reply_to_uid = str(reply_to_uid)

    return NormalizedMessage(
        message_id=str(item_id),
        thread_id=thread_id,
        sender_id=user_id,
        sender_username=username,
        text=text,
        timestamp=ts,
        item_type=item_type,
        is_sent_by_viewer=is_viewer,
        reply_to_message_id=reply_to_msg_id,
        reply_to_user_id=reply_to_uid,
        is_historical=False,
    )


def normalize_http(dm, thread_id: str, viewer_id: str, username_cache: dict) -> Optional[NormalizedMessage]:
    """
    Build a NormalizedMessage from an instagrapi DirectMessage object (HTTP poll path).

    dm: instagrapi.types.DirectMessage
    """
    try:
        msg_id = str(dm.id)
        user_id = str(dm.user_id) if dm.user_id else ""
        item_type = dm.item_type or "unknown"
        text = dm.text
        is_viewer = bool(dm.is_sent_by_viewer)
        ts = _ensure_utc(dm.timestamp)
        username = username_cache.get(user_id, user_id)

        # Extract reply metadata from the HTTP reply field
        reply_to_msg_id = None
        reply_to_uid = None
        if dm.reply is not None:
            reply_to_msg_id = str(dm.reply.id) if dm.reply.id else None
            reply_to_uid = str(dm.reply.user_id) if dm.reply.user_id else None

        if not msg_id or not user_id:
            logger.debug("[NORM] http: missing required fields msg_id=%s user_id=%s", msg_id, user_id)
            return None

        return NormalizedMessage(
            message_id=msg_id,
            thread_id=str(thread_id),
            sender_id=user_id,
            sender_username=username,
            text=text,
            timestamp=ts,
            item_type=item_type,
            is_sent_by_viewer=is_viewer,
            reply_to_message_id=reply_to_msg_id,
            reply_to_user_id=reply_to_uid,
            is_historical=False,
            raw_dm=dm,
        )
    except Exception as e:
        logger.error("[NORM] http normalization failed: %s", e)
        return None
