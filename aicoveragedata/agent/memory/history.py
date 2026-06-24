import json
import re
import threading
from pathlib import Path


_HISTORY_LOCK = threading.Lock()


def normalize_session_id(session_id):
    value = str(session_id or "default").strip()
    value = re.sub(r"[^A-Za-z0-9_.:-]", "_", value)
    return value[:80] or "default"


def trim_message(text, max_chars):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[Message trimmed.]"


def _read_history(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_history(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(path)


def get_session_messages(session_id, path, max_messages=None):
    session_id = normalize_session_id(session_id)
    with _HISTORY_LOCK:
        data = _read_history(path)
        messages = data.get(session_id, [])
    if not isinstance(messages, list):
        return []
    if max_messages:
        return messages[-max_messages:]
    return messages


def append_exchange(session_id, user_text, assistant_text, path, max_messages=16, max_chars=4000):
    session_id = normalize_session_id(session_id)
    exchange = [
        {"role": "user", "content": trim_message(user_text, max_chars)},
        {"role": "assistant", "content": trim_message(assistant_text, max_chars)},
    ]
    with _HISTORY_LOCK:
        data = _read_history(path)
        messages = data.get(session_id, [])
        if not isinstance(messages, list):
            messages = []
        messages.extend(exchange)
        data[session_id] = messages[-max_messages:]
        _write_history(path, data)
    return data[session_id]


def clear_session(session_id, path):
    session_id = normalize_session_id(session_id)
    with _HISTORY_LOCK:
        data = _read_history(path)
        removed = bool(data.pop(session_id, None))
        _write_history(path, data)
    return removed


def format_history(messages):
    lines = []
    for message in messages:
        role = message.get("role", "message")
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n\n".join(lines)

