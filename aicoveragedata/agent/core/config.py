from dataclasses import dataclass
import os
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = AGENT_DIR / ".env"
STATE_DIR = AGENT_DIR / "state"
DEFAULT_OPENAI_MODEL = "gpt-5.2"


def load_env_file(path=ENV_FILE):
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AgentConfig:
    api_key: str | None
    model: str
    reasoning_effort: str
    text_verbosity: str
    use_dspy: bool
    dspy_fallback: bool
    max_context_chars: int
    chat_history_path: Path
    max_history_messages: int
    max_stored_message_chars: int

    @classmethod
    def from_env(cls):
        load_env_file()
        return cls(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low"),
            text_verbosity=os.getenv("OPENAI_TEXT_VERBOSITY", "low"),
            use_dspy=env_bool("AGENT_USE_DSPY", True),
            dspy_fallback=env_bool("AGENT_DSPY_FALLBACK", True),
            max_context_chars=int(os.getenv("AGENT_MAX_CONTEXT_CHARS", "18000")),
            chat_history_path=Path(os.getenv("AGENT_CHAT_HISTORY_PATH", STATE_DIR / "chat_history.json")),
            max_history_messages=int(os.getenv("AGENT_MAX_HISTORY_MESSAGES", "16")),
            max_stored_message_chars=int(os.getenv("AGENT_MAX_STORED_MESSAGE_CHARS", "4000")),
        )
