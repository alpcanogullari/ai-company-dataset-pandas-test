"""Dataset question-answering agent for the AI coverage project."""

from .core.agent import answer_question
from .core.config import AgentConfig

__all__ = ["AgentConfig", "answer_question"]
