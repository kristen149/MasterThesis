"""Prompt strategies.

A ``PromptStrategy`` consumes a ``Sample`` and a schema description and
returns a list of chat messages ready for any ``ModelAdapter``. The runner
holds one ``PromptStrategy`` instance per run; swapping strategies is a CLI
flag, not a code change in the runner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..data_loader import Sample


@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# Shared system prompt — every strategy uses the same instruction header so
# differences across strategies are isolated to the user turn.
SYSTEM_PROMPT = (
    "You are an expert SQLite query writer for an event-log database used in "
    "process mining. Produce a single valid SQLite SQL statement that answers "
    "the user's question. Output ONLY the SQL — no prose, no markdown fences, "
    "no comments. End with a semicolon."
)


class PromptStrategy(ABC):
    """Abstract base. Subclass with a single ``build`` method."""

    name: str = "base"

    # Some strategies (notably the OpenAI/NumberSign zero-shot) send a
    # completion-style prompt that ends with a token the model is expected
    # to continue from (e.g. "SELECT "). Subclasses declare their prefix
    # here; the runner passes it to the postprocessor so it can re-prepend
    # the prefix when the model continues directly. Default is empty
    # (i.e. the model is expected to emit a full SQL statement on its own).
    completion_prefix: str = ""

    def __init__(self, schema_description: str, schema_notes: str = "") -> None:
        self.schema_description = schema_description
        self.schema_notes = schema_notes

    @abstractmethod
    def build(self, sample: Sample) -> list[ChatMessage]:
        """Return the ordered chat messages to send to the model."""
        raise NotImplementedError

    # Helpers shared by subclasses ------------------------------------------
    def _schema_block(self) -> str:
        block = f"Database schema:\n  {self.schema_description}"
        if self.schema_notes:
            block += f"\n\n{self.schema_notes}"
        return block

    def _system_message(self) -> ChatMessage:
        return ChatMessage(role="system", content=SYSTEM_PROMPT)
