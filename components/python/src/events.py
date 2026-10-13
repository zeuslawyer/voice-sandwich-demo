"""
Events the voice pipeline sends to the browser.

Each event is a small record that says "this happened". The server sends
each one to the browser as JSON with `dataclasses.asdict(event)`, so field
names here are the JSON keys. They match the TypeScript backend
(`components/typescript/src/types.ts`).

Every event has `type` (what happened) and `ts` (when the server made it,
in milliseconds since 1970). Both are filled in for you:
`AgentChunkEvent(text="Hi")`.
"""

import time
from dataclasses import dataclass, field
from typing import Literal, Union


def _now_ms() -> int:
    """Return the time now, in milliseconds since 1970."""
    return int(time.time() * 1000)


TurnType = Literal[
    "turn.start", "turn.update", "turn.eager_end", "turn.resume", "turn.end"
]


@dataclass(kw_only=True)
class TurnEvent:
    """
    One turn event from Cartesia Ink 2 speech-to-text, passed on unchanged.

    A "turn" is one stretch of the user talking. Ink 2 detects it and sends:
    turn.start -> turn.update (repeats) -> [turn.eager_end -> turn.resume] -> turn.end

    transcript: the turn's text so far. It only grows and is never revised;
    turn.end has the final text. Empty on turn.start and turn.resume.
    """

    type: TurnType
    transcript: str = ""
    ts: int = field(default_factory=_now_ms)


@dataclass(kw_only=True)
class AgentChunkEvent:
    """A few words of the agent's reply, sent as the LLM writes them."""

    text: str
    type: Literal["agent_chunk"] = "agent_chunk"
    ts: int = field(default_factory=_now_ms)


@dataclass(kw_only=True)
class AgentEndEvent:
    """The agent's reply for this turn is complete. No more text follows."""

    type: Literal["agent_end"] = "agent_end"
    ts: int = field(default_factory=_now_ms)


@dataclass(kw_only=True)
class ToolCallEvent:
    """
    The agent called a tool.

    id: the call's id. name: the tool's name. args: what the agent passed in.
    """

    id: str
    name: str
    args: dict
    type: Literal["tool_call"] = "tool_call"
    ts: int = field(default_factory=_now_ms)


@dataclass(kw_only=True)
class ToolResultEvent:
    """
    A tool finished.

    toolCallId: the `id` of the matching ToolCallEvent. Written in camelCase
    because it is the JSON key the browser and the TypeScript backend use.
    name: the tool's name. result: what the tool returned, as text.
    """

    toolCallId: str
    name: str
    result: str
    type: Literal["tool_result"] = "tool_result"
    ts: int = field(default_factory=_now_ms)


@dataclass(kw_only=True)
class TTSChunkEvent:
    """
    A piece of the agent's voice from Cartesia Sonic, ready to play.

    audio: raw 16-bit PCM, mono, 24 kHz, as base64 text (JSON can't hold bytes).
    """

    audio: str
    type: Literal["tts_chunk"] = "tts_chunk"
    ts: int = field(default_factory=_now_ms)


AgentEvent = Union[AgentChunkEvent, AgentEndEvent, ToolCallEvent, ToolResultEvent]
VoiceAgentEvent = Union[TurnEvent, AgentEvent, TTSChunkEvent]
