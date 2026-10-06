import asyncio
import contextlib
import os
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from cartesia import AsyncCartesia
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.checkpoint.memory import InMemorySaver
from starlette.staticfiles import StaticFiles

from cartesia_prompts import CARTESIA_TTS_SYSTEM_PROMPT
from events import (
    AgentChunkEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    TTSChunkEvent,
    TurnEvent,
    VoiceAgentEvent,
    event_to_dict,
)
from utils import merge_async_iters

load_dotenv()

# Static files are served from the shared web build output
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Web build not found at {STATIC_DIR}. "
        "Run 'make build-web' or 'make dev-py' from the project root."
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."


def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


system_prompt = f"""
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.

{CARTESIA_TTS_SYSTEM_PROMPT}
"""

cartesia = AsyncCartesia(api_key=os.environ["CARTESIA_API_KEY"])
VOICE_ID = "f6ff7c0c-e396-40a9-a70b-f7607edb6937"

agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[add_to_order, confirm_order],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)


async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (bytes) → Turn events, via Cartesia Ink 2.

    audio_stream: PCM audio from the browser (16-bit, mono, 16 kHz).
    Yields a TurnEvent for each Ink 2 turn.* message (see events.TurnEvent).
    """
    async with cartesia.stt.auto_finalize.websocket(
        model="ink-2", encoding="pcm_s16le", sample_rate=16000
    ) as connection:

        async def send_audio():
            # Runs alongside the receive loop below, so transcripts arrive
            # while the user is still talking.
            try:
                async for chunk in audio_stream:
                    await connection.send_raw(chunk)
            finally:
                # Ask Ink 2 to finish any buffered audio. It closes the socket
                # when done, which ends the receive loop.
                await connection.send({"type": "close"})

        send_task = asyncio.create_task(send_audio())
        try:
            async for event in connection:
                if event.type.startswith("turn."):
                    transcript = getattr(event, "transcript", "")
                    yield TurnEvent.create(event.type, transcript)
                elif event.type == "error":
                    print(f"Ink 2 error: {event.message}")
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                send_task.cancel()
                await send_task


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Agent Responses)

    Passes every upstream event through. On each turn.end (the user finished
    speaking) it sends the transcript to the LangChain agent and yields the
    reply as agent_chunk, tool_call and tool_result events, then agent_end.
    """
    # Generate a unique thread ID for this conversation session
    # This allows the agent to maintain conversation context across multiple turns
    # using the checkpointer (InMemorySaver) configured in the agent
    thread_id = str(uuid4())

    # TODO:@zeuslawyer barge-in handling to be added. While the agent replies
    # below, this loop can't read the next event, so a turn.start that arrives
    # mid-reply waits until the reply ends.
    async for event in event_stream:
        # Pass through all events to downstream consumers
        yield event

        if event.type == "turn.end" and event.transcript:
            # Stream the agent's response using LangChain's astream method.
            # stream_mode="messages" yields message chunks as they're generated.
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            # Iterate through the agent's streaming response. The stream yields
            # tuples of (message, metadata), but we only need the message.
            # Text after a tool call starts with no space ("for you." + "Your"),
            # and the browser and TTS join chunks as-is. So we add one, but only
            # if the reply already has text.
            has_text = after_tool = False
            async for message, metadata in stream:
                # Emit agent chunks (AI messages)
                if isinstance(message, AIMessage):
                    text = message.text
                    if after_tool and text:
                        text, after_tool = " " + text.lstrip(), False
                    has_text = has_text or bool(text)
                    yield AgentChunkEvent.create(text)
                    # Emit tool calls if present
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            yield ToolCallEvent.create(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )

                # Emit tool results (tool messages)
                if isinstance(message, ToolMessage):
                    after_tool = has_text
                    yield ToolResultEvent.create(
                        tool_call_id=getattr(message, "tool_call_id", ""),
                        name=getattr(message, "name", "unknown"),
                        result=str(message.content) if message.content else "",
                    )

            # Signal that the agent has finished responding for this turn
            yield AgentEndEvent.create()


async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Audio), via Cartesia Sonic.

    Passes every upstream event through. For each agent reply, sends the text
    to Sonic and yields the speech as tts_chunk events (raw PCM, 24 kHz).
    A Sonic "context" is one generation; we use one per agent reply.
    """
    async with cartesia.tts.websocket_connect() as ws:
        # Sending and receiving run at the same time (merge_async_iters).
        # This queue hands each reply's context from the sender to the
        # receiver, in order. None means no more replies.
        contexts: asyncio.Queue = asyncio.Queue()

        async def forward_events_and_send_text() -> AsyncIterator[VoiceAgentEvent]:
            buffer: list[str] = []
            try:
                async for event in event_stream:
                    yield event
                    if event.type == "agent_chunk":
                        buffer.append(event.text)
                    if event.type == "agent_end":
                        text, buffer = "".join(buffer).strip(), []
                        if not text:
                            continue
                        ctx = ws.context(
                            model_id="sonic-3.6",
                            voice=VOICE_ID,
                            output_format={
                                "container": "raw",
                                "encoding": "pcm_s16le",
                                "sample_rate": 24000,
                            },
                            language="en",
                        )
                        await contexts.put(ctx)
                        await ctx.push(text)
                        await ctx.no_more_inputs()
            finally:
                # Ends receive_audio; otherwise merge_async_iters waits forever.
                await contexts.put(None)

        async def receive_audio() -> AsyncIterator[VoiceAgentEvent]:
            while (ctx := await contexts.get()) is not None:
                async for response in ctx.receive():
                    if response.type == "chunk" and response.audio:
                        yield TTSChunkEvent.create(response.audio)
                    elif response.type == "error":
                        print(f"Sonic error: {response.message}")

        async for event in merge_async_iters(
            forward_events_and_send_text(), receive_audio()
        ):
            yield event


pipeline = (
    RunnableGenerator(_stt_stream)  # Audio -> STT events
    | RunnableGenerator(_agent_stream)  # STT events -> STT + Agent events
    | RunnableGenerator(_tts_stream)  # STT + Agent events -> All events
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        """Async generator that yields audio bytes from the websocket."""
        try:
            while True:
                yield await websocket.receive_bytes()
        except WebSocketDisconnect:
            # Browser closed: end the stream, so the STT stage tells Ink 2 to close.
            return

    output_stream = pipeline.atransform(websocket_audio_stream())

    # Process all events from the pipeline, sending events back to the client
    async for event in output_stream:
        await websocket.send_json(event_to_dict(event))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
