# Voice Sandwich Demo 🥪

A real-time, voice-to-voice AI pipeline demo featuring a sandwich shop order assistant. Built with LangChain/LangGraph agents, Cartesia Ink 2 for speech-to-text, and Cartesia Sonic 3.6 for text-to-speech.

## Architecture

The pipeline processes audio through three transform stages using async generators with a producer-consumer pattern:

```mermaid
flowchart LR
    subgraph Client [Browser]
        Mic[🎤 Microphone] -->|PCM Audio| WS_Out[WebSocket]
        WS_In[WebSocket] -->|Audio + Events| Speaker[🔊 Speaker]
    end

    subgraph Server [Node.js / Python]
        WS_Receiver[WS Receiver] --> Pipeline

        subgraph Pipeline [Voice Agent Pipeline]
            direction LR
            STT[Cartesia Ink 2 STT] -->|Transcripts| Agent[LangChain Agent]
            Agent -->|Text Chunks| TTS[Cartesia Sonic 3.6 TTS]
        end

        Pipeline -->|Events| WS_Sender[WS Sender]
    end

    WS_Out --> WS_Receiver
    WS_Sender --> WS_In
```

### Pipeline Stages

Each stage is an async generator that transforms a stream of events:

1. **STT Stage** (`sttStream`): Streams audio to Cartesia Ink 2, yields its turn events (`turn.start`, `turn.update`, `turn.end`, ...)
2. **Agent Stage** (`agentStream`): Passes upstream events through, invokes LangChain agent on each `turn.end`, yields agent responses (`agent_chunk`, `tool_call`, `tool_result`, `agent_end`)
3. **TTS Stage** (`ttsStream`): Passes upstream events through, streams each `agent_chunk` to Cartesia Sonic as it arrives (one context per reply, with continuations), yields audio events (`tts_chunk`). Voice starts before the agent finishes its reply.

### Event sequence (one turn)

Each label uses the names in the code: TypeScript first, then Python (`sttStream / _stt_stream`). After `→` is the function the browser calls in `components/web/src/lib/websocket.ts`.

Agent stage = the LLM with its tools and chat memory. It is one stage in the pipeline, not the whole voice app.

```mermaid
sequenceDiagram
    participant B as Browser<br/>websocket.ts
    participant WS as /ws handler<br/>index.ts / main.py
    participant STT as STT stage<br/>sttStream / _stt_stream
    participant INK as Cartesia Ink 2<br/>(STT websocket)
    participant A as Agent stage (LLM + tools)<br/>agentStream / _agent_stream
    participant T as TTS stage<br/>ttsStream / _tts_stream
    participant SON as Cartesia Sonic<br/>(TTS websocket)

    Note over B,SON: Arrow into Browser = event sent by the /ws handler (currentSocket.send / send_json).<br/>Every stage yields every upstream event unchanged, so all events pass STT → Agent → TTS → /ws handler.

    loop every 100 ms while the session is open
        B->>WS: binary PCM chunk (audioCapture → ws.send)
        WS->>STT: inputStream.push / websocket_audio_stream yields bytes
        STT->>INK: sendRaw / send_raw
    end

    INK-->>STT: connected (ignored)
    INK-->>STT: turn.start
    STT->>B: turn.start → currentTurn.startTurn()

    loop while the user speaks
        INK-->>STT: turn.update {transcript}
        STT->>B: turn.update → currentTurn.sttChunk()
    end

    opt Ink 2 thinks the user may be done
        INK-->>STT: turn.eager_end, then turn.resume if the user goes on
        STT->>B: turn.eager_end / turn.resume (no case, ignored)
    end

    INK-->>STT: turn.end {transcript}
    STT->>A: turn.end {transcript}
    STT->>B: turn.end → currentTurn.sttEnd(), activities.add("stt")

    A->>A: agent.stream / agent.astream (LangChain + Claude)
    par Agent stage writes the reply
        loop each piece of reply text
            A->>B: agent_chunk {text} → currentTurn.agentChunk()
            A->>T: agent_chunk {text}
            T->>SON: ctx.push {transcript, continue: true} (first chunk opens the reply's context, ws.context)
        end
        opt agent uses a tool (runs on the server)
            A->>B: tool_call {name, args} → activities.add("tool")
            A->>B: tool_result {name, result} → activities.add("tool")
        end
        A->>B: agent_end → activities.add("agent", "Agent Response")
        A->>T: agent_end
        T->>SON: ctx.no_more_inputs {transcript: "", continue: false}
    and Sonic speaks while text still arrives (managed buffering, max_buffer_delay_ms 3000)
        loop audio arrives (ctx.receive)
            SON-->>T: chunk {data = base64 audio}
            T->>B: tts_chunk {audio} → audioPlayback.push(), currentTurn.ttsChunk()
        end
        SON-->>T: done
    end
    B->>B: 300 ms after last tts_chunk → finishTurn()

    Note over B,SON: Stop button
    B->>WS: ws.close()
    WS->>STT: inputStream.cancel() / WebSocketDisconnect ends the audio stream
    STT->>INK: send {"type": "close"}
    INK-->>STT: last turn events, then Ink 2 closes the socket
```

## Prerequisites

- **Node.js** (v18+) or **Python** (3.11+)
- **pnpm** or **uv** (Python package manager)

### API Keys

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| Cartesia | `CARTESIA_API_KEY` | Speech-to-Text (Ink 2) and Text-to-Speech (Sonic) |
| Anthropic | `ANTHROPIC_API_KEY` | LangChain Agent (Claude) |

## Quick Start

### Using Make (Recommended)

```bash
# Install all dependencies
make bootstrap

# Run TypeScript implementation (with hot reload)
make dev-ts

# Or run Python implementation (with hot reload)
make dev-py
```

The app will be available at `http://localhost:8000`

### Manual Setup

#### TypeScript

```bash
cd components/typescript
pnpm install
cd ../web
pnpm install && pnpm build
cd ../typescript
pnpm run server
```

#### Python

```bash
cd components/python
uv sync --dev
cd ../web
pnpm install && pnpm build
cd ../python
uv run src/main.py
```

## Project Structure

```
components/
├── web/                 # Svelte frontend (shared by both backends)
│   └── src/
├── typescript/          # Node.js backend
│   └── src/
│       ├── index.ts     # Main server & pipeline
│       ├── cartesia/    # TTS system prompt (STT and TTS use the Cartesia SDK in index.ts)
│       └── elevenlabs/  # Alternate TTS client
└── python/              # Python backend
    └── src/
        ├── main.py             # Main server & pipeline
        ├── cartesia_prompts.py # TTS system prompt
        ├── elevenlabs_tts.py   # Alternate TTS client
        └── events.py           # Event type definitions
```

## Event Types

The pipeline communicates via a unified event stream:

| Event | Made by | Used by | Description |
|-------|---------|---------|-------------|
| `turn.*` | STT stage | Agent stage (`turn.end`), browser | Cartesia Ink 2 turn events: `turn.start`, `turn.update` (transcript so far), `turn.eager_end`, `turn.resume`, `turn.end` (final transcript, triggers the agent) |
| `agent_chunk` | Agent stage | TTS stage, browser | Text chunk from agent response |
| `tool_call` | Agent stage | browser | Tool invocation |
| `tool_result` | Agent stage | browser | Tool execution result |
| `agent_end` | Agent stage | TTS stage, browser | End of the agent's reply |
| `tts_chunk` | TTS stage | browser | Audio chunk for playback |
