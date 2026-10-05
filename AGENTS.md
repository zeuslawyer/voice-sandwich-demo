# AGENTS.md

Voice-to-voice demo: browser mic → STT → LangChain agent → TTS → browser speaker.
Two interchangeable backends share one Svelte frontend.

## Layout

- `components/web/` — Svelte frontend (pnpm). Captures mic as 16 kHz PCM s16le.
- `components/typescript/` — Node backend (pnpm). Pipeline in `src/index.ts`.
- `components/python/` — Python backend (uv). Pipeline in `src/main.py`.

Both backends implement the same three stages (`sttStream`/`_stt_stream`,
`agentStream`/`_agent_stream`, `ttsStream`/`_tts_stream`) and emit the same
event types (see `README.md` → Event Types). Keep the two backends in parity.

## Commands

- Install: `make bootstrap`
- Run: `make dev-ts` or `make dev-py` (serves http://localhost:8000)
- Typecheck: `make check`
- Python lint: `cd components/python && uv run ruff check`

## Local context

If a `CLAUDE.local.md` exists at the repo root, read it before starting work.
It is gitignored and holds maintainer-local context. Never commit it, and never
copy its contents into tracked files.
