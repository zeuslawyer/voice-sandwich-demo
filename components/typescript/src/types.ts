/**
 * Events the voice pipeline sends to the browser, as JSON.
 *
 * Each event is a small record that says "this happened". Field names are
 * the JSON keys, and match the Python backend (`components/python/src/events.py`).
 *
 * Every event has `type` (what happened) and `ts` (when the server made it,
 * in milliseconds since 1970).
 */

/**
 * One turn event from Cartesia Ink 2 speech-to-text, passed on unchanged.
 *
 * A "turn" is one stretch of the user talking. Ink 2 detects it and sends:
 * turn.start -> turn.update (repeats) -> [turn.eager_end -> turn.resume] -> turn.end
 *
 * transcript: the turn's text so far. It only grows and is never revised;
 * turn.end has the final text. Empty on turn.start and turn.resume.
 */
export interface TurnEvent {
  type:
    | "turn.start"
    | "turn.update"
    | "turn.eager_end"
    | "turn.resume"
    | "turn.end";
  transcript: string;
  ts: number;
}

/** A few words of the agent's reply, sent as the LLM writes them. */
export interface AgentChunk {
  type: "agent_chunk";
  text: string;
  ts: number;
}

/** The agent's reply for this turn is complete. No more text follows. */
export interface AgentEnd {
  type: "agent_end";
  ts: number;
}

/**
 * The agent called a tool.
 *
 * id: the call's id. name: the tool's name. args: what the agent passed in.
 */
export interface ToolCall {
  type: "tool_call";
  id: string;
  name: string;
  args: Record<string, unknown>;
  ts: number;
}

/**
 * A tool finished.
 *
 * toolCallId: the `id` of the matching ToolCall. name: the tool's name.
 * result: what the tool returned, as text.
 */
export interface ToolResult {
  type: "tool_result";
  toolCallId: string;
  name: string;
  result: string;
  ts: number;
}

/**
 * A piece of the agent's voice from Cartesia Sonic, ready to play.
 *
 * audio: raw 16-bit PCM, mono, 24 kHz, as base64 text (JSON can't hold bytes).
 */
export interface TTSChunk {
  type: "tts_chunk";
  audio: string;
  ts: number;
}

export type AgentEvent = AgentChunk | AgentEnd | ToolCall | ToolResult;

/** Any event. Check `event.type` to know which one it is. */
export type VoiceAgentEvent = TurnEvent | AgentEvent | TTSChunk;
