export namespace VoiceAgentEvent {
  /**
   * Base interface for all voice agent events.
   *
   * All events in the pipeline share these common properties to enable
   * consistent handling, logging, and debugging across the system.
   */
  interface BaseEvent {
    /**
     * Discriminator field identifying the specific event type.
     * Used for type narrowing in TypeScript and event routing.
     */
    type: string;

    /**
     * Unix timestamp (milliseconds since epoch) when the event was created.
     * Useful for measuring latency between pipeline stages and debugging timing issues.
     */
    ts: number;
  }

  /**
   * Event emitted when raw audio data is received from the user.
   *
   * This is the entry point of the voice agent pipeline. Audio should be
   * in PCM format (16-bit, mono, 16kHz) for optimal processing by the STT stage.
   */
  export interface UserInput extends BaseEvent {
    readonly type: "user_input";

    /**
     * Raw PCM audio bytes from the user's microphone.
     * Expected format: 16-bit signed integer, mono channel, 16kHz sample rate.
     */
    audio: Uint8Array;
  }

  /**
   * One turn event from Cartesia Ink 2 speech-to-text, passed to the browser as-is.
   *
   * A "turn" is one stretch of the user talking. Ink 2 detects it and emits:
   * turn.start -> turn.update (repeats) -> [turn.eager_end -> turn.resume] -> turn.end
   */
  export interface TurnEvent extends BaseEvent {
    readonly type:
      | "turn.start"
      | "turn.update"
      | "turn.eager_end"
      | "turn.resume"
      | "turn.end";

    /**
     * The turn's text so far. It only grows and is never revised, and
     * turn.end carries the final text. Empty on turn.start and turn.resume.
     */
    transcript: string;
  }

  /**
   * Event emitted during agent response generation for streaming text chunks.
   *
   * As the LLM generates its response, it streams tokens incrementally.
   * These chunks enable real-time display of the agent's response and allow
   * the TTS stage to begin synthesis before the complete response is generated,
   * reducing overall latency.
   */
  export interface AgentChunk extends BaseEvent {
    readonly type: "agent_chunk";

    /**
     * Partial text chunk from the agent's streaming response.
     * Multiple chunks combine to form the complete agent output.
     */
    text: string;
  }

  /**
   * Event emitted when the agent has finished generating its response for a turn.
   *
   * This signals downstream consumers (like TTS) that no more text is coming
   * for this turn and they should flush any buffered content.
   */
  export interface AgentEnd extends BaseEvent {
    readonly type: "agent_end";
  }

  /**
   * Event emitted when the agent invokes a tool.
   *
   * This event provides visibility into the agent's decision-making process,
   * showing which tools are being called and with what arguments.
   */
  export interface ToolCall extends BaseEvent {
    readonly type: "tool_call";

    /**
     * Unique identifier for this tool invocation.
     */
    id: string;

    /**
     * Name of the tool being invoked.
     */
    name: string;

    /**
     * Arguments passed to the tool, serialized as JSON.
     */
    args: Record<string, unknown>;
  }

  /**
   * Event emitted when a tool completes execution and returns a result.
   *
   * This event contains the output from the tool, allowing tracking of
   * the full tool execution lifecycle.
   */
  export interface ToolResult extends BaseEvent {
    readonly type: "tool_result";

    /**
     * The tool call ID this result corresponds to.
     */
    toolCallId: string;

    /**
     * Name of the tool that was executed.
     */
    name: string;

    /**
     * The result returned by the tool.
     */
    result: string;
  }

  // interface AgentInterrupt extends BaseEvent {}

  /**
   * Union type of all agent-related events.
   *
   * This type encompasses all events emitted during agent processing, including
   * streaming text chunks, tool invocations, and completion signals. It enables
   * type-safe handling of the various stages of agent response generation.
   */
  export type AgentEvent = AgentChunk | AgentEnd | ToolCall | ToolResult;

  /**
   * Event emitted during text-to-speech synthesis for streaming audio chunks.
   *
   * As the TTS service synthesizes speech, it streams audio incrementally.
   * These chunks enable real-time playback of the agent's response, allowing
   * audio to begin playing before the complete synthesis is finished, which
   * significantly improves perceived responsiveness.
   */
  export interface TTSChunk extends BaseEvent {
    readonly type: "tts_chunk";

    /**
     * PCM audio bytes encoded as base64 string synthesized from the agent's
     * text response. Format: 16-bit signed integer, mono channel, 16kHz sample
     * rate. Can be played immediately as it arrives for low-latency audio
     * output.
     */
    audio: string;
  }
}

/**
 * Union type of all possible voice agent events.
 *
 * This type enables type-safe event handling throughout the voice agent pipeline.
 * TypeScript can use the discriminator `type` field to narrow the union to the
 * specific event interface, providing full type safety and autocomplete for
 * event-specific properties.
 *
 * @example
 * ```typescript
 * function handleEvent(event: VoiceAgentEvent) {
 *   switch (event.type) {
 *     case "user_input":
 *       processAudio(event.audio);
 *       break;
 *     case "turn.end":
 *       sendToAgent(event.transcript);
 *       break;
 *     // ... handle other event types
 *   }
 * }
 * ```
 */
export type VoiceAgentEvent =
  | VoiceAgentEvent.UserInput
  | VoiceAgentEvent.TurnEvent
  | VoiceAgentEvent.AgentEvent
  | VoiceAgentEvent.TTSChunk;
