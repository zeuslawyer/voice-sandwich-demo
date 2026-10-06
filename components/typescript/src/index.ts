import "dotenv/config";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";
import { createAgent, AIMessage, ToolMessage } from "langchain";
import path from "node:path";
import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { cors } from "hono/cors";
import { createNodeWebSocket } from "@hono/node-ws";
import type { WSContext } from "hono/ws";
import type WebSocket from "ws";
import { iife, writableIterator } from "./utils";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { v4 as uuidv4 } from "uuid";
import { CARTESIA_TTS_SYSTEM_PROMPT } from "./cartesia";
import Cartesia from "@cartesia/cartesia-js";
import type { VoiceAgentEvent } from "./types";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const STATIC_DIR = path.join(__dirname, "../../web/dist");
const PORT = parseInt(process.env.PORT ?? "8000");

if (!existsSync(STATIC_DIR)) {
  console.error(
    `Web build not found at ${STATIC_DIR}.\n` +
      "Run 'make build-web' or 'make dev-ts' from the project root."
  );
  process.exit(1);
}

const app = new Hono();
const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });

app.use("/*", cors());

const addToOrder = tool(
  async ({ item, quantity }) => {
    return `Added ${quantity} x ${item} to the order.`;
  },
  {
    name: "add_to_order",
    description: "Add an item to the customer's sandwich order.",
    schema: z.object({
      item: z.string(),
      quantity: z.number(),
    }),
  }
);

const confirmOrder = tool(
  async ({ orderSummary }) => {
    return `Order confirmed: ${orderSummary}. Sending to kitchen.`;
  },
  {
    name: "confirm_order",
    description: "Confirm the final order with the customer.",
    schema: z.object({
      orderSummary: z.string().describe("Summary of the order"),
    }),
  }
);

const systemPrompt = `
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.

${CARTESIA_TTS_SYSTEM_PROMPT}
`;

const cartesia = new Cartesia({ apiKey: process.env.CARTESIA_API_KEY });
const VOICE_ID = "f6ff7c0c-e396-40a9-a70b-f7607edb6937";

const agent = createAgent({
  model: "claude-haiku-4-5",
  tools: [addToOrder, confirmOrder],
  checkpointer: new MemorySaver(),
  systemPrompt: systemPrompt,
});

/**
 * Transform stream: Audio (Uint8Array) → Turn events, via Cartesia Ink 2.
 *
 * @param audioStream - PCM audio from the browser (16-bit, mono, 16kHz)
 * @returns Async generator yielding a TurnEvent for each Ink 2 turn.* message
 */
async function* sttStream(
  audioStream: AsyncIterable<Uint8Array>
): AsyncGenerator<VoiceAgentEvent> {
  const ws = cartesia.stt.autoFinalize.websocket({
    model: "ink-2",
    encoding: "pcm_s16le",
    sample_rate: 16000,
  });

  // Runs alongside the receive loop below, so transcripts arrive while the
  // user is still talking.
  const producer = iife(async () => {
    try {
      for await (const audioChunk of audioStream) {
        ws.sendRaw(audioChunk);
      }
    } finally {
      // Ask Ink 2 to finish any buffered audio. It closes the socket when
      // done, which ends the receive loop.
      ws.send({ type: "close" });
    }
  });

  try {
    for await (const e of ws.stream()) {
      if (e.type === "error") {
        console.error("Ink 2 error:", e.error.message);
      } else if (e.type === "message") {
        const m = e.message;
        if (m.type === "connected" || m.type === "error") continue;
        // turn.start and turn.resume carry no text.
        const transcript = "transcript" in m ? m.transcript : "";
        yield { type: m.type, transcript, ts: Date.now() };
      }
    }
  } finally {
    await producer;
  }
}

/**
 * Transform stream: Voice Events → Voice Events (with Agent Responses)
 *
 * Passes every upstream event through. On each turn.end (the user finished
 * speaking) it sends the transcript to the LangChain agent and yields the
 * reply as agent_chunk, tool_call and tool_result events, then agent_end.
 */
async function* agentStream(
  eventStream: AsyncIterable<VoiceAgentEvent>
): AsyncGenerator<VoiceAgentEvent> {
  // Generate a unique thread ID for this conversation session
  // This allows the agent to maintain conversation context across multiple turns
  // using the checkpointer (MemorySaver) configured in the agent
  const threadId = uuidv4();

  // TODO:@zeuslawyer barge-in handling to be added. While the agent replies
  // below, this loop can't read the next event, so a turn.start that arrives
  // mid-reply waits until the reply ends.
  for await (const event of eventStream) {
    yield event;
    if (event.type === "turn.end" && event.transcript) {
      const stream = await agent.stream(
        { messages: [new HumanMessage(event.transcript)] },
        {
          configurable: { thread_id: threadId },
          streamMode: "messages",
        }
      );

      // Text after a tool call starts with no space ("for you." + "Your"),
      // and the browser and TTS join chunks as-is. So we add one, but only
      // if the reply already has text.
      let hasText = false;
      let afterTool = false;
      for await (const [message] of stream) {
        if (AIMessage.isInstance(message) && message.tool_calls) {
          let text = message.text;
          if (afterTool && text) {
            text = " " + text.trimStart();
            afterTool = false;
          }
          hasText ||= text.length > 0;
          yield { type: "agent_chunk", text, ts: Date.now() };
          for (const toolCall of message.tool_calls) {
            yield {
              type: "tool_call",
              id: toolCall.id ?? uuidv4(),
              name: toolCall.name,
              args: toolCall.args,
              ts: Date.now(),
            };
          }
        }
        if (ToolMessage.isInstance(message)) {
          afterTool = hasText;
          yield {
            type: "tool_result",
            toolCallId: message.tool_call_id ?? "",
            name: message.name ?? "unknown",
            result:
              typeof message.content === "string"
                ? message.content
                : JSON.stringify(message.content),
            ts: Date.now(),
          };
        }
      }

      // Signal that the agent has finished responding for this turn
      yield { type: "agent_end", ts: Date.now() };
    }
  }
}

/**
 * Transform stream: Voice Events → Voice Events (with Audio), via Cartesia Sonic.
 *
 * Passes every upstream event through. For each agent reply, sends the text
 * to Sonic and yields the speech as tts_chunk events (base64 PCM, 24 kHz).
 * A Sonic "context" is one generation; we use one per agent reply.
 */
async function* ttsStream(
  eventStream: AsyncIterable<VoiceAgentEvent>
): AsyncGenerator<VoiceAgentEvent> {
  const ws = await cartesia.tts.websocket();
  const passthrough = writableIterator<VoiceAgentEvent>();
  // The producer and consumer below run at the same time. This hands each
  // reply's context from the producer to the consumer, in order.
  const contexts = writableIterator<ReturnType<typeof ws.context>>();

  // Passes events through and sends each reply's text to Sonic.
  const producer = iife(async () => {
    try {
      let buffer: string[] = [];
      for await (const event of eventStream) {
        passthrough.push(event);
        if (event.type === "agent_chunk") {
          buffer.push(event.text);
        }
        if (event.type === "agent_end") {
          const text = buffer.join("").trim();
          buffer = [];
          if (!text) continue;
          const ctx = ws.context({
            model_id: "sonic-3.6",
            voice: VOICE_ID,
            output_format: {
              container: "raw",
              encoding: "pcm_s16le",
              sample_rate: 24000,
            },
            language: "en",
          });
          contexts.push(ctx);
          await ctx.push({ transcript: text });
          await ctx.no_more_inputs();
        }
      }
    } finally {
      // Ends the consumer's loop; otherwise it waits forever.
      contexts.cancel();
    }
  });

  // Reads each reply's audio, in order.
  const consumer = iife(async () => {
    for await (const ctx of contexts) {
      try {
        for await (const e of ctx.receive()) {
          if (e.type === "chunk") {
            // e.data is base64, which is what the browser expects.
            passthrough.push({ type: "tts_chunk", audio: e.data, ts: Date.now() });
          }
        }
      } catch (err) {
        console.error("Sonic error:", err);
      }
    }
  });

  void Promise.allSettled([producer, consumer]).then(() => passthrough.cancel());

  try {
    yield* passthrough;
  } finally {
    ws.close();
  }
}

app.get("/*", serveStatic({ root: STATIC_DIR }));

app.get(
  "/ws",
  upgradeWebSocket(async () => {
    let currentSocket: WSContext<WebSocket> | undefined;

    // Create a writable stream for incoming WebSocket audio data
    const inputStream = writableIterator<Uint8Array>();

    // Define the voice processing pipeline as a chain of async generators
    // Audio -> STT events
    const transcriptEventStream = sttStream(inputStream);
    // STT events -> STT Events + Agent events
    const agentEventStream = agentStream(transcriptEventStream);
    // STT events + Agent events -> STT Events + Agent Events + TTS events
    const outputEventStream = ttsStream(agentEventStream);

    const flushPromise = iife(async () => {
      // Process all events from the pipeline, sending events back to the client
      for await (const event of outputEventStream) {
        currentSocket?.send(JSON.stringify(event));
      }
    });

    return {
      onOpen(_, ws) {
        currentSocket = ws;
      },
      onMessage(event) {
        // Push incoming audio data into the pipeline's input stream
        const data = event.data;
        if (Buffer.isBuffer(data)) {
          inputStream.push(new Uint8Array(data));
        } else if (data instanceof ArrayBuffer) {
          inputStream.push(new Uint8Array(data));
        }
      },
      async onClose() {
        // Signal end of stream when socket closes
        inputStream.cancel();
        await flushPromise;
      },
    };
  })
);

const server = serve({
  fetch: app.fetch,
  port: PORT,
});

injectWebSocket(server);

console.log(`Server is running on port ${PORT}`);
