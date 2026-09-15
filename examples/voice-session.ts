// The same registry and role, driven by a realtime model instead of a
// language model. Nothing about the tools, the role or the transcript changes
// — only the `model` passed to `init()`, and the session surface that follows
// from it (`.voice()` instead of `.prompt()`).
//
//   XAI_API_KEY=... bun run examples/voice-session.ts
//
// This example feeds silence and stops, so it exercises connect/seed/teardown
// without a microphone. Feed real 24kHz mono PCM to `pushAudio` for a live
// turn, or use `sendText` for a typed turn with no audio at all.

import { createRegistry, harnessTool, init, type Session } from "../src/index";
import { grok } from "../src/voice/providers/grok/index";
import { z } from "zod";

const { registry, role } = createRegistry({
  lookupOrder: harnessTool({
    description: "Look up an order by its id.",
    inputSchema: z.object({ id: z.string() }),
    contextSchema: z.object({ userId: z.string() }),
    execute: async (
      input: { id: string },
      { context }: { context: { userId: string }; session: Session },
    ) => {
      console.log(`  [tool] lookupOrder ${input.id} for ${context.userId}`);
      return { id: input.id, status: "in transit" };
    },
  }),
});

const support = role({
  name: "support",
  systemPrompt: "You are a support agent on a phone call. Keep replies short.",
  tools: ["lookupOrder"],
});

const harness = await init({
  registry: registry({ userId: "user_1" }),
  // A `RealtimeModelV1` here is what makes `harness.session()` return a voice
  // session. The type follows the spec — `.prompt()` is not on this session.
  model: grok("grok-voice-latest"),
  role: support(),
});

const session = await harness.session();

// `voice()` seeds the prior conversation, exposes the same role-gated tools
// and persists each finalized turn. It returns the live session without
// connecting.
const call = await session.voice({
  onStatus: (status) => console.log("[status]", status),
  // A realtime message is either a finalized text turn or a completed tool
  // round-trip — discriminate on `type`, not on a parts array.
  onMessage: (message) =>
    message.type === "text"
      ? console.log(`[${message.role}]`, message.text)
      : console.log(`[tool] ${message.name} ->`, message.output),
  onError: (error) => console.error("[error]", error),
});

await call.start();

// A typed user turn — the text analogue of a spoken utterance. Finalized and
// persisted like any other message, with no audio involved.
call.sendText("Where is order 42?");

await new Promise((resolve) => setTimeout(resolve, 8_000));
await call.stop();

console.log("messages persisted:", session.messages.length);

// To bridge this to a browser instead, hand the session a duplex wire:
//
//   const handle = call.serve((frame) => ws.send(frame));
//   ws.on("message", (frame) => handle.receive(frame));
//
// That is the voice analogue of `result.toUIMessageStream()`. The browser side
// lives in `ai-sdk-harness/voice/ui` and `ai-sdk-harness/voice/react`.
