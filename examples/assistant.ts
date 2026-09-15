// `createAssistant` — one reusable assistant that answers on text *and* voice
// over the same durable conversation. This is the higher-order usage: it calls
// `init()` underneath, once per channel, and everything here could be written
// against `init()` directly.
//
// What it buys you is the shape of a per-request agent. `prepare()` resolves a
// conversation's storage and application context once; `text()` and `voice()`
// then say how each drive is configured from that context. Without it, every
// visitor message re-resolves all of that inline.
//
//   AI_GATEWAY_API_KEY=... XAI_API_KEY=... bun run examples/assistant.ts
//
// Runs a text turn, then opens a voice call on the *same* session id, and
// shows the voice side seeing the text turn in its seeded history.

import {
  createAssistant,
  createRegistry,
  harnessTool,
  InMemorySessionStorage,
  type Session,
} from "../src/index";
import { grok } from "../src/voice/providers/grok/index";
import { z } from "zod";

// The caller-supplied identity/authorization input. It reaches `prepare` and
// both channel factories; the harness never interprets it.
type Scope = { userId: string; locale: string };

// What the application resolves once per session. One object serves both
// audiences: the tools take the keys their own contextSchemas declare
// (`userId` here), and the drives read the rest when building roles.
type Context = { userId: string; displayName: string; openOrders: number };

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
      return { id: input.id, status: "in transit", eta: "Thursday" };
    },
  }),
});

// Two roles over one registry: the drives differ in how they should *speak*,
// not in what they can do. Both are built from the same resolved context.
const chat = role({
  name: "chat",
  argsSchema: z.object({ displayName: z.string(), openOrders: z.number() }),
  systemPrompt:
    "You are the support assistant for {{displayName}}, who has " +
    "{{openOrders}} open orders. Look orders up before answering.",
  tools: ["lookupOrder"],
});

const spoken = role({
  name: "spoken",
  argsSchema: z.object({ displayName: z.string() }),
  systemPrompt:
    "You are on a phone call with {{displayName}}. One short sentence per " +
    "turn, never read out ids digit by digit.",
  tools: ["lookupOrder"],
});

// One durable store for this example so the text turn and the voice call share
// a transcript. A real application returns a per-conversation store from
// `prepare` — that is the point of resolving it there.
const storage = new InMemorySessionStorage();

// The `<Scope>()` first call is a partial-inference workaround: it pins the
// scope type so `TContext`, `TTools` and `TOutput` can still be inferred from
// the config object in the second call.
const assistant = createAssistant<Scope>()({
  // Once per session, before either drive is configured. Resolve the context
  // once and hand it to the registry — `registry(context)` takes exactly the
  // same bound form `init({ registry })` does, and each tool still receives
  // only the keys its own contextSchema declares, so `displayName` never
  // reaches `lookupOrder`. The registry is built here rather than on the
  // assistant because the context it binds is per-conversation.
  async prepare({ sessionId, scope, signal }) {
    console.log(`[prepare] session=${sessionId} user=${scope.userId}`);
    signal?.throwIfAborted();
    const context: Context = {
      userId: scope.userId,
      displayName: "Ada",
      openOrders: 2,
    };
    return { registry: registry(context), storage, context };
  },

  // How the text drive is configured from that context.
  text({ context }) {
    return {
      model: "anthropic/claude-sonnet-5",
      role: chat({
        displayName: context.displayName,
        openOrders: context.openOrders,
      }),
    };
  },

  // And the voice drive. Same registry, same storage, different model and a
  // role written for speech.
  voice({ context }) {
    return {
      model: grok("grok-voice-latest"),
      role: spoken({ displayName: context.displayName }),
    };
  },
});

// One durable conversation, addressed by id. `prepare` runs here, once.
const session = await assistant.session({
  sessionId: "conversation_1",
  scope: { userId: "user_1", locale: "en" },
  title: "Order support",
});

console.log("\n--- text ---");
const turn = await session.prompt("Where is order 42?");
for await (const part of turn.textStream) process.stdout.write(part);
process.stdout.write("\n");

// One response-producing operation at a time. A text turn releases the session
// when `committed` settles; a voice call releases it when `stop()` settles.
// Each activation reloads durable history, so the drive that starts next
// carries the whole transcript forward.
await turn.committed;

console.log("\n--- voice, same session ---");
const call = await session.voice({
  onStatus: (status) => console.log("[status]", status),
  onMessage: (message) =>
    message.type === "text"
      ? console.log(`[${message.role}]`, message.text)
      : console.log(`[tool] ${message.name} ->`, message.output),
});

// The text turn above is already in here — the history the voice drive opened
// with. Note the ordering: unlike the low-level `session.voice()`, which hands
// back an unstarted session, the assistant calls `start()` for you, so the
// callbacks above have already been firing by the time you hold `call`. Don't
// gate a callback on something you read off `call` — it is decided too late.
console.log("seeded from the text turn:", call.initialMessages.length, "messages");

call.sendText("And what about order 43?");
await new Promise((resolve) => setTimeout(resolve, 8_000));

// `stop()` settles finalized turns and waits for their persistence.
await call.stop();

console.log("\ntranscript:", (await storage.loadMessages(session.id)).length, "messages");
