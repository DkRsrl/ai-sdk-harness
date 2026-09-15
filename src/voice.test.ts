import assert from "node:assert/strict";
import { test } from "node:test";
import { tool } from "ai";
import { MockLanguageModelV4, convertArrayToReadableStream } from "ai/test";
import z from "zod";
import type {
  RealtimeEvent,
  RealtimeModelV1,
  RealtimeOutbound,
} from "./voice";
import { codeModeTool, createRegistry, harnessTool, init, InMemorySessionStorage, role } from "./index";
import type { SessionMessage } from "./index";

// A fake realtime provider: it records the `call` (instructions/seed/tools) and
// hands back the `emit`/`send` channels so a test can drive a turn by hand —
// the realtime analogue of MockLanguageModelV4.
function fakeRealtimeModel() {
  const state: {
    call?: import("./voice").RealtimeCall;
    emit?: (e: RealtimeEvent) => void;
    sends: RealtimeOutbound[];
  } = { sends: [] };
  const model: RealtimeModelV1 = {
    specificationVersion: "realtime-v1",
    provider: "fake",
    modelId: "fake-voice-1",
    async connect(args) {
      state.call = args.call;
      state.emit = args.emit;
      args.emit({ type: "transport", status: "connected" });
      return {
        send: (item) => state.sends.push(item),
        pushAudio: () => {},
        requestResponse: () => {},
        close: async () => {},
      };
    },
  };
  return { model, state };
}

function textStream(text: string) {
  return convertArrayToReadableStream([
    { type: "stream-start" as const, warnings: [] },
    { type: "text-start" as const, id: "1" },
    { type: "text-delta" as const, id: "1", delta: text },
    { type: "text-end" as const, id: "1" },
    {
      type: "finish" as const,
      finishReason: { unified: "stop" as const, raw: undefined },
      usage: {
        inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 1, text: 1, reasoning: 0 },
      },
    },
  ]);
}

function mockTextModel(reply: string) {
  return new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream(reply) }),
  });
}

async function until(
  cond: () => boolean | Promise<boolean>,
  ms = 1000,
): Promise<void> {
  const start = Date.now();
  while (!(await cond())) {
    if (Date.now() - start > ms) throw new Error("timed out waiting");
    await new Promise((r) => setTimeout(r, 5));
  }
}

const readFile = tool({
  description: "Read a file.",
  inputSchema: z.object({ path: z.string() }),
  execute: async () => "ok",
});

// Registered in the harness but NOT in the role's baseline — must stay gated out.
const createArtifact = tool({
  description: "Create an artifact.",
  inputSchema: z.object({ title: z.string() }),
  execute: async () => "made",
});

const helper = role({
  name: "helper",
  systemPrompt: "You are a helper.",
  tools: ["readFile"],
});

test("the model spec picks the drive: realtime → .voice, text → .prompt", async () => {
  const { model } = fakeRealtimeModel();
  const voiceHarness = await init({
    registry: { readFile },
    model,
    role: helper(),
  });
  const voiceSession = await voiceHarness.session({ sessionId: "v1" });
  assert.equal(typeof voiceSession.voice, "function");
  assert.equal("prompt" in voiceSession, false);

  const textHarness = await init({
    registry: { readFile },
    model: mockTextModel("hi"),
    role: helper(),
  });
  const textSession = await textHarness.session({ sessionId: "t1" });
  assert.equal(typeof textSession.prompt, "function");
  assert.equal("voice" in textSession, false);
});

test("voice() seeds prior history, advertises only active tools, and persists finalized turns to the same storage", async () => {
  const { model, state } = fakeRealtimeModel();
  const storage = new InMemorySessionStorage();
  const seed: SessionMessage[] = [
    {
      id: "u1",
      role: "user",
      parts: [{ type: "text", text: "Hi, I'm Marco.", state: "done" }],
    },
    {
      id: "a1",
      role: "assistant",
      parts: [{ type: "text", text: "Hello, Marco!", state: "done" }],
    },
  ];
  await storage.saveMessages("chatX", seed);

  const harness = await init({
    registry: { readFile, createArtifact },
    model,
    role: helper(),
    storage,
  });
  const session = await harness.session({ sessionId: "chatX" });
  const realtime = await session.voice();
  await realtime.start();

  // The whole registry is handed over, but only the role's active set is
  // advertised — `createArtifact` is gated out. The prior chat is the seed.
  assert.deepEqual(
    state.call?.tools.map((t) => t.name).sort(),
    ["readFile"],
  );
  assert.ok(
    state.call?.seed.some(
      (s) => s.type === "text" && s.text.includes("Marco"),
    ),
  );

  // A finalized assistant turn lands in the same storage, in UIMessage shape.
  state.emit?.({ type: "transcript.delta", role: "assistant", text: "Sure thing." });
  state.emit?.({ type: "transcript.done", role: "assistant" });

  await until(async () => (await storage.loadMessages("chatX")).length > 2);
  const stored = await storage.loadMessages("chatX");
  const last = stored[stored.length - 1]!;
  assert.equal(last.role, "assistant");
  assert.ok(
    last.parts.some((p) => p.type === "text" && p.text === "Sure thing."),
  );
  // The realtime ordering stamp is lifted to the top-level `createdAt`.
  assert.ok(last.createdAt instanceof Date);
});

test("stop() resolves only once the turn it settled is in storage", async () => {
  const { model, state } = fakeRealtimeModel();
  const storage = new InMemorySessionStorage();
  const harness = await init({
    registry: { readFile },
    model,
    role: helper(),
    storage,
  });
  const session = await harness.session({ sessionId: "chatY" });
  const realtime = await session.voice();
  await realtime.start();

  state.emit?.({ type: "transcript.final", role: "user", text: "What time is it?" });
  state.emit?.({ type: "transcript.delta", role: "assistant", text: "Three o'clock." });
  await realtime.stop();

  // No polling: whoever loads the transcript next does so on this edge — a
  // successor session on the same conversation starts exactly here.
  const stored = await storage.loadMessages("chatY");
  assert.deepEqual(
    stored.map((m) => m.role),
    ["user", "assistant"],
  );
});

test("a voice tool reaches the voice session via currentSession() and delegates to a text subsession", async () => {
  let seenId: string | undefined;
  let workerAnswer: string | undefined;

  const consult = harnessTool({
    description: "Look something up with the powerful text model.",
    inputSchema: z.object({ q: z.string() }),
    execute: async ({ q }, { session }) => {
      seenId = session.id;
      const worker = await session.subsession({
        model: mockTextModel("from the worker"),
        role: helper(),
      });
      const res = await worker.prompt(q);
      let text = "";
      for await (const part of res.textStream) text += part;
      workerAnswer = text;
      return { answer: text };
    },
  });

  const voiceHost = role({
    name: "voice-host",
    systemPrompt: "You are a voice host.",
    tools: ["consult"],
  });

  const { model, state } = fakeRealtimeModel();
  const session = await (
    await init({ registry: { readFile, consult }, model, role: voiceHost() })
  ).session({ sessionId: "v2" });

  const realtime = await session.voice();
  await realtime.start();

  state.emit?.({
    type: "tool.call",
    callId: "c1",
    name: "consult",
    input: { q: "what's up" },
  });

  await until(() => state.sends.some((s) => s.type === "tool.result"));

  // currentSession() inside the realtime-driven tool is the voice session...
  assert.equal(seenId, session.id);
  // ...and the subsession ran the text worker and returned to the provider.
  assert.equal(workerAnswer, "from the worker");
  const result = state.sends.find((s) => s.type === "tool.result");
  assert.deepEqual(result?.output, { answer: "from the worker" });
});

test("voice() binds the sandbox at connect: routed tools hidden, their API advertised inside codeMode", async () => {
  const { model, state } = fakeRealtimeModel();
  const bound = createRegistry({
    lookup: tool({
      description: "Look up an answer.",
      inputSchema: z.object({ q: z.string() }),
      outputSchema: z.object({ answer: z.number() }),
      execute: async () => ({ answer: 42 }),
    }),
    chat: tool({
      description: "Small talk.",
      inputSchema: z.object({ text: z.string() }),
      execute: async () => "ok",
    }),
    code: codeModeTool(),
  });
  const analyst = bound.role({
    name: "analyst",
    systemPrompt: "You analyse.",
    tools: ["lookup", "chat", "code"],
    toolCallers: { lookup: ["code"] },
  });

  const harness = await init({
    registry: bound.registry(),
    model,
    role: analyst(),
  });
  const session = await harness.session({ sessionId: "vc1" });
  const realtime = await session.voice();
  await realtime.start();

  // The routed tool disappears from the advertised set; the sandbox appears,
  // already bound — the routed tool's typed API rides in the instructions,
  // the sandbox description stays invariant.
  assert.deepEqual(
    state.call?.tools.map((t) => t.name).sort(),
    ["chat", "code"],
  );
  const sandbox = state.call?.tools.find((t) => t.name === "code");
  // The catalog reaches the model at connect, wherever it is carried.
  const sent = `${state.call?.instructions ?? ""}\n${sandbox?.description ?? ""}`;
  assert.match(sent, /tools\.lookup/);
  assert.match(sent, /Look up an answer/);
});

test("voice() grounds time per message: seeded turns carry <message_metadata>, live turns get injected stamps", async () => {
  const { model, state } = fakeRealtimeModel();
  const storage = new InMemorySessionStorage();
  await storage.saveMessages("chatT", [
    {
      id: "u1",
      role: "user",
      createdAt: new Date("2026-08-20T08:00:00Z"),
      metadata: { timezone: "Europe/Rome" },
      parts: [{ type: "text", text: "quanto ad agosto?", state: "done" }],
    },
  ]);
  const harness = await init({
    registry: { readFile },
    model,
    role: helper(),
    storage,
  });
  const session = await harness.session({ sessionId: "chatT" });
  const realtime = await session.voice({ timezone: "Europe/Rome" });
  await realtime.start();

  // The seeded turn replays with the same decoration the text drive applies,
  // merged into the one utterance it was.
  const seeded = state.call?.seed.find((s) => s.type === "text");
  if (!seeded || seeded.type !== "text") throw new Error("expected a text seed");
  assert.ok(seeded.text.startsWith("<message_metadata>"));
  assert.ok(seeded.text.includes("time: 2026-08-20T10:00:00+02:00"));
  assert.ok(seeded.text.endsWith("quanto ad agosto?"));

  // Live turns are transcribed provider-side, so their stamp travels as an
  // injected item: one lands at connect, rendered in the session's zone.
  const stamp = state.sends.find(
    (s) => s.type === "text" && s.text.startsWith("<message_metadata>"),
  );
  if (!stamp || stamp.type !== "text") throw new Error("expected a clock stamp");
  assert.ok(stamp.text.includes("timezone: Europe/Rome"));
});
