import { test } from "node:test";
import assert from "node:assert/strict";
import {
  gateway,
  gatewayDefaultVoice,
  gatewayOutboundItems,
  gatewaySessionUpdate,
  type GatewayWebSocketFactory,
} from "./index";
import { createRealtimeSession, type RealtimeMessage } from "../../session";
import type { RealtimeCall, RealtimeEvent } from "../../spec";

function makeCall(over: Partial<RealtimeCall> = {}): RealtimeCall {
  return {
    instructions: "be brief",
    tools: [],
    seed: [],
    triggerResponse: false,
    ...over,
  };
}

type Listener = (ev?: unknown) => void;

// Simulates an instantly-connecting socket: fires `open` on a microtask once
// the provider subscribes, so callers never have to time a manual open against
// the core's async start().
class FakeWS {
  readyState = 1;
  sent: string[] = [];
  private listeners = new Map<string, Listener[]>();

  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
    this.fire("close");
  }
  addEventListener(type: string, l: Listener) {
    const arr = this.listeners.get(type) ?? [];
    arr.push(l);
    this.listeners.set(type, arr);
    if (type === "open") queueMicrotask(() => this.fire("open"));
  }
  removeEventListener(type: string, l: Listener) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((x) => x !== l));
  }

  fire(type: string, ev?: unknown) {
    for (const l of [...(this.listeners.get(type) ?? [])]) l(ev);
  }
  message(obj: unknown) {
    this.fire("message", { data: JSON.stringify(obj) });
  }
  sentTypes() {
    return this.sent.map((s) => JSON.parse(s).type as string);
  }
  sentOf(type: string) {
    return this.sent.map((s) => JSON.parse(s)).filter((m) => m.type === type);
  }
}

function fakeModel(fake: FakeWS, capture?: { url?: string; protocols?: string[] }) {
  const createWebSocket: GatewayWebSocketFactory = (url, protocols) => {
    if (capture) {
      capture.url = url;
      capture.protocols = protocols;
    }
    return fake as unknown as ReturnType<GatewayWebSocketFactory>;
  };
  return gateway("xai/grok-voice-think-fast-2.0", { apiKey: "k", createWebSocket });
}

test("gatewayDefaultVoice follows the creator prefix", () => {
  assert.equal(gatewayDefaultVoice("xai/grok-voice-think-fast-2.0"), "eve");
  assert.equal(gatewayDefaultVoice("openai/gpt-realtime-2"), "marin");
  assert.equal(gatewayDefaultVoice("acme/voice-1"), undefined);
});

test("gatewaySessionUpdate maps voice, tools, formats and providerOptions", () => {
  const update = gatewaySessionUpdate(
    makeCall({
      tools: [{ name: "search", description: "find", parametersJsonSchema: { type: "object" } }],
    }),
    {
      voice: "eve",
      sampleRate: 24000,
      providerOptions: { audio: { input: { transcription: { keyterms: ["Acme"] } } } },
    },
  );
  const config = update.config as Record<string, unknown>;
  assert.equal(config.voice, "eve");
  assert.equal(config.instructions, "be brief");
  assert.deepEqual(config.inputAudioFormat, { type: "audio/pcm", rate: 24000 });
  assert.deepEqual(config.tools, [
    { type: "function", name: "search", description: "find", parameters: { type: "object" } },
  ]);
  // The keyterms vehicle: provider-native fields ride through
  // `providerOptions` as-is — the gateway merges them un-namespaced into the
  // underlying provider's session. Undocumented beta semantics, pinned here.
  assert.deepEqual(config.providerOptions, {
    audio: { input: { transcription: { keyterms: ["Acme"] } } },
  });
  // Only `type` by default: server VAD on, every threshold left to the
  // underlying provider's defaults.
  assert.deepEqual(config.turnDetection, { type: "server-vad" });
});

test("gatewaySessionUpdate carries keyterms in the provider-native shape", () => {
  const update = gatewaySessionUpdate(makeCall(), {
    sampleRate: 24000,
    language: "it",
    keyterms: ["Acme", "Acme", "  "],
  });
  const config = update.config as Record<string, unknown>;
  // The language hint has a normalized home; key terms do not, so they ride
  // bare through providerOptions while both reach the provider.
  assert.deepEqual(config.inputAudioTranscription, { language: "it" });
  assert.deepEqual(config.providerOptions, {
    audio: { input: { transcription: { keyterms: ["Acme"] } } },
  });
});

test("gatewaySessionUpdate merges keyterms into caller-supplied providerOptions", () => {
  const update = gatewaySessionUpdate(makeCall(), {
    sampleRate: 24000,
    keyterms: ["Acme"],
    providerOptions: {
      reasoning_effort: "none",
      audio: { input: { transcription: { language_hint: "it" } } },
    },
  });
  const config = update.config as Record<string, unknown>;
  assert.deepEqual(config.providerOptions, {
    reasoning_effort: "none",
    audio: { input: { transcription: { language_hint: "it", keyterms: ["Acme"] } } },
  });
});

test("gatewaySessionUpdate forwards an explicit turnDetection override", () => {
  const update = gatewaySessionUpdate(makeCall(), {
    sampleRate: 24000,
    turnDetection: { type: "server-vad", threshold: 0.9 },
  });
  const config = update.config as Record<string, unknown>;
  assert.deepEqual(config.turnDetection, { type: "server-vad", threshold: 0.9 });
});

test("gatewayOutboundItems maps text and tool.result, drops tool.call", () => {
  assert.deepEqual(
    gatewayOutboundItems({ type: "tool.result", callId: "c1", output: { ok: true } }),
    [
      {
        type: "conversation-item-create",
        item: { type: "function-call-output", callId: "c1", output: JSON.stringify({ ok: true }) },
      },
    ],
  );
  // A failed tool result goes out as prose, not JSON — the wire has no error
  // channel, so the model must be able to read the failure as text.
  assert.deepEqual(
    gatewayOutboundItems({
      type: "tool.result",
      callId: "c1",
      output: { error: "boom" },
      isError: true,
    }),
    [
      {
        type: "conversation-item-create",
        item: { type: "function-call-output", callId: "c1", output: "Error: boom" },
      },
    ],
  );
  assert.deepEqual(
    gatewayOutboundItems({ type: "text", role: "assistant", text: "ciao" }),
    [
      {
        type: "conversation-item-create",
        item: { type: "text-message", role: "assistant", text: "ciao" },
      },
    ],
  );
  // No function-call item in the normalized union; the gateway drops one
  // silently, and the paired output carries the context (probed live).
  assert.deepEqual(
    gatewayOutboundItems({ type: "tool.call", callId: "c1", name: "t", input: {} }),
    [],
  );
});

test("connect dials the gateway endpoint with the auth subprotocol", async () => {
  const fake = new FakeWS();
  const capture: { url?: string; protocols?: string[] } = {};
  await fakeModel(fake, capture).connect({
    call: makeCall(),
    emit: () => {},
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  assert.equal(
    capture.url,
    "wss://ai-gateway.vercel.sh/v4/ai/realtime-model?ai-model-id=xai%2Fgrok-voice-think-fast-2.0",
  );
  assert.deepEqual(capture.protocols, ["ai-gateway-realtime.v1", "ai-gateway-auth.k"]);
});

test("connect sends session-update + seed + trigger on open", async () => {
  const fake = new FakeWS();
  await fakeModel(fake).connect({
    call: makeCall({
      seed: [{ type: "text", role: "user", text: "prior turn" }],
      triggerResponse: true,
    }),
    emit: () => {},
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  const types = fake.sentTypes();
  assert.ok(types.includes("session-update"));
  assert.ok(types.includes("conversation-item-create"));
  assert.equal(types.at(-1), "response-create");
});

test("seeds a tool round-trip as text turns plus the tool output", async () => {
  const fake = new FakeWS();
  await fakeModel(fake).connect({
    call: makeCall({
      seed: [
        { type: "text", role: "user", text: "what time is it" },
        { type: "tool.call", callId: "c1", name: "get_time", input: { tz: "Rome" } },
        { type: "tool.result", callId: "c1", output: { now: "15:00" } },
        { type: "text", role: "assistant", text: "it's 3pm" },
      ],
    }),
    emit: () => {},
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  const items = fake.sentOf("conversation-item-create").map((m) => m.item.type);
  assert.deepEqual(items, ["text-message", "function-call-output", "text-message"]);
});

test("inbound audio + transcript map to spec events", async () => {
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  const audio: ArrayBuffer[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: (a) => audio.push(a),
    signal: new AbortController().signal,
  });

  const pcm = Buffer.from([1, 2, 3, 4]).toString("base64");
  fake.message({ type: "audio-delta", responseId: "r1", itemId: "a1", delta: pcm, raw: {} });
  fake.message({ type: "audio-transcript-delta", responseId: "r1", itemId: "a1", delta: "Ci", raw: {} });
  fake.message({ type: "audio-transcript-delta", responseId: "r1", itemId: "a1", delta: "ao", raw: {} });
  fake.message({ type: "audio-transcript-done", responseId: "r1", itemId: "a1", transcript: "Ciao", raw: {} });

  assert.equal(audio.length, 1);
  assert.equal(audio[0]?.byteLength, 4);
  assert.ok(events.some((e) => e.type === "speech.start"));
  assert.equal(events.filter((e) => e.type === "transcript.delta").length, 2);
  // transcript settles on its own done signal, before response-done (audio).
  assert.ok(events.some((e) => e.type === "transcript.done" && e.role === "assistant"));
});

test("the terminal user transcript finalizes with the item as the utterance key", async () => {
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.message({
    type: "input-transcription-completed",
    itemId: "i1",
    transcript: "Ehi Grok, ciao.",
    raw: { status: "completed" },
  });
  assert.deepEqual(events.at(-1), {
    type: "transcript.final",
    role: "user",
    text: "Ehi Grok, ciao.",
    utterance: "i1",
  });
});

test("a forwarded in-progress transcript stays a snapshot, not a final", async () => {
  // The gateway swallows live snapshots today; if it ever forwards them, the
  // raw status is the guard that keeps them from finalizing the turn early.
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.message({
    type: "input-transcription-completed",
    itemId: "i1",
    transcript: "Ehi",
    raw: { status: "in_progress" },
  });
  assert.deepEqual(events.at(-1), {
    type: "transcript.update",
    role: "user",
    text: "Ehi",
    utterance: "i1",
  });
});

test("tool call → result → response-create continuation", async () => {
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  const handle = await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.message({ type: "session-updated", raw: {} });
  fake.message({
    type: "function-call-arguments-done",
    responseId: "r1",
    itemId: "f1",
    callId: "c1",
    name: "search",
    arguments: JSON.stringify({ q: "x" }),
    raw: {},
  });
  assert.deepEqual(events.at(-1), {
    type: "tool.call",
    callId: "c1",
    name: "search",
    input: { q: "x" },
  });

  fake.sent.length = 0;
  handle.send({ type: "tool.result", callId: "c1", output: { ok: true } });
  // Verified live: the model does not continue on its own after a tool output.
  assert.ok(!fake.sentTypes().includes("response-create"));
  fake.message({ type: "response-done", responseId: "r1", status: "completed", raw: {} });
  assert.ok(fake.sentTypes().includes("response-create"));
});

test("barge-in cancels an in-flight response and emits speech.interrupted", async () => {
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.message({ type: "response-created", responseId: "r1", raw: {} });
  fake.sent.length = 0;
  fake.message({ type: "speech-started", itemId: "i1", raw: {} });
  assert.ok(fake.sentTypes().includes("response-cancel"));
  assert.deepEqual(events.at(-1), { type: "speech.interrupted", utterance: "i1" });
});

test("does not cancel when no response is active", async () => {
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.sent.length = 0;
  fake.message({ type: "speech-started", itemId: "i1", raw: {} });
  assert.ok(!fake.sentTypes().includes("response-cancel"));
  assert.deepEqual(events.at(-1), { type: "speech.interrupted", utterance: "i1" });
});

test("benign cancellation errors are not surfaced", async () => {
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.message({
    type: "error",
    message: "Cancellation failed: no active response found",
    raw: {},
  });
  assert.equal(events.filter((e) => e.type === "error").length, 0);
});

test("pushAudio buffers until session-updated, then flushes", async () => {
  const fake = new FakeWS();
  const handle = await fakeModel(fake).connect({
    call: makeCall(),
    emit: () => {},
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.sent.length = 0;
  handle.pushAudio(new Uint8Array([9, 9]).buffer);
  assert.equal(fake.sentOf("input-audio-append").length, 0);
  fake.message({ type: "session-updated", raw: {} });
  assert.equal(fake.sentOf("input-audio-append").length, 1);
});

test("missing API key throws on connect", async () => {
  const fake = new FakeWS();
  const model = gateway("xai/grok-voice-think-fast-2.0", {
    createWebSocket: () => fake as unknown as ReturnType<GatewayWebSocketFactory>,
  });
  const prevKey = process.env.AI_GATEWAY_API_KEY;
  delete process.env.AI_GATEWAY_API_KEY;
  try {
    await assert.rejects(
      model.connect({
        call: makeCall(),
        emit: () => {},
        onAudio: () => {},
        signal: new AbortController().signal,
      }),
      /AI_GATEWAY_API_KEY/,
    );
  } finally {
    if (prevKey !== undefined) process.env.AI_GATEWAY_API_KEY = prevKey;
  }
});

test("end-to-end through the core: a repeated terminal transcript stores one turn", async () => {
  // xAI repeats/reattributes terminal transcripts (observed through the
  // gateway too, with cumulative text under one item id): the utterance key
  // must keep the turn single, exactly as with the direct provider.
  const fake = new FakeWS();
  const messages: RealtimeMessage[] = [];
  const session = createRealtimeSession({
    model: fakeModel(fake),
    instructions: "x",
    seed: [],
    triggerResponse: false,
    onMessage: (m) => messages.push(m),
  });
  await session.start();
  fake.message({ type: "session-updated", raw: {} });

  fake.message({
    type: "input-transcription-completed",
    itemId: "i1",
    transcript: "Ciao",
    raw: { status: "completed" },
  });
  fake.message({
    type: "input-transcription-completed",
    itemId: "i1",
    transcript: "Ciao",
    raw: { status: "completed" },
  });
  assert.equal(messages.length, 1);
});

test("end-to-end through the core: a gateway tool round-trip", async () => {
  const { tool } = await import("ai");
  const { z } = await import("zod");
  const tick = () => new Promise<void>((r) => setTimeout(r, 0));
  const fake = new FakeWS();
  const messages: { type: string }[] = [];
  const session = createRealtimeSession({
    model: fakeModel(fake),
    instructions: "x",
    seed: [],
    triggerResponse: false,
    tools: {
      search: tool({
        description: "search",
        inputSchema: z.object({ q: z.string() }),
        execute: async ({ q }) => ({ found: q }),
      }),
    },
    onMessage: (m) => messages.push(m),
  });

  await session.start();
  fake.message({ type: "session-updated", raw: {} });
  fake.message({
    type: "function-call-arguments-done",
    responseId: "r1",
    itemId: "f1",
    callId: "c1",
    name: "search",
    arguments: JSON.stringify({ q: "customers" }),
    raw: {},
  });
  await tick();

  const out = fake.sentOf("conversation-item-create").map((m) => m.item.type);
  assert.ok(out.includes("function-call-output"));
  assert.ok(messages.some((m) => m.type === "tool"));
});

test("a transcription that fails still ends the utterance", async () => {
  // No completion follows a failure, so without this the utterance the core
  // opened on speech-started is never closed and anything waiting on it waits
  // for the rest of the call.
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.message({ type: "speech-started", itemId: "i1" });
  assert.deepEqual(events.at(-1), { type: "speech.interrupted", utterance: "i1" });
  // The gateway has no failure event of its own: the provider's native one
  // comes through as `custom`, which is where this has to be caught.
  fake.message({
    type: "custom",
    rawType: "conversation.item.input_audio_transcription.failed",
    raw: { item_id: "i1" },
  });
  assert.deepEqual(events.at(-1), {
    type: "transcript.done",
    role: "user",
    utterance: "i1",
  });
});
