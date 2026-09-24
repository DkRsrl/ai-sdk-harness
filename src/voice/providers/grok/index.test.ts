import { test } from "node:test";
import assert from "node:assert/strict";
import {
  grok,
  grokOutboundItems,
  grokSessionUpdate,
  type GrokWebSocketFactory,
} from "./index";
import { createRealtimeSession, type RealtimeMessage } from "../../session";
import type { RealtimeCall, RealtimeEvent } from "../../spec";

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

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

function fakeModel(fake: FakeWS) {
  const createWebSocket: GrokWebSocketFactory = () =>
    fake as unknown as ReturnType<GrokWebSocketFactory>;
  return grok("grok-realtime", { apiKey: "k", voice: "ada", createWebSocket });
}

test("grokSessionUpdate maps voice, tools and audio formats", () => {
  const update = grokSessionUpdate(
    makeCall({
      tools: [{ name: "search", description: "find", parametersJsonSchema: { type: "object" } }],
    }),
    { voice: "ada", sampleRate: 24000 },
  );
  const session = update.session as Record<string, unknown>;
  assert.equal(session.voice, "ada");
  assert.equal(session.instructions, "be brief");
  assert.deepEqual(session.tools, [
    { type: "function", name: "search", description: "find", parameters: { type: "object" } },
  ]);
  // Only `type` by default: server VAD on, every threshold left to xAI's
  // defaults.
  assert.deepEqual(session.turn_detection, { type: "server_vad" });
});

test("grokSessionUpdate sends key terms and the language hint side by side", () => {
  const update = grokSessionUpdate(makeCall(), {
    voice: "ada",
    sampleRate: 24000,
    language: "en",
    keyterms: ["Acme", " Acme "],
  });
  const audio = (update.session as Record<string, unknown>).audio as Record<string, unknown>;
  const input = audio.input as Record<string, unknown>;
  assert.deepEqual(input.transcription, { language_hint: "en", keyterms: ["Acme"] });
});

test("grokSessionUpdate omits keyterms when the list is empty", () => {
  const update = grokSessionUpdate(makeCall(), { voice: "ada", sampleRate: 24000, keyterms: [] });
  const audio = (update.session as Record<string, unknown>).audio as Record<string, unknown>;
  assert.deepEqual((audio.input as Record<string, unknown>).transcription, {});
});

test("grokSessionUpdate forwards an explicit turnDetection override", () => {
  const update = grokSessionUpdate(makeCall(), {
    voice: "ada",
    sampleRate: 24000,
    turnDetection: { type: "server_vad", threshold: 0.9 },
  });
  const session = update.session as Record<string, unknown>;
  assert.deepEqual(session.turn_detection, { type: "server_vad", threshold: 0.9 });
});

test("grokOutboundItems maps text, tool.call and tool.result", () => {
  assert.deepEqual(grokOutboundItems({ type: "tool.result", callId: "c1", output: { ok: true } }), [
    {
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: "c1", output: JSON.stringify({ ok: true }) },
    },
  ]);
  const user = grokOutboundItems({ type: "text", role: "user", text: "ciao" })[0] as {
    item: { content: { type: string }[] };
  };
  assert.equal(user.item.content[0]?.type, "input_text");
});

test("a failed tool result goes out as prose, not JSON", () => {
  // The wire has no error channel, so the model must be able to read the
  // failure as text.
  assert.deepEqual(
    grokOutboundItems({
      type: "tool.result",
      callId: "c1",
      output: { error: "boom" },
      isError: true,
    }),
    [
      {
        type: "conversation.item.create",
        item: { type: "function_call_output", call_id: "c1", output: "Error: boom" },
      },
    ],
  );
});

test("connect sends session.update + seed + trigger on open", async () => {
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
  assert.ok(types.includes("session.update"));
  assert.ok(types.includes("conversation.item.create"));
  assert.equal(types.at(-1), "response.create");
});

test("seeds a full tool round-trip natively as conversation items in order", async () => {
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

  const items = fake.sentOf("conversation.item.create").map((m) => m.item.type);
  assert.deepEqual(items, ["message", "function_call", "function_call_output", "message"]);
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
  fake.message({ type: "response.output_audio.delta", delta: pcm });
  fake.message({ type: "response.output_audio_transcript.delta", delta: "Ci" });
  fake.message({ type: "response.output_audio_transcript.delta", delta: "ao" });
  fake.message({ type: "response.output_audio_transcript.done", transcript: "Ciao" });

  assert.equal(audio.length, 1);
  assert.equal(audio[0]?.byteLength, 4);
  assert.ok(events.some((e) => e.type === "speech.start"));
  assert.equal(events.filter((e) => e.type === "transcript.delta").length, 2);
  // transcript settles on its own done signal, before response.done (audio).
  assert.ok(events.some((e) => e.type === "transcript.done" && e.role === "assistant"));
});

test("user transcription streams as cumulative snapshots, finalizing on completed", async () => {
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  // xAI re-sends the whole transcript-so-far (and revises it); the partial
  // `completed` events carry status "in_progress", the terminal one "completed".
  fake.message({ type: "conversation.item.input_audio_transcription.updated", transcript: "Hey" });
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    transcript: "Hey Grok",
    status: "in_progress",
  });
  fake.message({
    type: "conversation.item.input_audio_transcription.updated",
    transcript: "Ehi Grok, ciao",
  });
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    transcript: "Ehi Grok, ciao.",
    status: "completed",
  });

  // Snapshots REPLACE (so the revision wins); only the terminal event finalizes.
  const updates = events.filter((e) => e.type === "transcript.update" && e.role === "user");
  assert.equal(updates.length, 3);
  const last = updates.at(-1);
  assert.equal(last?.type === "transcript.update" ? last.text : "", "Ehi Grok, ciao");
  assert.deepEqual(events.at(-1), {
    type: "transcript.final",
    role: "user",
    text: "Ehi Grok, ciao.",
  });
});

test("tool call → result → response.create continuation", async () => {
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  const handle = await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.message({ type: "session.updated" });
  fake.message({
    type: "response.function_call_arguments.done",
    call_id: "c1",
    name: "search",
    arguments: JSON.stringify({ q: "x" }),
  });
  assert.deepEqual(events.at(-1), {
    type: "tool.call",
    callId: "c1",
    name: "search",
    input: { q: "x" },
  });

  fake.sent.length = 0;
  handle.send({ type: "tool.result", callId: "c1", output: { ok: true } });
  assert.ok(!fake.sentTypes().includes("response.create"));
  fake.message({ type: "response.done" });
  assert.ok(fake.sentTypes().includes("response.create"));
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

  fake.message({ type: "response.created" }); // a response is now in flight
  fake.sent.length = 0;
  fake.message({ type: "input_audio_buffer.speech_started" });
  assert.ok(fake.sentTypes().includes("response.cancel"));
  assert.deepEqual(events.at(-1), { type: "speech.interrupted" });
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
  fake.message({ type: "input_audio_buffer.speech_started" }); // nothing in flight
  assert.ok(!fake.sentTypes().includes("response.cancel"));
  assert.deepEqual(events.at(-1), { type: "speech.interrupted" });
});

test("transcription events carry the conversation item as the utterance key", async () => {
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.message({
    type: "conversation.item.input_audio_transcription.updated",
    item_id: "i1",
    transcript: "Ciao",
  });
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "i1",
    transcript: "Ciao",
    status: "completed",
  });
  assert.deepEqual(events.at(-2), {
    type: "transcript.update",
    role: "user",
    text: "Ciao",
    utterance: "i1",
  });
  assert.deepEqual(events.at(-1), {
    type: "transcript.final",
    role: "user",
    text: "Ciao",
    utterance: "i1",
  });
});

test("end-to-end through the core: a repeated terminal completed stores one turn", async () => {
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
  fake.message({ type: "session.updated" });

  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "i1",
    transcript: "Ciao",
    status: "completed",
  });
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "i1",
    transcript: "Ciao",
    status: "completed",
  });
  assert.equal(messages.length, 1);
});

test("end-to-end through the core: an utterance whose terminal never arrives still becomes its own turn", async () => {
  const fake = new FakeWS();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const session = createRealtimeSession({
    model: fakeModel(fake),
    instructions: "x",
    seed: [],
    triggerResponse: false,
    now: () => t,
    onMessage: (m) => messages.push(m),
  });
  await session.start();
  fake.message({ type: "session.updated" });

  // The first utterance streams but its terminal completed gets lost.
  fake.message({
    type: "conversation.item.input_audio_transcription.updated",
    item_id: "i1",
    transcript: "I'm talking to a customer, I need to show them",
  });

  // The next utterance starting is the boundary that settles the previous one
  // from its last snapshot, under its own identity.
  t = 2000;
  fake.message({
    type: "conversation.item.input_audio_transcription.updated",
    item_id: "i2",
    transcript: "Looking for aluminium",
  });
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "i2",
    transcript: "Looking for aluminium presets",
    status: "completed",
  });

  const texts = messages.flatMap((m) => (m.type === "text" ? [m.text] : []));
  assert.deepEqual(texts, [
    "I'm talking to a customer, I need to show them",
    "Looking for aluminium presets",
  ]);
  const [first, second] = messages;
  assert.ok(first && second && first.id !== second.id);
  assert.ok(first.createdAt < second.createdAt);
});

test("end-to-end through the core: a late terminal repairs the early-settled turn in place", async () => {
  const fake = new FakeWS();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const session = createRealtimeSession({
    model: fakeModel(fake),
    instructions: "x",
    seed: [],
    triggerResponse: false,
    now: () => t,
    onMessage: (m) => messages.push(m),
  });
  await session.start();
  fake.message({ type: "session.updated" });

  // Utterance 1 streams; its terminal lags behind the next utterance's start.
  fake.message({
    type: "conversation.item.input_audio_transcription.updated",
    item_id: "i1",
    transcript: "I'm talking to a",
  });
  t = 2000;
  fake.message({
    type: "conversation.item.input_audio_transcription.updated",
    item_id: "i2",
    transcript: "Looking for",
  });
  // The lagging terminal for utterance 1 arrives with the full text: it must
  // repair the turn that settled early — same id, same stamp — and leave the
  // open utterance 2 alone.
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "i1",
    transcript: "I'm talking to a customer",
    status: "completed",
  });
  t = 3000;
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "i2",
    transcript: "Looking for aluminium",
    status: "completed",
  });

  const texts = messages.flatMap((m) =>
    m.type === "text" ? [[m.id, m.text, m.createdAt] as const] : [],
  );
  assert.equal(texts.length, 3);
  const [flushed, repaired, second] = texts;
  assert.ok(flushed && repaired && second);
  assert.equal(flushed[1], "I'm talking to a");
  // The repair reuses the flushed turn's identity, so an id-keyed store ends
  // up with exactly two turns, full text, in spoken order.
  assert.equal(repaired[0], flushed[0]);
  assert.equal(repaired[1], "I'm talking to a customer");
  assert.equal(repaired[2], flushed[2]);
  assert.equal(second[1], "Looking for aluminium");
  assert.ok(second[0] !== flushed[0]);
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

  fake.message({ type: "error", error: { message: "Cancellation failed: no active response found" } });
  assert.equal(events.filter((e) => e.type === "error").length, 0);
});

test("pushAudio buffers until session.updated, then flushes", async () => {
  const fake = new FakeWS();
  const handle = await fakeModel(fake).connect({
    call: makeCall(),
    emit: () => {},
    onAudio: () => {},
    signal: new AbortController().signal,
  });

  fake.sent.length = 0;
  handle.pushAudio(new Uint8Array([9, 9]).buffer);
  assert.equal(fake.sentOf("input_audio_buffer.append").length, 0);
  fake.message({ type: "session.updated" });
  assert.equal(fake.sentOf("input_audio_buffer.append").length, 1);
});

test("missing API key throws on connect", async () => {
  const fake = new FakeWS();
  const model = grok("grok-realtime", {
    createWebSocket: () => fake as unknown as ReturnType<GrokWebSocketFactory>,
  });
  const prevKey = process.env.XAI_API_KEY;
  delete process.env.XAI_API_KEY;
  try {
    await assert.rejects(
      model.connect({
        call: makeCall(),
        emit: () => {},
        onAudio: () => {},
        signal: new AbortController().signal,
      }),
      /XAI_API_KEY/,
    );
  } finally {
    if (prevKey !== undefined) process.env.XAI_API_KEY = prevKey;
  }
});

test("end-to-end through the core: a repeated item_id never folds the next utterance into a stale turn", async () => {
  // Observed production sequence: the terminal
  // `completed` for one utterance repeats an already-finalized item_id, so the
  // turn it should close never closes; the next utterance then rides the stale
  // turn — its text lands under the earlier id and createdAt, and the earlier
  // question vanishes from the transcript.
  const fake = new FakeWS();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const session = createRealtimeSession({
    model: fakeModel(fake),
    instructions: "x",
    seed: [],
    triggerResponse: false,
    now: () => t,
    onMessage: (m) => messages.push(m),
  });
  await session.start();
  fake.message({ type: "session.updated" });

  // Utterance 1 transcribes and finalizes normally.
  fake.message({
    type: "conversation.item.input_audio_transcription.updated",
    transcript: "Ciao, mi senti?",
  });
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "i1",
    transcript: "Ciao, mi senti?",
    status: "completed",
  });

  // Utterance 2: its terminal completed carries an already-finalized item_id,
  // so nothing ever closes the turn.
  t = 2000;
  fake.message({
    type: "conversation.item.input_audio_transcription.updated",
    transcript: "Looking for parts for a truck bed",
  });
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "i1",
    transcript: "Looking for parts for a truck bed",
    status: "completed",
  });

  // The model heard the audio natively and answers regardless.
  t = 3000;
  fake.message({ type: "response.created" });
  fake.message({
    type: "response.output_audio_transcript.delta",
    delta: "Here are the compatible presets",
  });
  fake.message({ type: "response.output_audio_transcript.done" });
  fake.message({ type: "response.done" });

  // Utterance 3 arrives with a fresh item_id and finalizes.
  t = 4000;
  fake.message({
    type: "conversation.item.input_audio_transcription.updated",
    transcript: "Ma cosa sono TR1, TR2, TR3?",
  });
  fake.message({
    type: "conversation.item.input_audio_transcription.completed",
    item_id: "i2",
    transcript: "Ma cosa sono TR1, TR2, TR3?",
    status: "completed",
  });

  const texts = messages.flatMap((m) => (m.type === "text" ? [[m.role, m.text]] : []));
  assert.deepEqual(texts, [
    ["user", "Ciao, mi senti?"],
    ["user", "Looking for parts for a truck bed"],
    ["assistant", "Here are the compatible presets"],
    ["user", "Ma cosa sono TR1, TR2, TR3?"],
  ]);
  // Every turn keeps its own identity and stamp: no id is reused across turns,
  // and no turn sorts before the answer that followed it.
  const ids = messages.map((m) => m.id);
  assert.equal(new Set(ids).size, ids.length);
  const stamps = messages.map((m) => m.createdAt);
  assert.deepEqual(stamps, [...stamps].sort((a, b) => a - b));
});

test("end-to-end through the core: a Grok tool round-trip", async () => {
  const { tool } = await import("ai");
  const { z } = await import("zod");
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
  fake.message({ type: "session.updated" });
  fake.message({
    type: "response.function_call_arguments.done",
    call_id: "c1",
    name: "search",
    arguments: JSON.stringify({ q: "customers" }),
  });
  await tick();

  const out = fake.sentOf("conversation.item.create").map((m) => m.item.type);
  assert.ok(out.includes("function_call_output"));
  assert.ok(messages.some((m) => m.type === "tool"));
});

test("a transcription that fails still ends the utterance", async () => {
  // No `completed` follows a `failed`, so without this the utterance the core
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

  fake.message({ type: "input_audio_buffer.speech_started", item_id: "i1" });
  assert.deepEqual(events.at(-1), { type: "speech.interrupted", utterance: "i1" });
  fake.message({
    type: "conversation.item.input_audio_transcription.failed",
    item_id: "i1",
    error: { message: "asr unavailable" },
  });
  assert.deepEqual(events.at(-1), {
    type: "transcript.done",
    role: "user",
    utterance: "i1",
  });
});

test("an unexpected close reports its code and reason before the disconnect", async () => {
  // An upstream drop (the gateway relays xAI's as 1011) must reach the host
  // with its cause, not as a bare disconnect.
  const fake = new FakeWS();
  const events: RealtimeEvent[] = [];
  await fakeModel(fake).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });
  events.length = 0;

  fake.readyState = 3;
  fake.fire("close", { code: 1011, reason: "Upstream connection closed", wasClean: true });

  assert.equal(events.length, 2);
  const [error, disconnect] = events;
  assert.equal(error?.type, "error");
  assert.ok(error?.type === "error" && !error.fatal);
  assert.match(
    error?.type === "error" ? error.message : "",
    /closed: code=1011 reason=Upstream connection closed$/,
  );
  assert.deepEqual(disconnect, { type: "transport", status: "disconnected" });
});

test("a close the host asked for stays silent", async () => {
  for (const how of ["abort", "handle"] as const) {
    const fake = new FakeWS();
    const events: RealtimeEvent[] = [];
    const controller = new AbortController();
    const handle = await fakeModel(fake).connect({
      call: makeCall(),
      emit: (e) => events.push(e),
      onAudio: () => {},
      signal: controller.signal,
    });
    events.length = 0;

    if (how === "abort") controller.abort();
    else await handle.close();

    assert.deepEqual(events, [{ type: "transport", status: "disconnected" }], how);
  }
});
