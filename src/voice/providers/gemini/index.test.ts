import { test } from "node:test";
import assert from "node:assert/strict";
import {
  gemini,
  geminiToolDefs,
  renderSeedTranscript,
  type GeminiConnect,
  type GeminiLiveCallbacks,
  type GeminiLiveSession,
  type GeminiServerMessage,
} from "./index";
import type { RealtimeCall, RealtimeEvent } from "../../spec";

function makeCall(over: Partial<RealtimeCall> = {}): RealtimeCall {
  return { instructions: "be brief", tools: [], seed: [], triggerResponse: false, ...over };
}

function fakeLive() {
  let callbacks: GeminiLiveCallbacks = {};
  let config: Record<string, unknown> = {};
  const realtimeInput: Array<{ audio: { data: string; mimeType: string } }> = [];
  const clientContent: unknown[] = [];
  const toolResponses: Array<{
    functionResponses: Array<{ id?: string; name?: string; response: Record<string, unknown> }>;
  }> = [];
  let closed = false;

  const session: GeminiLiveSession = {
    sendRealtimeInput: (i) => realtimeInput.push(i),
    sendClientContent: (i) => clientContent.push(i),
    sendToolResponse: (i) => toolResponses.push(i),
    close: () => {
      closed = true;
    },
  };
  const connect: GeminiConnect = async (args) => {
    callbacks = args.callbacks;
    config = args.config;
    return session;
  };
  return {
    connect,
    realtimeInput,
    clientContent,
    toolResponses,
    get config() {
      return config;
    },
    get closed() {
      return closed;
    },
    open: () => callbacks.onopen?.(),
    setup: () => callbacks.onmessage?.({ setupComplete: {} }),
    close: (reason?: string) => callbacks.onclose?.(reason ? { reason } : undefined),
    fire: (m: GeminiServerMessage) => callbacks.onmessage?.(m),
  };
}

function systemText(config: Record<string, unknown>): string {
  const si = config.systemInstruction as { parts?: Array<{ text?: string }> } | undefined;
  return si?.parts?.[0]?.text ?? "";
}

test("geminiToolDefs maps to functionDeclarations (undefined when empty)", () => {
  assert.equal(geminiToolDefs([]), undefined);
  const defs = geminiToolDefs([
    { name: "search", description: "find", parametersJsonSchema: { type: "object" } },
  ]);
  assert.deepEqual(defs, [
    {
      functionDeclarations: [
        { name: "search", description: "find", parametersJsonSchema: { type: "object" } },
      ],
    },
  ]);
});

test("maps audio, transcripts, tool calls and barge-in to spec events", async () => {
  const fake = fakeLive();
  const events: RealtimeEvent[] = [];
  const audio: ArrayBuffer[] = [];
  await gemini("m", { apiKey: "k", connect: fake.connect }).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: (a) => audio.push(a),
    signal: new AbortController().signal,
  });
  fake.open();
  assert.ok(events.some((e) => e.type === "transport" && e.status === "connected"));

  // user speaks (incremental), then the model responds
  fake.fire({ serverContent: { inputTranscription: { text: "che ore " } } });
  fake.fire({ serverContent: { inputTranscription: { text: "sono" } } });
  const pcm = Buffer.from([1, 2, 3, 4]).toString("base64");
  fake.fire({ serverContent: { modelTurn: { parts: [{ inlineData: { data: pcm } }] } } });
  fake.fire({ serverContent: { outputTranscription: { text: "It's 3 o'clock" } } });
  fake.fire({ serverContent: { generationComplete: true } });
  fake.fire({ serverContent: { turnComplete: true } });

  const userDeltas = events.filter((e) => e.type === "transcript.delta" && e.role === "user");
  assert.equal(userDeltas.length, 2); // streamed live as the user spoke
  const userFinals = events.filter((e) => e.type === "transcript.final" && e.role === "user");
  assert.equal(userFinals.length, 1); // then settled into one
  assert.equal(userFinals[0]?.type === "transcript.final" ? userFinals[0].text : "", "what time is it");
  assert.equal(audio.length, 1);
  assert.ok(events.some((e) => e.type === "speech.start"));
  assert.ok(events.some((e) => e.type === "transcript.delta" && e.role === "assistant"));
  // transcript settles on generationComplete, before turnComplete (audio/turn).
  assert.ok(events.some((e) => e.type === "transcript.done" && e.role === "assistant"));
  assert.ok(events.some((e) => e.type === "speech.stop"));
});

test("tool call → result maps to sendToolResponse with the function name", async () => {
  const fake = fakeLive();
  const events: RealtimeEvent[] = [];
  const handle = await gemini("m", { apiKey: "k", connect: fake.connect }).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });
  fake.fire({ toolCall: { functionCalls: [{ id: "c1", name: "get_time", args: { tz: "Rome" } }] } });
  assert.deepEqual(events.at(-1), {
    type: "tool.call",
    callId: "c1",
    name: "get_time",
    input: { tz: "Rome" },
  });

  handle.send({ type: "tool.result", callId: "c1", output: { now: "x" } });
  assert.equal(fake.toolResponses.length, 1);
  const fr = fake.toolResponses[0]?.functionResponses[0];
  assert.equal(fr?.id, "c1");
  assert.equal(fr?.name, "get_time");
  assert.deepEqual(fr?.response, { now: "x" });
});

test("buffers mic frames until setupComplete, then flushes them", async () => {
  const fake = fakeLive();
  const handle = await gemini("m", { apiKey: "k", sampleRate: 24000, connect: fake.connect }).connect({
    call: makeCall(),
    emit: () => {},
    onAudio: () => {},
    signal: new AbortController().signal,
  });
  // Before the handshake, frames are held back (Gemini would reject them).
  handle.pushAudio(new Int16Array(30).fill(500).buffer);
  assert.equal(fake.realtimeInput.length, 0);
  fake.setup();
  assert.equal(fake.realtimeInput.length, 1);
  assert.match(fake.realtimeInput[0]?.audio.mimeType ?? "", /rate=24000/);
  // After setup, frames go straight through.
  handle.pushAudio(new Int16Array(30).fill(500).buffer);
  assert.equal(fake.realtimeInput.length, 2);
});

test("seeds the prior conversation as one user-turn transcript on setupComplete", async () => {
  const fake = fakeLive();
  await gemini("m", { apiKey: "k", connect: fake.connect }).connect({
    call: makeCall({
      instructions: "be brief",
      seed: [
        { type: "text", role: "user", text: "my name is Marco" },
        { type: "tool.call", callId: "c1", name: "get_time", input: {} },
        { type: "tool.result", callId: "c1", output: { now: "15:00" } },
      ],
    }),
    emit: () => {},
    onAudio: () => {},
    signal: new AbortController().signal,
  });
  // System instruction stays clean — just the instructions.
  assert.equal(systemText(fake.config), "be brief");
  // Nothing is sent until the handshake; then the seed lands as one user turn.
  assert.equal(fake.clientContent.length, 0);
  fake.setup();
  assert.equal(fake.clientContent.length, 1);
  const sent = fake.clientContent[0] as {
    turns: Array<{ role: string; parts: Array<{ text: string }> }>;
    turnComplete?: boolean;
  };
  assert.equal(sent.turns.length, 1);
  assert.equal(sent.turns[0]?.role, "user");
  assert.equal(sent.turnComplete, false); // context only — no generation
  const text = sent.turns[0]?.parts[0]?.text ?? "";
  assert.match(text, /<previous_conversation_messages>[\s\S]*<\/previous_conversation_messages>/);
  assert.match(text, /User: my name is Marco/);
  assert.match(text, /Assistant called tool "get_time"/);
  assert.match(text, /Tool "get_time" returned \{"now":"15:00"\}/);
});

test("a triggered response waits for setupComplete", async () => {
  const fake = fakeLive();
  await gemini("m", { apiKey: "k", connect: fake.connect }).connect({
    call: makeCall({ triggerResponse: true }),
    emit: () => {},
    onAudio: () => {},
    signal: new AbortController().signal,
  });
  assert.equal(fake.clientContent.length, 0); // not yet — handshake pending
  fake.setup();
  assert.equal(fake.clientContent.length, 1); // trigger sent after setupComplete
  assert.deepEqual(fake.clientContent[0], { turnComplete: true }); // no empty turns[]
});

test("surfaces a close reason as an error before disconnect", async () => {
  const fake = fakeLive();
  const events: RealtimeEvent[] = [];
  await gemini("m", { apiKey: "k", connect: fake.connect }).connect({
    call: makeCall(),
    emit: (e) => events.push(e),
    onAudio: () => {},
    signal: new AbortController().signal,
  });
  fake.close("Request contains an invalid argument.");
  assert.ok(
    events.some(
      (e) => e.type === "error" && /invalid argument/.test(e.message),
    ),
  );
  assert.ok(events.some((e) => e.type === "transport" && e.status === "disconnected"));
});

test("renderSeedTranscript renders text + tool round-trips, null when empty", () => {
  assert.equal(renderSeedTranscript([]), null);
  const text = renderSeedTranscript([
    { type: "text", role: "user", text: "what time is it" },
    { type: "text", role: "assistant", text: "let me check" },
    { type: "tool.call", callId: "c1", name: "get_time", input: { tz: "Rome" } },
    { type: "tool.result", callId: "c1", output: { now: "15:00" } },
    { type: "text", role: "assistant", text: "it's 3pm" },
  ]);
  assert.equal(
    text,
    [
      "User: what time is it",
      "Assistant: let me check",
      'Assistant called tool "get_time" with {"tz":"Rome"}',
      'Tool "get_time" returned {"now":"15:00"}',
      "Assistant: it's 3pm",
    ].join("\n"),
  );
});

test("missing API key throws on connect", async () => {
  const prev = process.env.GOOGLE_API_KEY;
  const prevG = process.env.GEMINI_API_KEY;
  const prevGen = process.env.GOOGLE_GENAI_API_KEY;
  delete process.env.GOOGLE_API_KEY;
  delete process.env.GEMINI_API_KEY;
  delete process.env.GOOGLE_GENAI_API_KEY;
  try {
    await assert.rejects(
      gemini("m").connect({
        call: makeCall(),
        emit: () => {},
        onAudio: () => {},
        signal: new AbortController().signal,
      }),
      /Google API key/,
    );
  } finally {
    if (prev !== undefined) process.env.GOOGLE_API_KEY = prev;
    if (prevG !== undefined) process.env.GEMINI_API_KEY = prevG;
    if (prevGen !== undefined) process.env.GOOGLE_GENAI_API_KEY = prevGen;
  }
});

test("buffers text items until setupComplete, then flushes them in order", async () => {
  const fake = fakeLive();
  const handle = await gemini("m", { apiKey: "k", connect: fake.connect }).connect({
    call: makeCall(),
    emit: () => {},
    onAudio: () => {},
    signal: new AbortController().signal,
  });
  // Before the handshake, items are held back (Gemini would reject them).
  handle.send({ type: "text", role: "user", text: "<message_metadata>t</message_metadata>" });
  assert.equal(fake.clientContent.length, 0);
  fake.setup();
  assert.equal(fake.clientContent.length, 1);
  const sent = fake.clientContent[0] as {
    turns: Array<{ role: string; parts: Array<{ text: string }> }>;
    turnComplete: boolean;
  };
  assert.equal(sent.turns[0]?.parts[0]?.text, "<message_metadata>t</message_metadata>");
  assert.equal(sent.turnComplete, false);
  // After setup, items go straight through.
  handle.send({ type: "text", role: "user", text: "live" });
  assert.equal(fake.clientContent.length, 2);
});
