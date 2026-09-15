import { test } from "node:test";
import assert from "node:assert/strict";
import { tool } from "ai";
import { z } from "zod";
import { createRealtimeSession, type RealtimeMessage } from "./session";
import { decodeServerEvent, encodeClientEvent } from "./protocol";
import type {
  RealtimeCall,
  RealtimeConnectArgs,
  RealtimeEvent,
  RealtimeHandle,
  RealtimeModelV1,
  RealtimeOutbound,
  VoiceStatus,
} from "./spec";

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

function makeCall(): Omit<RealtimeCall, "tools"> {
  return { instructions: "be brief", seed: [], triggerResponse: false };
}

function mockModel() {
  let emit: ((e: RealtimeEvent) => void) | null = null;
  let onAudio: ((pcm: ArrayBuffer) => void) | null = null;
  let call: RealtimeCall | null = null;
  const sent: RealtimeOutbound[] = [];
  const pushed: ArrayBuffer[] = [];
  let closed = false;
  let responsesRequested = 0;

  const handle: RealtimeHandle = {
    send: (item) => sent.push(item),
    pushAudio: (pcm) => pushed.push(pcm),
    requestResponse: () => {
      responsesRequested++;
    },
    close: async () => {
      closed = true;
    },
  };

  const model: RealtimeModelV1 = {
    specificationVersion: "realtime-v1",
    provider: "mock",
    modelId: "mock",
    async connect(connectArgs: RealtimeConnectArgs) {
      emit = connectArgs.emit;
      onAudio = connectArgs.onAudio;
      call = connectArgs.call;
      return handle;
    },
  };

  return {
    model,
    emit: (e: RealtimeEvent) => emit?.(e),
    pushAudio: (pcm: ArrayBuffer) => onAudio?.(pcm),
    sent,
    pushed,
    handle,
    get call() {
      return call;
    },
    get closed() {
      return closed;
    },
    get responsesRequested() {
      return responsesRequested;
    },
  };
}

test("connecting → listening on start", async () => {
  const m = mockModel();
  const states: VoiceStatus[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onStatus: (st) => states.push(st),
  });
  await s.start();
  assert.deepEqual(states, ["connecting", "listening"]);
  assert.equal(s.status, "listening");
});

test("concurrent start calls share one provider connection", async () => {
  const m = mockModel();
  let connects = 0;
  const model: RealtimeModelV1 = {
    ...m.model,
    async connect(args) {
      connects += 1;
      return m.model.connect(args);
    },
  };
  const s = createRealtimeSession({
    model,
    ...makeCall(),
    tools: {
      ping: tool({
        description: "Ping.",
        inputSchema: z.object({}),
        execute: async () => "pong",
      }),
    },
  });

  await Promise.all([s.start(), s.start()]);

  assert.equal(connects, 1);
  assert.equal(s.status, "listening");
});

test("stop during startup closes a provider that connects late", async () => {
  let resolveHandle: ((handle: RealtimeHandle) => void) | undefined;
  let emit: ((event: RealtimeEvent) => void) | undefined;
  let closed = 0;
  const model: RealtimeModelV1 = {
    specificationVersion: "realtime-v1",
    provider: "deferred",
    modelId: "deferred",
    connect(args) {
      emit = args.emit;
      return new Promise((resolve) => {
        resolveHandle = resolve;
      });
    },
  };
  const s = createRealtimeSession({ model, ...makeCall() });
  const starting = s.start();
  await tick();
  const stopping = s.stop();

  resolveHandle?.({
    send: () => {},
    pushAudio: () => {},
    requestResponse: () => {},
    close: async () => {
      closed += 1;
    },
  });
  await Promise.all([starting, stopping]);
  emit?.({ type: "transport", status: "connected" });

  assert.equal(closed, 1);
  assert.equal(s.status, "idle");
});

test("tool setup failure moves startup to error", async () => {
  const m = mockModel();
  const errors: string[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: {
      broken: tool({
        description: "Broken.",
        inputSchema: {
          "~standard": {
            version: 1,
            vendor: "broken",
            validate: () => ({ value: {} }),
          },
        } as never,
        execute: async () => "never",
      }),
    },
    onError: (message) => errors.push(message),
  });

  await assert.rejects(s.start());

  assert.equal(s.status, "error");
  assert.equal(errors.length, 1);
});

test("assembles assistant transcript deltas and flushes on speech.stop", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "speech.start" });
  m.emit({ type: "transcript.delta", role: "assistant", text: "Ciao" });
  m.emit({ type: "transcript.delta", role: "assistant", text: " Marco" });
  assert.equal(s.status, "speaking");
  m.emit({ type: "speech.stop" });

  assert.equal(messages.length, 1);
  const msg = messages[0];
  if (!msg || msg.type !== "text") throw new Error("expected a text message");
  assert.equal(msg.role, "assistant");
  assert.equal(msg.text, "Ciao Marco");
  assert.equal(s.status, "listening");
});

test("transcript.done settles the turn without waiting for speech.stop", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "speech.start" });
  m.emit({ type: "transcript.delta", role: "assistant", text: "Ciao" });
  m.emit({ type: "transcript.done", role: "assistant" });

  // The text is finalized while audio is still "speaking" (no speech.stop yet).
  assert.equal(s.status, "speaking");
  const settled = messages.at(-1);
  if (!settled || settled.type !== "text") throw new Error("expected a text message");
  assert.equal(settled.text, "Ciao");

  // speech.stop now only moves status; the transcript is already settled.
  m.emit({ type: "speech.stop" });
  assert.equal(messages.length, 1);
  assert.equal(s.status, "listening");
});

test("transcript.update replaces (cumulative snapshots) instead of appending", async () => {
  const m = mockModel();
  const fulls: string[] = [];
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onTranscriptDelta: (d) => fulls.push(d.full),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.update", role: "user", text: "Hey" });
  m.emit({ type: "transcript.update", role: "user", text: "Hey Grok, ciao" });
  m.emit({ type: "transcript.final", role: "user", text: "Hey Grok, ciao." });

  // Each snapshot REPLACES — the streamed full text is the latest, not concatenated.
  assert.deepEqual(fulls, ["Hey", "Hey Grok, ciao"]);
  const final = messages.at(-1);
  if (!final || final.type !== "text") throw new Error("expected a text message");
  assert.equal(final.text, "Hey Grok, ciao.");
});

test("emits a user turn from transcript.final", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "which customers do I have?" });

  const msg = messages[0];
  if (!msg || msg.type !== "text") throw new Error("expected a text message");
  assert.equal(msg.role, "user");
  assert.equal(msg.text, "which customers do I have?");
});

test("stamps createdAt at turn start (from the injected clock) and reuses it", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const deltas: number[] = [];
  let t = 1000;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => t,
    onTranscriptDelta: (d) => deltas.push(d.createdAt),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.delta", role: "assistant", text: "Ciao" });
  // Clock advances mid-turn — the final message must keep the turn's start stamp.
  t = 5000;
  m.emit({ type: "transcript.done", role: "assistant" });

  const msg = messages.at(-1);
  if (!msg || msg.type !== "text") throw new Error("expected a text message");
  assert.equal(msg.createdAt, 1000);
  assert.deepEqual(deltas, [1000]);
});

test("the filler spoken before a tool call is settled before the tool runs", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const search = tool({
    description: "search",
    inputSchema: z.object({}),
    execute: async () => "ok",
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { search },
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  // The response speaks a filler, then calls the tool — no transcript.done yet.
  m.emit({ type: "transcript.delta", role: "assistant", utterance: "u1", text: "One moment." });
  m.emit({ type: "tool.call", callId: "c1", name: "search", input: {} });
  await tick();

  // A conversation-reading tool must find the filler on the record: the text
  // turn settles at tool start, before the tool's own message.
  assert.deepEqual(
    messages.map((msg) => msg.type),
    ["text", "tool"],
  );
  const filler = messages[0]!;
  if (filler.type !== "text") throw new Error("expected the filler text turn");
  assert.equal(filler.text, "One moment.");
});

test("a tool that throws reports in-band and leaves the session alive", async () => {
  const m = mockModel();
  const errors: string[] = [];
  const failing = tool({
    description: "always fails",
    inputSchema: z.object({}),
    execute: async (): Promise<string> => {
      throw new Error("the assistant came back with nothing to say");
    },
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { failing },
    onError: (message) => errors.push(message),
  });
  await s.start();
  m.emit({ type: "tool.call", callId: "c1", name: "failing", input: {} });
  await tick();

  // The model gets the failure as the call's result and recovers in-band; no
  // error frame reaches the client, which would treat it as fatal and close.
  assert.deepEqual(m.sent.at(-1), {
    type: "tool.result",
    callId: "c1",
    output: { error: "the assistant came back with nothing to say" },
    isError: true,
  });
  assert.deepEqual(errors, []);
  assert.notEqual(s.status, "error");
});

test("dispatches a real tool() call and sends the result back", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const pending: Array<{ callId: string; name: string; input: unknown }> = [];
  const search = tool({
    description: "search the knowledge base",
    inputSchema: z.object({ q: z.string() }),
    execute: async ({ q }) => ({ found: q }),
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { search },
    onMessage: (msg) => messages.push(msg),
    onToolPending: (p) => pending.push(p),
  });
  await s.start();
  m.emit({ type: "tool.call", callId: "c1", name: "search", input: { q: "x" } });
  assert.equal(s.status, "thinking");
  await tick();

  assert.deepEqual(m.sent.at(-1), {
    type: "tool.result",
    callId: "c1",
    output: { found: "x" },
  });
  const msg = messages.at(-1);
  if (!msg || msg.type !== "tool") throw new Error("expected a tool message");
  assert.equal(msg.callId, "c1");
  assert.deepEqual(msg.input, { q: "x" });
  assert.deepEqual(msg.output, { found: "x" });
  // The running call is surfaced (with input) before the result lands, and
  // shares the call's ordering stamp with the finalized turn.
  assert.deepEqual(pending, [{ callId: "c1", name: "search", input: { q: "x" }, createdAt: msg.createdAt }]);
});

test("tool messages are typed per the ToolSet", async () => {
  // This only compiles if `msg.input`/`msg.output` are inferred from the tool's
  // schema after narrowing on `msg.name` — the AI-SDK-style typing.
  const m = mockModel();
  const search = tool({
    description: "search",
    inputSchema: z.object({ q: z.string() }),
    execute: async ({ q }) => ({ found: q }),
  });
  const seen: string[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { search },
    onMessage: (msg) => {
      if (msg.type === "tool" && msg.name === "search") {
        seen.push(`${msg.input.q}:${msg.output.found}`);
      }
    },
  });
  await s.start();
  m.emit({ type: "tool.call", callId: "c1", name: "search", input: { q: "x" } });
  await tick();
  assert.deepEqual(seen, ["x:x"]);
});

test("activeTools advertises and runs only the active subset of tools", async () => {
  const m = mockModel();
  const a = tool({
    description: "tool a",
    inputSchema: z.object({}),
    execute: async () => ({ ok: "a" }),
  });
  const b = tool({
    description: "tool b",
    inputSchema: z.object({}),
    execute: async () => ({ ok: "b" }),
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { a, b },
    activeTools: ["a"],
  });
  await s.start();

  // Only `a` is advertised to the provider.
  assert.deepEqual(m.call?.tools.map((t) => t.name), ["a"]);

  // `b` is gated out: calling it is rejected without executing.
  m.emit({ type: "tool.call", callId: "c1", name: "b", input: {} });
  await tick();
  const rejected = m.sent.at(-1);
  assert.ok(
    rejected?.type === "tool.result" &&
      typeof rejected.output === "object" &&
      rejected.output !== null &&
      "error" in rejected.output,
  );

  // `a` runs normally.
  m.emit({ type: "tool.call", callId: "c2", name: "a", input: {} });
  await tick();
  assert.deepEqual(m.sent.at(-1), {
    type: "tool.result",
    callId: "c2",
    output: { ok: "a" },
  });
});

test("rejects invalid tool input via the tool's schema", async () => {
  const m = mockModel();
  let executed = false;
  const search = tool({
    description: "search",
    inputSchema: z.object({ q: z.string() }),
    execute: async () => {
      executed = true;
      return { ok: true };
    },
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { search },
  });
  await s.start();
  m.emit({ type: "tool.call", callId: "c1", name: "search", input: { q: 123 } });
  await tick();

  assert.equal(executed, false);
  const result = m.sent.at(-1);
  if (!result || result.type !== "tool.result") throw new Error("expected a tool.result");
  assert.ok(result.output && typeof result.output === "object" && "error" in result.output);
});

test("barge-in retracts a tool that honors the abort", async () => {
  const m = mockModel();
  let aborted = false;
  const slow = tool({
    description: "slow",
    inputSchema: z.object({}),
    execute: (_input, { abortSignal }) =>
      new Promise((_resolve, reject) => {
        abortSignal?.addEventListener("abort", () => {
          aborted = true;
          reject(new Error("aborted"));
        });
      }),
  });
  const messages: RealtimeMessage[] = [];
  const cancelled: string[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { slow },
    onMessage: (msg) => messages.push(msg),
    onToolCancel: ({ callId }) => cancelled.push(callId),
  });
  await s.start();
  m.emit({ type: "tool.call", callId: "c1", name: "slow", input: {} });
  await tick();
  assert.equal(s.status, "thinking");

  m.emit({ type: "speech.interrupted" });
  await tick();
  assert.equal(aborted, true);
  assert.equal(s.status, "listening");
  assert.equal(m.sent.find((x) => x.type === "tool.result"), undefined);
  assert.equal(messages.find((x) => x.type === "tool"), undefined);
  // The announced running tool is retracted so the UI doesn't hang on it.
  assert.deepEqual(cancelled, ["c1"]);
});

test("a tool that outlives the barge-in still reaches the record", async () => {
  const m = mockModel();
  // Acts on the world and never looks at the abort signal — the shape of
  // every database-backed tool.
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  const act = tool({
    description: "acts, ignoring the abort",
    inputSchema: z.object({}),
    execute: async () => {
      await gate;
      return { deleted: 3 };
    },
  });
  const messages: RealtimeMessage[] = [];
  const cancelled: string[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { act },
    onMessage: (msg) => messages.push(msg),
    onToolCancel: ({ callId }) => cancelled.push(callId),
  });
  await s.start();
  m.emit({ type: "tool.call", callId: "c1", name: "act", input: {} });
  await tick();
  m.emit({ type: "speech.interrupted" });
  release();
  await tick();

  // What the tool did is done: the round-trip must be on the record even
  // though the call was aborted — only the provider hand-off is cancelled.
  const msg = messages.find((x) => x.type === "tool");
  if (!msg || msg.type !== "tool") throw new Error("expected a tool message");
  assert.equal(msg.callId, "c1");
  assert.deepEqual(msg.output, { deleted: 3 });
  assert.equal(m.sent.find((x) => x.type === "tool.result"), undefined);
  // A message came, so the running render is superseded, not retracted.
  assert.deepEqual(cancelled, []);
});

test("stop waits for an aborted tool to finish acting, and records it", async () => {
  const m = mockModel();
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  const act = tool({
    description: "acts, ignoring the abort",
    inputSchema: z.object({}),
    execute: async () => {
      await gate;
      return { deleted: 3 };
    },
  });
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { act },
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "tool.call", callId: "c1", name: "act", input: {} });
  await tick();

  const stopping = s.stop();
  release();
  await stopping;

  // Whoever loads the transcript next — a successor session included — must
  // see what the tool did, or the record ends on an action that "never
  // happened".
  const msg = messages.find((x) => x.type === "tool");
  if (!msg || msg.type !== "tool") throw new Error("expected a tool message");
  assert.deepEqual(msg.output, { deleted: 3 });
  assert.equal(s.status, "idle");
});

test("stop closes the model and returns to idle", async () => {
  const m = mockModel();
  const s = createRealtimeSession({ model: m.model, ...makeCall() });
  await s.start();
  await s.stop();
  assert.equal(m.closed, true);
  assert.equal(s.status, "idle");
});

test("stop settles the reply that was in flight", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "which customers do I have?" });
  m.emit({ type: "speech.start" });
  m.emit({ type: "transcript.delta", role: "assistant", text: "You have three" });
  await s.stop();

  // Otherwise the transcript ends on the question, and whoever loads it next
  // reads it as still open.
  assert.deepEqual(
    messages.map((msg) => (msg.type === "text" ? [msg.role, msg.text] : msg.type)),
    [
      ["user", "which customers do I have?"],
      ["assistant", "You have three"],
    ],
  );
});

test("stop emits nothing when the assistant had not started replying", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "which customers do I have?" });
  await s.stop();
  assert.equal(messages.length, 1);
});

test("a response that called a tool is not the end of the turn", async () => {
  const m = mockModel();
  const search = tool({
    description: "search",
    inputSchema: z.object({}),
    execute: async () => ({ hits: 1 }),
  });
  let turns = 0;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { search },
    onTurnDone: () => turns++,
  });
  await s.start();

  // The answer the seller waits for spans two responses: the first speaks a
  // filler and calls the tool, the second speaks the result.
  m.emit({ type: "response.start" });
  m.emit({ type: "transcript.delta", role: "assistant", text: "One moment" });
  m.emit({ type: "tool.call", callId: "c1", name: "search", input: {} });
  m.emit({ type: "transcript.done", role: "assistant" });
  m.emit({ type: "response.done" });
  await tick();
  assert.equal(turns, 0);
  assert.equal(s.status, "thinking");

  m.emit({ type: "response.start" });
  // The silence here is the model composing, not an ended turn.
  assert.equal(s.status, "thinking");
  m.emit({ type: "transcript.delta", role: "assistant", text: "I found one" });
  m.emit({ type: "transcript.done", role: "assistant" });
  m.emit({ type: "response.done" });
  await tick();
  assert.equal(turns, 1);
  assert.equal(s.status, "listening");
});

test("a tool still running holds the turn open past its response", async () => {
  const m = mockModel();
  let release: (() => void) | undefined;
  const slow = tool({
    description: "slow",
    inputSchema: z.object({}),
    execute: () => new Promise((resolve) => (release = () => resolve({ ok: true }))),
  });
  let turns = 0;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { slow },
    onTurnDone: () => turns++,
  });
  await s.start();
  m.emit({ type: "response.start" });
  m.emit({ type: "tool.call", callId: "c1", name: "slow", input: {} });
  m.emit({ type: "response.done" });
  await tick();
  assert.equal(turns, 0);

  release?.();
  await tick();
  m.emit({ type: "response.start" });
  m.emit({ type: "response.done" });
  await tick();
  assert.equal(turns, 1);
});

test("audio outliving the response holds the turn until speech stops", async () => {
  const m = mockModel();
  const order: string[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => order.push(msg.type === "text" ? `text:${msg.text}` : msg.type),
    onTurnDone: () => order.push("turn.done"),
  });
  await s.start();
  m.emit({ type: "response.start" });
  m.emit({ type: "speech.start" });
  m.emit({ type: "transcript.delta", role: "assistant", text: "You have three" });
  m.emit({ type: "response.done" });
  await tick();
  assert.deepEqual(order, []);
  assert.equal(s.status, "speaking");

  m.emit({ type: "speech.stop" });
  await tick();
  assert.deepEqual(order, ["text:You have three", "turn.done"]);
  assert.equal(s.status, "listening");
});

test("barge-in ends the turn without an end-of-turn signal", async () => {
  const m = mockModel();
  let turns = 0;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onTurnDone: () => turns++,
  });
  await s.start();
  m.emit({ type: "response.start" });
  m.emit({ type: "speech.start" });
  m.emit({ type: "speech.interrupted" });
  await tick();
  assert.equal(turns, 0);
  assert.equal(s.status, "listening");
});

test("serve() bridges the end of a turn to the wire", async () => {
  const m = mockModel();
  const frames: (string | ArrayBuffer)[] = [];
  const s = createRealtimeSession({ model: m.model, ...makeCall() });
  s.serve((f) => frames.push(f));
  await s.start();
  m.emit({ type: "response.start" });
  m.emit({ type: "transcript.delta", role: "assistant", text: "Ciao" });
  m.emit({ type: "transcript.done", role: "assistant" });
  m.emit({ type: "response.done" });

  const events = frames
    .filter((f): f is string => typeof f === "string")
    .map(decodeServerEvent);
  // The settled turn must reach the client before the signal that nothing
  // else is coming, or a client that stops listening on it loses the answer.
  const messageIdx = events.findIndex((e) => e.t === "message");
  const doneIdx = events.findIndex((e) => e.t === "turn.done");
  assert.ok(messageIdx !== -1);
  assert.ok(doneIdx > messageIdx);
});

test("serve() bridges session events to the wire", async () => {
  const m = mockModel();
  const frames: (string | ArrayBuffer)[] = [];
  const s = createRealtimeSession({ model: m.model, ...makeCall() });
  s.serve((f) => frames.push(f));
  await s.start();
  m.emit({ type: "speech.start" });
  m.emit({ type: "transcript.delta", role: "assistant", text: "Ciao" });
  m.emit({ type: "speech.stop" });

  const events = frames
    .filter((f): f is string => typeof f === "string")
    .map(decodeServerEvent);
  assert.ok(events.some((e) => e.t === "status" && e.status === "speaking"));
  assert.ok(events.some((e) => e.t === "transcript"));
  assert.ok(events.some((e) => e.t === "message" && e.message.type === "text"));
});

test("serve() reports the current status when attached after startup", async () => {
  const m = mockModel();
  const frames: (string | ArrayBuffer)[] = [];
  const s = createRealtimeSession({ model: m.model, ...makeCall() });
  await s.start();

  s.serve((frame) => frames.push(frame));

  assert.deepEqual(
    frames
      .filter((frame): frame is string => typeof frame === "string")
      .map(decodeServerEvent),
    [{ t: "status", status: "listening" }],
  );
});

test("speech.interrupted fires onInterrupted and serve() flushes before the status drops", async () => {
  const m = mockModel();
  let interrupted = 0;
  const frames: (string | ArrayBuffer)[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onInterrupted: () => interrupted++,
  });
  s.serve((f) => frames.push(f));
  await s.start();
  m.emit({ type: "speech.start" });
  m.emit({ type: "speech.interrupted" });
  await tick();

  assert.equal(interrupted, 1);
  assert.equal(s.status, "listening");
  const events = frames
    .filter((f): f is string => typeof f === "string")
    .map(decodeServerEvent);
  // The flush frame must precede "listening" so the client clears buffered
  // audio before the status settles — otherwise the reply plays on.
  const flushIdx = events.findIndex((e) => e.t === "speech.interrupted");
  assert.ok(flushIdx !== -1);
  assert.ok(
    events.slice(flushIdx + 1).some((e) => e.t === "status" && e.status === "listening"),
  );
});

test("serve() relays assistant audio as binary frames", async () => {
  const m = mockModel();
  const frames: (string | ArrayBuffer)[] = [];
  const s = createRealtimeSession({ model: m.model, ...makeCall() });
  s.serve((f) => frames.push(f));
  await s.start();
  m.pushAudio(new ArrayBuffer(8));
  assert.equal(frames.filter((f) => typeof f !== "string").length, 1);
});

test("serve().receive pushes inbound binary as mic audio, ignores non-command strings", async () => {
  const m = mockModel();
  const s = createRealtimeSession({ model: m.model, ...makeCall() });
  const bridge = s.serve(() => {});
  await s.start();
  bridge.receive(new Uint8Array([1, 2, 3, 4]));
  bridge.receive(JSON.stringify({ t: "noop" }));
  bridge.receive("not json");
  assert.equal(m.pushed.length, 1);
  assert.equal(m.pushed[0]?.byteLength, 4);
  assert.equal(m.sent.length, 0);
});

test("sendText finalizes a user turn, forwards it, and requests a response", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => 42,
    onMessage: (msg) => messages.push(msg),
  });

  // Before start: no live handle, so nothing happens.
  s.sendText("troppo presto");
  assert.equal(messages.length, 0);

  await s.start();
  s.sendText("  which customers do I have?  ");

  const msg = messages[0];
  if (!msg || msg.type !== "text") throw new Error("expected a text message");
  assert.equal(msg.role, "user");
  assert.equal(msg.text, "which customers do I have?");
  assert.equal(msg.createdAt, 42);
  assert.deepEqual(m.sent, [
    { type: "text", role: "user", text: "which customers do I have?" },
  ]);
  assert.equal(m.responsesRequested, 1);

  // Blank text is dropped entirely.
  s.sendText("   ");
  assert.equal(messages.length, 1);
  assert.equal(m.responsesRequested, 1);
});

test("serve().receive routes a say frame into the session as a typed user turn", async () => {
  const m = mockModel();
  const frames: (string | ArrayBuffer)[] = [];
  const s = createRealtimeSession({ model: m.model, ...makeCall() });
  const bridge = s.serve((f) => frames.push(f));
  await s.start();
  bridge.receive(encodeClientEvent({ t: "say", text: "ciao" }));

  assert.deepEqual(m.sent, [{ type: "text", role: "user", text: "ciao" }]);
  assert.equal(m.responsesRequested, 1);
  // The finalized user turn goes back out on the wire like any other message.
  const events = frames
    .filter((f): f is string => typeof f === "string")
    .map(decodeServerEvent);
  assert.ok(
    events.some(
      (e) =>
        e.t === "message" &&
        e.message.type === "text" &&
        e.message.role === "user" &&
        e.message.text === "ciao",
    ),
  );
});

test("serve() and creation callbacks both receive events", async () => {
  const m = mockModel();
  const states: VoiceStatus[] = [];
  const frames: string[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onStatus: (st) => states.push(st),
  });
  s.serve((f) => {
    if (typeof f === "string") frames.push(f);
  });
  await s.start();
  assert.ok(states.includes("listening"));
  assert.ok(
    frames.map(decodeServerEvent).some((e) => e.t === "status" && e.status === "listening"),
  );
});

test("relays assistant audio frames through onAudio", async () => {
  const m = mockModel();
  const frames: ArrayBuffer[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onAudio: (pcm) => frames.push(pcm),
  });
  await s.start();
  const frame = new ArrayBuffer(8);
  m.pushAudio(frame);
  assert.equal(frames.length, 1);
  assert.equal(frames[0], frame);
});

// ── Turn timings ────────────────────────────────────────────────────────────

test("an assistant turn carries ttft (from injected text) and total duration", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => t,
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  s.sendText("which customers do I have?");
  t = 1300;
  m.emit({ type: "transcript.delta", role: "assistant", text: "Hai" });
  t = 1450;
  m.emit({ type: "transcript.delta", role: "assistant", text: " two customers" });
  t = 2100;
  m.emit({ type: "transcript.done", role: "assistant" });

  const user = messages[0];
  if (!user || user.type !== "text") throw new Error("expected the user turn");
  assert.equal(user.timings, undefined); // user turns carry no timings

  const reply = messages.at(-1);
  if (!reply || reply.type !== "text") throw new Error("expected the reply");
  assert.equal(reply.role, "assistant");
  // Both run from the end of the user's turn (1000): first assistant signal at
  // 1300, settled at 2100. ttft is inside total, as the shared contract says.
  assert.deepEqual(reply.timings, { ttftMs: 300, totalMs: 1100 });
});

test("ttft settles on first audio when speech precedes the transcript", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => t,
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  s.sendText("ciao");
  t = 1250;
  m.emit({ type: "speech.start" }); // audio arrives before any transcript
  t = 1400;
  m.emit({ type: "transcript.delta", role: "assistant", text: "Ciao" });
  t = 2000;
  m.emit({ type: "transcript.done", role: "assistant" });

  const reply = messages.at(-1);
  if (!reply || reply.type !== "text") throw new Error("expected the reply");
  assert.deepEqual(reply.timings, { ttftMs: 250, totalMs: 1000 });
});

test("a settled mic utterance anchors the next reply's ttft", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => t,
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  m.emit({ type: "transcript.final", role: "user", text: "what time is it?" });
  t = 1500;
  m.emit({ type: "transcript.delta", role: "assistant", text: "Sono" });
  t = 1900;
  m.emit({ type: "transcript.done", role: "assistant" });

  const reply = messages.at(-1);
  if (!reply || reply.type !== "text") throw new Error("expected the reply");
  // The defect this contract closed: ttft used to exceed total, because the two
  // timed disjoint segments instead of nesting.
  assert.deepEqual(reply.timings, { ttftMs: 500, totalMs: 900 });
});

test("every message of one answer measures from the same user turn", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => t,
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  s.sendText("what discount does Northwind get?");
  // A filler while the assistant goes looking...
  t = 1400;
  m.emit({ type: "transcript.delta", role: "assistant", text: "One moment" });
  t = 1900;
  m.emit({ type: "transcript.done", role: "assistant" });
  // ...then the answer itself, seconds later.
  t = 6000;
  m.emit({ type: "transcript.delta", role: "assistant", text: "They get 12%" });
  t = 8000;
  m.emit({ type: "transcript.done", role: "assistant" });

  const spoken = messages.filter((msg) => msg.type === "text" && msg.role === "assistant");
  assert.deepEqual(
    spoken.map((msg) => (msg.type === "text" ? msg.timings : undefined)),
    [
      { ttftMs: 400, totalMs: 900 },
      // Only the first message opens the answer, so only it carries ttft; the
      // last one carries the latency the user actually waited through.
      { totalMs: 7000 },
    ],
  );
});

test("a new user turn restarts the measurement", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => t,
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  s.sendText("prima domanda");
  t = 1500;
  m.emit({ type: "transcript.delta", role: "assistant", text: "Ecco" });
  t = 2000;
  m.emit({ type: "transcript.done", role: "assistant" });

  t = 5000;
  m.emit({ type: "transcript.final", role: "user", text: "seconda domanda" });
  t = 5200;
  m.emit({ type: "transcript.delta", role: "assistant", text: "Anche" });
  t = 5600;
  m.emit({ type: "transcript.done", role: "assistant" });

  const reply = messages.at(-1);
  if (!reply || reply.type !== "text") throw new Error("expected the reply");
  assert.deepEqual(reply.timings, { ttftMs: 200, totalMs: 600 });
});

test("a proactive assistant turn (no user anchor) has no ttftMs", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => t,
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  m.emit({ type: "transcript.delta", role: "assistant", text: "Benvenuto!" });
  t = 1200;
  m.emit({ type: "transcript.done", role: "assistant" });

  const reply = messages.at(-1);
  if (!reply || reply.type !== "text") throw new Error("expected the reply");
  assert.deepEqual(reply.timings, { totalMs: 200 });
});

test("a tool message carries its round-trip duration", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => t,
    tools: {
      clock: tool({
        description: "Read the clock.",
        inputSchema: z.object({}),
        execute: async () => {
          t = 1750; // the clock advances while the tool runs
          return { ok: true };
        },
      }),
    },
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  m.emit({ type: "tool.call", callId: "c1", name: "clock", input: {} });
  await tick();
  await tick();

  const toolMsg = messages.find((msg) => msg.type === "tool");
  if (!toolMsg || toolMsg.type !== "tool") throw new Error("expected a tool message");
  assert.deepEqual(toolMsg.timings, { totalMs: 750 });
});

// ── A corrected user transcript ─────────────────────────────────
// The ASR sends a second `completed` for the SAME conversation item when it
// revises what it heard, and the gateway relays both (verified live: two
// events, one item id, "what were the" then the full sentence). The
// revision belongs to the message already shown.

test("a corrected final for a settled utterance revises it in place", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "what were the", utterance: "i1" });
  m.emit({
    type: "transcript.final",
    role: "user",
    text: "what were the March sales figures",
    utterance: "i1",
  });

  const [first, second] = messages;
  if (first?.type !== "text" || second?.type !== "text") throw new Error("expected text messages");
  assert.equal(messages.length, 2);
  // Same id and stamp — the UI and the store both upsert by id, so the reader
  // sees one message whose text was corrected.
  assert.equal(second.id, first.id);
  assert.equal(second.createdAt, first.createdAt);
  assert.equal(second.text, "what were the March sales figures");
});

test("successive corrections keep revising the same message", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "what", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "what were", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "what were the March sales", utterance: "i1" });

  assert.equal(new Set(messages.map((msg) => msg.id)).size, 1);
  const last = messages.at(-1);
  if (last?.type !== "text") throw new Error("expected a text message");
  assert.equal(last.text, "what were the March sales");
});

test("the same key reused after the assistant answered starts a new turn", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "which customers do I have?", utterance: "i1" });
  m.emit({ type: "transcript.delta", role: "assistant", text: "You have five" });
  m.emit({ type: "transcript.done", role: "assistant" });
  // The wire reuses the key for genuinely new speech — a separate question.
  m.emit({ type: "transcript.final", role: "user", text: "and Northwind?", utterance: "i1" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(users.length, 2);
  assert.notEqual(users[0]?.id, users[1]?.id);
});

test("no revision once the assistant is speaking", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "prima domanda", utterance: "i1" });
  m.emit({ type: "speech.start" }); // audio reached the user
  m.emit({ type: "speech.interrupted" }); // barge-in
  m.emit({ type: "transcript.final", role: "user", text: "seconda domanda", utterance: "i1" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(users.length, 2);
  assert.notEqual(users[0]?.id, users[1]?.id);
});

test("a correction while a tool is still running revises the question in place", async () => {
  // A pending tool is not a perceptible answer: the ASR correcting what it
  // heard while the tool runs still names the question the tool is working
  // on, and must not duplicate it.
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const slow = tool({
    description: "slow",
    inputSchema: z.object({}),
    execute: () => new Promise(() => {}), // never settles
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { slow },
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "cerca Rossi", utterance: "i1" });
  m.emit({ type: "tool.call", callId: "c1", name: "slow", input: {} });
  await tick();
  m.emit({ type: "transcript.final", role: "user", text: "cerca Rossini", utterance: "i1" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(new Set(users.map((msg) => msg.id)).size, 1);
  const last = users.at(-1);
  if (last?.type !== "text") throw new Error("expected a text message");
  assert.equal(last.text, "cerca Rossini");
});

test("a completed tool round-trip makes the reused key a new turn", async () => {
  // Once the round-trip is on the record, the question it answered must not
  // be rewritten under it.
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const search = tool({
    description: "search",
    inputSchema: z.object({}),
    execute: async () => ({ found: true }),
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { search },
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "cerca Rossi", utterance: "i1" });
  m.emit({ type: "tool.call", callId: "c1", name: "search", input: {} });
  await tick();
  assert.ok(messages.some((msg) => msg.type === "tool")); // the result landed
  m.emit({ type: "transcript.update", role: "user", text: "cerca Rossi e Bianchi", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "cerca Rossi e Bianchi", utterance: "i1" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(users.length, 2);
  assert.notEqual(users[0]?.id, users[1]?.id);
});

test("resumed speech that aborts the pending tool folds into the same message", async () => {
  // The barge-in kills the tool before it answered: nothing of it remains,
  // so the continuation still belongs to the message already shown.
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const slow = tool({
    description: "slow",
    inputSchema: z.object({}),
    execute: () => new Promise(() => {}), // never settles
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { slow },
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "first part", utterance: "i1" });
  m.emit({ type: "tool.call", callId: "c1", name: "slow", input: {} });
  await tick();
  m.emit({ type: "speech.interrupted" }); // resume aborts the tool
  m.emit({ type: "transcript.update", role: "user", text: "first part and second", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "first part and second", utterance: "i1" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(new Set(users.map((msg) => msg.id)).size, 1);
  const last = users.at(-1);
  if (last?.type !== "text") throw new Error("expected a text message");
  assert.equal(last.text, "first part and second");
  assert.ok(!messages.some((msg) => msg.type === "tool")); // aborted, never settled
});

test("a late correction repairs its message even while the next utterance streams", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "first sentence", utterance: "i1" });
  // The next utterance already streams when i1's corrected final lands.
  m.emit({ type: "speech.interrupted" });
  m.emit({ type: "transcript.update", role: "user", text: "second sentence", utterance: "i2" });
  m.emit({ type: "transcript.final", role: "user", text: "first sentence complete", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "second sentence", utterance: "i2" });

  const users = messages.filter(
    (msg): msg is Extract<RealtimeMessage, { type: "text" }> => msg.type === "text" && msg.role === "user",
  );
  const first = users[0];
  const repair = users.find((msg, i) => i > 0 && msg.id === first?.id);
  assert.ok(repair, "expected the correction to repair the first message");
  assert.equal(repair?.text, "first sentence complete");
  // The second utterance keeps its own turn, untouched by the repair.
  const second = users.find((msg) => msg.text === "second sentence");
  assert.ok(second);
  assert.notEqual(second?.id, first?.id);
});

test("a wire that appends after the pause cannot lose the head of the message", async () => {
  // The reopened buffer starts from the settled text, so a keyed delta that
  // appends (permitted by the spec) extends the message instead of
  // rebuilding it from the tail alone.
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "first part", utterance: "i1" });
  m.emit({ type: "speech.interrupted" });
  m.emit({ type: "transcript.delta", role: "user", text: " and second", utterance: "i1" });
  m.emit({ type: "transcript.done", role: "user" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(new Set(users.map((msg) => msg.id)).size, 1);
  const last = users.at(-1);
  if (last?.type !== "text") throw new Error("expected a text message");
  assert.equal(last.text, "first part and second");
});

test("a revision re-anchors the reply's latency to the corrected end-of-turn", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  let t = 1000;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    now: () => t,
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "what were the", utterance: "i1" });
  t = 2000;
  m.emit({ type: "transcript.final", role: "user", text: "what were the March sales", utterance: "i1" });
  t = 2300;
  m.emit({ type: "transcript.delta", role: "assistant", text: "Ecco" });
  t = 2500;
  m.emit({ type: "transcript.done", role: "assistant" });

  const reply = messages.at(-1);
  if (reply?.type !== "text") throw new Error("expected the reply");
  // From the correction (2000), not the first partial (1000) — the user was
  // still talking, so the wait they felt starts at the later settle.
  assert.deepEqual(reply.timings, { ttftMs: 300, totalMs: 500 });
});

test("an exact repeat of a settled utterance still changes nothing", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "which customers do I have?", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "which customers do I have?", utterance: "i1" });
  assert.equal(messages.length, 1);
});

test("two genuinely distinct utterances stay two messages", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "what were the", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "March sales figures", utterance: "i2" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(users.length, 2);
  assert.notEqual(users[0]?.id, users[1]?.id);
});

test("a keyless final is unaffected by the revision path", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "prima" });
  m.emit({ type: "transcript.final", role: "user", text: "seconda" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(users.length, 2);
  assert.notEqual(users[0]?.id, users[1]?.id);
});

test("a late final still repairs a turn that settled early", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  // i1 streams, then i2 opens without i1's final ever arriving: i1 settles
  // early from its buffer. i1's final lands afterwards and repairs it.
  m.emit({ type: "transcript.update", role: "user", text: "first sentence", utterance: "i1" });
  m.emit({ type: "transcript.update", role: "user", text: "second sentence", utterance: "i2" });
  m.emit({ type: "transcript.final", role: "user", text: "first sentence complete", utterance: "i1" });

  const first = messages[0];
  const repair = messages.find((msg, i) => i > 0 && msg.type === "text" && msg.id === first?.id);
  if (first?.type !== "text" || repair?.type !== "text") throw new Error("expected a repair");
  assert.equal(repair.text, "first sentence complete");
});

test("an assistant turn is never revised by a repeated final", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "assistant", text: "prima", utterance: "a1" });
  m.emit({ type: "transcript.final", role: "assistant", text: "seconda", utterance: "a1" });

  const replies = messages.filter((msg) => msg.type === "text" && msg.role === "assistant");
  assert.equal(replies.length, 2);
  assert.notEqual(replies[0]?.id, replies[1]?.id);
});

// ── An utterance resuming across a pause ────────────────────────
// Verified live: a pause long enough for VAD to commit the turn settles it by
// final and a response starts; resuming speech fires barge-in (cancelling the
// answer before any audio) and the ASR keeps growing the SAME item — live
// snapshots first, then a corrected terminal final carrying the whole
// utterance. Those snapshots must reopen the settled turn, or they mint the
// sibling the final then settles under, and the reader sees the utterance
// twice.

test("snapshots resuming a settled utterance reopen it, and the final revises in place", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  // The pause: VAD commits, the turn settles by its own final.
  m.emit({ type: "transcript.update", role: "user", text: "yesterday I visited", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "yesterday I visited the customer", utterance: "i1" });
  // Speech resumes: barge-in with nothing audible in flight, then the same
  // item keeps growing.
  m.emit({ type: "speech.interrupted" });
  m.emit({
    type: "transcript.update",
    role: "user",
    text: "yesterday I visited the customer and we talked about pricing",
    utterance: "i1",
  });
  m.emit({
    type: "transcript.final",
    role: "user",
    text: "yesterday I visited the customer and we talked about pricing",
    utterance: "i1",
  });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(new Set(users.map((msg) => msg.id)).size, 1);
  const last = users.at(-1);
  if (last?.type !== "text") throw new Error("expected a text message");
  assert.equal(last.text, "yesterday I visited the customer and we talked about pricing");
  assert.equal(last.createdAt, (users[0] as { createdAt: number }).createdAt);
});

test("every pause of a longer utterance keeps revising the same message", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "first part", utterance: "i1" });
  m.emit({ type: "speech.interrupted" });
  m.emit({ type: "transcript.update", role: "user", text: "first part and second", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "first part and second", utterance: "i1" });
  m.emit({ type: "speech.interrupted" });
  m.emit({ type: "transcript.update", role: "user", text: "first part and second and third", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "first part and second and third", utterance: "i1" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(new Set(users.map((msg) => msg.id)).size, 1);
  const last = users.at(-1);
  if (last?.type !== "text") throw new Error("expected a text message");
  assert.equal(last.text, "first part and second and third");
});

test("snapshots under a settled key after the assistant spoke stay a new turn", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "which customers do I have?", utterance: "i1" });
  m.emit({ type: "speech.start" }); // the answer audibly began
  m.emit({ type: "speech.interrupted" }); // barge-in
  // The wire reuses the key for new speech — the reader heard an answer in
  // between, so this is a genuine new turn.
  m.emit({ type: "transcript.update", role: "user", text: "one more thing", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "one more thing", utterance: "i1" });

  const users = messages.filter((msg) => msg.type === "text" && msg.role === "user");
  assert.equal(users.length, 2);
  assert.notEqual(users[0]?.id, users[1]?.id);
});

test("turnMetadata stamps the clock: at connect, before a typed turn, after a settled spoken turn", async () => {
  const m = mockModel();
  let t = 0;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    turnMetadata: () => `clock:${++t}`,
  });
  await s.start();
  // Connect-time stamp, right after the seed replay: covers the first turn.
  assert.deepEqual(m.sent, [{ type: "text", role: "user", text: "clock:1" }]);

  // A typed turn's stamp precedes the injected text, as in the text drive.
  s.sendText("ciao");
  assert.deepEqual(m.sent.slice(1), [
    { type: "text", role: "user", text: "clock:2" },
    { type: "text", role: "user", text: "ciao" },
  ]);

  // A spoken turn settling re-stamps, covering the next turn.
  m.emit({ type: "transcript.final", role: "user", text: "quanto ad agosto?" });
  assert.deepEqual(m.sent.at(-1), { type: "text", role: "user", text: "clock:3" });
});

test("a revised spoken turn does not re-stamp the clock", async () => {
  const m = mockModel();
  let t = 0;
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    turnMetadata: () => `clock:${++t}`,
  });
  await s.start();
  m.emit({ type: "transcript.final", role: "user", text: "quanto ad agosto", utterance: "u1" });
  const stamps = () => m.sent.filter((i) => i.type === "text" && i.text.startsWith("clock:")).length;
  assert.equal(stamps(), 2); // connect + settle
  // The ASR corrects the same utterance in place: same turn, no new stamp.
  m.emit({ type: "transcript.final", role: "user", text: "quanto ad agosto?", utterance: "u1" });
  assert.equal(stamps(), 2);
});

// A pause while the model writes its own answer is not a lookup. Rendering
// them alike made a plain greeting sound and look exactly like a query to the
// ERP: the seller heard the tool cue and had nothing to look at.
test("composing while the model writes, thinking only while a tool runs", async () => {
  const m = mockModel();
  const s = createRealtimeSession({ model: m.model, ...makeCall() });
  await s.start();
  assert.equal(s.status, "listening");

  // A greeting: produced, then spoken, with no tool anywhere in it.
  m.emit({ type: "response.start" });
  assert.equal(s.status, "composing");
  m.emit({ type: "speech.start" });
  assert.equal(s.status, "speaking");
  m.emit({ type: "speech.stop" });
  await tick();
  assert.equal(s.status, "listening");
});

test("a tool result hands the turn back to composing, not to thinking", async () => {
  const m = mockModel();
  const search = tool({
    description: "search",
    inputSchema: z.object({}),
    execute: async () => ({ ok: true }),
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { search },
  });
  await s.start();

  m.emit({ type: "response.start" });
  assert.equal(s.status, "composing");
  m.emit({ type: "tool.call", callId: "c1", name: "search", input: {} });
  // The tool is out: the wait now has a cause outside the model.
  assert.equal(s.status, "thinking");
  await tick();

  // It came back. Nothing is running any more, so writing the answer from the
  // result is composing again — the cue stops with the lookup, not with the turn.
  m.emit({ type: "response.done" });
  await tick();
  assert.equal(s.status, "composing");
});

test("a tool waits for the utterance it is answering to reach the transcript", async () => {
  // The model answers from the audio and can reach the tool before the ASR has
  // settled the words. A tool that reads the conversation would then read one
  // turn short, with no error anywhere.
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const seen: string[] = [];
  const read = tool({
    description: "reads the transcript",
    inputSchema: z.object({}),
    execute: () => {
      seen.push(messages.map((msg) => (msg.type === "text" ? msg.text : "")).join("|"));
      return "ok";
    },
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { read },
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  m.emit({ type: "speech.interrupted" }); // the mic hears speech: an utterance opens
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await tick();
  assert.equal(seen.length, 0, "the tool ran before its question was on the record");

  m.emit({ type: "transcript.final", role: "user", text: "which customers do I have?", utterance: "i1" });
  await tick();
  assert.deepEqual(seen, ["which customers do I have?"]);
});

test("with no utterance open a tool runs without waiting", async () => {
  const m = mockModel();
  const ran: string[] = [];
  const quick = tool({
    description: "quick",
    inputSchema: z.object({}),
    execute: () => {
      ran.push("c1");
      return "ok";
    },
  });
  const s = createRealtimeSession({ model: m.model, ...makeCall(), tools: { quick } });
  await s.start();

  m.emit({ type: "transcript.final", role: "user", text: "which customers do I have?", utterance: "i1" });
  m.emit({ type: "tool.call", callId: "c1", name: "quick", input: {} });
  await tick();
  assert.deepEqual(ran, ["c1"]);
});

test("a barge-in while a tool waits for the transcript drops the tool", async () => {
  const m = mockModel();
  const ran: string[] = [];
  const read = tool({
    description: "reads",
    inputSchema: z.object({}),
    execute: () => {
      ran.push("c1");
      return "ok";
    },
  });
  const s = createRealtimeSession({ model: m.model, ...makeCall(), tools: { read } });
  await s.start();

  m.emit({ type: "speech.interrupted" });
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await tick();
  m.emit({ type: "speech.interrupted" }); // the user carries on: the response is gone
  await tick();
  m.emit({ type: "transcript.final", role: "user", text: "anzi, lascia stare", utterance: "i1" });
  await tick();

  assert.deepEqual(ran, [], "an aborted tool must not run once its response is cancelled");
  assert.equal(
    m.sent.some((o) => o.type === "tool.result"),
    false,
  );
});

test("an utterance nobody closes holds the tool for as long as the call lasts", async () => {
  // What the wait costs when it is not capped, stated plainly: a provider that
  // opens an utterance and never closes it strands every tool behind it. This
  // is why each provider closes the utterance it opened even when the ASR
  // produced nothing — the guarantee lives there, not in a
  // deadline here.
  const m = mockModel();
  const ran: string[] = [];
  const read = tool({
    description: "reads",
    inputSchema: z.object({}),
    execute: () => {
      ran.push("c1");
      return "ok";
    },
  });
  const s = createRealtimeSession({ model: m.model, ...makeCall(), tools: { read } });
  await s.start();

  m.emit({ type: "speech.interrupted" }); // opens an utterance
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await new Promise((r) => setTimeout(r, 60));
  assert.deepEqual(ran, [], "nothing releases a wait that is never closed");

  // Any of the three closing routes lets it go. This is the one a provider
  // reaches when the ASR heard nothing at all.
  m.emit({ type: "transcript.done", role: "user" });
  await tick();
  assert.deepEqual(ran, ["c1"]);
});

test("the user's transcript.done ends the utterance even with no words in it", async () => {
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const ran: string[] = [];
  const read = tool({
    description: "reads",
    inputSchema: z.object({}),
    execute: () => {
      ran.push("c1");
      return "ok";
    },
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { read },
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  m.emit({ type: "speech.interrupted" });
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await tick();
  m.emit({ type: "transcript.done", role: "user" });
  await tick();

  assert.deepEqual(ran, ["c1"]);
  assert.equal(
    messages.some((msg) => msg.type === "text" && msg.role === "user"),
    false,
    "an utterance with no words must not become a turn",
  );
});

test("stopping the session releases a tool still waiting for the transcript", async () => {
  // Teardown awaits the running tools, so an uncapped wait must not be able to
  // hold `stop()` open. The abort that ends the tool ends the wait with it.
  const m = mockModel();
  const ran: string[] = [];
  const read = tool({
    description: "reads",
    inputSchema: z.object({}),
    execute: () => {
      ran.push("c1");
      return "ok";
    },
  });
  const s = createRealtimeSession({ model: m.model, ...makeCall(), tools: { read } });
  await s.start();

  m.emit({ type: "speech.interrupted" }); // opens an utterance nothing will close
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await tick();

  await s.stop(); // must not hang
  assert.deepEqual(ran, [], "a tool released by teardown must not go on to run");
});

test("a late transcript for the previous utterance does not release the current one", async () => {
  // The bug the gate exists to stop, reachable through the gate itself: the
  // user speaks A, the ASR lags, the user cuts in with B. A tool called for B
  // must not be released by A's final finally landing.
  const m = mockModel();
  const messages: RealtimeMessage[] = [];
  const seen: string[] = [];
  const read = tool({
    description: "reads",
    inputSchema: z.object({}),
    execute: () => {
      seen.push(messages.filter((x) => x.type === "text" && x.role === "user").length.toString());
      return "ok";
    },
  });
  const s = createRealtimeSession({
    model: m.model,
    ...makeCall(),
    tools: { read },
    onMessage: (msg) => messages.push(msg),
  });
  await s.start();

  m.emit({ type: "speech.interrupted", utterance: "a" }); // utterance A opens
  // A pause mid-sentence re-arms the detector: the same utterance, not a new one.
  m.emit({ type: "speech.interrupted", utterance: "a" });
  m.emit({ type: "speech.interrupted", utterance: "b" }); // the user cuts in
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await tick();

  m.emit({ type: "transcript.final", role: "user", text: "prima domanda", utterance: "a" });
  await tick();
  assert.deepEqual(seen, [], "A's late transcript must not release a tool waiting for B");

  m.emit({ type: "transcript.final", role: "user", text: "seconda domanda", utterance: "b" });
  await tick();
  assert.deepEqual(seen, ["2"], "the tool must see both turns");
});

test("a lost wire releases the utterances it was going to end", async () => {
  const m = mockModel();
  const ran: string[] = [];
  const read = tool({
    description: "reads",
    inputSchema: z.object({}),
    execute: () => {
      ran.push("c1");
      return "ok";
    },
  });
  const s = createRealtimeSession({ model: m.model, ...makeCall(), tools: { read } });
  await s.start();

  m.emit({ type: "speech.interrupted" });
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await tick();
  assert.deepEqual(ran, []);

  m.emit({ type: "transport", status: "disconnected" });
  await tick();
  assert.deepEqual(ran, ["c1"], "no transcript can arrive over a wire that is gone");
});

test("a restarted session does not inherit an utterance from the last one", async () => {
  const m = mockModel();
  const ran: string[] = [];
  const read = tool({
    description: "reads",
    inputSchema: z.object({}),
    execute: () => {
      ran.push("c1");
      return "ok";
    },
  });
  const s = createRealtimeSession({ model: m.model, ...makeCall(), tools: { read } });
  await s.start();
  m.emit({ type: "speech.interrupted" }); // stopped mid-utterance
  await s.stop();

  await s.start();
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await tick();
  assert.deepEqual(ran, ["c1"], "the new session starts with nothing pending");
});

test("one call's open utterance does not hold another call's tools", async () => {
  // Two sellers on the phone at once. The utterance state is the session's
  // own, so a turn still being transcribed in one call must not stop a tool
  // in the other.
  const a = mockModel();
  const b = mockModel();
  const ran: string[] = [];
  const read = (who: string) =>
    tool({
      description: "reads",
      inputSchema: z.object({}),
      execute: () => {
        ran.push(who);
        return "ok";
      },
    });
  const sa = createRealtimeSession({ model: a.model, ...makeCall(), tools: { read: read("a") } });
  const sb = createRealtimeSession({ model: b.model, ...makeCall(), tools: { read: read("b") } });
  await sa.start();
  await sb.start();

  a.emit({ type: "speech.interrupted" }); // only A has someone talking
  a.emit({ type: "tool.call", callId: "a1", name: "read", input: {} });
  b.emit({ type: "tool.call", callId: "b1", name: "read", input: {} });
  await tick();
  assert.deepEqual(ran, ["b"], "B must not wait on A's utterance");

  // And A's own transcript releases only A.
  a.emit({ type: "transcript.final", role: "user", text: "which customers do I have?", utterance: "i1" });
  await tick();
  assert.deepEqual(ran, ["b", "a"]);
});

test("speech that re-arms mid-sentence is one utterance, not several", async () => {
  // What hung a real call: a pause mid-sentence makes the detector fire again,
  // but the ASR still delivers one transcript for the one utterance. Counting
  // the starts left opens nothing would ever answer, and every tool after that
  // waited for the rest of the call.
  const m = mockModel();
  const ran: string[] = [];
  const read = tool({
    description: "reads",
    inputSchema: z.object({}),
    execute: () => {
      ran.push("c1");
      return "ok";
    },
  });
  const s = createRealtimeSession({ model: m.model, ...makeCall(), tools: { read } });
  await s.start();

  m.emit({ type: "speech.interrupted", utterance: "i1" });
  m.emit({ type: "speech.interrupted", utterance: "i1" });
  m.emit({ type: "speech.interrupted", utterance: "i1" });
  m.emit({ type: "transcript.final", role: "user", text: "what tools do you have?", utterance: "i1" });
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await tick();

  assert.deepEqual(ran, ["c1"], "one transcript must answer every start of its own utterance");
});

test("an ending that names no utterance releases whatever is open", async () => {
  // A provider that keys the two ends differently, or an ASR reattributing its
  // terminal event, must not be able to strand a tool. A stale read is
  // recoverable; a session that stops answering is not.
  const m = mockModel();
  const ran: string[] = [];
  const read = tool({
    description: "reads",
    inputSchema: z.object({}),
    execute: () => {
      ran.push("c1");
      return "ok";
    },
  });
  const s = createRealtimeSession({ model: m.model, ...makeCall(), tools: { read } });
  await s.start();

  m.emit({ type: "speech.interrupted", utterance: "i1" });
  m.emit({ type: "tool.call", callId: "c1", name: "read", input: {} });
  await tick();
  assert.deepEqual(ran, []);

  m.emit({ type: "transcript.final", role: "user", text: "boh", utterance: "somethingelse" });
  await tick();
  assert.deepEqual(ran, ["c1"]);
});
