import { test } from "node:test";
import assert from "node:assert/strict";
import { VoiceChat } from "./voice-chat";
import type { VoiceAudio } from "./audio";
import type { ServerEvent } from "../protocol";
import type {
  VoiceCredentials,
  VoiceTransportClose,
  VoiceTransport,
  VoiceTransportHandlers,
} from "./transport";

function mockTransport() {
  const handlers: VoiceTransportHandlers[] = [];
  let credentials: VoiceCredentials | null = null;
  const sent: ArrayBuffer[] = [];
  let closed = false;
  let connections = 0;
  const transport: VoiceTransport = {
    async connect(h, c) {
      connections++;
      handlers.push(h);
      credentials = c;
      return {
        sendAudio: (p) => sent.push(p),
        close: () => {
          closed = true;
        },
      };
    },
  };
  return {
    transport,
    sent,
    get credentials() {
      return credentials;
    },
    get closed() {
      return closed;
    },
    get connections() {
      return connections;
    },
    open: () => handlers.at(-1)?.onOpen(),
    event: (e: ServerEvent) => handlers.at(-1)?.onEvent(e),
    audio: (p: ArrayBuffer) => handlers.at(-1)?.onAudio(p),
    error: (m: string) => handlers.at(-1)?.onError(m),
    closeWire: (
      connection = handlers.length - 1,
      event: VoiceTransportClose = {
        code: 1006,
        reason: "",
        wasClean: false,
      },
    ) => handlers[connection]?.onClose(event),
  };
}

function stubAudio() {
  let onFrame: ((pcm: ArrayBuffer) => void) | null = null;
  const played: ArrayBuffer[] = [];
  let muted = false;
  let playing = false;
  let cleared = 0;
  let endHandler: (() => void) | null = null;
  const audio: VoiceAudio = {
    start: async (cb) => {
      onFrame = cb;
    },
    play: (pcm) => played.push(pcm),
    clearPlayback: () => {
      cleared++;
      playing = false;
    },
    stop: () => {},
    setMuted: (m) => {
      muted = m;
    },
    inputLevel: () => 0,
    outputLevel: () => 0,
    isPlaying: () => playing,
    onPlaybackEnd: (h) => {
      endHandler = h;
    },
  };
  return {
    audio,
    played,
    emitFrame: (b: ArrayBuffer) => onFrame?.(b),
    setPlaying: (p: boolean) => {
      playing = p;
    },
    drain: () => {
      playing = false;
      endHandler?.();
    },
    get cleared() {
      return cleared;
    },
    get muted() {
      return muted;
    },
  };
}

function delayedFirstAudio() {
  let firstFrame: ((pcm: ArrayBuffer) => void) | undefined;
  let rejectFirst: ((error: Error) => void) | undefined;
  let starts = 0;
  let stops = 0;
  const firstStart = new Promise<void>((_, reject) => {
    rejectFirst = reject;
  });
  const audio: VoiceAudio = {
    start: async (onFrame) => {
      starts++;
      if (starts === 1) {
        firstFrame = onFrame;
        return firstStart;
      }
    },
    play: () => {},
    stop: () => {
      stops++;
    },
    setMuted: () => {},
    inputLevel: () => 0,
    outputLevel: () => 0,
  };
  return {
    audio,
    emitFirstFrame: (pcm: ArrayBuffer) => firstFrame?.(pcm),
    rejectFirst: (error: Error) => rejectFirst?.(error),
    get starts() {
      return starts;
    },
    get stops() {
      return stops;
    },
  };
}

function makeChat() {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({ transport: t.transport, audio: a.audio });
  return { chat, t, a };
}

const timers = () => new Promise<void>((resolve) => setTimeout(resolve, 1));
const retry = async () => {
  await timers();
  await timers();
};

test("resolves auth with the session id and applies the credentials", async () => {
  const t = mockTransport();
  const a = stubAudio();
  let authedId = "";
  const chat = new VoiceChat({
    id: "sess-123",
    transport: t.transport,
    audio: a.audio,
    auth: () => async ({ id }) => {
      authedId = id;
      return { params: { id, t: "tok" } };
    },
  });
  await chat.start();
  assert.equal(chat.id, "sess-123");
  assert.equal(authedId, "sess-123");
  assert.deepEqual(t.credentials?.params, { id: "sess-123", t: "tok" });
});

test("connects with empty credentials when no auth is given", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({ transport: t.transport, audio: a.audio });
  await chat.start();
  assert.deepEqual(t.credentials, {});
});

test("generates a session id when none is provided", () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({ transport: t.transport, audio: a.audio });
  assert.match(chat.id, /[A-Za-z0-9-]{8,}/);
});

test("accumulates a streaming turn into one message, keyed by id", async () => {
  const { chat, t } = makeChat();
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });
  t.event({ t: "transcript", id: "a1", role: "assistant", delta: "Ci", full: "Ci", createdAt: 1000 });
  t.event({ t: "transcript", id: "a1", role: "assistant", delta: "ao", full: "Ciao", createdAt: 1000 });

  // Mid-stream: the text part is marked `streaming`, like an AI SDK text-delta.
  const live = chat.messages[0]?.parts[0];
  assert.equal(live?.type === "text" ? live.text : "", "Ciao");
  assert.equal(live?.type === "text" ? live.state : undefined, "streaming");

  t.event({ t: "message", message: { type: "text", id: "a1", role: "assistant", text: "Ciao", createdAt: 1000 } });

  assert.equal(chat.status, "listening");
  assert.equal(chat.messages.length, 1);
  const m = chat.messages[0];
  if (!m) throw new Error("expected one message");
  assert.equal(m.role, "assistant");
  const part = m.parts[0];
  if (!part || part.type !== "text") throw new Error("expected a text part");
  assert.equal(part.text, "Ciao");
  // Settled: the same id flips to `done`, like an AI SDK text-end.
  assert.equal(part.state, "done");
  // The turn's createdAt rides in metadata as the cross-mode ordering key.
  assert.equal((m.metadata as { createdAt?: number } | undefined)?.createdAt, 1000);
});

test("a tool message becomes a dynamic-tool UI part", async () => {
  const { chat, t } = makeChat();
  await chat.start();
  t.open();
  t.event({
    t: "message",
    message: { type: "tool", id: "c1", callId: "c1", name: "search", input: { q: "x" }, output: { ok: true }, createdAt: 2000 },
  });
  const m = chat.messages.at(-1);
  if (!m) throw new Error("expected a message");
  const part = m.parts[0] as { type: string; toolName?: string; output?: unknown };
  assert.equal(part.type, "dynamic-tool");
  assert.equal(part.toolName, "search");
  assert.deepEqual(part.output, { ok: true });
});

test("a pending tool renders running, and tool.cancel retracts it", async () => {
  const { chat, t } = makeChat();
  await chat.start();
  t.open();
  t.event({ t: "tool.pending", callId: "c1", name: "search", input: { description: "X" }, createdAt: 2000 });
  const running = chat.messages.at(-1)?.parts[0] as { type: string; state?: string } | undefined;
  assert.equal(running?.type, "dynamic-tool");
  assert.equal(running?.state, "input-available");

  t.event({ t: "tool.cancel", callId: "c1" });
  assert.equal(chat.messages.length, 0);
});

test("ending the session drops a running tool but keeps a completed one", async () => {
  const { chat, t } = makeChat();
  await chat.start();
  t.open();
  t.event({
    t: "message",
    message: { type: "tool", id: "done", callId: "done", name: "search", input: {}, output: { ok: true }, createdAt: 1000 },
  });
  t.event({ t: "tool.pending", callId: "run", name: "search", input: { description: "X" }, createdAt: 2000 });
  assert.equal(chat.messages.length, 2);

  chat.stop();
  assert.equal(chat.messages.length, 1);
  assert.equal(chat.messages[0]?.id, "done");
});

test("plays inbound audio and sends mic frames over the transport", async () => {
  const { chat, t, a } = makeChat();
  await chat.start();
  t.open(); // wires audio.start → sendAudio
  await timers();

  t.audio(new ArrayBuffer(8));
  assert.equal(a.played.length, 1);

  a.emitFrame(new ArrayBuffer(4));
  assert.equal(t.sent.length, 1);
  assert.equal(t.sent[0]?.byteLength, 4);
});

test("notifies status subscribers and toggles mute", async () => {
  const { chat, t, a } = makeChat();
  let statusChanges = 0;
  chat.subscribeStatus(() => statusChanges++);
  await chat.start();
  t.open();
  t.event({ t: "status", status: "speaking" });
  chat.setMuted(true);

  assert.ok(statusChanges > 0);
  assert.equal(a.muted, true);
  assert.equal(chat.muted, true);
  assert.equal(chat.status, "speaking");
});

test("messages subscription is separate from status", async () => {
  const { chat, t } = makeChat();
  let messageChanges = 0;
  let statusChanges = 0;
  chat.subscribeMessages(() => messageChanges++);
  chat.subscribeStatus(() => statusChanges++);
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" }); // status only
  t.event({ t: "transcript", id: "a1", role: "assistant", delta: "hi", full: "hi", createdAt: 1000 }); // messages only

  assert.equal(messageChanges, 1);
  assert.ok(statusChanges >= 1);
});

test("status flows through while the wire is live", async () => {
  const { chat, t } = makeChat();
  await chat.start();
  t.open();
  t.event({ t: "status", status: "speaking" });
  assert.equal(chat.status, "speaking");
});

test("an unexpected wire close reconnects with fresh credentials", async () => {
  const t = mockTransport();
  const a = stubAudio();
  let authCalls = 0;
  const chat = new VoiceChat({
    id: "recovering-session",
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
    auth: () => async () => ({ params: { t: `ticket-${++authCalls}` } }),
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  t.closeWire();
  await retry();

  assert.equal(t.connections, 2);
  assert.equal(authCalls, 2);
  assert.equal(chat.status, "connecting");
});

test("a provider death closed by the relay reconnects the live session", async () => {
  const t = mockTransport();
  const a = stubAudio();
  let authCalls = 0;
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
    auth: () => async () => ({ params: { t: `ticket-${++authCalls}` } }),
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  // The relay reports the dead runtime, then closes the wire (code 4003).
  t.event({ t: "status", status: "error" });
  t.closeWire(0, { code: 4003, reason: "voice runtime lost", wasClean: true });
  await retry();

  assert.equal(t.connections, 2);
  assert.equal(authCalls, 2);
  assert.equal(chat.status, "connecting");
});

test("a nonfatal provider error keeps the live session open", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  t.event({ t: "error", message: "one response failed" });
  await retry();

  assert.equal(t.closed, false);
  assert.equal(t.connections, 1);
  assert.equal(chat.status, "listening");
  assert.equal(chat.error, "one response failed");
});

test("intentional stop does not reconnect when the wire closes", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  chat.stop();
  t.closeWire();
  await timers();

  assert.equal(t.connections, 1);
  assert.equal(chat.status, "idle");
});

test("a stale close cannot retire a replacement connection", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  chat.stop();
  await chat.start();

  t.closeWire(0);
  await timers();

  assert.equal(t.connections, 2);
  assert.equal(chat.status, "connecting");
});

test("a superseded session does not reconnect", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  t.open();
  t.event({ t: "superseded" });
  t.closeWire();
  await timers();

  assert.equal(t.connections, 1);
  assert.equal(chat.status, "idle");
  assert.match(chat.error ?? "", /another voice session/);
});

test("a superseded close code suppresses recovery when its event was lost", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  t.closeWire(0, { code: 4001, reason: "superseded", wasClean: true });
  await retry();

  assert.equal(t.connections, 1);
  assert.equal(chat.status, "idle");
  assert.match(chat.error ?? "", /another voice session/);
});

test("a stale superseded close cannot poison the replacement", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0, 0],
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });
  t.closeWire();
  await retry();

  t.closeWire(0, { code: 4001, reason: "superseded", wasClean: true });
  t.open();
  t.event({ t: "status", status: "listening" });
  t.closeWire();
  await retry();

  assert.equal(t.connections, 3);
  assert.equal(chat.status, "connecting");
});

test("a takeover close during stale microphone teardown settles as superseded", async () => {
  const t = mockTransport();
  const a = delayedFirstAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  t.open();
  await timers();
  t.event({ t: "status", status: "listening" });

  t.closeWire(0, { code: 4001, reason: "superseded", wasClean: true });
  a.rejectFirst(new Error("stale microphone failure"));
  await retry();

  assert.equal(t.connections, 1);
  assert.equal(chat.status, "idle");
  assert.match(chat.error ?? "", /another voice session/);
});

test("a nonfatal runtime error does not become the disconnect reason", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [],
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  t.event({ t: "error", message: "one response failed" });
  t.closeWire(0, { code: 4003, reason: "voice runtime lost", wasClean: true });
  await timers();

  assert.equal(chat.status, "error");
  assert.equal(chat.error, "voice connection lost");
});

test("a pre-live runtime error keeps its explanatory message", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  t.open();

  // The relay's start-failure sequence: explanatory error frame, then a close
  // with the start-failed code.
  t.event({ t: "error", message: "provider authentication failed" });
  t.closeWire(0, { code: 4002, reason: "start failed", wasClean: true });
  await timers();

  assert.equal(t.connections, 1);
  assert.equal(chat.status, "error");
  assert.equal(chat.error, "provider authentication failed");
});

test("a start failure during recovery keeps the relay's reason when retries exhaust", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  t.closeWire();
  await retry();
  assert.equal(t.connections, 2);
  t.event({ t: "error", message: "voice session could not start" });
  t.closeWire(1, { code: 4002, reason: "start failed", wasClean: true });
  await retry();

  assert.equal(chat.status, "error");
  assert.equal(chat.error, "voice session could not start");
});

test("brief liveness does not reset the reconnect backoff", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  // The provider comes up and dies right away, twice: the second loss finds
  // the backoff spent instead of looping at zero delay forever.
  t.closeWire();
  await retry();
  t.open();
  t.event({ t: "status", status: "listening" });
  t.closeWire();
  await retry();

  assert.equal(t.connections, 2);
  assert.equal(chat.status, "error");
  assert.equal(chat.error, "voice connection lost");
});

test("sustained liveness resets the reconnect backoff", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
    reconnectResetAfterMs: 0,
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  t.closeWire();
  await retry();
  t.open();
  t.event({ t: "status", status: "listening" });
  await retry(); // the connection outlives the reset window
  t.closeWire();
  await retry();

  assert.equal(t.connections, 3);
  assert.equal(chat.status, "connecting");
});

test("exhausted reconnect attempts end in an explicit error", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0, 0],
  });
  await chat.start();
  t.open();
  t.event({ t: "status", status: "listening" });

  t.closeWire();
  await retry();
  t.closeWire();
  await retry();
  t.closeWire();
  await retry();

  assert.equal(t.connections, 3);
  assert.equal(chat.status, "error");
  assert.equal(chat.error, "voice connection lost");
});

test("an initial connection failure is terminal without retrying", async () => {
  const t = mockTransport();
  const a = stubAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0, 0],
  });
  await chat.start();

  t.closeWire();
  await timers();

  assert.equal(t.connections, 1);
  assert.equal(chat.status, "error");
});

test("delayed microphone startup cannot affect a replacement connection", async () => {
  const t = mockTransport();
  const a = delayedFirstAudio();
  const chat = new VoiceChat({
    transport: t.transport,
    audio: a.audio,
    reconnectDelays: [0],
  });
  await chat.start();
  t.open();
  await timers();
  t.event({ t: "status", status: "listening" });

  t.closeWire();
  await timers();
  assert.equal(t.connections, 1);
  a.emitFirstFrame(new ArrayBuffer(4));
  a.rejectFirst(new Error("stale microphone failure"));
  await timers();
  await timers();
  assert.equal(t.connections, 2);
  t.open();
  await timers();
  t.event({ t: "status", status: "listening" });

  assert.equal(t.sent.length, 0);
  assert.equal(a.starts, 2);
  assert.ok(a.stops >= 1);
  assert.equal(chat.status, "listening");
  assert.equal(chat.error, null);
});

test("manual reopen waits for delayed microphone teardown", async () => {
  const t = mockTransport();
  const a = delayedFirstAudio();
  const chat = new VoiceChat({ transport: t.transport, audio: a.audio });
  await chat.start();
  t.open();
  await timers();

  chat.stop();
  const reopened = chat.start();
  await timers();
  assert.equal(t.connections, 1);

  a.rejectFirst(new Error("stale microphone failure"));
  await reopened;

  assert.equal(t.connections, 2);
  assert.equal(chat.status, "connecting");
  assert.equal(chat.error, null);
});

test("holds 'speaking' until audio playback drains", async () => {
  const { chat, t, a } = makeChat();
  await chat.start();
  t.open();
  t.event({ t: "status", status: "speaking" });
  a.setPlaying(true); // assistant audio buffered and still playing
  assert.equal(chat.status, "speaking");

  // Provider signals generation-end → wire says listening, but audio plays on.
  t.event({ t: "status", status: "listening" });
  assert.equal(chat.status, "speaking"); // deferred

  a.drain(); // playback buffer empties
  assert.equal(chat.status, "listening"); // now settles
});

test("speech.interrupted flushes playback and drops to 'listening' immediately", async () => {
  const { chat, t, a } = makeChat();
  await chat.start();
  t.open();
  t.event({ t: "status", status: "speaking" });
  a.setPlaying(true); // assistant audio buffered and still playing
  assert.equal(chat.status, "speaking");

  // User barges in: the buffered reply must be dropped, not drained.
  t.event({ t: "speech.interrupted" });
  assert.equal(a.cleared, 1);
  assert.equal(chat.status, "listening"); // not deferred behind playback
});

test("applies a non-speaking status immediately when audio isn't playing", async () => {
  const { chat, t } = makeChat();
  await chat.start();
  t.open();
  t.event({ t: "status", status: "speaking" });
  t.event({ t: "status", status: "listening" }); // nothing playing → no deferral
  assert.equal(chat.status, "listening");
});

test("stop() closes the transport connection", async () => {
  const { chat, t } = makeChat();
  await chat.start();
  t.open();
  chat.stop();
  assert.equal(t.closed, true);
  assert.equal(chat.status, "idle");
});

test("snapshot identity is stable across no-op updates", () => {
  const { chat } = makeChat();
  const s1 = chat.snapshot;
  chat.setMuted(false); // already false → no-op → same identity
  assert.equal(chat.snapshot, s1);
  chat.setMuted(true); // real change → new identity
  assert.notEqual(chat.snapshot, s1);
});
