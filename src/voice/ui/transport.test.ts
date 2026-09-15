import { test } from "node:test";
import assert from "node:assert/strict";
import { WebSocketVoiceTransport } from "./transport";
import type { ServerEvent } from "../protocol";
import type { VoiceTransportClose } from "./transport";

type Listener = (ev?: any) => void;

class FakeWS {
  binaryType = "";
  readyState = 1;
  sent: (string | ArrayBuffer)[] = [];
  private listeners = new Map<string, Listener[]>();

  send(data: string | ArrayBuffer) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
    this.fire("close", { code: 1000, reason: "finished", wasClean: true });
  }
  addEventListener(type: "message", l: (ev: { data: unknown }) => void): void;
  addEventListener(type: "open" | "error", l: () => void): void;
  addEventListener(type: "close", l: (ev: VoiceTransportClose) => void): void;
  addEventListener(type: string, l: Listener): void {
    const a = this.listeners.get(type) ?? [];
    a.push(l);
    this.listeners.set(type, a);
  }

  fire(type: string, ev?: unknown) {
    for (const l of [...(this.listeners.get(type) ?? [])]) l(ev);
  }
  open() {
    this.fire("open");
  }
  message(obj: unknown) {
    this.fire("message", { data: JSON.stringify(obj) });
  }
  binary(buf: ArrayBuffer) {
    this.fire("message", { data: buf });
  }
}

test("decodes string frames to events and binary frames to audio", async () => {
  const fake = new FakeWS();
  const transport = new WebSocketVoiceTransport({
    url: "ws://relay/session",
    createWebSocket: () => fake,
  });
  const events: ServerEvent[] = [];
  const audio: ArrayBuffer[] = [];
  let opened = false;
  let closed: VoiceTransportClose | undefined;
  const conn = await transport.connect({
    onOpen: () => (opened = true),
    onEvent: (e) => events.push(e),
    onAudio: (a) => audio.push(a),
    onClose: (event) => (closed = event),
    onError: () => {},
  }, {});

  assert.equal(fake.binaryType, "arraybuffer");
  fake.open();
  assert.equal(opened, true);

  fake.message({ t: "status", status: "listening" });
  fake.binary(new ArrayBuffer(4));
  assert.equal(events.length, 1);
  assert.equal(events[0]?.t, "status");
  assert.equal(audio.length, 1);

  conn.sendAudio(new ArrayBuffer(2));
  assert.equal(fake.sent.filter((x) => typeof x !== "string").length, 1);

  fake.close();
  assert.deepEqual(closed, {
    code: 1000,
    reason: "finished",
    wasClean: true,
  });
});

test("applies credentials: params → URL query, protocols → subprotocols", async () => {
  const fake = new FakeWS();
  let connectedUrl = "";
  let connectedProtocols: string[] | undefined;
  const transport = new WebSocketVoiceTransport({
    url: "ws://relay/voice/session",
    createWebSocket: (u, p) => {
      connectedUrl = u;
      connectedProtocols = p;
      return fake;
    },
  });
  await transport.connect(
    {
      onOpen: () => {},
      onEvent: () => {},
      onAudio: () => {},
      onClose: () => {},
      onError: () => {},
    },
    { params: { id: "c1", t: "tok" }, protocols: ["ticket.tok"] },
  );
  assert.match(connectedUrl, /id=c1/);
  assert.match(connectedUrl, /t=tok/);
  assert.deepEqual(connectedProtocols, ["ticket.tok"]);
});

test("drops malformed JSON frames without throwing", async () => {
  const fake = new FakeWS();
  const transport = new WebSocketVoiceTransport({
    url: "ws://x",
    createWebSocket: () => fake,
  });
  const events: ServerEvent[] = [];
  await transport.connect({
    onOpen: () => {},
    onEvent: (e) => events.push(e),
    onAudio: () => {},
    onClose: () => {},
    onError: () => {},
  }, {});
  fake.fire("message", { data: "{not json" });
  assert.equal(events.length, 0);
});

test("does not send when the socket is not open", async () => {
  const fake = new FakeWS();
  fake.readyState = 0; // CONNECTING
  const transport = new WebSocketVoiceTransport({
    url: "ws://x",
    createWebSocket: () => fake,
  });
  const conn = await transport.connect({
    onOpen: () => {},
    onEvent: () => {},
    onAudio: () => {},
    onClose: () => {},
    onError: () => {},
  }, {});
  conn.sendAudio(new ArrayBuffer(2));
  assert.equal(fake.sent.length, 0);
});
