import { test } from "node:test";
import assert from "node:assert/strict";
import {
  decodeClientEvent,
  decodeServerEvent,
  encodeClientEvent,
  encodeServerEvent,
  type ServerEvent,
} from "./protocol";

test("ServerEvent round-trips through encode/decode", () => {
  const events: ServerEvent[] = [
    { t: "status", status: "speaking" },
    { t: "transcript", id: "m1", role: "assistant", delta: "Ci", full: "Ciao", createdAt: 1000 },
    { t: "tool.pending", callId: "c1", name: "search", input: { q: "x" }, createdAt: 1001 },
    { t: "tool.cancel", callId: "c1" },
    { t: "message", message: { type: "text", id: "m1", role: "user", text: "ciao", createdAt: 1000 } },
    {
      t: "message",
      message: { type: "tool", id: "c1", callId: "c1", name: "search", input: { q: "x" }, output: { ok: true }, createdAt: 1001 },
    },
    { t: "error", message: "boom" },
  ];
  for (const e of events) {
    assert.deepEqual(decodeServerEvent(encodeServerEvent(e)), e);
  }
});

test("ClientEvent round-trips; anything else decodes to null", () => {
  const say = { t: "say", text: "which customers do I have?" } as const;
  assert.deepEqual(decodeClientEvent(encodeClientEvent(say)), say);

  assert.equal(decodeClientEvent("not json"), null);
  assert.equal(decodeClientEvent("42"), null);
  assert.equal(decodeClientEvent("null"), null);
  assert.equal(decodeClientEvent(JSON.stringify({ t: "noop" })), null);
  assert.equal(decodeClientEvent(JSON.stringify({ t: "say" })), null);
  assert.equal(decodeClientEvent(JSON.stringify({ t: "say", text: 7 })), null);
});
