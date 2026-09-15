import { test } from "node:test";
import assert from "node:assert/strict";
import { tool, type UIMessage } from "ai";
import { z } from "zod";
import { convertToRealtimeSeed, toUIMessages } from "./convert";
import type { RealtimeMessage } from "./session";

test("converts text turns, dropping empty text", async () => {
  const messages: UIMessage[] = [
    { id: "1", role: "user", parts: [{ type: "text", text: "ciao" }] },
    { id: "2", role: "assistant", parts: [{ type: "text", text: "  " }] },
    { id: "3", role: "assistant", parts: [{ type: "text", text: "hello" }] },
  ];
  assert.deepEqual(await convertToRealtimeSeed(messages), [
    { type: "text", role: "user", text: "ciao" },
    { type: "text", role: "assistant", text: "hello" },
  ]);
});

test("joins consecutive text parts of one message into a single seed turn", async () => {
  const messages: UIMessage[] = [
    {
      id: "1",
      role: "user",
      parts: [
        { type: "text", text: "<message_metadata>\ntime: 2026-08-20T10:00:00+02:00\n</message_metadata>" },
        { type: "text", text: "ciao" },
      ],
    },
  ];
  assert.deepEqual(await convertToRealtimeSeed(messages), [
    {
      type: "text",
      role: "user",
      text: "<message_metadata>\ntime: 2026-08-20T10:00:00+02:00\n</message_metadata>\nciao",
    },
  ]);
});

test("expands a settled tool round-trip into a tool.call + tool.result pair", async () => {
  const messages: UIMessage[] = [
    { id: "1", role: "user", parts: [{ type: "text", text: "what time is it" }] },
    {
      id: "2",
      role: "assistant",
      parts: [
        {
          type: "dynamic-tool",
          toolName: "get_time",
          toolCallId: "c1",
          state: "output-available",
          input: { tz: "Rome" },
          output: { now: "15:00" },
        },
      ],
    },
  ];
  assert.deepEqual(await convertToRealtimeSeed(messages), [
    { type: "text", role: "user", text: "what time is it" },
    { type: "tool.call", callId: "c1", name: "get_time", input: { tz: "Rome" } },
    { type: "tool.result", callId: "c1", output: { now: "15:00" } },
  ]);
});

test("maps a tool result through the tool's toModelOutput when tools are passed", async () => {
  const tools = {
    get_time: tool({
      description: "time",
      inputSchema: z.object({}),
      execute: async () => ({ now: "15:00" }),
      // The model should see a terse string, not the raw object.
      toModelOutput: ({ output }) => ({
        type: "text" as const,
        value: `the time is ${(output as { now: string }).now}`,
      }),
    }),
  };
  const messages: UIMessage[] = [
    {
      id: "1",
      role: "assistant",
      parts: [
        {
          type: "dynamic-tool",
          toolName: "get_time",
          toolCallId: "c1",
          state: "output-available",
          input: {},
          output: { now: "15:00" },
        },
      ],
    },
  ];
  assert.deepEqual(await convertToRealtimeSeed(messages, { tools }), [
    { type: "tool.call", callId: "c1", name: "get_time", input: {} },
    { type: "tool.result", callId: "c1", output: "the time is 15:00" },
  ]);
});

test("carries an output-error through as { error }, skipping toModelOutput", async () => {
  const messages: UIMessage[] = [
    {
      id: "1",
      role: "assistant",
      parts: [
        {
          type: "dynamic-tool",
          toolName: "get_time",
          toolCallId: "c1",
          state: "output-error",
          input: {},
          errorText: "boom",
        },
      ],
    },
  ];
  assert.deepEqual(await convertToRealtimeSeed(messages), [
    { type: "tool.call", callId: "c1", name: "get_time", input: {} },
    { type: "tool.result", callId: "c1", output: { error: "boom" }, isError: true },
  ]);
});

test("skips tool parts with no settled outcome and non-user/assistant roles", async () => {
  const messages: UIMessage[] = [
    { id: "0", role: "system", parts: [{ type: "text", text: "be brief" }] },
    {
      id: "1",
      role: "assistant",
      parts: [
        {
          type: "dynamic-tool",
          toolName: "get_time",
          toolCallId: "c1",
          state: "input-available",
          input: {},
        },
      ],
    },
  ];
  assert.deepEqual(await convertToRealtimeSeed(messages), []);
});

test("round-trips a typed tool-<name> part the way getToolName resolves it", async () => {
  const messages: UIMessage[] = [
    {
      id: "1",
      role: "assistant",
      parts: [
        {
          type: "tool-search",
          toolCallId: "c9",
          state: "output-available",
          input: { q: "x" },
          output: { hits: 1 },
        },
      ],
    },
  ] as unknown as UIMessage[];
  assert.deepEqual(await convertToRealtimeSeed(messages), [
    { type: "tool.call", callId: "c9", name: "search", input: { q: "x" } },
    { type: "tool.result", callId: "c9", output: { hits: 1 } },
  ]);
});

test("toUIMessages: a text message becomes a settled text part with createdAt metadata", () => {
  const messages: RealtimeMessage[] = [
    { type: "text", id: "m1", role: "user", text: "ciao", createdAt: 1000 },
  ];
  assert.deepEqual(toUIMessages(messages), [
    {
      id: "m1",
      role: "user",
      parts: [{ type: "text", text: "ciao", state: "done" }],
      metadata: { createdAt: 1000 },
    },
  ]);
});

test("toUIMessages: a tool round-trip becomes a dynamic-tool part on an assistant turn", () => {
  const messages: RealtimeMessage[] = [
    {
      type: "tool",
      id: "c1",
      callId: "c1",
      name: "search",
      input: { q: "x" },
      output: { hits: 1 },
      createdAt: 2000,
    },
  ];
  assert.deepEqual(toUIMessages(messages), [
    {
      id: "c1",
      role: "assistant",
      parts: [
        {
          type: "dynamic-tool",
          toolName: "search",
          toolCallId: "c1",
          state: "output-available",
          input: { q: "x" },
          output: { hits: 1 },
        },
      ],
      metadata: { createdAt: 2000 },
    },
  ]);
});

test("toUIMessages round-trips with convertToRealtimeSeed (text + tool fidelity)", async () => {
  const messages: RealtimeMessage[] = [
    { type: "text", id: "u1", role: "user", text: "what time is it", createdAt: 1 },
    {
      type: "tool",
      id: "c1",
      callId: "c1",
      name: "get_time",
      input: { tz: "Rome" },
      output: { now: "15:00" },
      createdAt: 2,
    },
  ];
  // RealtimeMessage → UIMessage → RealtimeOutbound seed keeps the turns intact.
  assert.deepEqual(await convertToRealtimeSeed(toUIMessages(messages)), [
    { type: "text", role: "user", text: "what time is it" },
    { type: "tool.call", callId: "c1", name: "get_time", input: { tz: "Rome" } },
    { type: "tool.result", callId: "c1", output: { now: "15:00" } },
  ]);
});

test("toUIMessages: turn timings ride into metadata next to createdAt", () => {
  const messages: RealtimeMessage[] = [
    {
      type: "text",
      id: "a1",
      role: "assistant",
      text: "Ciao",
      createdAt: 1000,
      timings: { ttftMs: 300, totalMs: 800 },
    },
    {
      type: "tool",
      id: "c1",
      callId: "c1",
      name: "search",
      input: { q: "x" },
      output: { hits: 1 },
      createdAt: 2000,
      timings: { totalMs: 750 },
    },
  ];
  const [text, toolTurn] = toUIMessages(messages);
  assert.deepEqual(text!.metadata, {
    createdAt: 1000,
    timings: { ttftMs: 300, totalMs: 800 },
  });
  assert.deepEqual(toolTurn!.metadata, {
    createdAt: 2000,
    timings: { totalMs: 750 },
  });
});
