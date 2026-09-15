import assert from "node:assert/strict";
import { test } from "node:test";
import { MockLanguageModelV4, convertArrayToReadableStream } from "ai/test";
import type {
  RealtimeCall,
  RealtimeEvent,
  RealtimeModelV1,
} from "./voice";
import z from "zod";
import {
  Assistant,
  createAssistant,
  createRegistry,
  harnessTool,
  InMemorySessionStorage,
  role,
  type AssistantPrepareArgs,
} from "./index";

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

function fakeRealtimeModel() {
  const state: {
    emit?: (event: RealtimeEvent) => void;
    call?: RealtimeCall;
    connects: number;
    closes: number;
  } = { connects: 0, closes: 0 };
  const model: RealtimeModelV1 = {
    specificationVersion: "realtime-v1",
    provider: "fake",
    modelId: "fake-voice",
    async connect(args) {
      state.connects += 1;
      state.emit = args.emit;
      state.call = args.call;
      args.emit({ type: "transport", status: "connected" });
      return {
        send: () => {},
        pushAudio: () => {},
        requestResponse: () => {},
        close: async () => {
          state.closes += 1;
        },
      };
    },
  };
  return { model, state };
}

const helper = role({
  name: "helper",
  systemPrompt: "You are a helper.",
});

test("Assistant prepares context and runs a durable text turn", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hello") }),
  });
  const storage = new InMemorySessionStorage();
  const controller = new AbortController();
  const seen: string[] = [];
  const assistant = createAssistant<{ userId: string }>()({
    async prepare({ sessionId, scope }) {
      seen.push(`prepare:${sessionId}:${scope.userId}`);
      return { registry: {}, storage, context: { greeting: "hello" } };
    },
    text({ context }) {
      seen.push(`text:${context.greeting}`);
      return { model, role: helper() };
    },
    voice() {
      throw new Error("not used");
    },
  });

  const session = await assistant.session({
    sessionId: "conversation-1",
    scope: { userId: "user-1" },
    signal: controller.signal,
  });
  const turn = await session.prompt("Hi");
  await assert.rejects(session.voice(), /active text operation/);
  for await (const _ of turn.textStream) {
    // Drain the public stream while the harness independently assembles it.
  }
  await turn.committed;

  const secondTurn = await session.prompt("Again");
  for await (const _ of secondTurn.textStream) {
    // Drain the second turn.
  }
  await secondTurn.committed;

  assert.deepEqual(seen, [
    "prepare:conversation-1:user-1",
    "text:hello",
    "text:hello",
  ]);
  assert.equal(model.doStreamCalls[0]?.abortSignal, controller.signal);
  assert.deepEqual(
    (await storage.loadMessages("conversation-1")).map(
      (message) => message.role,
    ),
    ["user", "assistant", "user", "assistant"],
  );
});

test("Assistant voice resolves after startup and stop is a durability barrier", async () => {
  const { model, state } = fakeRealtimeModel();
  const storage = new InMemorySessionStorage();
  const controller = new AbortController();
  const assistant = new Assistant<
    { userId: string },
    { userName: string },
    Record<string, never>
  >({
    prepare: async () => ({
      registry: {},
      storage,
      context: { userName: "Ada" },
    }),
    text() {
      throw new Error("not used");
    },
    voice({ context }) {
      assert.equal(context.userName, "Ada");
      return { model, role: helper() };
    },
  });

  const session = await assistant.session({
    sessionId: "conversation-2",
    scope: { userId: "user-2" },
    signal: controller.signal,
  });
  const call = await session.voice();

  assert.equal(state.connects, 1);
  assert.equal(call.status, "listening");
  assert.equal("start" in call, false);
  await assert.rejects(session.prompt("overlap"), /active voice operation/);

  state.emit?.({ type: "transcript.final", role: "user", text: "Hello" });
  state.emit?.({ type: "transcript.delta", role: "assistant", text: "Hi" });
  controller.abort();
  await call.stop();
  await call.stop();

  assert.equal(state.closes, 1);
  assert.deepEqual(
    (await storage.loadMessages("conversation-2")).map(
      (message) => message.role,
    ),
    ["user", "assistant"],
  );
});

test("Assistant propagates preparation failures without resolving a channel", async () => {
  let resolvedText = false;
  const assistant = new Assistant<unknown, never, Record<string, never>>({
    prepare(_args: AssistantPrepareArgs<unknown>) {
      throw new Error("conversation unavailable");
    },
    text() {
      resolvedText = true;
      throw new Error("unreachable");
    },
    voice() {
      throw new Error("unreachable");
    },
  });

  await assert.rejects(
    assistant.session({
      sessionId: "conversation-3",
      scope: {},
    }),
    /conversation unavailable/,
  );
  assert.equal(resolvedText, false);
});

test("Assistant propagates history and commit failures", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hello") }),
  });
  class FailingStorage extends InMemorySessionStorage {
    failLoad = true;
    failAssistantSave = false;

    override async loadMessages(sessionId: string) {
      if (this.failLoad) throw new Error("history unavailable");
      return super.loadMessages(sessionId);
    }

    override async saveMessages(
      sessionId: string,
      messages: Parameters<InMemorySessionStorage["saveMessages"]>[1],
    ) {
      if (
        this.failAssistantSave &&
        messages.some((message) => message.role === "assistant")
      ) {
        throw new Error("commit unavailable");
      }
      return super.saveMessages(sessionId, messages);
    }
  }
  const storage = new FailingStorage();
  const assistant = new Assistant<unknown, undefined, Record<string, never>>({
    prepare: async () => ({ registry: {}, storage, context: undefined }),
    text: async () => ({ model, role: helper() }),
    voice() {
      throw new Error("not used");
    },
  });
  const session = await assistant.session({
    sessionId: "conversation-4",
    scope: {},
  });

  await assert.rejects(session.prompt("Hi"), /history unavailable/);

  storage.failLoad = false;
  storage.failAssistantSave = true;
  const turn = await session.prompt("Hi");
  for await (const _ of turn.textStream) {
    // Drain the public stream before observing the durability failure.
  }
  await assert.rejects(turn.committed, /commit unavailable/);
});

test("Assistant cancels voice preparation before connecting and releases the session", async () => {
  const { model: voiceModel, state } = fakeRealtimeModel();
  const textModel = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hello") }),
  });
  let releaseLoad: (() => void) | undefined;
  let markLoadStarted: (() => void) | undefined;
  const loadStarted = new Promise<void>((resolve) => {
    markLoadStarted = resolve;
  });
  class DelayedStorage extends InMemorySessionStorage {
    delayNextLoad = true;

    override async loadMessages(sessionId: string) {
      if (this.delayNextLoad) {
        this.delayNextLoad = false;
        markLoadStarted?.();
        await new Promise<void>((resolve) => {
          releaseLoad = resolve;
        });
      }
      return super.loadMessages(sessionId);
    }
  }
  const storage = new DelayedStorage();
  const assistant = createAssistant<unknown>()({
    prepare: async () => ({ registry: {}, storage, context: undefined }),
    text: async () => ({ model: textModel, role: helper() }),
    voice: async () => ({ model: voiceModel, role: helper() }),
  });
  const session = await assistant.session({
    sessionId: "conversation-5",
    scope: {},
  });
  const controller = new AbortController();
  const opening = session.voice({ abortSignal: controller.signal });
  await loadStarted;
  controller.abort();
  releaseLoad?.();

  await assert.rejects(opening, (error) => error === controller.signal.reason);
  assert.equal(state.connects, 0);

  const turn = await session.prompt("still available");
  for await (const _ of turn.textStream) {
    // Drain the recovery turn.
  }
  await turn.committed;
});

test("Assistant hands the durable transcript between text and voice", async () => {
  const { model: voiceModel, state } = fakeRealtimeModel();
  const textModel = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("written reply") }),
  });
  const storage = new InMemorySessionStorage();
  const assistant = createAssistant<unknown>()({
    prepare: async () => ({ registry: {}, storage, context: undefined }),
    text: async () => ({ model: textModel, role: helper() }),
    voice: async () => ({ model: voiceModel, role: helper() }),
  });
  const session = await assistant.session({
    sessionId: "conversation-6",
    scope: {},
  });

  const textTurn = await session.prompt("written question");
  for await (const _ of textTurn.textStream) {
    // Drain the text turn before switching drives.
  }
  await textTurn.committed;

  const call = await session.voice();
  assert.deepEqual(
    call.initialMessages.map((message) => message.role),
    ["user", "assistant"],
  );
  assert.ok(
    state.call?.seed.some(
      (item) => item.type === "text" && item.text.includes("written question"),
    ),
  );
  state.emit?.({ type: "transcript.final", role: "user", text: "spoken question" });
  state.emit?.({ type: "transcript.delta", role: "assistant", text: "spoken reply" });
  await call.stop();

  const resumedText = await session.prompt("continue in text");
  for await (const _ of resumedText.textStream) {
    // Drain the resumed text turn.
  }
  await resumedText.committed;
  assert.match(
    JSON.stringify(textModel.doStreamCalls.at(-1)?.prompt),
    /spoken question.*spoken reply/s,
  );
});

test("Assistant voice reports deferred persistence failure from stop", async () => {
  const { model, state } = fakeRealtimeModel();
  class FailingVoiceStorage extends InMemorySessionStorage {
    override async saveMessages(
      sessionId: string,
      messages: Parameters<InMemorySessionStorage["saveMessages"]>[1],
    ) {
      if (messages.some((message) => message.role === "assistant")) {
        throw new Error("voice commit unavailable");
      }
      return super.saveMessages(sessionId, messages);
    }
  }
  const assistant = createAssistant<unknown>()({
    prepare: async () => ({
      registry: {},
      storage: new FailingVoiceStorage(),
      context: undefined,
    }),
    text() {
      throw new Error("not used");
    },
    voice: async () => ({ model, role: helper() }),
  });
  const session = await assistant.session({
    sessionId: "conversation-7",
    scope: {},
  });
  const call = await session.voice();
  state.emit?.({ type: "transcript.delta", role: "assistant", text: "Hi" });
  state.emit?.({ type: "transcript.done", role: "assistant" });
  await new Promise((resolve) => setTimeout(resolve, 10));

  await assert.rejects(call.stop(), /voice commit unavailable/);
});

test("Assistant voice snapshots seeded messages before startup events", async () => {
  const storage = new InMemorySessionStorage();
  const model: RealtimeModelV1 = {
    specificationVersion: "realtime-v1",
    provider: "startup-message",
    modelId: "startup-message",
    async connect(args) {
      args.emit({
        type: "transcript.final",
        role: "user",
        text: "during startup",
      });
      return {
        send: () => {},
        pushAudio: () => {},
        requestResponse: () => {},
        close: async () => {},
      };
    },
  };
  const assistant = createAssistant<unknown>()({
    prepare: async () => ({ registry: {}, storage, context: undefined }),
    text() {
      throw new Error("not used");
    },
    voice: async () => ({ model, role: helper() }),
  });
  const session = await assistant.session({
    sessionId: "conversation-8",
    scope: {},
  });

  const call = await session.voice();

  assert.deepEqual(call.initialMessages, []);
  await call.stop();
  assert.equal((await storage.loadMessages("conversation-8")).length, 1);
});

test("one per-session context serves both the tools and the drives", async () => {
  const seen: Array<Record<string, unknown>> = [];
  const { registry, role: boundRole } = createRegistry({
    lookupOrder: harnessTool({
      description: "Look up an order.",
      inputSchema: z.object({ id: z.string() }),
      contextSchema: z.object({ userId: z.string() }),
      execute: async (
        _input: { id: string },
        { context }: { context: { userId: string } },
      ) => {
        seen.push(context);
        return "ok";
      },
    }),
  });
  const support = boundRole({
    name: "support",
    argsSchema: z.object({ displayName: z.string() }),
    systemPrompt: "You help {{displayName}}.",
    tools: ["lookupOrder"],
  });

  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? convertArrayToReadableStream([
              { type: "stream-start" as const, warnings: [] },
              {
                type: "tool-call" as const,
                toolCallId: "c1",
                toolName: "lookupOrder",
                input: JSON.stringify({ id: "42" }),
              },
              {
                type: "finish" as const,
                finishReason: { unified: "tool-calls" as const, raw: undefined },
                usage: {
                  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
                  outputTokens: { total: 1, text: 1, reasoning: 0 },
                },
              },
            ])
          : textStream("done"),
    }),
  });

  const storage = new InMemorySessionStorage();
  let rolePrompt = "";
  const assistant = createAssistant<{ userId: string }>()({
    async prepare({ scope }) {
      // Resolved once; handed to the registry and to the drives alike.
      const context = {
        userId: scope.userId,
        displayName: "Ada",
        secret: "must not reach a tool",
      };
      return { registry: registry(context), storage, context };
    },
    text({ context }) {
      rolePrompt = context.displayName;
      return { model, role: support({ displayName: context.displayName }) };
    },
    voice() {
      throw new Error("not used");
    },
  });

  const session = await assistant.session({
    sessionId: "one-context",
    scope: { userId: "u1" },
  });
  const result = await session.prompt("where is order 42?");
  const reader = result.textStream.getReader();
  while (!(await reader.read()).done) {
    /* drain */
  }
  await result.committed;

  // The drive read what it needed from the same object...
  assert.equal(rolePrompt, "Ada");
  // ...and the tool received only the key its own schema declared.
  assert.deepEqual(seen, [{ userId: "u1" }]);
});
