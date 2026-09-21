import assert from "node:assert/strict";
import { test } from "node:test";
import { tool } from "ai";
import { MockLanguageModelV4, convertArrayToReadableStream } from "ai/test";
import z from "zod";
import {
  codeModeTool,
  createRegistry,
  DIRECT_TOOL_CALL,
  harnessTool,
  init,
  role,
  skill,
  InMemorySessionStorage,
  skillLoaderTool,
  type BoundSkill,
  type SessionMessage,
  type SkillSource,
} from "./index";

// A mock model that always replies with one line of text and records every
// call. We assert on `doStreamCalls[i].tools` — the tools the SDK actually
// exposed to the model on step i, which is exactly the harness's active set.
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

function toolNames(call: { tools?: Array<{ name: string }> }): string[] {
  return (call.tools ?? []).map((t) => t.name).sort();
}

async function drain(result: { textStream: AsyncIterable<string> }) {
  for await (const _ of result.textStream) {
    /* consume */
  }
}

const registry = {
  readFile: tool({
    description: "Read a file.",
    inputSchema: z.object({ path: z.string() }),
    execute: async () => "ok",
  }),
  createArtifact: tool({
    description: "Create an artifact.",
    inputSchema: z.object({ title: z.string() }),
    execute: async () => "made",
  }),
};

const helper = role({
  name: "helper",
  systemPrompt: "You are a helper.",
  tools: ["readFile"], // baseline active set — createArtifact is registered but OFF
});

const artifacts = skill({
  name: "artifacts",
  description: "Create artifacts on request.",
  instructions: "When active, use createArtifact to persist results.",
  tools: ["createArtifact"], // toggled ON when the skill is invoked
});

/** Inline static source over ready skills — the shape the ai package's
 *  `staticSkillsSource` provides (the harness owns only the concept). The
 *  skills' name literals type the source, which types `session.skill`. */
function staticSource<TName extends string>(
  ...held: BoundSkill<TName>[]
): SkillSource<TName> {
  return {
    list: async () =>
      held.map(({ name, description }) => ({ name, description })),
    read: async (name) => {
      const found = held.find((s) => s.name === name);
      if (!found) return null;
      return {
        name: found.name,
        description: found.description,
        instructions: await found.resolveInstructions(),
        tools: [...found.tools],
        toolCallers: found.toolCallers,
      };
    },
  };
}

test("role exposes only its baseline tools; a skill toggles more on (sticky)", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hello") }),
  });
  const storage = new InMemorySessionStorage();

  const harness = await init({
    registry: loaderRegistry,
    model,
    role: helper(),
    loadableSkills: [staticSource(artifacts())],
    storage,
  });
  const session = await harness.session({ sessionId: "s1" });

  // Turn 1: role baseline only.
  await drain(await session.prompt("hi"));
  assert.deepEqual(session.activeTools.slice().sort(), ["readFile"]);
  assert.deepEqual(toolNames(model.doStreamCalls[0]!), ["readFile"]);

  // Invoke the skill by name: resolved through the source, its tool is now
  // in the active set.
  await session.skill("artifacts");
  assert.deepEqual(session.activeTools.slice().sort(), [
    "createArtifact",
    "readFile",
  ]);

  // Turn 2: the model now sees both tools (skill toggle is sticky).
  await drain(await session.prompt("make one"));
  assert.deepEqual(toolNames(model.doStreamCalls[1]!), [
    "createArtifact",
    "readFile",
  ]);

  // Persistence: user(hi), assistant, skill-init, user(make one), assistant.
  await new Promise((r) => setTimeout(r, 50));
  const saved = await storage.loadMessages("s1");
  assert.equal(saved.length, 5);
  assert.deepEqual(
    saved.map((m) => m.role),
    ["user", "assistant", "user", "user", "assistant"],
  );
  // The skill-init message carries the marker the model will read.
  const skillInit = saved[2]!;
  const text = skillInit.parts
    .map((p) => (p.type === "text" ? p.text : ""))
    .join("");
  assert.match(text, /<skill-init name="artifacts">/);
});

function promptText(call: { prompt?: unknown }): string {
  return JSON.stringify(call.prompt ?? null);
}

/** Everything the model can read on a call: the prompt plus the tool
 *  descriptions. The code-mode catalog is asserted against this, so the tests
 *  hold whichever of the two carries it. */
function modelFacingText(call: { prompt?: unknown; tools?: unknown }): string {
  const tools = Array.isArray(call.tools) ? call.tools : [];
  return [
    promptText(call),
    ...tools.map((t) => (t as { description?: string }).description ?? ""),
  ].join("\n");
}

test("each prompt injects <message_metadata> with the time for the model only", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("ok") }),
  });
  const storage = new InMemorySessionStorage();

  const harness = await init({
    registry,
    model,
    role: helper(),
    storage,
  });
  const session = await harness.session({ sessionId: "m1" });

  await drain(await session.prompt("what time is it?"));

  // The model sees the metadata block plus an ISO-8601 timestamp ahead of the
  // user's text.
  const sent = promptText(model.doStreamCalls[0]!);
  assert.match(sent, /<message_metadata>/);
  assert.match(sent, /time: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
  assert.match(sent, /what time is it\?/);

  // Persistence (and thus the UI) keeps the raw user text — no metadata leaks in.
  await new Promise((r) => setTimeout(r, 50));
  const saved = await storage.loadMessages("m1");
  const userText = saved
    .filter((m) => m.role === "user")
    .flatMap((m) => m.parts)
    .map((p) => (p.type === "text" ? p.text : ""))
    .join("");
  assert.equal(userText, "what time is it?");
  assert.doesNotMatch(userText, /<message_metadata>/);
});

test("a prompt's timezone is rendered as local time and frozen on the message", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("ok") }),
  });
  const storage = new InMemorySessionStorage();
  const harness = await init({ registry, model, role: helper(), storage });
  const session = await harness.session({ sessionId: "tz1" });

  await drain(await session.prompt("ciao", { timezone: "Europe/Rome" }));

  const sent = promptText(model.doStreamCalls[0]!);
  assert.match(sent, /timezone: Europe\/Rome/);
  // Local wall-clock with a numeric offset, not a trailing Z.
  assert.match(sent, /time: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}/);

  // The zone is frozen on the message (metadata round-trips through storage), so
  // a later replay re-renders the same local time regardless of the new request.
  await new Promise((r) => setTimeout(r, 50));
  const saved = await storage.loadMessages("tz1");
  const user = saved.find((m) => m.role === "user");
  assert.equal((user?.metadata as { timezone?: string }).timezone, "Europe/Rome");
});

test("metadata is stable across turns so the cached prefix holds", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("ok") }),
  });
  const harness = await init({ registry, model, role: helper() });
  const session = await harness.session({ sessionId: "m2" });

  await drain(await session.prompt("first"));
  // A later turn must not change how the first turn was rendered — otherwise the
  // prompt prefix shifts and provider-side prefix caching misses.
  await new Promise((r) => setTimeout(r, 5));
  await drain(await session.prompt("second"));

  // The first `time:` in each prompt belongs to the oldest (first) user turn.
  const firstTime = (s: string) => s.match(/time: ([0-9T:.Z-]+)/)?.[1];
  const turn1 = firstTime(promptText(model.doStreamCalls[0]!));
  const turn2 = firstTime(promptText(model.doStreamCalls[1]!));
  assert.ok(turn1, "turn 1 carries a metadata timestamp");
  assert.equal(turn1, turn2, "first turn's metadata is byte-identical on replay");
});

test("config reasoning effort is passed through to the model on every step", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("ok") }),
  });
  const harness = await init({
    registry,
    model,
    role: helper(),
    reasoning: "high",
  });
  const session = await harness.session({ sessionId: "r1" });

  await drain(await session.prompt("think hard"));
  assert.equal(model.doStreamCalls[0]!.reasoning, "high");
});

test("a subsession inherits the parent's reasoning effort and can override it", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("ok") }),
  });
  const harness = await init({
    registry,
    model,
    role: helper(),
    reasoning: "low",
  });
  const root = await harness.session({ sessionId: "r2" });

  const inherited = await root.subsession({ sessionId: "r2-a" });
  await drain(await inherited.prompt("x"));
  assert.equal(model.doStreamCalls[0]!.reasoning, "low");

  const overridden = await root.subsession({
    sessionId: "r2-b",
    reasoning: "xhigh",
  });
  await drain(await overridden.prompt("y"));
  assert.equal(model.doStreamCalls[1]!.reasoning, "xhigh");
});

test("a subsession is linked to its parent and filtered out of top-level listings", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("ok") }),
  });
  const storage = new InMemorySessionStorage();
  const harness = await init({ registry, model, role: helper(), storage });
  const root = await harness.session({ sessionId: "p1" });

  const worker = await root.subsession({ sessionId: "p1-sub" });
  await drain(await worker.prompt("x"));

  // The subsession records its parent; a top-level session has none.
  const all = await storage.listSessions();
  assert.equal(all.find((s) => s.id === "p1-sub")?.parentId, "p1");
  assert.equal(all.find((s) => s.id === "p1")?.parentId, null);

  // `parentId: null` hides subsessions; `parentId: "p1"` surfaces only them.
  const topLevel = await storage.listSessions({ parentId: null });
  assert.deepEqual(
    topLevel.map((s) => s.id),
    ["p1"],
  );
  const children = await storage.listSessions({ parentId: "p1" });
  assert.deepEqual(
    children.map((s) => s.id),
    ["p1-sub"],
  );
});

test("a role's outputSchema constrains the model and types result.output", async () => {
  const payload = { entities: [{ name: "Rossi", code: "C1" }] };
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream(JSON.stringify(payload)) }),
  });
  const storage = new InMemorySessionStorage();

  const extractor = role({
    name: "extractor",
    systemPrompt: "Extract entities.",
    outputSchema: z.object({
      entities: z.array(z.object({ name: z.string(), code: z.string() })),
    }),
  });

  const harness = await init({ registry, model, role: extractor(), storage });
  const session = await harness.session({ sessionId: "o1" });

  const result = await session.prompt("...transcript...");
  // `result.output` is the parsed, validated object (typed as the schema).
  // Awaiting it also drives the (lazy) model call so we can inspect it below.
  const output = await result.output;
  assert.deepEqual(output, payload);
  // The role's schema was advertised to the model as a JSON response format —
  // the model is constrained to the shape, not asked for free prose.
  assert.equal(model.doStreamCalls[0]!.responseFormat?.type, "json");
});

test("unknown tool names are rejected at session start", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("x") }),
  });
  const broken = role({
    name: "broken",
    systemPrompt: "x",
    tools: ["nope"],
  });
  const harness = await init({ registry, model, role: broken() });
  await assert.rejects(() => harness.session(), /not in the harness registry/);
});

test("a skill referencing an unknown tool is rejected on invocation", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("x") }),
  });
  const badSkill = skill({
    name: "bad",
    description: "references a missing tool",
    instructions: "x",
    tools: ["ghost"],
  });
  const session = await (
    await init({
      registry: loaderRegistry,
      model,
      role: helper(),
      loadableSkills: [staticSource(badSkill())],
    })
  ).session({ sessionId: "s2" });
  await assert.rejects(
    () => session.skill("bad"),
    /not in the harness registry/,
  );
});

test("session.skill with an unknown name fails naming the available skills", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("x") }),
  });
  const session = await (
    await init({
      registry: loaderRegistry,
      model,
      role: helper(),
      loadableSkills: [staticSource(artifacts())],
    })
  ).session({ sessionId: "s3" });
  await assert.rejects(
    () => session.skill("ghost"),
    /session\.skill\("ghost"\): unknown skill — available: artifacts/,
  );
  // Nothing was activated and no skill-init message was persisted.
  assert.deepEqual(session.activeTools.slice().sort(), ["readFile"]);
  assert.equal(session.messages.length, 0);
});

test("session.skill without any loadableSkills source is a clear error", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("x") }),
  });
  const session = await (
    await init({ registry, model, role: helper() })
  ).session({ sessionId: "s4" });
  await assert.rejects(
    () => session.skill("artifacts"),
    /no skills are available in this session/,
  );
});

test("session.skill's name parameter is typed from the static sources", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("x") }),
  });
  // The knowledge-base shape: names known only at runtime, so it declares no
  // static ones and must not widen the union to `string`.
  const knowledgeLike: SkillSource<never> = {
    list: async () => [],
    read: async () => null,
  };
  const session = await (
    await init({
      registry: loaderRegistry,
      model,
      role: helper(),
      loadableSkills: [staticSource(artifacts()), knowledgeLike],
    })
  ).session({ sessionId: "s5" });

  // Compile-time (checked by the package's tsc lint): the parameter is the
  // autocomplete-preserving union — the static names as literals plus plain
  // string for knowledge-base names, not collapsed to `string`.
  type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <
    T,
  >() => T extends B ? 1 : 2
    ? true
    : false;
  type Name = Parameters<(typeof session)["skill"]>[0];
  const unionSurvives: Equal<Name, "artifacts" | (string & {})> = true;
  void unionSurvives;

  // A plain string — a knowledge-base name — is accepted by the type; an
  // unknown one still fails loudly at runtime.
  await assert.rejects(() => session.skill("kb-authored"), /unknown skill/);

  // @ts-expect-error the old value-based call shape is rejected: skills are
  // resolved by name through the sources, not passed as values.
  await assert.rejects(() => session.skill(artifacts()), /unknown skill/);

  // A known static name resolves to the skill.
  const bound = await session.skill("artifacts");
  assert.equal(bound.name, "artifacts");
});

// ── Initial skills (host-supplied, at session creation) ─────────────────────

test("an initial skill joins the instructions and activates its tools from the first turn", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hello") }),
  });
  const storage = new InMemorySessionStorage();

  const harness = await init({
    registry,
    model,
    role: helper(),
    initialSkills: [artifacts()],
    storage,
  });

  // The skill's instructions ride in the system prompt, after the role's.
  assert.match(harness.instructions, /You are a helper\./);
  assert.match(
    harness.instructions,
    /When active, use createArtifact to persist results\./,
  );
  assert.ok(
    harness.instructions.indexOf("You are a helper.") <
      harness.instructions.indexOf("When active, use createArtifact"),
  );

  const session = await harness.session({ sessionId: "b1" });
  assert.deepEqual(session.activeTools.slice().sort(), [
    "createArtifact",
    "readFile",
  ]);

  // Turn 1 already sees the skill's tools — no invocation step needed.
  await drain(await session.prompt("hi"));
  assert.deepEqual(toolNames(model.doStreamCalls[0]!), [
    "createArtifact",
    "readFile",
  ]);

  // Unlike session.skill(), an initial skill leaves no <skill-init> message.
  await new Promise((r) => setTimeout(r, 50));
  const saved = await storage.loadMessages("b1");
  assert.deepEqual(
    saved.map((m) => m.role),
    ["user", "assistant"],
  );
});

test("a subsession takes its own initial skills: instructions after the role's, tools active from the first turn", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hello") }),
  });
  const harness = await init({ registry, model, role: helper() });
  const root = await harness.session({ sessionId: "bs1" });

  const worker = await root.subsession({
    role: helper(),
    initialSkills: [artifacts()],
    sessionId: "bs1-sub",
  });
  assert.deepEqual(worker.activeTools.slice().sort(), [
    "createArtifact",
    "readFile",
  ]);

  await drain(await worker.prompt("hi"));
  assert.deepEqual(toolNames(model.doStreamCalls[0]!), [
    "createArtifact",
    "readFile",
  ]);
  const wire = JSON.stringify(model.doStreamCalls[0]!.prompt ?? null);
  assert.ok(
    wire.indexOf("You are a helper.") <
      wire.indexOf("When active, use createArtifact"),
  );
});

test("an initial skill referencing an unknown tool is rejected at session start", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("x") }),
  });
  const badSkill = skill({
    name: "bad",
    description: "references a missing tool",
    instructions: "x",
    tools: ["ghost"],
  });
  const harness = await init({
    registry,
    model,
    role: helper(),
    initialSkills: [badSkill()],
  });
  await assert.rejects(() => harness.session(), /not in the harness registry/);
});

test("the skill catalog renders when an initial skill brings the loader tool", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const loaderSkill = skill({
    name: "loader-bearer",
    description: "turns on the skill loader",
    instructions: "Load skills when a covered domain comes up.",
    tools: ["loadSkill"],
  });
  const harness = await init({
    registry: loaderRegistry,
    model,
    role: helper(),
    initialSkills: [loaderSkill()],
    loadableSkills: [agendaSource],
  });
  assert.match(harness.instructions, /<available_skills>/);
  // The initial skill's init envelope follows the catalog, exactly where a
  // load off the catalog would put it.
  assert.ok(
    harness.instructions.indexOf("<available_skills>") <
      harness.instructions.indexOf("Load skills when a covered domain"),
  );
  assert.match(harness.instructions, /<skill-init name="loader-bearer">/);
});

// ── Tool approvals ──────────────────────────────────────────────────────────

function toolCallStream(
  toolName: string,
  input: unknown,
  toolCallId = "call-1",
) {
  return convertArrayToReadableStream([
    { type: "stream-start" as const, warnings: [] },
    {
      type: "tool-call" as const,
      toolCallId,
      toolName,
      input: JSON.stringify(input),
    },
    {
      type: "finish" as const,
      finishReason: { unified: "tool-calls" as const, raw: undefined },
      usage: {
        inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 1, text: 1, reasoning: 0 },
      },
    },
  ]);
}

/** A registry whose writes are observable, a role over it, and a model whose
 *  first call asks for `writeFile` and whose next call replies with text —
 *  the canonical gated turn. */
function approvalSetup() {
  const written: unknown[] = [];
  const gatedRegistry = {
    writeFile: tool({
      description: "Write a file.",
      inputSchema: z.object({ path: z.string(), content: z.string() }),
      execute: async (input) => {
        written.push(input);
        return "written";
      },
    }),
    readFile: tool({
      description: "Read a file.",
      inputSchema: z.object({ path: z.string() }),
      execute: async () => "contents",
    }),
  };
  const writer = role({
    name: "writer",
    systemPrompt: "You edit files.",
    tools: ["writeFile", "readFile"],
  });
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("writeFile", { path: "/a.md", content: "hi" })
          : textStream("done"),
    }),
  });
  const storage = new InMemorySessionStorage();
  return { written, gatedRegistry, writer, model, storage };
}

type ApprovalToolPart = {
  type: string;
  state?: string;
  approval?: { id: string; approved?: boolean; reason?: string };
};

function toolPart(
  message: { parts: unknown[] },
  name: string,
): ApprovalToolPart | undefined {
  return (message.parts as ApprovalToolPart[]).find(
    (p) => p.type === `tool-${name}`,
  );
}

/** What the client re-sends after `addToolApprovalResponse`: the stored
 *  assistant message with the answered request part(s) flipped to
 *  `approval-responded`. */
function answeredMessage(
  message: SessionMessage,
  answers: Record<string, { approved: boolean; reason?: string }>,
): SessionMessage {
  return {
    ...message,
    parts: message.parts.map((part) => {
      const p = part as ApprovalToolPart;
      const answer =
        p.state === "approval-requested" && p.approval
          ? answers[p.approval.id]
          : undefined;
      return answer
        ? ({
            ...p,
            state: "approval-responded",
            approval: { id: p.approval!.id, ...answer },
          } as unknown as typeof part)
        : part;
    }),
  };
}

test("a gated tool stops with an approval request; approving resumes and executes it", async () => {
  const { written, gatedRegistry, writer, model, storage } = approvalSetup();
  const harness = await init({
    registry: gatedRegistry,
    model,
    role: writer(),
    storage,
    toolApproval: ["writeFile"],
  });
  const session = await harness.session({ sessionId: "ap1" });

  await drain(await session.prompt("write hi to /a.md"));
  await new Promise((r) => setTimeout(r, 50));

  // The turn completed without executing: the tool part is a pending request.
  assert.equal(written.length, 0);
  const afterRequest = (await storage.loadMessages("ap1")).at(-1)!;
  const requested = toolPart(afterRequest, "writeFile")!;
  assert.equal(requested.state, "approval-requested");
  const approvalId = requested.approval!.id;
  assert.ok(approvalId);

  // Approving re-drives the loop, through the same `prompt` entry point the
  // wire uses: the client re-sends the answered assistant message. The tool
  // executes with its original input and the resumed reply persists.
  await drain(
    await session.prompt(
      answeredMessage(afterRequest, { [approvalId]: { approved: true } }),
    ),
  );
  await new Promise((r) => setTimeout(r, 50));

  assert.deepEqual(written, [{ path: "/a.md", content: "hi" }]);
  const saved = await storage.loadMessages("ap1");
  const resolved = saved
    .filter((m) => m.role === "assistant")
    .map((m) => toolPart(m, "writeFile"))
    .find((p) => p?.state === "output-available");
  assert.ok(resolved, "the approved call carries its output after the resume");
  const replyText = saved
    .filter((m) => m.role === "assistant")
    .flatMap((m) => m.parts)
    .map((p) => (p.type === "text" ? p.text : ""))
    .join("");
  assert.match(replyText, /done/);
});

test("a denied tool never executes and the model hears the denial", async () => {
  const { written, gatedRegistry, writer, model, storage } = approvalSetup();
  const harness = await init({
    registry: gatedRegistry,
    model,
    role: writer(),
    storage,
    toolApproval: ["writeFile"],
  });
  const session = await harness.session({ sessionId: "ap2" });

  await drain(await session.prompt("write hi to /a.md"));
  await new Promise((r) => setTimeout(r, 50));
  const afterRequest = (await storage.loadMessages("ap2")).at(-1)!;
  const requested = toolPart(afterRequest, "writeFile")!;

  await drain(
    await session.prompt(
      answeredMessage(afterRequest, {
        [requested.approval!.id]: { approved: false, reason: "not now" },
      }),
    ),
  );
  await new Promise((r) => setTimeout(r, 50));

  assert.equal(written.length, 0);
  // The resumed model call carries the denial (and its reason) instead of a
  // tool result.
  const resumedPrompt = promptText(model.doStreamCalls[1]!);
  assert.match(resumedPrompt, /execution-denied/);
  assert.match(resumedPrompt, /not now/);
});

test("tools outside the approval list still execute in a single turn", async () => {
  const { gatedRegistry, writer, storage } = approvalSetup();
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("readFile", { path: "/a.md" })
          : textStream("done"),
    }),
  });
  const harness = await init({
    registry: gatedRegistry,
    model,
    role: writer(),
    storage,
    toolApproval: ["writeFile"],
  });
  const session = await harness.session({ sessionId: "ap3" });

  await drain(await session.prompt("read /a.md"));
  await new Promise((r) => setTimeout(r, 50));

  // Two model calls inside ONE prompt — the loop never stopped for approval.
  assert.equal(model.doStreamCalls.length, 2);
  const last = (await storage.loadMessages("ap3")).at(-1)!;
  assert.equal(toolPart(last, "readFile")!.state, "output-available");
});

test("an approval turn must answer the pending requests exactly", async () => {
  const { gatedRegistry, writer, model, storage } = approvalSetup();
  const harness = await init({
    registry: gatedRegistry,
    model,
    role: writer(),
    storage,
    toolApproval: ["writeFile"],
  });
  const session = await harness.session({ sessionId: "ap4" });

  // An assistant message carrying an answer the session never asked for.
  const ghost: SessionMessage = {
    id: "forged",
    role: "assistant",
    parts: [
      {
        type: "tool-writeFile",
        toolCallId: "c1",
        state: "approval-responded",
        approval: { id: "ghost", approved: true },
      } as unknown as SessionMessage["parts"][number],
    ],
  };

  // Nothing pending yet.
  await assert.rejects(
    () => session.prompt(ghost),
    /no pending approval request/,
  );

  await drain(await session.prompt("write hi to /a.md"));
  await new Promise((r) => setTimeout(r, 50));

  // Unknown id.
  await assert.rejects(
    () => session.prompt(ghost),
    /no pending approval request with id/,
  );
  // Re-sending the request unanswered resolves nothing.
  const unanswered = (await storage.loadMessages("ap4")).at(-1)!;
  await assert.rejects(
    () => session.prompt(unanswered),
    /still need a response/,
  );
});

test("a custom GenericToolApprovalFunction reaches the agent unmodified", async () => {
  const { written, gatedRegistry, writer, model, storage } = approvalSetup();
  const harness = await init({
    registry: gatedRegistry,
    model,
    role: writer(),
    storage,
    toolApproval: ({ toolCall }) =>
      toolCall.toolName === "writeFile" ? "user-approval" : undefined,
  });
  const session = await harness.session({ sessionId: "ap5" });

  await drain(await session.prompt("write hi to /a.md"));
  await new Promise((r) => setTimeout(r, 50));

  assert.equal(written.length, 0);
  assert.equal(
    toolPart((await storage.loadMessages("ap5")).at(-1)!, "writeFile")!.state,
    "approval-requested",
  );
});

test("an unknown tool name in toolApproval is rejected at session start", async () => {
  const { gatedRegistry, writer, model } = approvalSetup();
  const harness = await init({
    registry: gatedRegistry,
    model,
    role: writer(),
    // The array is typed to the registry's keys; the cast simulates a caller
    // sidestepping the compile-time check.
    toolApproval: ["writeFil"] as unknown as ["writeFile"],
  });
  await assert.rejects(() => harness.session(), /not in the harness registry/);
});

test("providerOptions reach every model call in the session", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const harness = await init({
    registry,
    model,
    role: helper(),
    providerOptions: { gateway: { only: ["azure"] } },
    storage: new InMemorySessionStorage(),
  });
  const session = await harness.session();

  await drain(await session.prompt("hello"));

  assert.deepEqual(model.doStreamCalls[0]!.providerOptions, {
    gateway: { only: ["azure"] },
  });
});

test("a subsession inherits providerOptions, but not across a model change", async () => {
  const parentModel = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("parent") }),
  });
  const otherModel = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("worker") }),
  });
  const harness = await init({
    registry,
    model: parentModel,
    role: helper(),
    providerOptions: { gateway: { only: ["azure"] } },
    storage: new InMemorySessionStorage(),
  });
  const session = await harness.session();

  // Same model: the routing was chosen for it, so it carries over.
  const sameModel = await session.subsession();
  await drain(await sameModel.prompt("go"));
  assert.deepEqual(parentModel.doStreamCalls.at(-1)!.providerOptions, {
    gateway: { only: ["azure"] },
  });

  // A different model: a provider serves specific models, so the parent's
  // routing must not follow it.
  const switched = await session.subsession({ model: otherModel });
  await drain(await switched.prompt("go"));
  assert.equal(otherModel.doStreamCalls[0]!.providerOptions, undefined);

  // Unless the subsession names its own.
  const pinned = await session.subsession({
    model: otherModel,
    providerOptions: { gateway: { only: ["bedrock"] } },
  });
  await drain(await pinned.prompt("go"));
  assert.deepEqual(otherModel.doStreamCalls.at(-1)!.providerOptions, {
    gateway: { only: ["bedrock"] },
  });
});

// ── Turn timings ────────────────────────────────────────────────────────────

test("a turn's timings persist on the assistant message's metadata", async () => {
  const timedRegistry = {
    search: tool({
      description: "Search.",
      inputSchema: z.object({ q: z.string() }),
      execute: async () => {
        await new Promise((r) => setTimeout(r, 25));
        return "found";
      },
    }),
  };
  const searcher = role({
    name: "searcher",
    systemPrompt: "You search.",
    tools: ["search"],
  });
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("search", { q: "x" })
          : textStream("done"),
    }),
  });
  const storage = new InMemorySessionStorage();
  const session = await (
    await init({ registry: timedRegistry, model, role: searcher(), storage })
  ).session({ sessionId: "t1" });

  const result = await session.prompt("find x");
  await drain(result);
  await result.committed;

  const saved = await storage.loadMessages("t1");
  const assistant = saved.at(-1)!;
  assert.equal(assistant.role, "assistant");
  const timings = (
    assistant.metadata as {
      timings?: {
        ttftMs?: number;
        totalMs: number;
        steps?: Array<{ responseMs: number; ttftMs?: number }>;
        tools?: Array<{ callId: string; name: string; ms: number }>;
      };
    }
  ).timings;
  assert.ok(timings, "assistant message carries metadata.timings");

  // Two model calls (tool step + reply step), one timed tool execution.
  assert.equal(typeof timings.totalMs, "number");
  assert.equal(timings.steps?.length, 2);
  assert.equal(timings.tools?.length, 1);
  assert.equal(timings.tools![0]!.callId, "call-1");
  assert.equal(timings.tools![0]!.name, "search");
  // The tool slept ~25ms; the SDK-measured duration must reflect it, and the
  // turn total must cover everything.
  assert.ok(timings.tools![0]!.ms >= 15, `tool ms ${timings.tools![0]!.ms}`);
  assert.ok(timings.totalMs >= timings.tools![0]!.ms);
  // Streaming steps carry a time-to-first-output; the turn lifts the first.
  assert.equal(timings.ttftMs, timings.steps![0]!.ttftMs);

  // `result.timings()` snapshots the same recorder — what a route streams to
  // its client matches what persisted (totalMs is stamped per call, so it may
  // trail by a beat; steps/tools/ttft are the same records).
  const streamed = result.timings();
  assert.deepEqual(streamed.steps, timings.steps);
  assert.deepEqual(streamed.tools, timings.tools);
  assert.equal(streamed.ttftMs, timings.ttftMs);
});

test("a text-only turn records one step and no tools", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const storage = new InMemorySessionStorage();
  const session = await (
    await init({ registry, model, role: helper(), storage })
  ).session({ sessionId: "t2" });

  const result = await session.prompt("hello");
  await drain(result);
  await result.committed;

  const assistant = (await storage.loadMessages("t2")).at(-1)!;
  const timings = (
    assistant.metadata as {
      timings?: { totalMs: number; steps?: unknown[]; tools?: unknown[] };
    }
  ).timings;
  assert.ok(timings);
  assert.equal(timings.steps?.length, 1);
  assert.equal(timings.tools, undefined);
});

// ── Code mode ───────────────────────────────────────────────────────────────

/** A registry with the sandbox wired in, a routable tool with observable
 *  executions and a plain tool, plus roles over each split: the analyst lists
 *  `codeMode` and routes `lookup` through it, the chatty role does neither. */
function codeModeSetup() {
  const looked: unknown[] = [];
  const seenSessionIds: string[] = [];
  const tools = {
    lookup: harnessTool({
      description: "Look up an answer.",
      inputSchema: z.object({ q: z.string() }),
      execute: async (input: { q: string }, { session }) => {
        looked.push(input);
        seenSessionIds.push(session.id);
        return { answer: 42 };
      },
    }),
    chat: tool({
      description: "Small talk.",
      inputSchema: z.object({ text: z.string() }),
      execute: async () => "ok",
    }),
    code: codeModeTool(),
  };
  const bound = createRegistry(tools);
  const analyst = bound.role({
    name: "analyst",
    systemPrompt: "You analyse.",
    tools: ["lookup", "chat", "code"],
    toolCallers: { lookup: ["code"] },
  });
  const chatty = bound.role({
    name: "chatty",
    systemPrompt: "You chat.",
    tools: ["chat"],
  });
  return {
    codeRegistry: bound.registry(),
    // The bare toolset, for the case that extends it with another entry.
    codeTools: tools,
    boundSkill: bound.skill,
    looked,
    seenSessionIds,
    analyst,
    chatty,
  };
}

test("a code-routed tool reaches the model through the sandbox, not directly", async () => {
  const { codeRegistry, analyst } = codeModeSetup();
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const session = await (
    await init({ registry: codeRegistry, model, role: analyst() })
  ).session();

  await drain(await session.prompt("hi"));

  // The model sees the plain tool and the sandbox; the routed tool is only
  // reachable from generated code.
  assert.deepEqual(toolNames(model.doStreamCalls[0]!), ["chat", "code"]);
  // The routed tool's typed API reaches the model with the catalog.
  const sandbox = (model.doStreamCalls[0]!.tools ?? []).find(
    (t) => t.name === "code",
  ) as { description?: string } | undefined;
  assert.ok(sandbox);
  const sent = modelFacingText(model.doStreamCalls[0]!);
  assert.match(sent, /tools\.lookup/);
  assert.match(sent, /Look up an answer/);
  // The harness-facing views: full active set, and the code-routed subset.
  assert.deepEqual(session.activeTools.slice().sort(), [
    "chat",
    "code",
    "lookup",
  ]);
  assert.deepEqual(session.activeCodeTools.slice(), ["lookup"]);
});

test("a role without the sandbox gets none; a skill can bring it with its routing", async () => {
  const { codeTools, boundSkill, chatty } = codeModeSetup();
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const search = boundSkill({
    name: "search",
    description: "Look things up.",
    instructions: "Use lookup from code.",
    tools: ["lookup", "code"],
    toolCallers: { lookup: ["code"] },
  });
  const session = await (
    await init({
      registry: { ...codeTools, loadSkill: skillLoaderTool() },
      model,
      role: chatty(),
      loadableSkills: [staticSource(search())],
    })
  ).session();

  // Baseline: nothing routed, no sandbox exposed.
  await drain(await session.prompt("hi"));
  assert.deepEqual(toolNames(model.doStreamCalls[0]!), ["chat"]);
  assert.deepEqual(session.activeCodeTools.slice(), []);

  // The skill lists the sandbox and merges its routing in — the source hands
  // the routing map over, so a host bind by name honors it.
  await session.skill("search");
  await drain(await session.prompt("find it"));
  assert.deepEqual(toolNames(model.doStreamCalls[1]!), ["chat", "code"]);
  assert.deepEqual(session.activeCodeTools.slice(), ["lookup"]);
});

test("a tool routed both ways stays a direct call and joins the sandbox", async () => {
  const { codeRegistry, analyst } = codeModeSetup();
  void analyst;
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const both = role({
    name: "both-ways",
    systemPrompt: "You analyse.",
    tools: ["lookup", "chat", "code"],
    toolCallers: { lookup: ["code", DIRECT_TOOL_CALL] },
  });
  const session = await (
    await init({ registry: codeRegistry, model, role: both() })
  ).session();

  await drain(await session.prompt("hi"));
  assert.deepEqual(toolNames(model.doStreamCalls[0]!), [
    "chat",
    "code",
    "lookup",
  ]);
});

test("a codeMode call runs the program in the interpreter and executes the routed tool", async () => {
  const { codeRegistry, looked, seenSessionIds, analyst } = codeModeSetup();
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("code", {
              code: "const r = await tools.lookup({ q: 'x' }); return r.answer;",
            })
          : textStream("done"),
    }),
  });
  const storage = new InMemorySessionStorage();
  const session = await (
    await init({ registry: codeRegistry, model, role: analyst(), storage })
  ).session({ sessionId: "cm1" });

  await drain(await session.prompt("compute"));
  await new Promise((r) => setTimeout(r, 50));

  // The program called the routed tool with the generated input…
  assert.deepEqual(looked, [{ q: "x" }]);
  // …and `currentSession()` resolved inside it, through the interpreter.
  assert.deepEqual(seenSessionIds, ["cm1"]);

  // The persisted turn carries the run: the tool output is the interpreter's
  // Result — the program's return value plus the calls it admitted.
  const saved = await storage.loadMessages("cm1");
  const assistant = saved.at(-1)!;
  assert.equal(assistant.role, "assistant");
  const part = toolPart(assistant, "code") as
    | { state?: string; output?: { ok?: boolean; value?: unknown; toolCalls?: unknown[] } }
    | undefined;
  assert.ok(part, "assistant message carries the codeMode tool part");
  assert.equal(part.state, "output-available");
  assert.equal(part.output?.ok, true);
  assert.equal(part.output?.value, 42);
  assert.deepEqual(part.output?.toolCalls, [{ name: "lookup" }]);
});

test("a failing program comes back as a diagnostic the model can read, not a thrown error", async () => {
  const { codeRegistry, analyst } = codeModeSetup();
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("code", {
              code: "return await tools.nope({});",
            })
          : textStream("done"),
    }),
  });
  const storage = new InMemorySessionStorage();
  const session = await (
    await init({ registry: codeRegistry, model, role: analyst(), storage })
  ).session({ sessionId: "cm2" });

  await drain(await session.prompt("compute"));
  await new Promise((r) => setTimeout(r, 50));

  const saved = await storage.loadMessages("cm2");
  const part = toolPart(saved.at(-1)!, "code") as
    | { state?: string; output?: { ok?: boolean; error?: { kind?: string } } }
    | undefined;
  assert.equal(part?.state, "output-available");
  assert.equal(part?.output?.ok, false);
  assert.equal(part?.output?.error?.kind, "UnknownTool");
});

test("routing declarations are validated at session start", async () => {
  const { codeRegistry, analyst } = codeModeSetup();
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });

  // An approval-gated tool cannot be routed through the sandbox.
  await assert.rejects(
    (
      await init({
        registry: codeRegistry,
        model,
        role: analyst(),
        toolApproval: ["lookup"],
      })
    ).session(),
    /code mode/,
  );

  // Routing a tool the role does not itself activate is a declaration error.
  const foreign = role({
    name: "foreign",
    systemPrompt: "x",
    tools: ["chat", "code"],
    toolCallers: { lookup: ["code"] } as never,
  });
  await assert.rejects(
    (await init({ registry: codeRegistry, model, role: foreign() })).session(),
    /not in its own tools list/,
  );

  // A caller the harness doesn't support fails up front, not at stream time.
  const nonsense = role({
    name: "nonsense",
    systemPrompt: "x",
    tools: ["lookup", "code"],
    toolCallers: { lookup: ["nonsense"] } as never,
  });
  await assert.rejects(
    (await init({ registry: codeRegistry, model, role: nonsense() })).session(),
    /unknown caller/,
  );

  // Routing into the sandbox without listing it is a declaration error.
  const sandboxless = role({
    name: "sandboxless",
    systemPrompt: "x",
    tools: ["lookup", "chat"],
    toolCallers: { lookup: ["code"] },
  });
  await assert.rejects(
    (
      await init({ registry: codeRegistry, model, role: sandboxless() })
    ).session(),
    /requires "code" among the active tools/,
  );
});

// ── Tool namespaces ─────────────────────────────────────────────────────────

test("a namespaced tool nests in the sandbox: catalog path, call path, search", async () => {
  const looked: unknown[] = [];
  const bound = createRegistry({
    lookup: harnessTool({
      description: "Look up an answer.",
      metadata: { namespace: "api" },
      inputSchema: z.object({ q: z.string() }),
      execute: async (input: { q: string }) => {
        looked.push(input);
        return { answer: 42 };
      },
    }),
    code: codeModeTool(),
  });
  const grouped = bound.role({
    name: "grouped",
    systemPrompt: "You analyse.",
    tools: ["lookup", "code"],
    toolCallers: { lookup: ["code"] },
  });
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("code", {
              code: [
                "const found = await search({ query: 'lookup' });",
                "const r = await tools.api.lookup({ q: 'x' });",
                "return { paths: found.items.map((i) => i.path), answer: r.answer };",
              ].join("\n"),
            })
          : textStream("done"),
    }),
  });
  const storage = new InMemorySessionStorage();
  const session = await (
    await init({ registry: bound.registry(), model, role: grouped(), storage })
  ).session({ sessionId: "ns1" });

  await drain(await session.prompt("compute"));
  await new Promise((r) => setTimeout(r, 50));

  // The catalog names the namespaced path and is complete (no Search
  // section); the sandbox description always carries the invariant guidance.
  const sent = modelFacingText(model.doStreamCalls[0]!);
  assert.match(sent, /The Code Mode tool catalog below is complete\./);
  assert.match(sent, /tools\.api\.lookup/);
  assert.doesNotMatch(sent, /## Search/);
  const sandbox = (model.doStreamCalls[0]!.tools ?? []).find(
    (t) => t.name === "code",
  ) as { description?: string } | undefined;
  assert.match(
    sandbox?.description ?? "",
    /catalog instructions or returned by `search`/,
  );

  // The program found the tool through the built-in search and called it at
  // its namespaced path.
  assert.deepEqual(looked, [{ q: "x" }]);
  const saved = await storage.loadMessages("ns1");
  const part = toolPart(saved.at(-1)!, "code") as
    | { output?: { ok?: boolean; value?: { paths?: string[]; answer?: number } } }
    | undefined;
  assert.equal(part?.output?.ok, true);
  assert.deepEqual(part?.output?.value, { paths: ["tools.api.lookup"], answer: 42 });
});

test("a malformed namespace fails at registry creation", () => {
  assert.throws(
    () =>
      createRegistry({
        broken: tool({
          description: "x",
          metadata: { namespace: "api." },
          inputSchema: z.object({}),
          execute: async () => "ok",
        }),
      }),
    /empty segment/,
  );
});

test("an oversized catalog turns partial: namespace counts and the search signature", async () => {
  const blurb = (n: number) =>
    `Tool number ${n}. ${"Fetches a page of records and explains every field in detail. ".repeat(12)}`;
  const names = Array.from({ length: 80 }, (_, n) => `fetch${n}`);
  const wide = {
    code: codeModeTool(),
    ...Object.fromEntries(
      names.map((name, n) => [
        name,
        tool({
          description: blurb(n),
          // fetch9 sorts last in erp and would lose the budget cut without
          // its pin: it must keep its listing regardless.
          metadata: { namespace: n < 40 ? "erp" : "crm", ...(n === 9 ? { pinned: true } : {}) },
          inputSchema: z.object({ id: z.string() }),
          execute: async () => "ok",
        }),
      ]),
    ),
  };
  const wideRole = role({
    name: "wide",
    systemPrompt: "x",
    tools: [...names, "code"],
    toolCallers: Object.fromEntries(
      names.map((name) => [name, ["code" as const]]),
    ),
  });
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const session = await (
    await init({ registry: wide as never, model, role: wideRole() })
  ).session();
  await drain(await session.prompt("hi"));

  const sent = modelFacingText(model.doStreamCalls[0]!);
  // Partial: the catalog says so, every namespace appears with its count,
  // and the search signature is spelled out.
  assert.match(sent, /The Code Mode tool catalog below is partial\./);
  assert.match(sent, /- erp \(40 tools, \d+ shown\)/);
  assert.match(sent, /- crm \(40 tools, \d+ shown\)/);
  assert.match(sent, /## Search/);
  assert.match(sent, /search\(input:/);
  // Budgeted: not every signature made it inline — but the pinned one did.
  const shown = sent.match(/tools\.(erp|crm)\.fetch\d+\(/g) ?? [];
  assert.ok(shown.length < 80, `expected a partial listing, got ${shown.length} of 80`);
  assert.match(sent, /tools\.erp\.fetch9\(/);
});

// ── Model-triggered skills (skillLoaderTool) ─────────────────────────────────

/** A source holding one skill whose declared tool is registered but not in the
 *  role baseline — loading it must both return instructions and grow the set. */
const agendaSource: SkillSource = {
  list: async () => [{ name: "agenda", description: "Calendar work." }],
  read: async (name) =>
    name === "agenda"
      ? {
          name: "agenda",
          description: "Calendar work.",
          instructions: "Read the day before booking.",
          tools: ["createArtifact"],
        }
      : null,
};

const loaderRegistry = {
  ...registry,
  loadSkill: skillLoaderTool(),
};

test("init renders the skill catalog only for a role that lists the loader", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const skilled = role({
    name: "skilled",
    systemPrompt: "You are a helper.",
    tools: ["readFile", "loadSkill"],
  });

  const withLoader = await init({
    registry: loaderRegistry,
    model,
    role: skilled(),
    loadableSkills: [agendaSource],
  });
  assert.match(withLoader.instructions, /<available_skills>/);
  assert.match(withLoader.instructions, /agenda: Calendar work\.|<name>agenda<\/name>/);

  // Same source, but the role never lists the loader: no catalog.
  const withoutLoader = await init({
    registry: loaderRegistry,
    model,
    role: helper(),
    loadableSkills: [agendaSource],
  });
  assert.doesNotMatch(withoutLoader.instructions, /<available_skills>/);
});

test("init refuses a skills source when the loader tool isn't in the registry", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  await assert.rejects(
    () =>
      init({
        registry,
        model,
        role: helper(),
        loadableSkills: [agendaSource],
      }),
    /wire skillLoaderTool/,
  );
});

test("loading a skill returns its instructions as the tool result and grows the active set", async () => {
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("loadSkill", { name: "agenda" })
          : textStream("ok"),
    }),
  });
  const skilled = role({
    name: "skilled",
    systemPrompt: "You are a helper.",
    tools: ["readFile", "loadSkill"],
  });
  const storage = new InMemorySessionStorage();
  const session = await (
    await init({
      registry: loaderRegistry,
      model,
      role: skilled(),
      loadableSkills: [agendaSource],
      storage,
    })
  ).session({ sessionId: "sk1" });

  const result = await session.prompt("what do I have tomorrow?");
  await drain(result);
  await result.committed;

  // The activation was live mid-turn: the step after the tool call already
  // advertises the skill's tool.
  assert.ok(session.activeTools.includes("createArtifact"));
  assert.deepEqual(toolNames(model.doStreamCalls[1]!), [
    "createArtifact",
    "loadSkill",
    "readFile",
  ]);
  // The model read the instructions from the tool result, in the same
  // `<skill-init>` shape session.skill() injects.
  assert.match(promptText(model.doStreamCalls[1]!), /skill-init name=\\"agenda\\"|skill-init name="agenda"/);
  assert.match(promptText(model.doStreamCalls[1]!), /Read the day before booking\./);

  const resumed = await (
    await init({
      registry: loaderRegistry,
      model,
      role: skilled(),
      loadableSkills: [agendaSource],
      storage,
    })
  ).session({ sessionId: "sk1" });
  assert.ok(resumed.activeTools.includes("createArtifact"));
});

test("loading an unknown skill fails listing the skills that exist", async () => {
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("loadSkill", { name: "ghost" })
          : textStream("ok"),
    }),
  });
  const skilled = role({
    name: "skilled",
    systemPrompt: "You are a helper.",
    tools: ["readFile", "loadSkill"],
  });
  const session = await (
    await init({
      registry: loaderRegistry,
      model,
      role: skilled(),
      loadableSkills: [agendaSource],
    })
  ).session({ sessionId: "sk2" });

  await drain(await session.prompt("hi"));

  // The failure reached the model as tool output naming the available skills,
  // and nothing was activated.
  assert.match(promptText(model.doStreamCalls[1]!), /available: agenda/);
  assert.ok(!session.activeTools.includes("createArtifact"));
});

// ── Merged skill sources (loadableSkills: SkillSource[]) ────────────────────

/** A second source, disjoint from `agendaSource`, whose skill declares no
 *  tools — loading it must still return instructions through the merge. */
const filingSource: SkillSource = {
  list: async () => [{ name: "filing", description: "Filing work." }],
  read: async (name) =>
    name === "filing"
      ? {
          name: "filing",
          description: "Filing work.",
          instructions: "File under the client's folder.",
          tools: [],
        }
      : null,
};

test("the catalog lists every source's skills and reads resolve through the merge", async () => {
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("loadSkill", { name: "filing" })
          : textStream("ok"),
    }),
  });
  const skilled = role({
    name: "skilled",
    systemPrompt: "You are a helper.",
    tools: ["readFile", "loadSkill"],
  });
  const harness = await init({
    registry: loaderRegistry,
    model,
    role: skilled(),
    loadableSkills: [agendaSource, filingSource],
  });
  assert.match(harness.instructions, /<name>agenda<\/name>/);
  assert.match(harness.instructions, /<name>filing<\/name>/);

  // A skill held by the second source loads like one from the first.
  const session = await harness.session({ sessionId: "merged1" });
  await drain(await session.prompt("archive this"));
  assert.match(promptText(model.doStreamCalls[1]!), /File under the client's folder\./);
});

test("a source over ready skill values lists next to a parsed one in the exact prompt", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const skilled = role({
    name: "skilled",
    systemPrompt: "You are a helper.",
    tools: ["readFile", "loadSkill"],
  });
  // The channel skill's source entry, also bound as an initial skill: the
  // catalog simply lists it like every other entry.
  const channelSource: SkillSource = {
    list: async () => [
      { name: "artifacts", description: "Create artifacts on request." },
    ],
    read: async (name) =>
      name === "artifacts"
        ? {
            name: "artifacts",
            description: "Create artifacts on request.",
            instructions: "When active, use createArtifact to persist results.",
            tools: ["createArtifact"],
          }
        : null,
  };
  const session = await (
    await init({
      registry: loaderRegistry,
      model,
      role: skilled(),
      initialSkills: [artifacts()],
      loadableSkills: [channelSource, agendaSource],
    })
  ).session();
  await drain(await session.prompt("ciao"));

  const sent = promptText(model.doStreamCalls[0]!);
  assert.match(sent, /<name>artifacts<\/name>/);
  assert.match(sent, /<name>agenda<\/name>/);
});

test("model-loading a skill with routing activates its tools through code mode, and a resume keeps it", async () => {
  let calls = 0;
  const model = new MockLanguageModelV4({
    doStream: async () => ({
      stream:
        calls++ === 0
          ? toolCallStream("loadSkill", { name: "analysis" })
          : textStream("ok"),
    }),
  });
  const routedRegistry = {
    ...registry,
    code: codeModeTool(),
    loadSkill: skillLoaderTool(),
  };
  const skilled = role({
    name: "skilled",
    systemPrompt: "You are a helper.",
    tools: ["readFile", "loadSkill", "code"],
  });
  // A static source hands over the real skill value: tools AND routing.
  const routedSource: SkillSource = {
    list: async () => [{ name: "analysis", description: "Analyse artifacts." }],
    read: async (name) =>
      name === "analysis"
        ? {
            name: "analysis",
            description: "Analyse artifacts.",
            instructions: "Create artifacts from programs.",
            tools: ["createArtifact"],
            toolCallers: { createArtifact: ["code"] },
          }
        : null,
  };
  const storage = new InMemorySessionStorage();
  const session = await (
    await init({
      registry: routedRegistry,
      model,
      role: skilled(),
      loadableSkills: [routedSource],
      storage,
    })
  ).session({ sessionId: "routed1" });

  const result = await session.prompt("analizza");
  await drain(result);
  await result.committed;

  // The routing was honored, same as binding would: the tool is active but
  // reachable only from the sandbox, never as a direct call.
  assert.ok(session.activeTools.includes("createArtifact"));
  assert.deepEqual(session.activeCodeTools.slice(), ["createArtifact"]);
  assert.deepEqual(toolNames(model.doStreamCalls[1]!), [
    "code",
    "loadSkill",
    "readFile",
  ]);

  // A resumed session replays the load with its routing intact.
  const resumed = await (
    await init({
      registry: routedRegistry,
      model,
      role: skilled(),
      loadableSkills: [routedSource],
      storage,
    })
  ).session({ sessionId: "routed1" });
  assert.ok(resumed.activeTools.includes("createArtifact"));
  assert.deepEqual(resumed.activeCodeTools.slice(), ["createArtifact"]);
});

test("duplicate skill names across sources fail session init", async () => {
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hi") }),
  });
  const rivalAgendaSource: SkillSource = {
    list: async () => [{ name: "agenda", description: "Also calendar work." }],
    read: async () => null,
  };
  // The role doesn't list the loader, so no catalog renders — the collision
  // must still fail init.
  await assert.rejects(
    () =>
      init({
        registry: loaderRegistry,
        model,
        role: helper(),
        loadableSkills: [agendaSource, rivalAgendaSource],
      }),
    /"agenda".*more than one/,
  );
});

// --- registry context -------------------------------------------------------
// `registry(context)` supplies context once instead of once per tool. Each tool
// still only sees what its own `contextSchema` declares, because the SDK
// validates the value through that schema before `execute` runs.

/** Two tools that declare different context, one that declares none, and a
 *  model that calls whichever tool the test names. */
function contextSetup() {
  const seen: Record<string, unknown> = {};
  const bound = createRegistry({
    scoped: harnessTool({
      description: "Declares userId only.",
      inputSchema: z.object({}),
      contextSchema: z.object({ userId: z.string() }),
      execute: async (
        _input: Record<string, never>,
        { context }: { context: { userId: string } },
      ) => {
        seen.scoped = context;
        return "ok";
      },
    }),
    filed: harnessTool({
      description: "Declares fs only.",
      inputSchema: z.object({}),
      contextSchema: z.object({ fs: z.string() }),
      execute: async (
        _input: Record<string, never>,
        { context }: { context: { fs: string } },
      ) => {
        seen.filed = context;
        return "ok";
      },
    }),
    undeclared: tool({
      description: "Declares no context at all.",
      inputSchema: z.object({}),
      execute: async (_input, options) => {
        seen.undeclared = (options as { context?: unknown }).context;
        return "ok";
      },
    }),
  });
  const everything = bound.role({
    name: "everything",
    systemPrompt: "You call tools.",
    tools: ["scoped", "filed", "undeclared"],
  });
  const modelCalling = (toolName: string) => {
    let calls = 0;
    return new MockLanguageModelV4({
      doStream: async () => ({
        stream:
          calls++ === 0 ? toolCallStream(toolName, {}) : textStream("done"),
      }),
    });
  };
  return { bound, everything, seen, modelCalling };
}

test("registry context reaches a tool projected to what its schema declares", async () => {
  const { bound, everything, seen, modelCalling } = contextSetup();
  const session = await (
    await init({
      registry: bound.registry({ userId: "u1", fs: "/home" }),
      model: modelCalling("scoped"),
      role: everything(),
    })
  ).session();

  await drain(await session.prompt("go"));
  // `fs` was supplied but never declared by this tool, so it is stripped.
  assert.deepEqual(seen.scoped, { userId: "u1" });
});

test("registry context projects differently per tool from one declaration", async () => {
  const { bound, everything, seen, modelCalling } = contextSetup();
  const session = await (
    await init({
      registry: bound.registry({ userId: "u1", fs: "/home" }),
      model: modelCalling("filed"),
      role: everything(),
    })
  ).session();

  await drain(await session.prompt("go"));
  assert.deepEqual(seen.filed, { fs: "/home" });
});

test("a tool declaring no context receives none of the shared context", async () => {
  const { bound, everything, seen, modelCalling } = contextSetup();
  const session = await (
    await init({
      registry: bound.registry({ userId: "u1", fs: "/home" }),
      model: modelCalling("undeclared"),
      role: everything(),
    })
  ).session();

  await drain(await session.prompt("go"));
  // Declaring a context is what admits a tool to the shared one; without a
  // schema the SDK would pass it through untouched, so it must be withheld.
  assert.equal(seen.undeclared, undefined);
});

test("a registry override replaces the shared context for that tool", async () => {
  const { bound, everything, seen, modelCalling } = contextSetup();
  const session = await (
    await init({
      registry: bound.registry(
        { userId: "u1", fs: "/home" },
        { filed: { fs: "/sandbox" } },
      ),
      model: modelCalling("filed"),
      role: everything(),
    })
  ).session();

  await drain(await session.prompt("go"));
  assert.deepEqual(seen.filed, { fs: "/sandbox" });
});

test("config toolsContext wins over a registry override", async () => {
  const { bound, everything, seen, modelCalling } = contextSetup();
  const session = await (
    await init({
      registry: bound.registry(
        { userId: "u1", fs: "/home" },
        { filed: { fs: "/sandbox" } },
      ),
      toolsContext: { filed: { fs: "/narrowest" } },
      model: modelCalling("filed"),
      role: everything(),
    })
  ).session();

  await drain(await session.prompt("go"));
  assert.deepEqual(seen.filed, { fs: "/narrowest" });
});

test("a bare toolset still works, with no context at all", async () => {
  const { seen, modelCalling } = contextSetup();
  const bare = {
    undeclared: tool({
      description: "Declares no context at all.",
      inputSchema: z.object({}),
      execute: async () => {
        seen.bare = "ran";
        return "ok";
      },
    }),
  };
  const only = role({
    name: "only",
    systemPrompt: "You call tools.",
    tools: ["undeclared"],
  });
  const session = await (
    await init({
      registry: bare,
      model: modelCalling("undeclared"),
      role: only(),
    })
  ).session();

  await drain(await session.prompt("go"));
  assert.equal(seen.bare, "ran");
});

// --- toUIMessageStream ------------------------------------------------------
// The bridge to a browser. The defaults are the union of what hand-written
// route handlers got right separately: id alignment with the persisted row,
// createdAt on every part, and the turn's timings on the finish part.

/** Read a UI-message stream into an array of chunks. */
async function readChunks(
  stream: ReadableStream<{ type: string; [k: string]: unknown }>,
) {
  const chunks: Array<{ type: string; [k: string]: unknown }> = [];
  const reader = stream.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  return chunks;
}

function uiStreamSetup() {
  const storage = new InMemorySessionStorage();
  const model = new MockLanguageModelV4({
    doStream: async () => ({ stream: textStream("hello") }),
  });
  const helper = role({ name: "helper", systemPrompt: "You help." });
  return { storage, model, helper };
}

test("the streamed message id is the id the turn is persisted under", async () => {
  const { storage, model, helper } = uiStreamSetup();
  const session = await (
    await init({ registry: {}, model, role: helper(), storage })
  ).session({ sessionId: "ui1" });

  const result = await session.prompt("hi");
  const chunks = await readChunks(
    result.toUIMessageStream() as ReadableStream<{
      type: string;
      [k: string]: unknown;
    }>,
  );
  await result.committed;

  const start = chunks.find((c) => c.type === "start");
  assert.equal(
    start?.messageId,
    result.responseMessageId,
    "the start chunk must carry the response id",
  );

  const saved = await storage.loadMessages("ui1");
  const assistant = saved.at(-1)!;
  assert.equal(assistant.role, "assistant");
  // The whole point: the client-held message and the stored row agree.
  assert.equal(assistant.id, result.responseMessageId);
});

test("createdAt is stamped on every part and timings on the finish part", async () => {
  const { storage, model, helper } = uiStreamSetup();
  const session = await (
    await init({ registry: {}, model, role: helper(), storage })
  ).session({ sessionId: "ui2" });

  const result = await session.prompt("hi");
  const chunks = await readChunks(
    result.toUIMessageStream() as ReadableStream<{
      type: string;
      [k: string]: unknown;
    }>,
  );
  await result.committed;

  // Interim metadata arrives as its own `message-metadata` chunk; the finish
  // part's rides on the `finish` chunk itself.
  const metadata = chunks
    .filter((c) => c.type === "message-metadata" || c.type === "finish")
    .map((c) => c.messageMetadata as { createdAt?: number; timings?: unknown });
  assert.ok(metadata.length > 1, "expected metadata on the stream");
  assert.ok(
    metadata.every((m) => typeof m.createdAt === "number"),
    "every part carries createdAt",
  );
  const timed = metadata.filter((m) => m.timings !== undefined);
  assert.equal(timed.length, 1, "timings ride on the finish part alone");
  assert.equal(
    typeof (timed[0]!.timings as { totalMs: number }).totalMs,
    "number",
  );
  const finish = chunks.find((c) => c.type === "finish");
  assert.ok(
    (finish?.messageMetadata as { timings?: unknown })?.timings !== undefined,
    "the finish chunk is the one carrying timings",
  );
});

test("toUIMessageStream defaults are overridable", async () => {
  const { storage, model, helper } = uiStreamSetup();
  const session = await (
    await init({ registry: {}, model, role: helper(), storage })
  ).session({ sessionId: "ui3" });

  const result = await session.prompt("hi");
  const chunks = await readChunks(
    result.toUIMessageStream({
      messageMetadata: () => ({ mine: true }),
    }) as ReadableStream<{ type: string; [k: string]: unknown }>,
  );
  await result.committed;

  const metadata = chunks
    .filter((c) => c.type === "message-metadata" || c.type === "finish")
    .map((c) => c.messageMetadata as Record<string, unknown>);
  assert.ok(metadata.length > 0);
  assert.ok(
    metadata.every((m) => m.mine === true && m.createdAt === undefined),
    "a caller's messageMetadata replaces the harness default",
  );
});

test("the response copy does not starve the harness's own persistence", async () => {
  const { storage, model, helper } = uiStreamSetup();
  const session = await (
    await init({ registry: {}, model, role: helper(), storage })
  ).session({ sessionId: "ui4" });

  const result = await session.prompt("hi");
  // Drain the caller's copy fully, then assert the turn still persisted — the
  // stream tees per access, which is the fact the old hand-written glue needed
  // a comment to explain.
  await readChunks(
    result.toUIMessageStream() as ReadableStream<{
      type: string;
      [k: string]: unknown;
    }>,
  );
  await result.committed;

  const saved = await storage.loadMessages("ui4");
  assert.equal(saved.at(-1)?.role, "assistant");
  const text = saved
    .at(-1)!
    .parts.filter((p) => p.type === "text")
    .map((p) => (p as { text: string }).text)
    .join("");
  assert.equal(text, "hello");
});

test("toUIMessageStreamResponse returns a streaming Response with the caller's headers", async () => {
  const { storage, model, helper } = uiStreamSetup();
  const session = await (
    await init({ registry: {}, model, role: helper(), storage })
  ).session({ sessionId: "ui5" });

  const result = await session.prompt("hi");
  const response = result.toUIMessageStreamResponse({
    headers: { "access-control-allow-origin": "https://example.com" },
  });
  assert.equal(response.status, 200);
  assert.equal(
    response.headers.get("access-control-allow-origin"),
    "https://example.com",
  );
  assert.match(response.headers.get("content-type") ?? "", /event-stream/);

  const body = await response.text();
  await result.committed;
  assert.ok(body.includes(result.responseMessageId), "the id reaches the wire");
  assert.ok(body.includes("hello"), "the text reaches the wire");
});

// --- registry.only ----------------------------------------------------------
// The context a registry asks for is the intersection of its tools' schemas, so
// a registry holding every tool demands every tool's context at once. `only`
// narrows the registry to what one environment can serve, and the context
// narrows with it.

test("only narrows the context the registry asks for", async () => {
  const { bound, seen, modelCalling } = contextSetup();
  const scopedOnly = bound.registry.only("scoped");
  const reader = bound.role({
    name: "reader",
    systemPrompt: "You call tools.",
    tools: ["scoped"],
  });
  const session = await (
    await init({
      // No `fs`: the only tool that declares it is outside this registry.
      registry: scopedOnly({ userId: "u1" }),
      model: modelCalling("scoped"),
      role: reader(),
    })
  ).session();

  await drain(await session.prompt("go"));
  assert.deepEqual(seen.scoped, { userId: "u1" });
});

test("a tool outside the narrowing is not registered, so a role naming it fails", async () => {
  const { bound, everything, modelCalling } = contextSetup();
  await assert.rejects(
    async () =>
      await init({
        registry: bound.registry.only("scoped")({ userId: "u1" }),
        model: modelCalling("scoped"),
        // `everything` lists filed and undeclared too.
        role: everything(),
      }).then((harness) => harness.session()),
    /not in the harness registry: filed, undeclared/,
  );
});

test("only is closed: a narrowed registry narrows again", async () => {
  const { bound, seen, modelCalling } = contextSetup();
  const narrowed = bound.registry.only("scoped", "filed").only("scoped");
  const reader = bound.role({
    name: "reader",
    systemPrompt: "You call tools.",
    tools: ["scoped"],
  });
  const session = await (
    await init({
      registry: narrowed({ userId: "u1" }),
      model: modelCalling("scoped"),
      role: reader(),
    })
  ).session();

  await drain(await session.prompt("go"));
  assert.deepEqual(seen.scoped, { userId: "u1" });
});

test("only rejects a name the registry does not hold", () => {
  const { bound } = contextSetup();
  assert.throws(
    () => (bound.registry.only as (...n: string[]) => unknown)("nope"),
    /Unknown tool in registry\.only\(\): "nope"/,
  );
});

test("an override still applies inside a narrowing", async () => {
  const { bound, seen, modelCalling } = contextSetup();
  const filedOnly = bound.registry.only("filed");
  const reader = bound.role({
    name: "reader",
    systemPrompt: "You call tools.",
    tools: ["filed"],
  });
  const session = await (
    await init({
      registry: filedOnly({ fs: "/home" }, { filed: { fs: "/sandbox" } }),
      model: modelCalling("filed"),
      role: reader(),
    })
  ).session();

  await drain(await session.prompt("go"));
  assert.deepEqual(seen.filed, { fs: "/sandbox" });
});
