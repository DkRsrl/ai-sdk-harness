import { randomUUID } from "node:crypto";
import {
  convertToModelMessages,
  getToolName,
  isToolUIPart,
  Output,
  toUIMessageStream,
  ToolLoopAgent,
  type GenericToolApprovalFunction,
  type LanguageModel,
  type PrepareStepFunction,
  type Tool,
  type ToolApprovalResponse,
  type ToolSet,
} from "ai";
import type { Context, InferToolSetContext } from "@ai-sdk/provider-utils";
import {
  convertToRealtimeSeed,
  createRealtimeSession,
  toUIMessages,
  type RealtimeMessage,
  type RealtimeModelV1,
} from "./voice";
import {
  assertKnownCallers,
  CODE_MODE_TOOL_NAME,
  codeModeBinder,
  codeModeOnlyNames,
  codeModeToolNames,
  DIRECT_TOOL_CALL,
  routesToCodeMode,
  type ToolCaller,
  type ToolCallers,
} from "./code-mode";
import { formatMessageMetadata, withMessageMetadata } from "./metadata";
import { createTimingsRecorder, withTimings } from "./timings";
import {
  turnToUIMessageStream,
  turnToUIMessageStreamResponse,
  type UIMessageStreamOptions,
  type UIMessageStreamResponseOptions,
} from "./ui-stream";
import { runWithSession } from "./session-context";
import {
  assembleInstructions,
  formatSkillInit,
  mergeSkillSources,
  SKILL_LOADER_TOOL_NAME,
  skillLoaderBinder,
  type ParsedSkill,
} from "./skills";
import { InMemorySessionStorage, type SessionMessage } from "./storage";
import type {
  DriveModel,
  ResolvedHarnessConfig,
  PromptOptions,
  Session,
  SessionFor,
  SessionOptions,
  StreamResult,
  SubsessionOptions,
  TextSession,
  VoiceOptions,
  VoiceSession,
} from "./types";

/** Fail loudly if a role/skill references a tool the registry doesn't contain —
 *  a typo would otherwise silently expose nothing. */
function assertToolsRegistered(
  label: string,
  names: readonly string[],
  registry: ToolSet,
): void {
  const unknown = names.filter((name) => !(name in registry));
  if (unknown.length > 0) {
    throw new Error(
      `${label}: tool(s) not in the harness registry: ${unknown.join(", ")}. Known tools: ${Object.keys(registry).join(", ") || "(none)"}`,
    );
  }
}

/** Wrap every tool's `execute` so it runs inside `runWithSession`, making
 *  `currentSession()` resolve to this session during the tool call. */
function wrapToolsWithSession<TTools extends ToolSet>(
  tools: TTools,
  getSession: () => Session<TTools>,
): TTools {
  return Object.fromEntries(
    Object.entries(tools).map(([name, t]) => {
      const original = t as Tool;
      if (typeof original.execute !== "function") return [name, original];
      const wrapped: Tool = {
        ...original,
        execute: (input: unknown, opts: unknown) =>
          runWithSession(getSession() as unknown as Session<never>, () =>
            (original.execute as (i: unknown, o: unknown) => unknown)(
              input,
              opts,
            ),
          ),
      } as Tool;
      return [name, wrapped];
    }),
  ) as TTools;
}

/** The SDK approval responses carried by a client's re-sent assistant message:
 *  one `tool-approval-response` per tool part in `approval-responded` state
 *  (the state `addToolApprovalResponse` writes). */
function approvalResponsesIn(message: SessionMessage): ToolApprovalResponse[] {
  return message.parts.filter(isToolUIPart).flatMap((part) =>
    part.state === "approval-responded"
      ? [
          {
            type: "tool-approval-response" as const,
            approvalId: part.approval.id,
            approved: part.approval.approved,
            ...(part.approval.reason ? { reason: part.approval.reason } : {}),
          },
        ]
      : [],
  );
}

/** The routing map a persisted loader output carries, if any — replayed with
 *  the same shape-tolerance as the rest of the transcript scan: entries that
 *  don't parse as a routing declaration over the skill's own tools are
 *  dropped, never thrown on. */
function persistedToolCallers(
  output: object,
  ownTools: readonly string[],
): ToolCallers {
  const raw = (output as { toolCallers?: unknown }).toolCallers;
  if (typeof raw !== "object" || raw === null) return {};
  const known: readonly string[] = [CODE_MODE_TOOL_NAME, DIRECT_TOOL_CALL];
  return Object.fromEntries(
    Object.entries(raw).filter(
      ([name, callers]) =>
        ownTools.includes(name) &&
        Array.isArray(callers) &&
        callers.every((c) => typeof c === "string" && known.includes(c)),
    ),
  ) as ToolCallers;
}

/** A user message's text, flattened the way every chat surface sends it. */
function flattenedText(message: SessionMessage): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n")
    .trim();
}

/** The drive is chosen by the model's spec: realtime models carry the
 *  `"realtime-v1"` marker (a model-id string or a text LanguageModelV2 does
 *  not), and get the voice session; everything else gets the text session. */
function isRealtimeModel(model: DriveModel): model is RealtimeModelV1 {
  return (
    typeof model === "object" &&
    model !== null &&
    (model as RealtimeModelV1).specificationVersion === "realtime-v1"
  );
}

export function createSession<
  TTools extends ToolSet,
  TModel extends DriveModel = DriveModel,
  TOutput = never,
  TSkillName extends string = string,
>(
  config: ResolvedHarnessConfig<TTools, TModel, TOutput, TSkillName>,
  instructions: string,
  opts: SessionOptions = {},
): SessionFor<TTools, TModel, TOutput, TSkillName> {
  const generateId = config.generateId ?? randomUUID;
  const id = opts.sessionId ?? generateId();
  const messages: SessionMessage[] = [...(opts.messages ?? [])];
  // Always-present persistence (in-memory when the caller configured none), so
  // the session and its tools can rely on `session.storage`.
  const storage = config.storage ?? new InMemorySessionStorage();
  let pendingSave: Promise<void> = Promise.resolve();
  let sessionRef: Session<TTools>;

  // The active set: role tools form the baseline; invoking a skill unions its
  // tools in (sticky for the rest of the session). Alongside it, the session
  // merges the role's (then each skill's) `toolCallers` routing map — the
  // code-mode declarations, in the SDK's own vocabulary. The whole registry
  // is wrapped so `currentSession()` resolves inside any tool; only the
  // active subset is exposed (per step for text, advertised at connect for
  // voice, which binds the sandbox once at connect).
  assertToolsRegistered(
    `role("${config.role.name}")`,
    config.role.tools,
    config.registry,
  );
  if (Array.isArray(config.toolApproval)) {
    assertToolsRegistered("toolApproval", config.toolApproval, config.registry);
  }
  const activeTools = new Set<string>(config.role.tools);
  const toolCallers: Record<string, readonly ToolCaller[]> = {};

  /** Merge a role/skill's routing map in, holding the code-mode invariants:
   *  routing keys must come from the declarer's own tools, routing anything
   *  into the sandbox requires `code` among the session's active tools
   *  (the sandbox is a registry tool the role/skill lists explicitly), and an
   *  approval-gated tool must stay a direct call (the sandbox rejects a
   *  nested call that needs approval instead of pausing for the user). */
  function mergeToolCallers(
    label: string,
    map: ToolCallers,
    ownTools: readonly string[],
  ): void {
    const names = Object.keys(map);
    if (names.length === 0) return;
    assertKnownCallers(label, map);
    const foreign = names.filter((name) => !ownTools.includes(name));
    if (foreign.length > 0) {
      throw new Error(
        `${label}: toolCallers name(s) not in its own tools list: ${foreign.join(", ")}`,
      );
    }
    const codeRouted = names.filter((name) => routesToCodeMode(map[name]));
    if (codeRouted.length > 0 && !activeTools.has(CODE_MODE_TOOL_NAME)) {
      throw new Error(
        `${label}: routing ${codeRouted.join(", ")} through code mode requires "${CODE_MODE_TOOL_NAME}" among the active tools — list it in the role's (or an earlier skill's) tools`,
      );
    }
    if (
      codeRouted.length > 0 &&
      codeModeBinder(config.registry[CODE_MODE_TOOL_NAME]) === undefined
    ) {
      throw new Error(
        `${label}: the "${CODE_MODE_TOOL_NAME}" registry entry is not the harness sandbox — wire codeModeTool() from "@dkr/ai-sdk-harness" under that key`,
      );
    }
    if (Array.isArray(config.toolApproval)) {
      const gated = codeRouted.filter((name) =>
        (config.toolApproval as readonly string[]).includes(name),
      );
      if (gated.length > 0) {
        throw new Error(
          `${label}: ${gated.join(", ")} cannot be routed through code mode — the sandbox has no approval flow`,
        );
      }
    }
    Object.assign(toolCallers, map);
  }
  mergeToolCallers(
    `role("${config.role.name}")`,
    config.role.toolCallers,
    config.role.tools,
  );

  /** The shared half of injecting a skill, whichever route it arrives by:
   *  union its tools into the active set and merge its routing map. */
  function activateSkillTools(
    label: string,
    tools: readonly string[],
    callers: ToolCallers,
  ): void {
    assertToolsRegistered(label, tools, config.registry);
    for (const name of tools) activeTools.add(name);
    mergeToolCallers(label, callers, tools);
  }

  // Initial skills shape the baseline: their tools and routing are active
  // from the first turn (their instructions already ride in `instructions`,
  // resolved by init).
  for (const initial of config.initialSkills ?? []) {
    activateSkillTools(
      `initial skill("${initial.name}")`,
      initial.tools,
      initial.toolCallers,
    );
  }

  for (const message of messages) {
    for (const part of message.parts) {
      if (
        !isToolUIPart(part) ||
        getToolName(part) !== SKILL_LOADER_TOOL_NAME ||
        part.state !== "output-available"
      ) {
        continue;
      }
      const loaded = part.output;
      if (
        typeof loaded !== "object" ||
        loaded === null ||
        !("name" in loaded) ||
        typeof loaded.name !== "string" ||
        !("tools" in loaded) ||
        !Array.isArray(loaded.tools) ||
        !loaded.tools.every((name) => typeof name === "string")
      ) {
        continue;
      }
      activateSkillTools(
        `loaded skill("${loaded.name}")`,
        loaded.tools,
        persistedToolCallers(loaded, loaded.tools),
      );
    }
  }

  const wrappedTools = wrapToolsWithSession<TTools>(
    config.registry,
    () => sessionRef,
  );

  // The model-facing skill loader, bound once per session to the configured
  // sources (merged into one): a loaded skill's instructions return as the
  // tool result, and its declared tools join the active set. The toggle
  // reaches the model per step on text; voice advertises tools once at
  // connect, so there it only ever feeds code-mode routing declared elsewhere
  // — a skill meant for voice should declare no tools of its own.
  const skills = mergeSkillSources(config.loadableSkills ?? []);
  const loaderBind = skillLoaderBinder(config.registry[SKILL_LOADER_TOOL_NAME]);
  const boundSkillLoader =
    loaderBind !== undefined && skills !== undefined
      ? loaderBind({
          source: skills,
          activate: (skill) =>
            activateSkillTools(
              `skill("${skill.name}")`,
              skill.tools,
              skill.toolCallers ?? {},
            ),
        })
      : undefined;

  async function persist(toSave: SessionMessage[]): Promise<void> {
    await storage.saveMessages(id, toSave);
  }

  // ── Shared (drive-agnostic) session methods ──────────────────────────────

  async function skillImpl(name: string): Promise<ParsedSkill> {
    await pendingSave;
    const resolved = await skills?.read(name);
    if (!resolved) {
      const available = (await skills?.list()) ?? [];
      throw new Error(
        available.length
          ? `session.skill("${name}"): unknown skill — available: ${available.map((s) => s.name).join(", ")}`
          : `session.skill("${name}"): no skills are available in this session — declare the skill in a loadableSkills source`,
      );
    }
    activateSkillTools(
      `skill("${resolved.name}")`,
      resolved.tools,
      resolved.toolCallers ?? {},
    );

    const message: SessionMessage = {
      id: generateId(),
      role: "user",
      parts: [{ type: "text", text: formatSkillInit(resolved) }],
      createdAt: new Date(),
    };
    messages.push(message);
    await persist([message]);
    return resolved;
  }

  async function subsessionImpl<
    TSubModel extends DriveModel = LanguageModel,
    TSubOutput = never,
  >(
    callOpts: SubsessionOptions<TSubModel, TSubOutput> = {},
  ): Promise<SessionFor<TTools, TSubModel, TSubOutput, TSkillName>> {
    const subRole = callOpts.role ?? config.role;
    const subModel = (callOpts.model ?? config.model) as DriveModel;
    const subInitial = callOpts.initialSkills ?? [];
    const subInstructions = await assembleInstructions(
      subRole,
      subInitial,
      skills,
    );
    const subConfig = {
      registry: config.registry,
      loadableSkills: config.loadableSkills,
      model: subModel,
      reasoning: callOpts.reasoning ?? config.reasoning,
      // A subsession that switched model must not keep the parent's routing:
      // provider options are chosen for a specific model.
      providerOptions: callOpts.model
        ? callOpts.providerOptions
        : (callOpts.providerOptions ?? config.providerOptions),
      role: subRole,
      initialSkills: subInitial,
      toolsContext: config.toolsContext,
      hooks: config.hooks,
      storage,
      generateId: config.generateId,
    } as ResolvedHarnessConfig<TTools, TSubModel, TSubOutput, TSkillName>;

    const subSessionId = callOpts.sessionId ?? generateId();
    let subMessages: SessionMessage[] | undefined;
    if (callOpts.sessionId) {
      subMessages = await storage.loadMessages(callOpts.sessionId);
    }
    await storage.createSession?.(subSessionId, {
      title: callOpts.title ?? "Subsession",
      parentId: id,
    });

    return createSession(subConfig, subInstructions, {
      sessionId: subSessionId,
      messages: subMessages,
    }) as SessionFor<TTools, TSubModel, TSubOutput, TSkillName>;
  }

  // ── Voice drive ──────────────────────────────────────────────────────────

  if (isRealtimeModel(config.model)) {
    const voiceModel = config.model;

    /** Persist a finalized realtime turn into the same storage the text drive
     *  uses, in the same UIMessage shape (`toUIMessages` is the inverse of
     *  `convertToRealtimeSeed`). The ordering stamp rides in `metadata.createdAt`
     *  (epoch ms); lift it to top-level `createdAt` too so a later text turn's
     *  `<message_metadata>` decoration sees it. Serialized through `pendingSave`
     *  so concurrent turns don't race the store. */
    function appendVoiceTurn(message: RealtimeMessage<TTools>): void {
      pendingSave = pendingSave.then(async () => {
        const [ui] = toUIMessages([message as RealtimeMessage]);
        if (!ui) return;
        const createdAtMs = (ui.metadata as { createdAt?: number } | undefined)
          ?.createdAt;
        const stored: SessionMessage = {
          ...ui,
          ...(typeof createdAtMs === "number"
            ? { createdAt: new Date(createdAtMs) }
            : {}),
        };
        // The core may re-emit a turn (same id) to repair one it settled
        // early; mirror the storage upsert so the transcript holds one entry.
        const at = messages.findIndex((m) => m.id === stored.id);
        if (at >= 0) messages[at] = stored;
        else messages.push(stored);
        await persist([stored]);
      });
      void pendingSave.catch(() => {});
    }

    const voiceSession: VoiceSession<TTools> = {
      id,
      storage,
      get messages() {
        return messages;
      },
      get activeTools() {
        return [...activeTools];
      },
      get activeCodeTools() {
        return codeModeToolNames(toolCallers);
      },
      skill: skillImpl,
      subsession: subsessionImpl,

      async voice(voiceOpts: VoiceOptions<TTools> = {}) {
        const { triggerResponse, audio, timezone, onMessage, ...callbacks } =
          voiceOpts;
        // Symmetric with the text drive: hand over the whole (wrapped) registry
        // plus the role/skill-active set, and let the voice core advertise only
        // the active subset. The full registry feeds the seed so a replayed tool
        // round-trip keeps its `toModelOutput` even if it's no longer active.
        // Code mode binds ONCE, here at connect (realtime providers advertise
        // tools once, not per step): the registry's `code` entry is
        // rebound to the routed tools, and code-only routed names drop out of
        // the advertised set — the sandbox runs in this process, so the
        // realtime provider calls it like any other tool.
        const bind = codeModeBinder(config.registry[CODE_MODE_TOOL_NAME]);
        const bound =
          activeTools.has(CODE_MODE_TOOL_NAME) && bind !== undefined
            ? bind({
                tools: wrappedTools,
                names: codeModeToolNames(toolCallers).filter((name) =>
                  activeTools.has(name),
                ),
                toolsContext: config.toolsContext as
                  | Record<string, unknown>
                  | undefined,
              })
            : undefined;
        const codeOnly = new Set(codeModeOnlyNames(toolCallers));
        // Temporal grounding, per seam: seeded history turns carry the same
        // per-turn `<message_metadata>` decoration the text drive applies;
        // live turns — which the provider transcribes out of our reach — get
        // theirs as injected items, rendered by the same formatter.
        const realtime = await createRealtimeSession<TTools>({
          model: voiceModel,
          // The routed tools' catalog rides in the instructions, not in the
          // sandbox tool's description (opencode style) — rendered once here,
          // as everything voice advertises is.
          instructions: bound?.catalogInstructions
            ? `${instructions}\n\n${bound.catalogInstructions}`
            : instructions,
          tools: {
            ...wrappedTools,
            ...(bound ? { [CODE_MODE_TOOL_NAME]: bound.tool } : {}),
            ...(boundSkillLoader
              ? { [SKILL_LOADER_TOOL_NAME]: boundSkillLoader }
              : {}),
          } as TTools,
          activeTools: [...activeTools].filter(
            (name) => !codeOnly.has(name),
          ) as Array<keyof TTools & string>,
          toolsContext: config.toolsContext as
            | InferToolSetContext<TTools>
            | undefined,
          seed: await convertToRealtimeSeed(messages.map(withMessageMetadata), {
            tools: wrappedTools,
          }),
          turnMetadata: () => formatMessageMetadata(new Date(), timezone),
          triggerResponse: triggerResponse ?? false,
          audio,
          generateId,
          ...callbacks,
          // The harness owns persistence; a consumer `onMessage` is additive.
          onMessage: (message) => {
            appendVoiceTurn(message);
            onMessage?.(message);
          },
        });
        // The harness owns persistence, so it owns draining it. `stop` settles
        // the turn that was in flight, and whoever reads the transcript next
        // does so the moment `stop` resolves — a successor session on the same
        // conversation starts on exactly that edge.
        return {
          ...realtime,
          get status() {
            return realtime.status;
          },
          async stop() {
            await realtime.stop();
            await pendingSave;
          },
        };
      },
    };
    sessionRef = voiceSession;
    // A realtime drive has no structured output: `SessionFor` collapses to
    // `VoiceSession<TTools>` here regardless of the role's `outputSchema`.
    return voiceSession as SessionFor<TTools, TModel, TOutput, TSkillName>;
  }

  // ── Text drive ─────────────────────────────────────────────────────────────

  // Single control point for tool visibility: emit the current active set,
  // minus the tools routed into the sandbox with no direct call left — the
  // harness hides those from the model itself (they stay callable only from
  // inside `code`, which the role listed as a normal tool). Any
  // consumer-supplied prepareStep hook can override/extend the result.
  const prepareStep: PrepareStepFunction<TTools, Context> = async (params) => {
    const userResult = (await config.hooks?.prepareStep?.(params)) ?? {};
    const codeOnly = new Set(codeModeOnlyNames(toolCallers));
    const names = [...activeTools].filter((name) => !codeOnly.has(name));
    return {
      activeTools: names as Array<keyof TTools & string>,
      ...userResult,
    };
  };

  // A role's `outputSchema` becomes the model's structured-output spec: the
  // reply is constrained to the schema and surfaces (validated) on
  // `result.output`. Absent a schema the role yields free prose as before.
  const output = config.role.outputSchema
    ? Output.object<TOutput>({ schema: config.role.outputSchema })
    : undefined;

  // A name list becomes the generic approval function the SDK expects: listed
  // tools stop for the user, everything else runs untouched. A function passes
  // through as the caller's own policy.
  const toolApproval = Array.isArray(config.toolApproval)
    ? ((({ toolCall }) =>
        (config.toolApproval as readonly string[]).includes(toolCall.toolName)
          ? "user-approval"
          : undefined) satisfies GenericToolApprovalFunction<
        TTools,
        InferToolSetContext<TTools>,
        Context
      >)
    : config.toolApproval;

  // The agent is (re)built per turn: a skill invoked mid-session may have
  // grown the active set or the routing map since the last turn. When the
  // sandbox is active, the registry's `code` entry is rebound to the
  // session (AI SDK style: the tool carries its own binder) — the routed
  // tools' catalog re-rendered into the instructions, execute running the
  // confined interpreter over the same wrapped tool objects the direct path
  // uses.
  function buildAgent() {
    const bind = codeModeBinder(config.registry[CODE_MODE_TOOL_NAME]);
    const bound =
      activeTools.has(CODE_MODE_TOOL_NAME) && bind !== undefined
        ? bind({
            tools: wrappedTools,
            names: codeModeToolNames(toolCallers).filter((name) =>
              activeTools.has(name),
            ),
            toolsContext: config.toolsContext as
              | Record<string, unknown>
              | undefined,
          })
        : undefined;
    return new ToolLoopAgent<never, TTools, Context, never>({
      model: config.model as LanguageModel,
      reasoning: config.reasoning,
      ...(config.providerOptions
        ? { providerOptions: config.providerOptions }
        : {}),
      // The routed tools' catalog rides in the instructions, not in the
      // sandbox tool's description (opencode style); rebuilt per turn like
      // the rest of the agent, so a skill-grown tool set re-renders it.
      instructions: bound?.catalogInstructions
        ? `${instructions}\n\n${bound.catalogInstructions}`
        : instructions,
      tools: {
        ...wrappedTools,
        ...(bound ? { [CODE_MODE_TOOL_NAME]: bound.tool } : {}),
        ...(boundSkillLoader
          ? { [SKILL_LOADER_TOOL_NAME]: boundSkillLoader }
          : {}),
      },
      toolsContext: config.toolsContext,
      prepareCall: config.hooks?.prepareCall,
      prepareStep,
      ...(toolApproval ? { toolApproval } : {}),
      ...(output ? { output } : {}),
    } as ConstructorParameters<
      typeof ToolLoopAgent<never, TTools, Context, never>
    >[0]);
  }

  // Shared tail of every text turn (user prompt or approval response): rebuild
  // the model messages from the stored history, stream, and persist the run's
  // output in the background.
  async function run(
    runOpts: Pick<PromptOptions, "abortSignal">,
  ): Promise<StreamResult<TTools, TOutput>> {
    // Decorate user turns with `<message_metadata>` (the send time) for the
    // model only — `messages` (persisted + UI-rendered) keeps the raw text.
    const modelMessages = await convertToModelMessages(
      messages.map(withMessageMetadata),
    );

    // The SDK times every model call and tool execution itself; the recorder
    // aggregates those telemetry events into the turn's `TurnTimings`.
    const timings = createTimingsRecorder();

    const agent = buildAgent();
    const result = await agent.stream({
      messages: modelMessages,
      toolsContext: config.toolsContext,
      abortSignal: runOpts.abortSignal,
      onStepEnd: ({ performance }) => timings.step(performance),
      onToolExecutionEnd: ({ toolCall, toolExecutionMs }) =>
        timings.tool({
          callId: toolCall.toolCallId,
          name: toolCall.toolName,
          ms: toolExecutionMs,
        }),
    } as Parameters<typeof agent.stream>[0]);

    // Drain a UI stream in the background to drive onFinish, which gives us
    // the fully-assembled assistant message(s) to persist. The returned
    // result still streams independently to the caller (the SDK tees from the
    // underlying stream, so consuming `result.stream` here doesn't starve it).
    // Mint the assistant message's id up front instead of letting the SDK
    // generate it inside the persistence stream: the caller's copy of the
    // stream has to stamp the SAME id, or a client-held message and its stored
    // row disagree and per-message feedback has nothing to key on. Later
    // messages in the same turn, if any, keep generating their own.
    const responseMessageId = generateId();
    let responseIdTaken = false;
    const nextMessageId = () => {
      if (responseIdTaken) return generateId();
      responseIdTaken = true;
      return responseMessageId;
    };

    const committed = (pendingSave = (async () => {
      const uiStream = toUIMessageStream<TTools, SessionMessage>({
        stream: result.stream,
        originalMessages: messages,
        generateMessageId: nextMessageId,
        onFinish: async ({ messages: updated }) => {
          // The run's latency profile rides on its assistant message, next to
          // `createdAt`, so it persists with the transcript.
          const last = updated.at(-1);
          if (last?.role === "assistant") {
            updated[updated.length - 1] = {
              ...last,
              metadata: withTimings(last.metadata, timings.finish()),
            };
          }
          messages.splice(0, messages.length, ...updated);
          await persist(updated);
        },
      });
      const reader = uiStream.getReader();
      try {
        // drain — consuming the stream is what fires onFinish
        while (!(await reader.read()).done) {
          /* discard chunks */
        }
      } catch (err) {
        // An aborted run may tear the stream mid-read; that's the caller's
        // requested outcome, not a persistence failure to surface later.
        const aborted =
          runOpts.abortSignal?.aborted &&
          (err === runOpts.abortSignal.reason ||
            (err instanceof Error && err.name === "AbortError"));
        if (!aborted) throw err;
      } finally {
        reader.releaseLock();
      }
    })());
    void committed.catch(() => {});

    // The agent is typed with a `never` output spec (the generic is fixed at
    // construction); the role's schema makes the runtime result carry the
    // parsed `TOutput`, so bridge the static type here. `timings()` exposes
    // the run's recorder so a route can stream the numbers to its client.
    const uiSource = {
      stream: result.stream,
      responseMessageId,
      timings: () => timings.finish(),
      now: () => Date.now(),
    };
    return Object.assign(result, {
      timings: () => timings.finish(),
      committed,
      responseMessageId,
      // `result.stream` tees per access, so this copy is independent of the
      // persistence consumer draining above.
      toUIMessageStream: (options?: UIMessageStreamOptions<TTools>) =>
        turnToUIMessageStream(uiSource, options),
      toUIMessageStreamResponse: (
        options?: UIMessageStreamResponseOptions<TTools>,
      ) => turnToUIMessageStreamResponse(uiSource, options),
    }) as unknown as StreamResult<TTools, TOutput>;
  }

  // An approval turn: the SDK `ToolApprovalResponse`s are applied to the
  // session's own copy of the pending requests — never to caller-supplied
  // message content — so a client can only answer approvals this session
  // actually issued. All pending requests must be answered at once: the SDK
  // refuses to resume a loop with open requests, and failing here names the
  // ids instead of surfacing a MissingToolResultsError later.
  async function approvalTurn(
    responses: ToolApprovalResponse[],
    promptOpts: PromptOptions,
  ): Promise<StreamResult<TTools, TOutput>> {
    await pendingSave;

    const last = messages.at(-1);
    const pending = new Set<string>();
    if (last?.role === "assistant") {
      for (const part of last.parts) {
        if (isToolUIPart(part) && part.state === "approval-requested") {
          pending.add(part.approval.id);
        }
      }
    }
    if (pending.size === 0) {
      throw new Error("prompt: the session has no pending approval requests");
    }
    const unknown = responses.filter((r) => !pending.has(r.approvalId));
    if (unknown.length > 0) {
      throw new Error(
        `prompt: no pending approval request with id(s) ${unknown.map((r) => r.approvalId).join(", ")}`,
      );
    }
    const answered = new Set(responses.map((r) => r.approvalId));
    const unanswered = [...pending].filter((p) => !answered.has(p));
    if (unanswered.length > 0) {
      throw new Error(
        `prompt: approval request(s) ${unanswered.join(", ")} still need a response`,
      );
    }

    const byId = new Map(responses.map((r) => [r.approvalId, r]));
    // Same rewrite `addToolApprovalResponse` does client-side, applied to the
    // stored copy: the requested part flips to `approval-responded`.
    const updated: SessionMessage = {
      ...(last as SessionMessage),
      parts: (last as SessionMessage).parts.map((part) => {
        if (!isToolUIPart(part) || part.state !== "approval-requested") {
          return part;
        }
        const response = byId.get(part.approval.id)!;
        return {
          ...part,
          state: "approval-responded" as const,
          approval: {
            id: response.approvalId,
            approved: response.approved,
            reason: response.reason,
          },
        };
      }),
    };
    messages[messages.length - 1] = updated;
    await persist([updated]);

    return run(promptOpts);
  }

  const textSession: TextSession<TTools, TOutput> = {
    id,
    storage,
    get messages() {
      return messages;
    },
    get activeTools() {
      return [...activeTools];
    },
    get activeCodeTools() {
      return codeModeToolNames(toolCallers);
    },
    skill: skillImpl,
    subsession: subsessionImpl,

    async prompt(
      input: string | SessionMessage,
      promptOpts: PromptOptions = {},
    ): Promise<StreamResult<TTools, TOutput>> {
      // A re-sent assistant message is an approval turn — the only thing read
      // from it is its `tool-approval-response`s.
      if (typeof input !== "string" && input.role === "assistant") {
        return approvalTurn(approvalResponsesIn(input), promptOpts);
      }

      const text = typeof input === "string" ? input : flattenedText(input);
      if (typeof input !== "string" && text === "") {
        throw new Error("prompt: the message carries no text");
      }

      await pendingSave;

      const userMessage: SessionMessage = {
        id: generateId(),
        role: "user",
        parts: [{ type: "text", text }],
        createdAt: new Date(),
        ...(promptOpts.timezone
          ? { metadata: { timezone: promptOpts.timezone } }
          : {}),
      };
      messages.push(userMessage);
      await persist([userMessage]);

      return run(promptOpts);
    },
  };
  // `Session` is the output-erased, drive-agnostic view a tool sees via
  // `currentSession()`; the concrete `TOutput` widens away here.
  sessionRef = textSession as unknown as Session<TTools>;
  return textSession as SessionFor<TTools, TModel, TOutput, TSkillName>;
}
