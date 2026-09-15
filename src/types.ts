import type {
  Context,
  InferToolSetContext,
  ProviderOptions,
} from "@ai-sdk/provider-utils";
import type {
  DeepPartial,
  GenericToolApprovalFunction,
  LanguageModel,
  Output,
  PrepareStepFunction,
  StreamTextResult,
  ToolLoopAgentSettings,
  ToolSet,
} from "ai";
import type {
  RealtimeAudioConfig,
  RealtimeModelV1,
  RealtimeSession,
  RealtimeSessionCallbacks,
} from "./voice";
import type { RegistrySource } from "./registry";
import type { BoundRole } from "./role";
import type { BoundSkill, ParsedSkill, SkillSource } from "./skills";
import type { SessionMessage, SessionStorage } from "./storage";
import type { TurnTimings } from "./timings";

export type PrepareCallHook<TTools extends ToolSet> = NonNullable<
  ToolLoopAgentSettings<never, TTools, Context, never>["prepareCall"]
>;

export type PrepareStepHook<TTools extends ToolSet> = PrepareStepFunction<
  TTools,
  Context
>;

export interface HarnessHooks<TTools extends ToolSet = ToolSet> {
  prepareCall?: PrepareCallHook<TTools>;
  /** Per-step override. The harness already supplies `activeTools` from the
   *  role + invoked skills; a value returned here is merged on top (and can
   *  override `activeTools` if a consumer really needs to). Text drive only —
   *  realtime providers advertise tools once at connect, not per step. */
  prepareStep?: PrepareStepHook<TTools>;
}

/** The SDK `Output` spec the text drive feeds for a role's `outputSchema`:
 *  none (`never`) for a free-prose role, else an object output whose complete
 *  value is `TOutput` (so `result.output` resolves to `TOutput`). */
type HarnessOutput<TOutput> = [TOutput] extends [never]
  ? never
  : Output.Output<TOutput, DeepPartial<TOutput>, never>;

export type StreamResult<
  TTools extends ToolSet,
  TOutput = never,
> = StreamTextResult<TTools, Context, HarnessOutput<TOutput>> & {
  /** Snapshot of the turn's latency profile, from the same recorder that
   *  lands on the persisted message's `metadata.timings`. Take it when the
   *  stream has ended — the natural seam is a `messageMetadata` callback on
   *  the `finish` part, which puts the timings on the streamed message too. */
  timings(): TurnTimings;
  /** Resolves after the streamed assistant turn has been assembled and saved.
   *  Rejects when stream consumption or persistence fails. */
  readonly committed: Promise<void>;
};

/** A `HarnessConfig` after `init` has normalized it: the registry reduced to
 *  its toolset, `toolsContext` flattened from the three layers, and `storage`
 *  filled in. This is what a session runs against. */
export type ResolvedHarnessConfig<
  TTools extends ToolSet = ToolSet,
  TModel extends DriveModel = DriveModel,
  TOutput = never,
  TSkillName extends string = string,
> = Omit<HarnessConfig<TTools, TModel, TOutput, TSkillName>, "registry"> & {
  registry: TTools;
  storage: SessionStorage;
};

/** Portable reasoning-effort levels — the AI SDK v7 top-level `reasoning`
 *  parameter (`'none'`, `'low'`, `'high'`, …). Derived from the SDK so the
 *  union tracks whatever version is installed. */
export type ReasoningEffort = NonNullable<
  ToolLoopAgentSettings<never, ToolSet, Context, never>["reasoning"]
>;

/** Mints unique ids for sessions and messages. Defaults to a UUID v4
 *  generator (crypto.randomUUID); override at harness-creation time to plug
 *  in a different scheme (e.g. the AI SDK's short `generateId`). */
export type GenerateId = () => string;

/** A model that can drive a session. The *spec* decides the drive: a text
 *  `LanguageModel` runs through the AI SDK's ToolLoopAgent and gives a
 *  `TextSession` (`.prompt`); a `RealtimeModelV1` (e.g. `grok(...)`) runs
 *  through ai-sdk-voice and gives a `VoiceSession` (`.voice`). */
export type DriveModel = LanguageModel | RealtimeModelV1;

/** Selects the session shape from the model spec — the type-level half of the
 *  runtime dispatch in `createSession`. A realtime model yields a
 *  `VoiceSession`, anything else a `TextSession`. (Distributes over the
 *  `DriveModel` union to `AnySession` when the model isn't statically known.) */
export type SessionFor<
  TTools extends ToolSet,
  TModel extends DriveModel,
  TOutput = never,
  TSkillName extends string = string,
> = TModel extends RealtimeModelV1
  ? VoiceSession<TTools, TSkillName>
  : TextSession<TTools, TOutput, TSkillName>;

export interface HarnessConfig<
  TTools extends ToolSet = ToolSet,
  TModel extends DriveModel = DriveModel,
  TOutput = never,
  TSkillName extends string = string,
> {
  /** The complete tool registry — every tool the harness can ever expose, the
   *  `registry` from `createRegistry` (or a bare `ToolSet`). All of these are
   *  registered with the model; the role and any invoked skills decide which are
   *  *active* (callable). */
  registry: RegistrySource<TTools>;
  /** The session's driver, and the thing that picks its surface. A
   *  `LanguageModel` (or a model-id string) gives a text session with
   *  `.prompt()`; a `RealtimeModelV1` gives a voice session with `.voice()`.
   *  Subsessions name their own model, so a voice session escalates heavy work
   *  to a text worker. */
  model: TModel;
  /** Default reasoning effort for every step in the session — the AI SDK's
   *  portable top-level `reasoning` parameter. Each provider maps it to its
   *  native reasoning/thinking API; provider-specific `providerOptions` reasoning
   *  settings, if any, take precedence. Omit for the provider default. */
  reasoning?: ReasoningEffort;
  /** Provider-scoped call options, passed through to every model request in
   *  the session (AI SDK `providerOptions`): gateway routing, provider-native
   *  thinking settings, anything a provider reads for itself. Text drive only
   *  — the realtime models take their configuration through `voice()`. */
  providerOptions?: ProviderOptions;
  /** The role bound for the root session. Its `tools` (names into `tools`
   *  above) form the baseline active set, and its `outputSchema` (if any) types
   *  the session's `prompt()` result (`TOutput`). */
  role: BoundRole<TOutput>;
  /** Per-tool execution context, keyed by tool name (AI SDK `toolsContext`).
   *  The narrowest of the three layers: it wins over a bound registry's
   *  `overrides`, which win over the context handed to `registry(...)`.
   *  Prefer supplying context once through `registry(context)` — each tool's
   *  `contextSchema` projects it down to what that tool declared — and reach
   *  for this only where one tool's context genuinely differs. */
  toolsContext?: Partial<InferToolSetContext<TTools>>;
  /** Which tool executions need the user's explicit approval before running.
   *  The common case is a list of registry tool names: calling one of those
   *  completes the turn with an `approval-requested` tool part instead of
   *  executing, and `respondToApprovals` resumes the loop with the verdicts.
   *  Pass a `GenericToolApprovalFunction` instead for a custom policy — it
   *  reaches the agent unmodified. Text drive only: realtime models have no
   *  approval flow, so a voice session ignores this. */
  toolApproval?:
    | readonly (keyof TTools & string)[]
    | GenericToolApprovalFunction<TTools, InferToolSetContext<TTools>, Context>;
  /** The skills active from turn 1 — the entry point's interaction contract,
   *  supplied by the host when the session starts. Each one's instructions
   *  join the system prompt after the role's, and its tools and routing map
   *  join the baseline active set, so the very first turn already runs under
   *  it. No `<skill-init>` message is written: an initial skill is
   *  configuration, not a conversation event. Distinct from `session.skill()`
   *  (host, mid-session) and from `skills` below (model-loadable, mid-turn). */
  initialSkills?: readonly BoundSkill[];
  /** Where the model loads skills from on its own, through the loader tool
   *  (`skillLoaderTool()`, wired under the `loadSkill` registry key and listed
   *  in a role's `tools`): the session's skill universe, merged from these
   *  sources. The merged catalog renders into the instructions of every role
   *  that lists the loader; the loaded skill's instructions come back as the
   *  tool result, reaching the model mid-turn on both drives. Two sources
   *  declaring the same skill name fail session init. The same merged
   *  universe resolves `session.skill(name)`, which the HOST calls between
   *  turns; the sources' statically known names type that call. */
  loadableSkills?: readonly SkillSource<TSkillName>[];
  hooks?: HarnessHooks<TTools>;
  storage?: SessionStorage;
  /** Id generator for sessions and messages. Defaults to crypto.randomUUID
   *  (UUID v4). Supply your own to override the id scheme harness-wide. */
  generateId?: GenerateId;
}

export interface SessionOptions {
  sessionId?: string;
  messages?: SessionMessage[];
  title?: string;
}

export interface PromptOptions {
  model?: LanguageModel;
  /** IANA timezone of the user sending this turn (e.g. "Europe/Rome"). Frozen
   *  onto the message so the injected `<message_metadata>` renders local time. */
  timezone?: string;
  /** Forwarded to the SDK's agent run: aborting it cancels the in-flight
   *  model request and stops the tool loop. */
  abortSignal?: AbortSignal;
}

export interface SubsessionOptions<
  TModel extends DriveModel = LanguageModel,
  TOutput = never,
> {
  role?: BoundRole<TOutput>;
  /** The subsession's driver. Defaults to a text `LanguageModel` (the common
   *  case: a voice tool escalating to a text worker). Omit to inherit the
   *  parent's model at runtime. */
  model?: TModel;
  /** Reasoning effort for the subsession; falls back to the parent's. */
  reasoning?: ReasoningEffort;
  /** Provider call options for the subsession; falls back to the parent's.
   *  A subsession naming its own `model` should name these too — routing
   *  options are model-specific. */
  providerOptions?: ProviderOptions;
  /** The subsession's initial skills, exactly as `initialSkills` on the
   *  harness config: instructions join the system prompt after the role's,
   *  tools and routing join the baseline active set. Not inherited from the
   *  parent — a subsession that switches role names its own contract. */
  initialSkills?: readonly BoundSkill[];
  sessionId?: string;
  title?: string;
}

/** Options for the voice drive. The lifecycle callbacks
 *  (`onStatus`/`onAudio`/…) are optional — a relay typically wires the wire via
 *  the returned session's `serve()` instead, which composes with these. The
 *  harness always persists finalized turns; an `onMessage` here is called in
 *  addition. */
export type VoiceOptions<TTools extends ToolSet = ToolSet> = {
  /** Speak first after the prior conversation is seeded in (proactive open).
   *  Default `false` — wait for the user to talk. */
  triggerResponse?: boolean;
  /** Audio config (sample rates, transcription). Provider defaults apply. */
  audio?: RealtimeAudioConfig;
  /** IANA timezone of the user, for the session clock stamped into the
   *  instructions at connect (live spoken turns carry no per-turn metadata).
   *  Omitted, the clock renders in UTC. */
  timezone?: string;
} & RealtimeSessionCallbacks<TTools>;

/** Everything a session is, regardless of how it's driven: identity, history,
 *  role/skill tool gating, and forking. The drive method (`.prompt` / `.voice`)
 *  lives on the variant the model spec selects. */
export interface SessionCore<
  TTools extends ToolSet = ToolSet,
  TSkillName extends string = string,
> {
  readonly id: string;
  readonly messages: ReadonlyArray<SessionMessage>;
  /** The tool names currently exposed to the model (role baseline ∪ skills). */
  readonly activeTools: ReadonlyArray<string>;
  /** The subset the session's routing maps into code mode (role/skill
   *  `toolCallers`). Non-empty means the drive offers the sandbox tool —
   *  rebound per turn on text, bound at connect on voice — and these are
   *  called from generated programs, not directly. */
  readonly activeCodeTools: ReadonlyArray<string>;
  /** The session's persistence — the same instance the harness loads/saves
   *  through (defaults to an in-memory store when none is configured). Lets a
   *  tool reach sibling sessions (e.g. `listSessions()` for cross-session
   *  recall) without threading storage through its context. */
  readonly storage: SessionStorage;
  /** Inject a skill by name, resolved through the session's `loadableSkills`
   *  sources: append its `<skill-init>` message and turn on its tools. The
   *  statically known names autocomplete; a name no source holds is a runtime
   *  error. */
  skill(name: TSkillName | (string & {})): Promise<ParsedSkill>;
  /** Fork a child session over the same registry/context/storage. The model
   *  you pass picks its drive (defaults to a text `LanguageModel`), so a voice
   *  session can delegate heavy work to a powerful text worker. */
  subsession<TModel extends DriveModel = LanguageModel, TOutput = never>(
    opts?: SubsessionOptions<TModel, TOutput>,
  ): Promise<SessionFor<TTools, TModel, TOutput, TSkillName>>;
}

/** A session driven by a `LanguageModel`. `TOutput` is the role's structured
 *  output (`never` for a free-prose role), surfaced on the result's `.output`. */
export interface TextSession<
  TTools extends ToolSet = ToolSet,
  TOutput = never,
  TSkillName extends string = string,
> extends SessionCore<TTools, TSkillName> {
  /** Send the conversation's next turn and stream the assistant's reply (AI
   *  SDK ToolLoopAgent), persisting everything to storage. Takes either the
   *  user's text, or the client's latest message whole — the session decides
   *  what kind of turn it is:
   *  - a string, or a `user` message (prompting on its flattened text parts),
   *    starts a normal turn;
   *  - an `assistant` message whose pending tool-approval requests have been
   *    answered (`addToolApprovalResponse` client-side) is an approval turn:
   *    only the verdicts `{id, approved, reason}` are read from it and applied
   *    to the session's *stored* copy of the requests — history stays
   *    server-authoritative — then the loop resumes, executing approved tools
   *    and telling the model about denials. All pending requests must be
   *    answered in one turn; unknown or missing ids throw.
   *  When the role declared an `outputSchema`, the reply is constrained to it
   *  and `result.output` resolves to the validated `TOutput`. */
  prompt(
    input: string | SessionMessage,
    opts?: PromptOptions,
  ): Promise<StreamResult<TTools, TOutput>>;
}

/** A session driven by a `RealtimeModelV1`. */
export interface VoiceSession<
  TTools extends ToolSet = ToolSet,
  TSkillName extends string = string,
> extends SessionCore<TTools, TSkillName> {
  /** Drive this chat with the realtime model: seeds the prior conversation in,
   *  exposes the role-gated tools (so `currentSession()` works inside them, and
   *  a tool can `subsession()` to a text worker), and persists each finalized
   *  turn back to the same storage. Returns the live `RealtimeSession` to
   *  `serve()` over a wire and `start()`. */
  voice(opts?: VoiceOptions<TTools>): Promise<RealtimeSession>;
}

export type AnySession<TTools extends ToolSet = ToolSet> =
  | TextSession<TTools>
  | VoiceSession<TTools>;

/** The drive-agnostic session, as a tool sees it via `currentSession()`. Prefer
 *  `TextSession` / `VoiceSession` when the drive is known. */
export type Session<TTools extends ToolSet = ToolSet> = AnySession<TTools>;

export interface Harness<
  TTools extends ToolSet = ToolSet,
  TModel extends DriveModel = DriveModel,
  TOutput = never,
  TSkillName extends string = string,
> {
  readonly config: ResolvedHarnessConfig<TTools, TModel, TOutput, TSkillName>;
  readonly instructions: string;
  session(
    opts?: SessionOptions,
  ): Promise<SessionFor<TTools, TModel, TOutput, TSkillName>>;
}
