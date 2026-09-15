// The session core: provider-agnostic orchestration over a RealtimeProviderV1.
// It owns everything that is invariant across providers — the VoiceStatus
// machine, transcript assembly, tool dispatch with barge-in abort — and knows
// nothing about any wire format. The provider is injected, exactly as a
// LanguageModel is injected into streamText.
//
// Tools are a real AI SDK ToolSet (the same `tool()`-built tools the text
// harness uses): their definitions are advertised to the provider, and they
// are executed with the SDK's own `executeTool` — never a bespoke executor.

import { randomUUID } from "node:crypto";
import { asSchema, type ToolSet } from "ai";
import {
  executeTool,
  type InferToolInput,
  type InferToolOutput,
  type InferToolSetContext,
} from "@ai-sdk/provider-utils";
import { toRealtimeToolDefs } from "./tools";
import { decodeClientEvent, encodeServerEvent } from "./protocol";
import type {
  RealtimeAudioConfig,
  RealtimeCall,
  RealtimeChunk,
  RealtimeEvent,
  RealtimeHandle,
  RealtimeModelV1,
  RealtimeOutbound,
  Role,
  VoiceStatus,
} from "./spec";

/** A completed tool round-trip, typed per the tool that was called: one member
 *  of the union per tool name, with `input`/`output` inferred from its schema. */
type RealtimeToolMessage<TOOLS extends ToolSet> = {
  [NAME in keyof TOOLS & string]: {
    type: "tool";
    id: string;
    callId: string;
    /** The name of the tool that was called. */
    name: NAME;
    /** The validated input, typed from the tool's `inputSchema`. */
    input: InferToolInput<TOOLS[NAME]>;
    /** The tool's output, typed from its `execute`/`outputSchema`. */
    output: InferToolOutput<TOOLS[NAME]>;
    /** Epoch ms the core stamped this turn — the cross-mode ordering key.
     *  Server-authoritative (the relay's clock), carried into the persisted
     *  UIMessage's metadata so text and voice turns interleave by time. */
    createdAt: number;
    /** How long the tool ran (`tool.call` → result), integer ms. */
    timings?: { totalMs: number };
  };
}[keyof TOOLS & string];

/** Latency profile of an assistant voice turn — the voice counterpart of the
 *  text drive's `metadata.timings` (same field names, so readers treat text
 *  and voice turns identically). */
export interface RealtimeTurnTimings {
  /** End of the user's turn (transcript settled / injected text) → the first
   *  assistant signal (audio or transcript). Absent when the core saw no user
   *  turn to anchor on (e.g. a proactive opening), and on every message of the
   *  answer after the first — the answer only starts once. */
  ttftMs?: number;
  /** End of the user's turn → this message settling, so it contains `ttftMs`
   *  and matches the text drive's "dispatch → end". An answer spoken as several
   *  messages measures each from the same start, which makes the last one the
   *  latency the user felt. Falls back to this message's own span when there was
   *  no user turn to anchor on. */
  totalMs: number;
}

/**
 * A finalized turn the core hands up for rendering/persistence. Generic over the
 * session's `ToolSet`: tool messages carry a typed `name` (one of your tool
 * names) and `input`/`output` inferred from that tool — the same way the AI SDK
 * types tool calls from `streamText({ tools })`.
 */
export type RealtimeMessage<TOOLS extends ToolSet = ToolSet> =
  | {
      type: "text";
      id: string;
      role: Role;
      text: string;
      createdAt: number;
      /** Stamped on assistant turns only. */
      timings?: RealtimeTurnTimings;
    }
  | RealtimeToolMessage<TOOLS>;

export interface RealtimeSessionCallbacks<TOOLS extends ToolSet = ToolSet> {
  /** The derived status changed (connecting → listening → speaking → thinking →
   *  …), with transport health folded in. Drive your UI from this. */
  onStatus?: (status: VoiceStatus) => void;
  /** A finalized turn (assistant/user text, or a completed tool round-trip).
   *  Tool messages are typed from `TOOLS`. This is what you persist and render. */
  onMessage?: (message: RealtimeMessage<TOOLS>) => void;
  /** A tool call started but hasn't finished — render-only, so a UI can show it
   *  running before the result. `callId`/`createdAt` match the eventual
   *  `onMessage`, which supersedes it. Not a finalized turn; never persist it. */
  onToolPending?: (pending: {
    callId: string;
    name: string;
    input: unknown;
    createdAt: number;
  }) => void;
  /** A previously-pending tool was aborted (barge-in or session end) before it
   *  finished — its `onMessage` will never come, so drop the running render. */
  onToolCancel?: (cancel: { callId: string }) => void;
  /** The assistant is done answering: the model closed a response and left no
   *  tool result outstanding, so nothing more is coming for this user turn.
   *  A turn spans as many responses as it took tool calls, and the model can
   *  fall silent for seconds inside one — this is the only sound way to tell
   *  "finished" from "still working". Providers that never report their
   *  responses simply never fire it. */
  onTurnDone?: () => void;
  /** Barge-in: the user spoke over the assistant, so generation was cancelled.
   *  Any assistant audio already emitted via `onAudio` is now stale — a relay
   *  should tell the client to flush its playback buffer. */
  onInterrupted?: () => void;
  /** Live transcript growth for captions. `full` is the accumulated text so
   *  far for that turn; `id`/`createdAt` match the eventual `onMessage`, so a
   *  streaming caption already sorts at the right spot before it finalizes. */
  onTranscriptDelta?: (delta: {
    id: string;
    role: Role;
    text: string;
    full: string;
    createdAt: number;
  }) => void;
  /** An assistant audio frame (raw PCM) to play/relay to the client. */
  onAudio?: (pcm: ArrayBuffer) => void;
  /** Every raw frame off the provider wire, before normalization and tagged with
   *  the provider name — for debugging exactly what a provider returns (usage
   *  metadata, undocumented fields, ignored frames). Off the hot path: supply it
   *  only when inspecting/recording wire traffic. */
  onChunk?: (chunk: RealtimeChunk) => void;
  /** An error surfaced by the provider or a tool. Fatal provider errors also
   *  flip the state to `"error"`. */
  onError?: (message: string) => void;
}

/**
 * Arguments for `createRealtimeSession`, shaped like a `streamText` call: the
 * `model`, the call fields, `tools`, and the lifecycle callbacks (`onStatus`,
 * `onMessage`, `onChunk`, …) all live at the root. Fields are listed explicitly
 * (rather than via `Omit<RealtimeCall>`) so each one's docs surface on hover at
 * the call site. The callbacks are all optional — supply only what you render or
 * persist.
 */
export type CreateRealtimeSessionArgs<TOOLS extends ToolSet = ToolSet> = {
  /**
   * The realtime model, created by a provider factory like `grok("model-id")`.
   * Carries the provider, model id, voice, and connection config (API key,
   * etc.) — the voice analogue of `openai("gpt-4")`.
   */
  model: RealtimeModelV1;
  /** The system prompt for the session. */
  instructions: string;
  /**
   * Prior conversation replayed into the session before audio flows — a resume
   * history, or a synthetic opener (e.g. memory summaries). Each item is
   * replayed during connect with the same wire mapping as a live result. Pass
   * `[]` for a fresh session.
   */
  seed: RealtimeOutbound[];
  /**
   * Whether the model should speak first, right after the seed is replayed (a
   * proactive opening). `false` waits for the user to talk.
   */
  triggerResponse: boolean;
  /** Optional audio config (sample rates, transcription). Provider defaults apply. */
  audio?: RealtimeAudioConfig;
  /**
   * The agent's tools, as a standard AI SDK `ToolSet` built with `tool()` —
   * exactly what you would pass to `streamText`. Their schemas are advertised
   * to the model and they are executed with the SDK's own `executeTool`.
   */
  tools?: TOOLS;
  /**
   * Restrict which of `tools` are advertised to the model and executable this
   * session — the realtime analogue of the AI SDK's `activeTools`. `tools` is
   * the full set the session is built with (so a seed can reference any of
   * them); pass the subset that should be callable now (e.g. a harness role's
   * active set). Omit to keep all of `tools` active. Realtime providers
   * advertise tools once at connect, so this is a connect-time gate, not
   * per-step.
   */
  activeTools?: Array<keyof TOOLS & string>;
  /**
   * Per-tool execution context, keyed by tool name — the same shape the harness
   * passes as `toolsContext`. Each tool's `execute` receives its slice as
   * `options.context`.
   */
  toolsContext?: InferToolSetContext<TOOLS>;
  /**
   * Render the model-facing temporal stamp (e.g. a `<message_metadata>` block
   * carrying "now") injected as a user-role conversation item: after the seed
   * at connect, before each typed turn, and after each settled spoken user
   * turn — so every response the model produces holds a stamp at most one
   * turn old. Spoken turns are transcribed provider-side, out of our reach,
   * which is why the stamp travels as its own item. Model-facing only: never
   * persisted, never rendered. Omit for no stamps.
   */
  turnMetadata?: () => string;
  /** Id minter for turns/messages. Defaults to `crypto.randomUUID`. */
  generateId?: () => string;
  /** Clock for the per-turn `createdAt` ordering stamp. Defaults to `Date.now`;
   *  inject for deterministic tests. */
  now?: () => number;
} & RealtimeSessionCallbacks<TOOLS>;

export interface RealtimeServeHandle {
  /** Feed one inbound wire frame from the client: binary ⇒ mic PCM, string ⇒
   *  a JSON `ClientEvent` (e.g. a typed user turn). */
  receive(frame: string | ArrayBuffer | Uint8Array): void;
}

export interface RealtimeSession {
  /** The current derived status. Also delivered via `onStatus`. */
  readonly status: VoiceStatus;
  /** Connect the provider, replay the seed, and go live. Resolves once the
   *  transport is connected; safe to call once. */
  start(): Promise<void>;
  /** Feed a user mic frame (raw PCM) into the live session. No-op before
   *  `start` or after `stop`. */
  pushAudio(pcm: ArrayBuffer): void;
  /** Inject a typed user turn — the text analogue of a mic utterance. The
   *  turn is finalized immediately (persisted and fanned out like any other
   *  message) and the model is asked to respond. No-op before `start`, after
   *  `stop`, or on blank text. */
  sendText(text: string): void;
  /** Bridge this session to a client over a duplex wire — the analogue of the
   *  AI SDK's `result.toUIMessageStream()`. Outbound events/audio are encoded
   *  to `send` (JSON `ServerEvent`s + binary PCM); the returned handle's
   *  `receive` takes inbound frames. Derived from the session, so it inherits
   *  the tool types with no extra type args. */
  serve(send: (frame: string | ArrayBuffer) => void): RealtimeServeHandle;
  /** Tear down: abort in-flight tools, close the provider, return to `"idle"`. */
  stop(): Promise<void>;
}

/** The advertised/executable subset: `tools` narrowed to `activeTools`, or all
 *  of `tools` when no active set is given. The full `tools` is still used by the
 *  caller to build the seed, so a replayed tool round-trip keeps its mapping
 *  even if that tool is no longer active. */
function filterActiveTools<TOOLS extends ToolSet>(
  tools: TOOLS | undefined,
  activeTools: Array<keyof TOOLS & string> | undefined,
): TOOLS | undefined {
  if (!tools || !activeTools) return tools;
  const active = new Set<string>(activeTools);
  return Object.fromEntries(
    Object.entries(tools).filter(([name]) => active.has(name)),
  ) as TOOLS;
}

function toArrayBuffer(data: ArrayBuffer | Uint8Array): ArrayBuffer {
  if (data instanceof ArrayBuffer) return data;
  return data.buffer.slice(
    data.byteOffset,
    data.byteOffset + data.byteLength,
  ) as ArrayBuffer;
}

/**
 * Create a realtime voice session over an injected provider — the voice
 * analogue of `streamText`. The core owns the state machine, transcript
 * assembly, and tool dispatch; the provider only translates the wire. Returns
 * a handle you `start()`, feed mic audio via `pushAudio()`, and `stop()`.
 *
 * @example
 * const session = createRealtimeSession({
 *   model: grok("grok-voice-1"),
 *   instructions,
 *   seed: [],
 *   triggerResponse: false,
 *   tools: { search },
 *   onStatus, onAudio, onMessage,
 * });
 * await session.start();
 */
export function createRealtimeSession<TOOLS extends ToolSet = ToolSet>(
  args: CreateRealtimeSessionArgs<TOOLS>,
): RealtimeSession {
  const tools = filterActiveTools(args.tools, args.activeTools);
  const toolsContext = args.toolsContext as
    | Record<string, unknown>
    | undefined;
  const generateId = args.generateId ?? randomUUID;
  const now = args.now ?? Date.now;

  // Every event fans out to all registered sinks: the creation callbacks (which
  // live at the root of `args`) plus whatever `serve()` adds. This is what lets
  // the wire bridge live on the session instead of being attached through them.
  const sinks = new Set<RealtimeSessionCallbacks<TOOLS>>();
  sinks.add(args);
  const cb = {
    onStatus: (s: VoiceStatus) => {
      for (const k of sinks) k.onStatus?.(s);
    },
    onMessage: (m: RealtimeMessage<TOOLS>) => {
      for (const k of sinks) k.onMessage?.(m);
    },
    onToolPending: (p: {
      callId: string;
      name: string;
      input: unknown;
      createdAt: number;
    }) => {
      for (const k of sinks) k.onToolPending?.(p);
    },
    onToolCancel: (c: { callId: string }) => {
      for (const k of sinks) k.onToolCancel?.(c);
    },
    onTurnDone: () => {
      for (const k of sinks) k.onTurnDone?.();
    },
    onInterrupted: () => {
      for (const k of sinks) k.onInterrupted?.();
    },
    onTranscriptDelta: (d: {
      id: string;
      role: Role;
      text: string;
      full: string;
      createdAt: number;
    }) => {
      for (const k of sinks) k.onTranscriptDelta?.(d);
    },
    onAudio: (p: ArrayBuffer) => {
      for (const k of sinks) k.onAudio?.(p);
    },
    onChunk: (c: RealtimeChunk) => {
      for (const k of sinks) k.onChunk?.(c);
    },
    onError: (e: string) => {
      for (const k of sinks) k.onError?.(e);
    },
  };

  let status: VoiceStatus = "idle";
  let handle: RealtimeHandle | null = null;
  let abort: AbortController | null = null;
  let startPromise: Promise<void> | null = null;
  let generation = 0;

  const pendingTools = new Set<AbortController>();
  // The in-flight `runTool` promises, so teardown can wait for a tool that
  // ignored its abort to finish acting before the transcript is read as final.
  const runningTools = new Set<Promise<void>>();
  const buffers: Record<Role, string> = { user: "", assistant: "" };
  const turnIds: Record<Role, string | null> = { user: null, assistant: null };
  // Stamped when a turn's id is first minted, reused for its deltas and final
  // message so a streaming caption and its settled turn share one ordering key.
  const turnCreatedAt: Record<Role, number | null> = { user: null, assistant: null };
  // The provider's utterance key the open turn belongs to (when the provider
  // supplies one) — the boundary signal that keeps a turn whose final got lost
  // from absorbing the next utterance.
  const turnUtterance: Record<Role, string | null> = { user: null, assistant: null };
  // Turns already settled, by utterance key: late events for one of them repair
  // or dedupe against the emitted turn instead of minting a sibling. A voice
  // session spans minutes, so these stay tiny.
  const settledTurns: Record<
    Role,
    Map<string, { id: string; createdAt: number; text: string; byFinal: boolean }>
  > = { user: new Map(), assistant: new Map() };
  // Where the answer the assistant is now producing began: the end of the user's
  // turn (transcript settled or injected text). Exact for injected text; for mic
  // it tracks when the ASR settled the utterance, the closest thing the wire
  // gives us to "the user stopped talking". Held for the whole answer, not
  // consumed by the first message, because an answer is usually several messages
  // (a filler, a tool, the reply) and each of them measures from here.
  let responseStartedAt: number | null = null;
  // Whether this answer's first signal has already been timed, and the latency
  // it measured — held until the message that carries it settles.
  let ttftTaken = false;
  let assistantTtftMs: number | null = null;
  // Whether the assistant has perceptibly answered the user turn that last
  // settled: audio or caption text that reached the client, or a tool
  // round-trip that completed onto the record. A tool merely pending doesn't
  // count, and neither does one that finishes after a barge-in aborted it
  // (its round-trip is recorded, but the user was already speaking again) —
  // either way the continuing transcript still names the question the tool
  // was working on.
  let assistantReacted = false;
  // Tool calls the response now closing asked for. Their results go back to the
  // provider, which answers them with another response — so a response that
  // called tools is never the last one of the turn, however quiet it goes.
  let toolCallsThisResponse = 0;
  // The model closed its last response while audio was still coming down: the
  // turn is over as soon as that audio ends, not before. Providers differ on
  // which of the two lands first, so the turn needs both.
  let turnEndsWithSpeech = false;
  // The mic has opened a user utterance the ASR has not settled yet. Without
  // this the core cannot tell "no user turn is coming" from "one is coming and
  // the ASR has not spoken yet", which is what a tool reading the transcript
  // needs to know.
  //
  // The contract a provider owes: every `speech.interrupted` it emits must be
  // followed by a user `transcript.final` or `transcript.done` — the words, or
  // the admission that there are none. grok and the gateway honor it, deriving
  // the open from their speech-started event and closing it on the
  // transcription's completion or failure. Gemini does not yet: it emits the
  // open only on a real barge-in and nothing at all for a turn that produced
  // no words, so a tool can be left waiting there. It is not the configured
  // voice, and closing that gap is its own piece of work.
  //
  // Held by key, not counted, because neither end of the pair is once-per-turn.
  // They overlap, so a single flag would let the older utterance's transcript
  // release a tool waiting for the newer one — the very turn-short read this
  // exists to stop. And speech starts more than once inside one utterance
  // (a pause mid-sentence re-arms the detector) while the transcript arrives
  // once, so counting leaves opens that nothing will ever answer. The key is
  // the same one the transcript carries, so a re-arm is the utterance already
  // held and adds nothing.
  const openUtterances = new Set<string>();
  const utteranceWaiters = new Set<() => void>();
  // Providers that supply no key still have to be able to open one.
  const UNKEYED = "\u0000unkeyed";

  function openUtterance(utterance?: string) {
    openUtterances.add(utterance ?? UNKEYED);
  }

  function closeUtterance(utterance?: string) {
    // An ending we cannot place means the bookkeeping is already wrong — the
    // ASR reattributes its terminal event, and a provider may key the two ends
    // differently. Release everything rather than hold a tool on a key that is
    // never coming back: a stale read is recoverable, a session that stops
    // answering is not.
    if (utterance && openUtterances.has(utterance)) openUtterances.delete(utterance);
    else openUtterances.clear();
    if (openUtterances.size > 0 || utteranceWaiters.size === 0) return;
    const waiting = [...utteranceWaiters];
    utteranceWaiters.clear();
    for (const wake of waiting) wake();
  }

  /** Nothing more is coming: release every utterance and everyone waiting on
   *  one. The wire is the only source of an utterance's ending, so a wire that
   *  is gone would otherwise strand them. */
  function abandonUtterances() {
    openUtterances.clear();
    closeUtterance();
  }

  /** Resolve once no user utterance is open — immediately when none is. The
   *  wait is not capped: every provider closes the utterance it opened, on the
   *  transcript or on the ASR giving up, and the abort that ends the tool ends
   *  the wait with it. A wait that never resolves would be a provider emitting
   *  an open it never closes, which is a bug to fix there, not to paper over
   *  with a deadline here. */
  function awaitSettledUtterance(signal?: AbortSignal): Promise<void> {
    if (openUtterances.size === 0 || signal?.aborted) return Promise.resolve();
    return new Promise<void>((resolve) => {
      const finish = () => {
        utteranceWaiters.delete(finish);
        signal?.removeEventListener("abort", finish);
        resolve();
      };
      utteranceWaiters.add(finish);
      signal?.addEventListener("abort", finish, { once: true });
    });
  }

  /** Whether a late event for a settled turn reopens it: always, when the
   *  turn settled early (not by its own final); for a user turn settled by
   *  final, only while the assistant has not perceptibly reacted — the same
   *  key carrying text again is then the ASR continuing or correcting the
   *  utterance, not new speech. After a reaction, it stays its own turn. */
  function reopensSettled(role: Role, prev: { byFinal: boolean }): boolean {
    return !prev.byFinal || (role === "user" && !assistantReacted);
  }

  function captureTtft() {
    assistantReacted = true;
    if (responseStartedAt === null || ttftTaken) return;
    ttftTaken = true;
    assistantTtftMs = Math.round(now() - responseStartedAt);
  }

  /** Timings for the assistant text turn now settling; user turns get none.
   *
   *  `totalMs` runs from the end of the user's turn, so it nests `ttftMs` and
   *  means what it means for a text turn: dispatch to here. Across an answer
   *  made of several messages the numbers therefore grow, and the last one is
   *  the latency the user actually felt. Without a user turn to anchor on (a
   *  proactive opening) the message can only time itself. */
  function takeTurnTimings(role: Role, createdAt: number): { timings?: RealtimeTurnTimings } {
    if (role === "user") {
      responseStartedAt = now();
      ttftTaken = false;
      assistantTtftMs = null;
      assistantReacted = false;
      return {};
    }
    const timings: RealtimeTurnTimings = {
      ...(assistantTtftMs !== null ? { ttftMs: assistantTtftMs } : {}),
      totalMs: Math.max(0, Math.round(now() - (responseStartedAt ?? createdAt))),
    };
    assistantTtftMs = null;
    return { timings };
  }

  function setStatus(next: VoiceStatus) {
    if (status === next) return;
    status = next;
    cb.onStatus?.(next);
  }

  function turnId(role: Role): string {
    const existing = turnIds[role];
    if (existing) return existing;
    const id = generateId();
    turnIds[role] = id;
    turnCreatedAt[role] = now();
    return id;
  }

  /** Align the open turn with the utterance a streaming event belongs to. A
   *  new key while a differently-keyed turn is open means the open turn's
   *  final got lost — settle it from its buffered snapshot before the new
   *  utterance can write under its id. With no turn open, a key whose turn
   *  already settled reopens per `reopensSettled` — for the settled-by-final
   *  case that is the utterance continuing across a pause (barge-in cancels
   *  the pending answer before any of it reached the user, and the item keeps
   *  growing), and without the reopen those snapshots would mint a sibling
   *  turn that the eventual corrected final then settles under, showing the
   *  whole utterance twice.
   *  The reopened buffer starts from the settled text: the wires observed
   *  live re-stream the whole utterance (cumulative snapshots, which simply
   *  replace it), and a wire that appended instead would otherwise lose the
   *  head of the message. */
  function alignTurn(role: Role, utterance: string | undefined) {
    if (!utterance) return;
    if (turnIds[role] && turnUtterance[role] && turnUtterance[role] !== utterance) {
      flush(role);
    }
    if (!turnIds[role]) {
      const prev = settledTurns[role].get(utterance);
      if (prev && reopensSettled(role, prev)) {
        turnIds[role] = prev.id;
        turnCreatedAt[role] = prev.createdAt;
        buffers[role] = prev.text;
      }
    }
    turnUtterance[role] = utterance;
  }

  /** Settle the open turn with `text` and emit it — unless its utterance
   *  already settled with exactly this text (a repeat adds nothing). A
   *  re-settle of an already-emitted turn keeps its id and stamp, so id-keyed
   *  stores (persistence, UI) update the turn in place, and skips the timing
   *  side effects its first emission already took. */
  function settleTurn(role: Role, rawText: string, byFinal: boolean) {
    const text = rawText.trim();
    const id = turnIds[role];
    const createdAt = turnCreatedAt[role] ?? now();
    const utterance = turnUtterance[role];
    buffers[role] = "";
    turnIds[role] = null;
    turnCreatedAt[role] = null;
    turnUtterance[role] = null;
    if (!text || !id) return;
    const prev = utterance ? settledTurns[role].get(utterance) : undefined;
    if (prev && prev.text === text) return;
    const revises = prev?.id === id;
    if (utterance) {
      settledTurns[role].set(utterance, { id, createdAt, text, byFinal });
    }
    if (role === "user" && !revises) stampClock();
    cb.onMessage?.({
      type: "text",
      id,
      role,
      text,
      createdAt,
      // Re-timing a revised assistant turn would restate a latency its first
      // emission already reported; a revised user turn still re-anchors, since
      // the answer starts from this later end-of-turn.
      ...(revises && role === "assistant" ? {} : takeTurnTimings(role, createdAt)),
    });
  }

  function flush(role: Role) {
    settleTurn(role, buffers[role], false);
  }

  /** Inject the temporal stamp as a user-role item (see `turnMetadata`). */
  function stampClock() {
    if (!args.turnMetadata || !handle) return;
    const text = args.turnMetadata();
    if (text) handle.send({ type: "text", role: "user", text });
  }

  function abortPendingTools() {
    for (const c of pendingTools) c.abort();
    pendingTools.clear();
  }

  // Run a tool call through the AI SDK exactly as streamText would: validate
  // the model's input against the tool's schema, then `executeTool` (which
  // applies the tool's own execute + toModelOutput, draining any streamed
  // preliminary outputs to the final one).
  async function runTool(ev: Extract<RealtimeEvent, { type: "tool.call" }>) {
    const live = handle;
    if (!live) return;
    const controller = new AbortController();
    pendingTools.add(controller);
    setStatus("thinking");
    // The model answers from the audio and reaches the tool before the ASR has
    // settled the words; a tool that reads the conversation would then read one
    // turn short, silently. Hold every tool here until the
    // utterance it is answering is on the record — free when it already is,
    // and the barge-in that aborts the tool releases the wait with it.
    await awaitSettledUtterance(controller.signal);
    if (controller.signal.aborted) {
      pendingTools.delete(controller);
      return;
    }
    // The filler spoken ahead of this call belongs on the record before a
    // conversation-reading tool runs — the delegate must see what the user
    // was just told. Settle any assembled assistant speech now instead of
    // waiting for `transcript.done`/`speech.stop`.
    flush("assistant");
    // Stamp once at call-start and reuse for the final turn, so the rendered
    // tool line keeps a stable ordering key from running to settled.
    const createdAt = now();
    let completed = false;
    try {
      const t = tools?.[ev.name];
      if (!t || typeof t.execute !== "function") {
        throw new Error(`tool not available: ${ev.name}`);
      }
      const validated = await asSchema(t.inputSchema).validate?.(ev.input);
      if (validated && !validated.success) throw new Error(validated.error.message);
      const input = validated?.success ? validated.value : ev.input;

      // Surface the call as running before `executeTool` (which can take seconds
      // — `search` escalates to a full text agent). Render-only; not persisted.
      cb.onToolPending?.({ callId: ev.callId, name: ev.name, input, createdAt });

      let output: unknown;
      const run = executeTool({
        tool: t,
        input,
        options: {
          toolCallId: ev.callId,
          messages: [],
          abortSignal: controller.signal,
          context: toolsContext?.[ev.name],
        },
      } as unknown as Parameters<typeof executeTool>[0]);
      for await (const part of run) output = part.output;

      // The execute finished with an output, so whatever it did is done. Most
      // tools act on the world without watching the abort signal, and an
      // action missing from the transcript is a record that lies:
      // the round-trip is emitted regardless. An abort only cancels the
      // provider hand-off — the interrupted response it belonged to is gone.
      completed = true;
      if (!controller.signal.aborted) {
        live.send({ type: "tool.result", callId: ev.callId, output });
        // A completed round-trip is on the record: from here the user's turn
        // has been answered, and a reused key is new speech, not a correction.
        assistantReacted = true;
      }
      // The wire event's name/input/output are runtime values, so we re-assert
      // the typed shape the consumer's callback expects.
      cb.onMessage?.({
        type: "tool",
        id: ev.callId,
        callId: ev.callId,
        name: ev.name,
        input,
        output,
        createdAt,
        timings: { totalMs: Math.max(0, Math.round(now() - createdAt)) },
      } as RealtimeMessage<TOOLS>);
    } catch (e) {
      if (controller.signal.aborted) return;
      const message = e instanceof Error ? e.message : "tool execution failed";
      // The error goes to the model as the call's result and nowhere else: the
      // model recovers in-band (its prompt says what to tell the user), while
      // `onError` would reach clients that treat any error frame as fatal and
      // close the session over a single failed call.
      live.send({
        type: "tool.result",
        callId: ev.callId,
        output: { error: message },
        isError: true,
      });
    } finally {
      pendingTools.delete(controller);
      // Aborted without an output — the tool honored the abort — so no
      // `onMessage` comes: retract the running render we announced (no-op on
      // the client if it was never shown).
      if (controller.signal.aborted && !completed) {
        cb.onToolCancel?.({ callId: ev.callId });
      }
    }
  }

  // The whole state machine lives here: VoiceStatus is derived from the
  // normalized event stream, never set by a provider. Transport health folds
  // in — a connected wire advances to `listening`, a dropped one to `error`.
  function onEvent(ev: RealtimeEvent) {
    switch (ev.type) {
      case "transport":
        if (ev.status === "connected" && status === "connecting") setStatus("listening");
        if (ev.status === "disconnected") {
          // The wire was the only thing that could end an open utterance, so a
          // tool waiting on one would wait for a transcript that has no way of
          // arriving.
          abandonUtterances();
          if (status !== "idle") setStatus("error");
        }
        return;
      case "response.start":
        toolCallsThisResponse = 0;
        turnEndsWithSpeech = false;
        // The model is producing but hasn't spoken yet. That silence reads as
        // work rather than an idle wire, but it is not yet a tool:
        // until one is actually out, this is the model composing.
        if (status === "listening") setStatus("composing");
        return;
      case "response.done": {
        // Another response follows whenever this one leaves a tool result to
        // answer: one still running, or one already sent back.
        const running = pendingTools.size > 0;
        const continues = running || toolCallsThisResponse > 0;
        toolCallsThisResponse = 0;
        if (status === "speaking") {
          // Audio outlives the response on some wires; `speech.stop` finishes it.
          turnEndsWithSpeech = !continues;
          return;
        }
        // Still waiting on a tool, or already holding its result and writing
        // the answer from it — the second is composing, however it got there.
        setStatus(continues ? (running ? "thinking" : "composing") : "listening");
        if (!continues) cb.onTurnDone?.();
        return;
      }
      case "speech.start":
        // First audible assistant signal — settle the response latency here if
        // the transcript hasn't already.
        captureTtft();
        setStatus("speaking");
        return;
      case "speech.stop":
        // Status tracks audio; the transcript is normally already finalized by
        // `transcript.done`, so this flush is a fallback for any leftover buffer.
        flush("assistant");
        setStatus(pendingTools.size > 0 ? "thinking" : "listening");
        // The response had already closed; the turn was only waiting on this
        // audio. Fires after the flush, so the settled turn precedes it.
        if (turnEndsWithSpeech && pendingTools.size === 0) {
          turnEndsWithSpeech = false;
          cb.onTurnDone?.();
        }
        return;
      case "speech.interrupted":
        // Both providers derive this from their speech-started event, so it is
        // also where the utterance opens — not only where a reply
        // is cut off.
        openUtterance(ev.utterance);
        abortPendingTools();
        // The cancelled response gets no continuation, and the turn it belonged
        // to ends here — the user is talking.
        toolCallsThisResponse = 0;
        turnEndsWithSpeech = false;
        flush("assistant");
        // Tell the client to drop buffered assistant audio before the status
        // flips, so the in-flight reply stops the moment the user cuts in.
        cb.onInterrupted?.();
        setStatus("listening");
        return;
      case "transcript.delta": {
        if (ev.role === "assistant") captureTtft();
        alignTurn(ev.role, ev.utterance);
        const id = turnId(ev.role);
        const createdAt = turnCreatedAt[ev.role] ?? now();
        buffers[ev.role] += ev.text;
        cb.onTranscriptDelta?.({ id, role: ev.role, text: ev.text, full: buffers[ev.role], createdAt });
        return;
      }
      case "transcript.update": {
        // Cumulative snapshot — replace the buffer rather than append.
        if (ev.role === "assistant") captureTtft();
        alignTurn(ev.role, ev.utterance);
        const id = turnId(ev.role);
        const createdAt = turnCreatedAt[ev.role] ?? now();
        buffers[ev.role] = ev.text;
        cb.onTranscriptDelta?.({ id, role: ev.role, text: ev.text, full: ev.text, createdAt });
        return;
      }
      case "transcript.final": {
        if (ev.role === "user") closeUtterance(ev.utterance);
        if (ev.utterance) {
          const prev = settledTurns[ev.role].get(ev.utterance);
          // An exact repeat of a settled utterance adds nothing — and must not
          // settle whatever newer turn happens to be open.
          if (prev && prev.text === ev.text.trim()) return;
          const openKey = turnUtterance[ev.role];
          if (turnIds[ev.role] && openKey && openKey !== ev.utterance) {
            // The final targets another utterance than the open turn — never
            // settle the open turn with someone else's text. Repair the turn
            // it names in place when it still reopens (settled early, or a
            // late correction the assistant never reacted to, arriving while
            // the next utterance already streams); otherwise the text becomes
            // a turn of its own.
            const text = ev.text.trim();
            if (!text) return;
            if (prev && reopensSettled(ev.role, prev)) {
              settledTurns[ev.role].set(ev.utterance, { ...prev, text, byFinal: true });
              cb.onMessage?.({
                type: "text",
                id: prev.id,
                role: ev.role,
                text,
                createdAt: prev.createdAt,
              });
              return;
            }
            const id = generateId();
            const createdAt = now();
            settledTurns[ev.role].set(ev.utterance, { id, createdAt, text, byFinal: true });
            cb.onMessage?.({
              type: "text",
              id,
              role: ev.role,
              text,
              createdAt,
              ...takeTurnTimings(ev.role, createdAt),
            });
            return;
          }
          // The final targets the open turn (same key, or an unkeyed one it
          // adopts) — or reopens its settled turn to repair it in place, per
          // `reopensSettled`: the ASR sends a second `completed` for the same
          // item when it corrects itself (the gateway relays both), and that
          // revision belongs to the message already shown, not to a new one.
          if (!turnIds[ev.role] && prev && reopensSettled(ev.role, prev)) {
            turnIds[ev.role] = prev.id;
            turnCreatedAt[ev.role] = prev.createdAt;
          }
          turnUtterance[ev.role] = ev.utterance;
        }
        turnId(ev.role);
        settleTurn(ev.role, ev.text, true);
        return;
      }
      case "transcript.done":
        // The streamed transcript stopped growing — settle it now, decoupled
        // from `speech.stop` so the text reads "done" while audio may play on.
        // For the user it is also how an utterance ends without words: the ASR
        // gave up, or heard nothing. Either way it is over, and whoever waits
        // on it must be let go.
        if (ev.role === "user") closeUtterance();
        flush(ev.role);
        return;
      case "tool.call": {
        toolCallsThisResponse += 1;
        const run = runTool(ev);
        runningTools.add(run);
        void run.finally(() => runningTools.delete(run));
        return;
      }
      case "error":
        cb.onError?.(ev.message);
        if (ev.fatal) setStatus("error");
        return;
    }
  }

  // A typed user turn has no ASR transcript to finalize, so it becomes a
  // message directly: emit it (persistence + fan-out), hand it to the
  // provider, and ask for a response.
  function injectUserText(raw: string) {
    const text = raw.trim();
    if (!handle || !text) return;
    cb.onMessage?.({
      type: "text",
      id: generateId(),
      role: "user",
      text,
      createdAt: now(),
    });
    responseStartedAt = now();
    ttftTaken = false;
    assistantTtftMs = null;
    assistantReacted = false;
    stampClock();
    handle.send({ type: "text", role: "user", text });
    handle.requestResponse();
  }

  return {
    get status() {
      return status;
    },

    async start() {
      if (handle) return;
      if (startPromise) return startPromise;
      const currentGeneration = ++generation;
      setStatus("connecting");
      const controller = new AbortController();
      abort = controller;
      const starting = (async () => {
        try {
          const fullCall: RealtimeCall = {
            instructions: args.instructions,
            seed: args.seed,
            triggerResponse: args.triggerResponse,
            audio: args.audio,
            tools: tools ? await toRealtimeToolDefs(tools) : [],
          };
          if (
            generation !== currentGeneration ||
            controller.signal.aborted
          ) {
            return;
          }
          const connected = await args.model.connect({
            call: fullCall,
            emit: (event) => {
              if (generation === currentGeneration) onEvent(event);
            },
            onAudio: (pcm) => {
              if (generation === currentGeneration) cb.onAudio?.(pcm);
            },
            onChunk: (raw) => {
              if (generation === currentGeneration) {
                cb.onChunk?.({ provider: args.model.provider, raw });
              }
            },
            signal: controller.signal,
          });
          if (
            generation !== currentGeneration ||
            controller.signal.aborted
          ) {
            await connected.close();
            return;
          }
          handle = connected;
          stampClock();
          // A provider that emits `transport: connected` will already have moved
          // us on; this guard covers providers that don't.
          if (status === "connecting") setStatus("listening");
        } catch (error) {
          if (
            generation !== currentGeneration ||
            controller.signal.aborted
          ) {
            return;
          }
          abort = null;
          setStatus("error");
          cb.onError?.(
            error instanceof Error ? error.message : "connect failed",
          );
          throw error;
        }
      })();
      startPromise = starting;
      try {
        await starting;
      } finally {
        if (startPromise === starting) startPromise = null;
      }
    },

    pushAudio(pcm) {
      handle?.pushAudio(pcm);
    },

    sendText(text) {
      injectUserText(text);
    },

    serve(send) {
      sinks.add({
        onStatus: (status) => send(encodeServerEvent({ t: "status", status })),
        onTranscriptDelta: (d) =>
          send(
            encodeServerEvent({
              t: "transcript",
              id: d.id,
              role: d.role,
              delta: d.text,
              full: d.full,
              createdAt: d.createdAt,
            }),
          ),
        onMessage: (message) =>
          send(encodeServerEvent({ t: "message", message: message as RealtimeMessage })),
        onToolPending: (p) => send(encodeServerEvent({ t: "tool.pending", ...p })),
        onToolCancel: (c) => send(encodeServerEvent({ t: "tool.cancel", ...c })),
        onTurnDone: () => send(encodeServerEvent({ t: "turn.done" })),
        onInterrupted: () => send(encodeServerEvent({ t: "speech.interrupted" })),
        onAudio: (pcm) => send(pcm),
        onError: (message) => send(encodeServerEvent({ t: "error", message })),
      });
      if (status !== "idle") {
        send(encodeServerEvent({ t: "status", status }));
      }
      return {
        receive(frame) {
          if (typeof frame === "string") {
            const event = decodeClientEvent(frame);
            if (event?.t === "say") injectUserText(event.text);
            return;
          }
          handle?.pushAudio(toArrayBuffer(frame));
        },
      };
    },

    async stop() {
      generation += 1;
      abortPendingTools();
      abort?.abort();
      abort = null;
      const starting = startPromise;
      const live = handle;
      handle = null;
      if (starting) await Promise.allSettled([starting]);
      // A tool that ignored the abort may still be mid-action; wait for it,
      // so what it did reaches the record before whoever loads the transcript
      // next — a successor session starts on exactly this edge.
      await Promise.allSettled([...runningTools]);
      // Settle what the assistant had already said. Those words reached the
      // user, and dropping them would leave the transcript ending on a
      // question nobody answered — which whoever loads it next, a successor
      // session included, reads as still open.
      flush("assistant");
      buffers.user = "";
      turnIds.user = null;
      turnCreatedAt.user = null;
      turnUtterance.user = null;
      abandonUtterances();
      responseStartedAt = null;
      ttftTaken = false;
      assistantTtftMs = null;
      assistantReacted = false;
      toolCallsThisResponse = 0;
      turnEndsWithSpeech = false;
      await live?.close();
      setStatus("idle");
    },
  };
}
