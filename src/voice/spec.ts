// The realtime-voice spec: the contract every provider satisfies, plus the
// standardized message types the core speaks. The core (`createRealtimeSession`)
// depends ONLY on this module; concrete providers (openai/gemini/grok) implement
// it and are injected — nothing here imports a provider, exactly as a
// `LanguageModel` is injected into `streamText`.
//
// Two planes cross the provider boundary:
//   - semantic: RealtimeEvent (provider -> core) / RealtimeOutbound (core -> provider)
//   - audio:    raw PCM frames, kept off the event stream so the relay can
//               fast-path them between the client socket and the provider.

export type Role = "user" | "assistant";

/** The single user-facing status, derived by the core from the event stream:
 *  the conversation lifecycle with transport health folded in (a connecting
 *  wire reads as `connecting`, a dropped one as `error`). Providers never set
 *  it — they only emit the raw events below. */
export type VoiceStatus =
  | "idle"
  | "connecting"
  | "listening"
  | "speaking"
  // Producing, with nothing audible yet and no tool out: the model is writing
  // the answer. Distinct from `thinking` because they cost a listener different
  // things — one is a pause in a conversation, the other is a lookup whose
  // result will change the answer, and a client that renders them alike leaves
  // a plain greeting looking exactly like a query to a backend tool.
  | "composing"
  // A tool is running. The wait has a cause outside the model.
  | "thinking"
  | "error";

/** Provider transport health, reported provider → core. The core folds it into
 *  `VoiceStatus`; it is never surfaced to the client as a separate field. */
export type TransportStatus = "connected" | "connecting" | "disconnected";

/** A tool reduced to what a realtime provider advertises to the model. */
export interface RealtimeToolDef {
  /** The tool name the model calls. */
  name: string;
  /** What the tool does — shown to the model to decide when to call it. */
  description: string;
  /** JSON Schema of the tool input, derived from the AI SDK tool's `inputSchema`. */
  parametersJsonSchema: Record<string, unknown>;
}

export interface RealtimeAudioConfig {
  /** Mic sample rate sent to the provider. Default 16000. */
  inputSampleRate?: number;
  /** Assistant playback sample rate. Default 24000. */
  outputSampleRate?: number;
  /** Optional input-transcription settings (model/language), provider-dependent. */
  transcription?: { model?: string; language?: string };
}

/** The standardized call that opens a session — the voice analogue of a
 *  streamText call: instructions + tools + seed + audio config. Built
 *  server-side; the provider translates it into its own session config. */
export interface RealtimeCall {
  /** The system prompt for the session. */
  instructions: string;
  /** Tool definitions advertised to the model. The core derives these from the
   *  `ToolSet` you pass to `createRealtimeSession`, so you don't build them by
   *  hand (that function takes `Omit<RealtimeCall, "tools">`). */
  tools: RealtimeToolDef[];
  /** Prior conversation replayed into the session before audio flows — text and
   *  tool round-trips alike, so a resumed session keeps full fidelity. The
   *  provider replays these during `connect`, absorbing its own replay quirks: a
   *  provider with native history (Grok) reuses the live `RealtimeHandle.send`
   *  mapping, while one without (Gemini, whose Live API rejects replayed model
   *  turns) folds them into the system instruction as a transcript. */
  seed: RealtimeOutbound[];
  /** Whether the model should speak first after seeding (proactive open). */
  triggerResponse: boolean;
  /** Optional audio config (sample rates, transcription). Provider defaults apply. */
  audio?: RealtimeAudioConfig;
}

/** provider -> core. The normalized semantic event stream (no audio frames).
 *  The core derives VoiceStatus, assembles transcripts, and dispatches tools
 *  from these. */
export type RealtimeEvent =
  | { type: "transport"; status: TransportStatus }
  | { type: "speech.start" }
  | { type: "speech.stop" }
  // Barge-in: the assistant's turn was cut off. Providers emit this whenever
  // the user interrupts — over assistant audio OR while a tool call is still
  // running — so the core has one signal to cancel pending work.
  | { type: "speech.interrupted"; utterance?: string }
  // Transcript events may carry `utterance`: the provider's stable key for the
  // spoken utterance the event belongs to (e.g. xAI's conversation item id).
  // The core uses it to spot an utterance boundary when a turn's final never
  // arrived, to drop a repeated final, and to repair a turn it settled early —
  // without it, a lost final lets the next utterance write under the previous
  // turn's identity. Optional: providers without such a key omit it.
  | { type: "transcript.delta"; role: Role; text: string; utterance?: string }
  // A cumulative full-text snapshot of the in-progress transcript — REPLACES,
  // not appends. For providers that re-send the whole transcript-so-far (and may
  // revise it) instead of emitting incremental deltas, e.g. xAI's `updated` /
  // in-progress `completed` events.
  | { type: "transcript.update"; role: Role; text: string; utterance?: string }
  | { type: "transcript.final"; role: Role; text: string; utterance?: string }
  // The streamed transcript for this role stopped growing — finalize the
  // accumulated deltas now. The text-stream analogue of `speech.stop`, kept
  // separate so a turn settles when its TEXT is done, not when its audio ends.
  | { type: "transcript.done"; role: Role }
  // The model opened / closed a response — the only signal that says whether it
  // is still producing, since it can think for seconds between two audible
  // steps. One answer usually spans SEVERAL responses: a provider opens a new
  // one after every tool round-trip, so the answer is over at the
  // `response.done` that leaves no tool result outstanding, which is what the
  // core derives from these (`onTurnDone`).
  | { type: "response.start" }
  | { type: "response.done" }
  | { type: "tool.call"; callId: string; name: string; input: unknown }
  | { type: "error"; message: string; fatal: boolean };

/** core -> provider. Items injected into the live session: seed replay of a
 *  prior conversation, and results of live tool calls. */
export type RealtimeOutbound =
  | { type: "text"; role: Role; text: string }
  | { type: "tool.call"; callId: string; name: string; input: unknown }
  // `isError` marks a failed execution. Providers whose wire has no error
  // channel (function_call_output carries one output string) render it as
  // plain "Error: …" prose the model can read; Gemini keeps its documented
  // `{ error }` response shape.
  | { type: "tool.result"; callId: string; output: unknown; isError?: boolean };

/** A raw, un-normalized frame straight off the provider wire — the parsed JSON
 *  server message (Gemini) or event (Grok), exactly as received. Untyped on
 *  purpose: it is the provider's own shape, including fields the provider maps
 *  to nothing (e.g. Gemini `usageMetadata`, Grok `response.done.usage`). */
export interface RealtimeChunk {
  /** The provider that produced it (`"gemini"` | `"grok"` | …). */
  provider: string;
  /** The raw parsed frame, before any normalization. */
  raw: unknown;
}

export interface RealtimeConnectArgs {
  call: RealtimeCall;
  /** Semantic events out. */
  emit: (event: RealtimeEvent) => void;
  /** Assistant audio frames out (raw PCM), relayed to the client socket. */
  onAudio: (pcm: ArrayBuffer) => void;
  /** Optional debug sink: every raw frame off the provider wire, handed over
   *  BEFORE normalization — including frames the provider otherwise ignores.
   *  Wire it only to inspect exactly what a provider returns; off the hot path
   *  when unset. */
  onChunk?: (raw: unknown) => void;
  /** Aborted when the core tears the session down. */
  signal: AbortSignal;
}

/** The live control handle the core drives. Muting and volume metering are
 *  absent by design: in a server relay they are client-side concerns (the
 *  browser holds the mic and plays the audio). */
export interface RealtimeHandle {
  /** Inject a conversation item (seed replay or a tool result). */
  send(item: RealtimeOutbound): void;
  /** Feed a user mic frame (raw PCM) into the session. */
  pushAudio(pcm: ArrayBuffer): void;
  /** Ask the model to produce a response now (proactive open / post-seed). */
  requestResponse(): void;
  close(): Promise<void>;
}

/** The provider contract. One implementation per provider (openai/gemini/grok),
 *  living outside the core and injected into it. */
/**
 * A realtime model instance — the voice analogue of the AI SDK's
 * `LanguageModelV2`. Created by a provider factory (e.g. `grok("model-id")`),
 * it carries the provider name, the model id, and its connection config (API
 * key, voice, …), and knows how to open a live session.
 */
export interface RealtimeModelV1 {
  /** The spec version this model implements. */
  readonly specificationVersion: "realtime-v1";
  /** Stable provider name, e.g. `"grok"` | `"gemini"` | `"openai"`. */
  readonly provider: string;
  /** The provider's model id this instance is bound to. */
  readonly modelId: string;
  /** Open a live session for `call`, emit events + audio via the supplied
   *  sinks, and return the control handle. Called by the core, not by you. */
  connect(args: RealtimeConnectArgs): Promise<RealtimeHandle>;
}
