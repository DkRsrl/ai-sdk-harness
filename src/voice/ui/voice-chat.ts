// Framework-agnostic client for a server-relayed voice session — the voice
// analogue of the AI SDK's `Chat` class. It owns the live transcript + state
// and ties together a `VoiceTransport` (the wire) and `VoiceAudio` (the device).
//
// Like the AI SDK, state is exposed as a per-field subscribable store
// (`messages` separate from `status`) so consumers re-render granularly and can
// throttle the high-frequency message stream. `useVoiceChat` binds it with
// `useSyncExternalStore`.

import type { UIMessage } from "ai";
import { toUIMessages } from "../convert";
import type { ServerEvent } from "../protocol";
import {
  VOICE_CLOSE_SUPERSEDED,
  VOICE_SUPERSEDED_MESSAGE,
} from "../protocol";
import type { Role, VoiceStatus } from "../spec";
import { createBrowserAudio, type VoiceAudio } from "./audio";
import type {
  VoiceConnectRequest,
  VoiceCredentials,
  VoiceTransport,
  VoiceTransportConnection,
} from "./transport";

/** Resolves connection credentials for a session. Returned by the `auth` factory
 *  and called on each connect with the request context (the id) — mint/refresh
 *  your ticket here, like `prepareSendMessagesRequest`. */
export type VoiceAuthResolver = (
  request: VoiceConnectRequest,
) => VoiceCredentials | Promise<VoiceCredentials>;

/** The non-message reactive state, grouped so a single change is one
 *  notification with one stable object identity (read via `snapshot`). */
export interface VoiceStatusSnapshot {
  /** The single derived status (transport health folded in), pushed over the
   *  wire and refined by local connect/stop transitions. */
  status: VoiceStatus;
  /** Whether the mic is muted (client-side). */
  muted: boolean;
  /** The last error surfaced, if any. */
  error: string | null;
}

export interface VoiceChatOptions {
  /** The session/chat id. Stable across reconnects — resume a conversation by
   *  passing its id, or omit to generate one. Flows to `auth` as `request.id`. */
  id?: string;
  /** The wire to the relay. Build one with `new WebSocketVoiceTransport({ url })`,
   *  or inject a custom/mock transport. */
  transport: VoiceTransport;
  /** Auth — a factory whose resolver mints credentials (a ticket, etc.) for the
   *  session on each connect. Decoupled from the transport, which only applies
   *  the credentials it returns. Omit for an unauthenticated wire. */
  auth?: () => VoiceAuthResolver;
  /** Audio I/O. Defaults to an AudioWorklet browser implementation; inject a
   *  stub for tests or a custom platform. */
  audio?: VoiceAudio;
  /** Capture + playback sample rate (Hz) for the default audio. Default 24000. */
  sampleRate?: number;
  /** getUserMedia constraints for the default audio. */
  audioConstraints?: MediaStreamConstraints["audio"];
  /** Delays before successive reconnect attempts after an unexpected wire loss. */
  reconnectDelays?: readonly number[];
  /** How long a connection must stay live before the reconnect backoff resets.
   *  Resetting on the first live status would turn a provider that comes up and
   *  dies repeatedly into a zero-delay reconnect loop. Default 10s. */
  reconnectResetAfterMs?: number;
}

const IDLE_SNAPSHOT: VoiceStatusSnapshot = {
  status: "idle",
  muted: false,
  error: null,
};

const DEFAULT_RECONNECT_DELAYS = [0, 500, 1500, 5000] as const;
const DEFAULT_RECONNECT_RESET_AFTER_MS = 10_000;

// Leading + trailing throttle: coalesces the rapid transcript-delta stream into
// at most one notification per `wait` ms.
function throttle(fn: () => void, wait: number): () => void {
  let last = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  return () => {
    const elapsed = Date.now() - last;
    if (elapsed >= wait) {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      last = Date.now();
      fn();
    } else if (timer === null) {
      timer = setTimeout(() => {
        last = Date.now();
        timer = null;
        fn();
      }, wait - elapsed);
    }
  };
}

export class VoiceChat<UI_MESSAGE extends UIMessage = UIMessage> {
  #id: string;
  #transport: VoiceTransport;
  #auth?: () => VoiceAuthResolver;
  #resolveAuth?: VoiceAuthResolver;
  #audio: VoiceAudio;
  #conn: VoiceTransportConnection | null = null;
  #running = false;
  #lifecycle = 0;
  #attempt = 0;
  #retry = 0;
  #retryTimer: ReturnType<typeof setTimeout> | null = null;
  #backoffResetTimer: ReturnType<typeof setTimeout> | null = null;
  #reconnectDelays: readonly number[];
  #reconnectResetAfter: number;
  #superseded = false;
  #wasLive = false;
  // The *current* connection reached a live status — unlike #wasLive, which is
  // sticky for the lifecycle. Gates what an error frame means: before liveness
  // it explains why this connection is failing; after, it's an in-session error.
  #connLive = false;
  #connectionError = "voice connection lost";
  #audioOperation = Promise.resolve();

  #messages: UI_MESSAGE[] = [];
  #snapshot: VoiceStatusSnapshot = IDLE_SNAPSHOT;
  // A status the wire told us to enter while audio was still playing — applied
  // once playback drains (see #applyStatus).
  #pendingStatus: VoiceStatus | null = null;

  #messagesCallbacks = new Set<() => void>();
  #statusCallbacks = new Set<() => void>();

  constructor(options: VoiceChatOptions) {
    this.#id = options.id ?? crypto.randomUUID();
    this.#transport = options.transport;
    this.#auth = options.auth;
    this.#audio =
      options.audio ??
      createBrowserAudio({
        sampleRate: options.sampleRate,
        constraints: options.audioConstraints,
      });
    this.#reconnectDelays =
      options.reconnectDelays ?? DEFAULT_RECONNECT_DELAYS;
    this.#reconnectResetAfter =
      options.reconnectResetAfterMs ?? DEFAULT_RECONNECT_RESET_AFTER_MS;
    // When the buffered assistant audio finishes playing, settle any status the
    // wire deferred (e.g. "listening" that arrived mid-playback).
    this.#audio.onPlaybackEnd?.(() => {
      if (this.#pendingStatus === null) return;
      const next = this.#pendingStatus;
      this.#pendingStatus = null;
      this.#patch({ status: next });
    });
  }

  // --- store surface (per field, like the AI SDK's useChat) ---

  /** The session/chat id this conversation is bound to. */
  get id(): string {
    return this.#id;
  }

  /** The live transcript as AI-SDK `UIMessage`s — same shape `useChat` returns,
   *  so voice and text render through one path. Streaming turns update in place
   *  (keyed by id); the array identity changes only when messages change. */
  get messages(): UI_MESSAGE[] {
    return this.#messages;
  }
  /** The non-message reactive bundle as one object whose identity changes only
   *  when a field changes — the snapshot `useVoiceChat` reads. */
  get snapshot(): VoiceStatusSnapshot {
    return this.#snapshot;
  }
  /** The single derived status (transport health folded in). */
  get status(): VoiceStatus {
    return this.#snapshot.status;
  }
  get muted(): boolean {
    return this.#snapshot.muted;
  }
  get error(): string | null {
    return this.#snapshot.error;
  }

  /** Subscribe to message-list changes. `throttleMs` coalesces the high-frequency
   *  transcript stream. Returns an unsubscribe. */
  subscribeMessages = (onChange: () => void, throttleMs?: number): (() => void) => {
    const cb = throttleMs ? throttle(onChange, throttleMs) : onChange;
    this.#messagesCallbacks.add(cb);
    return () => this.#messagesCallbacks.delete(cb);
  };

  /** Subscribe to snapshot changes (status / muted / error). */
  subscribeStatus = (onChange: () => void): (() => void) => {
    this.#statusCallbacks.add(onChange);
    return () => this.#statusCallbacks.delete(onChange);
  };

  /** Mic input level, 0..1. Poll from a rAF loop — intentionally not part of the
   *  reactive store so it doesn't churn renders. */
  inputLevel(): number {
    return this.#audio.inputLevel();
  }
  /** Assistant output level, 0..1. */
  outputLevel(): number {
    return this.#audio.outputLevel();
  }

  async start(): Promise<void> {
    if (this.#running) return;
    this.#running = true;
    this.#superseded = false;
    this.#wasLive = false;
    this.#connectionError = "voice connection lost";
    this.#retry = 0;
    const lifecycle = ++this.#lifecycle;
    await this.#audioOperation;
    await this.#connect(lifecycle);
  }

  stop(): void {
    this.#settle("idle", null);
    this.#queueAudioStop();
    this.#conn?.close();
    this.#conn = null;
    this.#pendingStatus = null;
    this.#clearRunningTools();
  }

  async #connect(lifecycle: number): Promise<void> {
    if (!this.#running || lifecycle !== this.#lifecycle) return;
    const attempt = ++this.#attempt;
    this.#connLive = false;
    this.#patch({ status: "connecting", error: null });
    // A wire closes once; the mock-driven duplicate must not re-enter recovery.
    let closed = false;
    try {
      // Resolve auth → credentials, then hand them to the transport to apply.
      // The factory inits once (may hold state); the resolver runs per connect.
      if (this.#auth && !this.#resolveAuth) this.#resolveAuth = this.#auth();
      const credentials = (await this.#resolveAuth?.({ id: this.#id })) ?? {};
      if (!this.#current(lifecycle, attempt)) return;
      const conn = await this.#transport.connect({
        onOpen: () => {
          if (!this.#current(lifecycle, attempt)) return;
          this.#queueAudioStart(lifecycle, attempt);
        },
        onEvent: (e) => {
          if (!this.#current(lifecycle, attempt)) return;
          if (
            e.t === "status" &&
            e.status !== "idle" &&
            e.status !== "connecting" &&
            e.status !== "error" &&
            !this.#connLive
          ) {
            this.#connLive = true;
            this.#wasLive = true;
            this.#connectionError = "voice connection lost";
            // Backoff resets only once the connection has proven itself for a
            // while — an immediate reset would turn a provider that comes up
            // and dies repeatedly into a zero-delay reconnect loop.
            this.#backoffResetTimer = setTimeout(() => {
              this.#backoffResetTimer = null;
              if (this.#current(lifecycle, attempt)) this.#retry = 0;
            }, this.#reconnectResetAfter);
          }
          this.#handle(e);
        },
        onAudio: (pcm) => {
          if (this.#current(lifecycle, attempt)) this.#audio.play(pcm);
        },
        onClose: (event) => {
          if (closed) return;
          closed = true;
          if (!this.#current(lifecycle, attempt)) return;
          if (event.code === VOICE_CLOSE_SUPERSEDED) {
            this.#superseded = true;
            this.#connectionError = VOICE_SUPERSEDED_MESSAGE;
          }
          this.#recover(lifecycle, attempt);
        },
        onError: (message) => {
          if (!this.#current(lifecycle, attempt)) return;
          this.#connectionError = message;
          this.#patch({ error: message });
        },
      }, credentials);
      if (!this.#current(lifecycle, attempt)) {
        conn.close();
        return;
      }
      this.#conn = conn;
    } catch (e) {
      this.#recover(
        lifecycle,
        attempt,
        e instanceof Error ? e.message : "connection failed",
      );
    }
  }

  #current(lifecycle: number, attempt: number): boolean {
    return (
      this.#running &&
      lifecycle === this.#lifecycle &&
      attempt === this.#attempt
    );
  }

  #queueAudioStart(lifecycle: number, attempt: number): void {
    const operation = this.#audioOperation.then(async () => {
      if (!this.#current(lifecycle, attempt)) return;
      await this.#audio.start((pcm) => {
        if (this.#current(lifecycle, attempt)) this.#conn?.sendAudio(pcm);
      });
      if (!this.#current(lifecycle, attempt)) this.#audio.stop();
    });
    this.#audioOperation = operation.catch((e) => {
      this.#audio.stop();
      if (!this.#current(lifecycle, attempt)) return;
      this.#patch({
        error: e instanceof Error ? e.message : "microphone error",
      });
    });
  }

  #queueAudioStop(): Promise<void> {
    this.#audioOperation = this.#audioOperation.then(
      () => this.#audio.stop(),
      () => this.#audio.stop(),
    );
    return this.#audioOperation;
  }

  #recover(
    lifecycle: number,
    attempt: number,
    terminalError = this.#connectionError,
  ): void {
    if (!this.#current(lifecycle, attempt)) return;
    this.#attempt++;
    this.#connLive = false;
    if (this.#backoffResetTimer) clearTimeout(this.#backoffResetTimer);
    this.#backoffResetTimer = null;
    const audioStopped = this.#queueAudioStop();
    this.#conn = null;
    this.#pendingStatus = null;
    this.#clearRunningTools();

    if (this.#superseded) {
      this.#settle("idle", this.#connectionError);
      return;
    }

    if (!this.#wasLive) {
      this.#settle("error", terminalError);
      return;
    }

    const delay = this.#reconnectDelays[this.#retry++];
    if (delay === undefined) {
      this.#settle("error", terminalError);
      return;
    }

    this.#patch({ status: "connecting", error: null });
    void audioStopped.then(() => {
      if (!this.#running || lifecycle !== this.#lifecycle) return;
      this.#retryTimer = setTimeout(() => {
        this.#retryTimer = null;
        void this.#connect(lifecycle);
      }, delay);
    });
  }

  // The one way a lifecycle ends: nothing from it stays current, no timer keeps
  // running, and the store reflects the terminal state.
  #settle(status: VoiceStatus, error: string | null): void {
    this.#running = false;
    this.#lifecycle++;
    this.#attempt++;
    this.#connLive = false;
    if (this.#retryTimer) clearTimeout(this.#retryTimer);
    this.#retryTimer = null;
    if (this.#backoffResetTimer) clearTimeout(this.#backoffResetTimer);
    this.#backoffResetTimer = null;
    this.#patch({ status, error });
  }

  setMuted(muted: boolean): void {
    this.#audio.setMuted(muted);
    this.#patch({ muted });
  }

  #handle(event: ServerEvent): void {
    switch (event.t) {
      case "status":
        this.#applyStatus(event.status);
        break;
      case "transcript":
        this.#upsertText(event.id, event.role, event.full, "streaming", event.createdAt);
        break;
      case "tool.pending":
        this.#upsertTool(event.callId, event.name, event.input, event.createdAt);
        break;
      case "tool.cancel":
        this.#removeMessage(event.callId);
        break;
      case "speech.interrupted":
        // Barge-in: drop the assistant audio still scheduled ahead of real time,
        // then let the status fall to "listening" now (nothing left to drain, so
        // `#applyStatus` won't defer it behind playback).
        this.#audio.clearPlayback?.();
        this.#applyStatus("listening");
        break;
      case "message": {
        // Same converter the relay persists with, so the rendered turn is byte-
        // for-byte the stored shape (incl. the `createdAt` ordering stamp).
        const [ui] = toUIMessages([event.message]);
        if (ui) this.#upsert(ui as unknown as UI_MESSAGE);
        break;
      }
      case "superseded":
        // Another session is speaking this conversation now. The socket closes
        // right behind this frame, so drop the audio still queued ahead of real
        // time and say why before the wire closes.
        this.#audio.clearPlayback?.();
        this.#superseded = true;
        this.#connectionError = VOICE_SUPERSEDED_MESSAGE;
        this.#patch({ error: this.#connectionError });
        break;
      case "error":
        // Before this connection is live, the message explains why it is
        // failing (e.g. the relay's "voice session could not start") and must
        // survive as the terminal reason; after, it's an in-session error.
        if (!this.#connLive) this.#connectionError = event.message;
        this.#patch({ error: event.message });
        break;
    }
  }

  // Providers signal generation-end (response.done / turnComplete) before the
  // browser finishes playing the buffered audio. Hold "speaking" until playback
  // drains so the UI's speaking indicator matches what's actually audible.
  #applyStatus(next: VoiceStatus): void {
    if (
      next !== "speaking" &&
      this.#snapshot.status === "speaking" &&
      this.#audio.isPlaying?.()
    ) {
      this.#pendingStatus = next;
      return;
    }
    this.#pendingStatus = null;
    this.#patch({ status: next });
  }

  // A live caption is a `streaming` text part; the settled turn flips the same
  // id to `done` — the per-part lifecycle the AI SDK uses for streamed text, so
  // a renderer can show a cursor while `state === "streaming"`.
  #upsertText(
    id: string,
    role: Role,
    text: string,
    state: "streaming" | "done",
    createdAt: number,
  ): void {
    this.#upsert({
      id,
      role,
      parts: [{ type: "text", text, state }],
      metadata: { createdAt },
    } as unknown as UI_MESSAGE);
  }

  // A running tool call is a `dynamic-tool` part with no output yet (`state:
  // "input-available"`); the eventual `message` flips the same callId to
  // `output-available` in place — mirroring the streaming→done text lifecycle.
  #upsertTool(
    callId: string,
    name: string,
    input: unknown,
    createdAt: number,
  ): void {
    this.#upsert({
      id: callId,
      role: "assistant",
      parts: [
        {
          type: "dynamic-tool",
          toolName: name,
          toolCallId: callId,
          state: "input-available",
          input,
        },
      ],
      metadata: { createdAt },
    } as unknown as UI_MESSAGE);
  }

  // Immutable upsert keyed by id: a streaming transcript becomes the settled
  // message in place, and the new array identity drives the subscription.
  #upsert(message: UI_MESSAGE): void {
    const next = this.#messages.slice();
    const i = next.findIndex((m) => m.id === message.id);
    if (i >= 0) next[i] = message;
    else next.push(message);
    this.#messages = next;
    for (const cb of this.#messagesCallbacks) cb();
  }

  // Drop a message by id — used to retract a cancelled tool's pending render.
  #removeMessage(id: string): void {
    const next = this.#messages.filter((m) => m.id !== id);
    if (next.length === this.#messages.length) return;
    this.#messages = next;
    for (const cb of this.#messagesCallbacks) cb();
  }

  // A pending tool is a lone `dynamic-tool` part still short of its result.
  #isRunningTool(message: UI_MESSAGE): boolean {
    const parts = (message as { parts?: Array<{ type?: string; state?: string }> })
      .parts;
    return (
      !!parts &&
      parts.length > 0 &&
      parts.every(
        (p) =>
          p.type === "dynamic-tool" &&
          p.state !== "output-available" &&
          p.state !== "output-error",
      )
    );
  }

  // Forget any tool left mid-flight when the session tears down — the result
  // can no longer arrive, so the running render would otherwise hang.
  #clearRunningTools(): void {
    if (!this.#messages.some((m) => this.#isRunningTool(m))) return;
    this.#messages = this.#messages.filter((m) => !this.#isRunningTool(m));
    for (const cb of this.#messagesCallbacks) cb();
  }

  #patch(patch: Partial<VoiceStatusSnapshot>): void {
    const next: VoiceStatusSnapshot = { ...this.#snapshot, ...patch };
    if (
      next.status === this.#snapshot.status &&
      next.muted === this.#snapshot.muted &&
      next.error === this.#snapshot.error
    ) {
      return; // no change — keep the stable identity, skip notifying
    }
    this.#snapshot = next;
    for (const cb of this.#statusCallbacks) cb();
  }
}
