// The client↔relay wire contract for a voice session over a single WebSocket —
// the analogue of the AI SDK's UI message stream protocol. Pure types both
// sides import, so the relay (server) and the RelayVoice transport (browser)
// never drift.
//
// Two frame kinds share the socket:
//   - binary frames = raw PCM audio (mic up / assistant down). Audio is the
//                     only binary payload, so `typeof data !== "string"` ⇒ audio.
//   - text frames   = JSON. relay→client is a ServerEvent; client→relay is a
//                     ClientEvent (typed user turns — the browser still sends
//                     only mic audio, but a headless client can speak in text).

import type { RealtimeMessage } from "./session";
import type { Role, VoiceStatus } from "./spec";

// App-range close codes, so a client can tell "you were taken over", "this
// session never came up" and "the provider died under a live session" from a
// transport failure without parsing the reason.
export const VOICE_CLOSE_SUPERSEDED = 4001;
export const VOICE_CLOSE_START_FAILED = 4002;
export const VOICE_CLOSE_RUNTIME_LOST = 4003;

/** The one user-facing sentence for a takeover, shared by every path that can
 *  learn of one (the `superseded` event, or its close code when the event was
 *  lost). */
export const VOICE_SUPERSEDED_MESSAGE =
  "another voice session took over this conversation";

/** relay → client. Everything the browser needs to render a live session,
 *  minus the assistant audio (which rides as binary frames). */
export type ServerEvent =
  // The derived status changed (transport health is folded into it).
  | { t: "status"; status: VoiceStatus }
  // Streaming caption for the in-progress turn; `full` is the text so far.
  // `createdAt` is the turn's ordering stamp (matches the eventual `message`).
  | { t: "transcript"; id: string; role: Role; delta: string; full: string; createdAt: number }
  // A tool call has started but not yet finished — render-only, so the UI can
  // show it running before the result lands. `callId`/`createdAt` match the
  // eventual `message`, which replaces it in place. Never persisted.
  | { t: "tool.pending"; callId: string; name: string; input: unknown; createdAt: number }
  // A running tool was aborted (barge-in / session end) and will never finish —
  // drop its pending render. The inverse of `tool.pending`; never persisted.
  | { t: "tool.cancel"; callId: string }
  // Barge-in: the user started talking over the assistant. The provider has
  // already cancelled generation, but assistant audio frames already sent are
  // queued ahead of real time in the client's playback buffer — the client must
  // flush them on this signal, or the agent audibly keeps talking. Never persisted.
  | { t: "speech.interrupted" }
  // A finalized turn (text or completed tool round-trip) to render/persist.
  | { t: "message"; message: RealtimeMessage }
  // The assistant is done answering: nothing more is coming for the user's
  // turn. A client that has to know when to stop waiting reads this instead of
  // timing the silence, which a model thinking between two steps looks exactly
  // like. Never persisted.
  | { t: "turn.done" }
  // A newer session took this conversation over (a reload, a second tab): this
  // one has stopped and the socket closes immediately after. Not an error —
  // the conversation is simply being spoken elsewhere.
  | { t: "superseded" }
  // An error to surface to the user.
  | { t: "error"; message: string };

/** client → relay. Commands a client can send as text frames. */
export type ClientEvent =
  // Inject a typed user turn — the text analogue of a mic utterance. The
  // session records it as a finalized user message and asks the model to
  // respond, so it exercises the real voice agent without audio.
  { t: "say"; text: string };

/** Serialize a `ServerEvent` to a JSON text frame (relay → client). */
export function encodeServerEvent(event: ServerEvent): string {
  return JSON.stringify(event);
}

/** Parse a JSON text frame back into a `ServerEvent` (client side). */
export function decodeServerEvent(data: string): ServerEvent {
  return JSON.parse(data) as ServerEvent;
}

/** Serialize a `ClientEvent` to a JSON text frame (client → relay). */
export function encodeClientEvent(event: ClientEvent): string {
  return JSON.stringify(event);
}

/** Parse an inbound text frame into a `ClientEvent`, or null when it isn't
 *  one — the relay receives arbitrary client strings, so this validates
 *  rather than trusts. */
export function decodeClientEvent(data: string): ClientEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const event = parsed as { t?: unknown; text?: unknown };
  if (event.t === "say" && typeof event.text === "string") {
    return { t: "say", text: event.text };
  }
  return null;
}
