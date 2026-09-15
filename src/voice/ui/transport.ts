// The voice transport — the duplex wire between VoiceChat and the relay, the
// analogue of the AI SDK's ChatTransport. VoiceChat is transport-agnostic; the
// default implementation is a WebSocket, but you can swap a mock (tests) or a
// different wire (WebTransport, etc.) without touching the class.

import { decodeServerEvent, type ServerEvent } from "../protocol";

export interface VoiceTransportHandlers {
  /** The wire is open; safe to start sending mic audio. */
  onOpen(): void;
  /** A decoded server event (state / transport / transcript / message / error). */
  onEvent(event: ServerEvent): void;
  /** An assistant audio frame (raw PCM) to play. */
  onAudio(pcm: ArrayBuffer): void;
  /** The wire closed. The event is required so every transport forwards close
   *  metadata — the close code is how a client tells a takeover from a
   *  transport failure; a transport without metadata passes `{}`. */
  onClose(event: VoiceTransportClose): void;
  /** A transport-level error. */
  onError(message: string): void;
}

export interface VoiceTransportClose {
  code?: number;
  reason?: string;
  wasClean?: boolean;
}

export interface VoiceTransportConnection {
  /** Send a mic frame (raw PCM) up to the relay. */
  sendAudio(pcm: ArrayBuffer): void;
  /** Close the wire. */
  close(): void;
}

/** The auth request context — passed to the `auth` resolver so it can mint
 *  credentials for this session. Analogous to what `prepareSendMessagesRequest`
 *  receives. */
export interface VoiceConnectRequest {
  /** The session/chat id this connection is for. */
  id: string;
}

/** Connection credentials produced by `auth` (on `VoiceChat`) and applied by the
 *  transport. Auth is transport-neutral; each transport decides how to attach
 *  these. */
export interface VoiceCredentials {
  /** Params to attach (e.g. session id + ticket). The WS transport puts them in
   *  the URL query; another transport may attach them differently. */
  params?: Record<string, string>;
  /** WebSocket subprotocols — e.g. a ticket kept out of the URL. Ignored by
   *  transports without subprotocols. */
  protocols?: string[];
}

export interface VoiceTransport {
  /** Open the connection with the resolved `credentials`, wiring inbound frames
   *  to `handlers`. The transport decides how to apply the credentials. Resolves
   *  with a handle; `handlers.onOpen` fires when the wire is live. */
  connect(
    handlers: VoiceTransportHandlers,
    credentials: VoiceCredentials,
  ): Promise<VoiceTransportConnection>;
}

const WS_OPEN = 1;

// Minimal WebSocket surface so we don't depend on a specific runtime's typings
// and can inject a fake in tests.
interface WSLike {
  binaryType: string;
  readyState: number;
  send(data: ArrayBuffer | string): void;
  close(): void;
  addEventListener(type: "message", l: (ev: { data: unknown }) => void): void;
  addEventListener(type: "open" | "error", l: () => void): void;
  addEventListener(
    type: "close",
    l: (ev: VoiceTransportClose) => void,
  ): void;
}

export interface WebSocketVoiceTransportOptions {
  /** The relay endpoint, e.g. `wss://host:3001/voice/session`. */
  url: string;
  /** WebSocket constructor seam. Defaults to the global `WebSocket`. */
  createWebSocket?: (url: string, protocols?: string[]) => WSLike;
}

const defaultCreateWebSocket = (url: string, protocols?: string[]): WSLike =>
  new (WebSocket as unknown as { new (u: string, p?: string[]): WSLike })(url, protocols);

/** The default transport: a single WebSocket carrying JSON `ServerEvent`s and
 *  binary PCM, both directions. Binary frames are audio; text frames are events.
 *  A future `WebRtcVoiceTransport` would implement the same `VoiceTransport`
 *  interface over a data channel + media tracks, with no change to `VoiceChat`. */
export class WebSocketVoiceTransport implements VoiceTransport {
  #url: string;
  #create: (url: string, protocols?: string[]) => WSLike;

  constructor(options: WebSocketVoiceTransportOptions) {
    this.#url = options.url;
    this.#create = options.createWebSocket ?? defaultCreateWebSocket;
  }

  async connect(
    handlers: VoiceTransportHandlers,
    credentials: VoiceCredentials,
  ): Promise<VoiceTransportConnection> {
    // Apply the resolved credentials: params → URL query, protocols → subprotocols.
    const url = new URL(this.#url);
    for (const [key, value] of Object.entries(credentials.params ?? {})) {
      url.searchParams.set(key, value);
    }
    const ws = this.#create(url.toString(), credentials.protocols);
    ws.binaryType = "arraybuffer";
    ws.addEventListener("message", (ev) => {
      if (typeof ev.data === "string") {
        let event: ServerEvent;
        try {
          event = decodeServerEvent(ev.data);
        } catch {
          return;
        }
        handlers.onEvent(event);
      } else if (ev.data instanceof ArrayBuffer) {
        handlers.onAudio(ev.data);
      }
    });
    ws.addEventListener("open", () => handlers.onOpen());
    ws.addEventListener("close", (event) => handlers.onClose(event));
    ws.addEventListener("error", () => handlers.onError("connection error"));
    return {
      sendAudio(pcm) {
        if (ws.readyState === WS_OPEN) ws.send(pcm);
      },
      close() {
        ws.close();
      },
    };
  }
}
