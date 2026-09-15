// Public type vocabulary for the Vercel AI Gateway realtime provider — the
// values a caller actually types out (model ids, options), kept apart from the
// implementation so the option surface reads at a glance. The `(string & {})`
// arm keeps the model union open to ids the catalog serves before the docs
// catch up, while still offering autocomplete on the known values.

export type { WSLike } from "../ws";
import type { WSLike } from "../ws";

/** The gateway authenticates the WebSocket upgrade via subprotocols (no
 *  headers), so the factory takes the protocol list instead of an options bag. */
export type GatewayWebSocketFactory = (url: string, protocols: string[]) => WSLike;

/** Realtime models the gateway serves, as `creator/model` gateway ids — the
 *  same id shape every text task uses. The catalog (`type: "realtime"`) is the
 *  authoritative list. */
export type GatewayVoiceModel =
  | "xai/grok-voice-think-fast-1.0"
  | "xai/grok-voice-think-fast-2.0"
  | "openai/gpt-realtime-2"
  | (string & {});

export interface GatewayModelOptions {
  /** Gateway API key, carried in the auth subprotocol of the WS upgrade.
   *  Defaults to `process.env.AI_GATEWAY_API_KEY`. Keep it server-side (this
   *  model is never run in the browser; the gateway itself rejects browser
   *  connections). */
  apiKey?: string;
  /** Output voice id, passed through to the underlying provider. Defaults by
   *  the model's creator prefix: `"eve"` for `xai/…`, `"marin"` for
   *  `openai/…`, otherwise the provider's own default. */
  voice?: string;
  /** Override the realtime endpoint. Defaults to
   *  `wss://ai-gateway.vercel.sh/v4/ai/realtime-model`. */
  url?: string;
  /** PCM sample rate for both the mic input and assistant output, in Hz.
   *  Defaults to 24000. Must match what the client captures/plays. */
  sampleRate?: number;
  /** Override the normalized turn-detection block. Defaults to
   *  `{ type: "server-vad" }` — VAD on, thresholds left to the underlying
   *  provider's defaults. */
  turnDetection?: Record<string, unknown>;
  /** Language hint for input transcription, forwarded as the normalized
   *  transcription config's `language`. */
  language?: string;
  /** Proper nouns to bias input transcription toward (product and client
   *  names). The normalized dialect has no field for them, so they ride to the
   *  provider through `providerOptions` in xAI's native shape; a provider that
   *  does not know key terms ignores them. Normalized to 100 terms of up to 50
   *  characters. */
  keyterms?: readonly string[];
  /** Provider-native session fields, merged by the gateway into the underlying
   *  provider's session config as-is (un-namespaced) — e.g. xAI's
   *  `{ audio: { input: { transcription: { keyterms: [...] } } } }`. This
   *  merge shape is undocumented beta behavior, pinned by a regression test. */
  providerOptions?: Record<string, unknown>;
  /** Runtime/test seam for the WebSocket constructor. Defaults to the global
   *  WebSocket. */
  createWebSocket?: GatewayWebSocketFactory;
}
