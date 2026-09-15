// Public type vocabulary for the Grok (xAI) realtime provider — the values a
// caller actually types out (model ids, voice, api options), kept apart from the
// implementation so the option surface reads at a glance. The `(string & {})`
// arms keep each union open to new/custom ids the docs haven't caught up with
// while still offering autocomplete on the known values.

export type { WSLike } from "../ws";
import type { WSLike } from "../ws";

export type GrokWebSocketFactory = (
  url: string,
  options: { headers: Record<string, string> },
) => WSLike;

/** xAI realtime voice models. `grok-voice-latest` tracks the current
 *  recommendation; `grok-voice-fast-1.0` is the deprecated legacy model. */
export type GrokVoiceModel =
  | "grok-voice-think-fast-1.0"
  | "grok-voice-latest"
  | "grok-voice-fast-1.0"
  | (string & {});

/** Built-in xAI voices; a custom voice id string is also accepted. */
export type GrokVoice = "eve" | "ara" | "rex" | "sal" | "leo" | (string & {});

/** Reasoning depth. Only `grok-voice-think-fast-1.0` honors it; xAI defaults to
 *  `"high"` when unset. */
export type GrokReasoningEffort = "high" | "none";

/** Audio container/codec. `audio/pcm` is linear 16-bit; `audio/pcmu`/`audio/pcma`
 *  are G.711 µ-law/A-law, fixed at an 8000 Hz sample rate. */
export type GrokAudioFormat = "audio/pcm" | "audio/pcmu" | "audio/pcma";

/** PCM sample rates xAI accepts (Hz). The G.711 formats are fixed at 8000. */
export type GrokSampleRate = 8000 | 16000 | 22050 | 24000 | 32000 | 44100 | 48000;

export interface GrokModelOptions {
  /** xAI API key. Sent as a `Bearer` Authorization header on the WS upgrade.
   *  Defaults to `process.env.XAI_API_KEY`. Keep it server-side (this model is
   *  never run in the browser). */
  apiKey?: string;
  /** Output voice id. Defaults to `"eve"`. */
  voice?: GrokVoice;
  /** Override the realtime endpoint. Defaults to `wss://api.x.ai/v1/realtime`. */
  url?: string;
  /** PCM sample rate for both the mic input and assistant output, in Hz.
   *  Defaults to 24000. Must match what the client captures/plays. */
  sampleRate?: GrokSampleRate;
  /** Audio container/codec for both directions. Defaults to `"audio/pcm"`.
   *  The G.711 formats (`audio/pcmu`/`audio/pcma`) require `sampleRate: 8000`. */
  format?: GrokAudioFormat;
  /** Reasoning depth, forwarded as the session's `reasoning_effort`. Only
   *  `grok-voice-think-fast-1.0` honors it; omit to take xAI's `"high"` default. */
  reasoningEffort?: GrokReasoningEffort;
  /** BCP-47 hint to bias input transcription (e.g. `"it"`, `"es-MX"`), sent as
   *  the session's `audio.input.transcription.language_hint`. */
  language?: string;
  /** Proper nouns to bias input transcription toward (product and client
   *  names), sent as the session's `audio.input.transcription.keyterms`. The
   *  list is normalized to xAI's limits — 100 terms of up to 50 characters. */
  keyterms?: readonly string[];
  /** Override the `turn_detection` block. Defaults to
   *  `{ type: "server_vad" }` — VAD on, thresholds left to xAI's defaults. */
  turnDetection?: Record<string, unknown>;
  /** Runtime/test seam for the WebSocket constructor. Defaults to the global
   *  WebSocket (Bun's, which honors the `headers` option for server auth). */
  createWebSocket?: GrokWebSocketFactory;
}
