// Public type vocabulary for the Gemini Live realtime provider — the caller's
// options plus the minimal slice of the @google/genai live API we use, so the
// provider doesn't fight the SDK's types and tests can inject a fake. Kept apart
// from the implementation in ./index.

export interface GeminiFunctionCall {
  id?: string;
  name?: string;
  args?: Record<string, unknown>;
}

export interface GeminiServerMessage {
  /** Handshake ack — the session is ready and will now accept client content,
   *  realtime input, and tool responses. Sending any of those before it arrives
   *  is rejected with "invalid argument", so the provider gates on it. */
  setupComplete?: object;
  toolCall?: { functionCalls?: GeminiFunctionCall[] };
  serverContent?: {
    modelTurn?: { parts?: Array<{ inlineData?: { data?: string } }> };
    interrupted?: boolean;
    generationComplete?: boolean;
    turnComplete?: boolean;
    inputTranscription?: { text?: string };
    outputTranscription?: { text?: string };
  };
}

export interface GeminiLiveSession {
  sendRealtimeInput(input: { audio: { data: string; mimeType: string } }): void;
  // Live `clientContent` only accepts `role: "user"` turns — a `model` turn is
  // rejected with "invalid argument". Prior assistant/tool history can't be
  // replayed here; it's folded into the system instruction instead.
  // `turns` is optional: omit it (with `turnComplete: true`) to ask the model to
  // respond now — passing an empty array makes the SDK throw "Failed to parse".
  sendClientContent(input: {
    turns?: Array<{ role: string; parts: Array<{ text: string }> }>;
    turnComplete?: boolean;
  }): void;
  sendToolResponse(input: {
    functionResponses: Array<{ id?: string; name?: string; response: Record<string, unknown> }>;
  }): void;
  close(): void;
}

export interface GeminiLiveCallbacks {
  onopen?: () => void;
  onmessage?: (message: GeminiServerMessage) => void;
  onerror?: (e: { message?: string }) => void;
  onclose?: (e?: { reason?: string }) => void;
}

export type GeminiConnect = (args: {
  model: string;
  config: Record<string, unknown>;
  callbacks: GeminiLiveCallbacks;
}) => Promise<GeminiLiveSession>;

// --- public option types: the Live-API models and voices, so callers get
// autocomplete on the known values while `(string & {})` keeps each union open
// to new/custom ids the docs haven't caught up with. ---

/** Gemini Live API model ids. `gemini-3.1-flash-live-preview` is the current
 *  flagship; the 2.5 entries are the prior native-audio / live previews. */
export type GeminiVoiceModel =
  | "gemini-3.1-flash-live-preview"
  | "gemini-2.5-flash-native-audio-preview-12-2025"
  | "gemini-2.5-flash-live-preview"
  | (string & {});

/** Gemini prebuilt voices for Live audio output. A custom/new voice name is
 *  also accepted. */
export type GeminiVoice =
  | "Zephyr"
  | "Puck"
  | "Charon"
  | "Kore"
  | "Fenrir"
  | "Leda"
  | "Orus"
  | "Aoede"
  | "Callirrhoe"
  | "Autonoe"
  | "Enceladus"
  | "Iapetus"
  | "Umbriel"
  | "Algieba"
  | "Despina"
  | "Erinome"
  | "Algenib"
  | "Rasalgethi"
  | "Laomedeia"
  | "Achernar"
  | "Alnilam"
  | "Schedar"
  | "Gacrux"
  | "Pulcherrima"
  | "Achird"
  | "Zubenelgenubi"
  | "Vindemiatrix"
  | "Sadachbia"
  | "Sadaltager"
  | "Sulafat"
  | (string & {});

export interface GeminiModelOptions {
  /** Google API key. Defaults to `GOOGLE_API_KEY` / `GEMINI_API_KEY`. */
  apiKey?: string;
  /** Output voice id. Defaults to `"Kore"`. */
  voice?: GeminiVoice;
  /** Client capture rate (Hz). Gemini resamples internally, so any rate is
   *  accepted (hence `number`, not a fixed union); just tag the real one.
   *  Default 24000. */
  sampleRate?: number;
  /** Gemini API version. Defaults to `"v1alpha"`. */
  apiVersion?: string;
  /** Test/runtime seam for the live connection. Defaults to `@google/genai`. */
  connect?: GeminiConnect;
}
