// Gemini Live realtime provider — server-side, over the @google/genai SDK's
// `live.connect`. Implements RealtimeModelV1: it only translates the
// standardized call/events to and from Gemini's wire. The session core owns
// state, transcripts, and tool execution. The caller-facing types (options and
// the SDK slice we depend on) live in ./types.
//
// Two Gemini-specific quirks are absorbed here so the client and protocol stay
// provider-agnostic:
//   - Gemini resamples input itself ("natively 16kHz, but the Live API will
//     resample so any sample rate can be sent"), so we just tag the client's
//     rate in the mimeType — no client-side resampling. Output is always 24kHz.
//   - Gemini emits transcription incrementally — we buffer the user transcript
//     and finalize it when the model starts responding.

import type {
  RealtimeCall,
  RealtimeConnectArgs,
  RealtimeHandle,
  RealtimeModelV1,
  RealtimeOutbound,
  RealtimeToolDef,
} from "../../spec";
import type {
  GeminiConnect,
  GeminiLiveCallbacks,
  GeminiLiveSession,
  GeminiModelOptions,
  GeminiServerMessage,
  GeminiVoice,
  GeminiVoiceModel,
} from "./types";

export type {
  GeminiConnect,
  GeminiFunctionCall,
  GeminiLiveCallbacks,
  GeminiLiveSession,
  GeminiModelOptions,
  GeminiServerMessage,
  GeminiVoice,
  GeminiVoiceModel,
} from "./types";

const DEFAULT_VOICE: GeminiVoice = "Kore";
const DEFAULT_SAMPLE_RATE = 24_000;

// --- pure mappers (exported for tests) ---

export function geminiToolDefs(
  tools: RealtimeToolDef[],
): Array<{ functionDeclarations: unknown[] }> | undefined {
  if (tools.length === 0) return undefined;
  return [
    {
      functionDeclarations: tools.map((t) => ({
        name: t.name,
        description: t.description,
        parametersJsonSchema: t.parametersJsonSchema,
      })),
    },
  ];
}

function buildConfig(call: RealtimeCall, voice: string): Record<string, unknown> {
  const tools = geminiToolDefs(call.tools);
  return {
    responseModalities: ["AUDIO"],
    systemInstruction: { parts: [{ text: call.instructions }] },
    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } },
    ...(tools ? { tools } : {}),
    inputAudioTranscription: {},
    outputAudioTranscription: {},
  };
}

// Framing for the seeded history. Gemini Live can't replay model turns, so the
// prior conversation rides one `user` turn as a transcript wrapped in a
// <previous_conversation_messages> tag; this preface tells the model to treat it
// as real shared history rather than text to act on.
const SEED_PREFACE =
  "The transcript of our earlier conversation is below, inside <previous_conversation_messages>. Treat it as our real shared history and continue naturally from it — don't restate or summarize it unless I ask.";

function b64ToArrayBuffer(b64: string): ArrayBuffer {
  const buf = Buffer.from(b64, "base64");
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function toGeminiResponse(output: unknown): Record<string, unknown> {
  if (output && typeof output === "object") return output as Record<string, unknown>;
  return { output };
}

function geminiRole(role: "user" | "assistant"): string {
  return role === "assistant" ? "model" : "user";
}

/**
 * Render a prior conversation as a plain-text transcript for Gemini, or `null`
 * for an empty seed. Gemini Live cannot replay history structurally: its
 * `clientContent.turns` only accepts `user` turns (a `model` turn — an assistant
 * reply or a `functionCall` — is rejected with "invalid argument"), and its
 * `sessionResumption` handle is an opaque, time-limited live-reconnect token,
 * not a portable transcript. So the whole history is collapsed into one `user`
 * turn carrying this transcript (see `SEED_PREFACE`). Tool round-trips are
 * rendered inline — the call's input then its result, matched by `callId` — so
 * they resume too, not just text.
 */
export function renderSeedTranscript(seed: RealtimeOutbound[]): string | null {
  if (seed.length === 0) return null;
  const names = new Map<string, string>();
  const lines: string[] = [];
  for (const item of seed) {
    switch (item.type) {
      case "text":
        lines.push(`${item.role === "assistant" ? "Assistant" : "User"}: ${item.text}`);
        break;
      case "tool.call":
        names.set(item.callId, item.name);
        lines.push(`Assistant called tool "${item.name}" with ${JSON.stringify(item.input ?? {})}`);
        break;
      case "tool.result": {
        const name = names.get(item.callId);
        lines.push(`Tool${name ? ` "${name}"` : ""} returned ${JSON.stringify(item.output ?? null)}`);
        break;
      }
    }
  }
  return lines.join("\n");
}

function defaultConnect(options: GeminiModelOptions): GeminiConnect {
  return async ({ model, config, callbacks }) => {
    const apiKey =
      options.apiKey ??
      process.env.GOOGLE_API_KEY ??
      process.env.GEMINI_API_KEY ??
      process.env.GOOGLE_GENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "Missing Google API key: set GOOGLE_API_KEY or pass gemini(model, { apiKey }).",
      );
    }
    const { GoogleGenAI } = await import("@google/genai/node");
    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: { apiVersion: options.apiVersion ?? "v1alpha" },
    });
    return (await (
      ai.live as unknown as {
        connect: (p: {
          model: string;
          config: unknown;
          callbacks: GeminiLiveCallbacks;
        }) => Promise<GeminiLiveSession>;
      }
    ).connect({ model, config, callbacks })) as GeminiLiveSession;
  };
}

/**
 * Create a Gemini Live realtime model — the voice analogue of `openai("gpt-4")`.
 * Pass it as `model` to `createRealtimeSession`.
 *
 * @param modelId - The Gemini Live model id (e.g. `"gemini-3.1-flash-live-preview"`).
 * @param options - API key (defaults to `GOOGLE_API_KEY`), voice, sample rate, etc.
 */
export function gemini(modelId: GeminiVoiceModel, options: GeminiModelOptions = {}): RealtimeModelV1 {
  const voice = options.voice ?? DEFAULT_VOICE;
  const sampleRate = options.sampleRate ?? DEFAULT_SAMPLE_RATE;
  const liveConnect = options.connect ?? defaultConnect(options);

  return {
    specificationVersion: "realtime-v1",
    provider: "gemini",
    modelId,

    async connect({ call, emit, onAudio, onChunk, signal }: RealtimeConnectArgs): Promise<RealtimeHandle> {
      let speaking = false;
      let userBuffer = "";
      const toolNames = new Map<string, string>();
      // Gemini announces no "response created": generation is simply under way
      // once the model produces something, and over at `turnComplete`.
      let responding = false;
      const startResponding = () => {
        if (responding) return;
        responding = true;
        emit({ type: "response.start" });
      };

      // Gemini rejects any client message sent before `setupComplete` with
      // "invalid argument" (and closes), so mic frames, the seed, and a proactive
      // trigger all wait for the handshake — mirroring Grok's `ready` gate.
      let ready = false;
      const pendingAudio: string[] = [];
      const pendingItems: RealtimeOutbound[] = [];
      let pendingTrigger = false;
      const seedTranscript = renderSeedTranscript(call.seed);

      const sendAudio = (data: string) =>
        session.sendRealtimeInput({ audio: { data, mimeType: `audio/pcm;rate=${sampleRate}` } });
      // Omit `turns` (don't pass `[]`) — an empty array trips the SDK's parser.
      const sendTrigger = () => session.sendClientContent({ turnComplete: true });

      function flushUser() {
        const text = userBuffer.trim();
        userBuffer = "";
        if (text) emit({ type: "transcript.final", role: "user", text });
      }

      function handleMessage(m: GeminiServerMessage) {
        onChunk?.(m); // raw frame, incl. usageMetadata the mapping below drops
        if (m.setupComplete) {
          ready = true;
          // Seed first, as one `user` turn (context only, no generation yet), so
          // it lands before any buffered mic audio or the proactive trigger.
          if (seedTranscript) {
            const text = `${SEED_PREFACE}\n\n<previous_conversation_messages>\n${seedTranscript}\n</previous_conversation_messages>`;
            session.sendClientContent({
              turns: [{ role: "user", parts: [{ text }] }],
              turnComplete: false,
            });
          }
          for (const item of pendingItems) sendItem(item);
          pendingItems.length = 0;
          for (const data of pendingAudio) sendAudio(data);
          pendingAudio.length = 0;
          if (pendingTrigger) {
            pendingTrigger = false;
            sendTrigger();
          }
          return;
        }
        if (m.toolCall?.functionCalls?.length) {
          flushUser();
          startResponding();
          for (const fc of m.toolCall.functionCalls) {
            const name = fc.name ?? "";
            const callId = fc.id ?? name;
            toolNames.set(callId, name);
            emit({ type: "tool.call", callId, name, input: fc.args ?? {} });
          }
        }

        const c = m.serverContent;
        if (!c) return;

        if (c.inputTranscription?.text) {
          userBuffer += c.inputTranscription.text;
          emit({ type: "transcript.delta", role: "user", text: c.inputTranscription.text });
        }
        if (c.outputTranscription?.text) {
          flushUser(); // the model is responding — the user turn is done
          startResponding();
          emit({ type: "transcript.delta", role: "assistant", text: c.outputTranscription.text });
        }
        if (c.modelTurn?.parts) {
          flushUser();
          startResponding();
          for (const p of c.modelTurn.parts) {
            if (p.inlineData?.data) {
              if (!speaking) {
                speaking = true;
                emit({ type: "speech.start" });
              }
              onAudio(b64ToArrayBuffer(p.inlineData.data));
            }
          }
        }
        if (c.interrupted) {
          speaking = false;
          responding = false;
          emit({ type: "speech.interrupted" });
        }
        // Generation finished — the transcript is complete even though buffered
        // audio may still be playing. Settle the text now, before turnComplete.
        if (c.generationComplete) emit({ type: "transcript.done", role: "assistant" });
        if (c.turnComplete) {
          flushUser();
          if (speaking) {
            speaking = false;
            emit({ type: "speech.stop" });
          }
          if (responding) {
            responding = false;
            emit({ type: "response.done" });
          }
        }
      }

      emit({ type: "transport", status: "connecting" });
      const session = await liveConnect({
        model: modelId,
        config: buildConfig(call, voice),
        callbacks: {
          onopen: () => emit({ type: "transport", status: "connected" }),
          onclose: (e) => {
            // Gemini reports rejections (bad config/message) as a close reason —
            // surface it before the disconnect so failures aren't silent.
            if (e?.reason) emit({ type: "error", message: `Gemini closed: ${e.reason}`, fatal: false });
            emit({ type: "transport", status: "disconnected" });
          },
          onerror: (e) =>
            emit({ type: "error", message: e?.message ?? "Gemini connection error", fatal: false }),
          onmessage: (m) => handleMessage(m),
        },
      });

      signal.addEventListener("abort", () => {
        try {
          session.close();
        } catch {
          /* already closed */
        }
      });

      // Seed + trigger are sent on `setupComplete` (see `handleMessage`), since
      // Gemini rejects pre-handshake messages. If setup already landed, fire now.
      if (call.triggerResponse) {
        if (ready) sendTrigger();
        else pendingTrigger = true;
      }

      const sendItem = (item: RealtimeOutbound) => {
        // Same pre-handshake rule as audio/trigger: hold text items until
        // `setupComplete` (e.g. the temporal stamp sent right after connect).
        // Tool results can only exist post-handshake and pass straight through.
        if (!ready && item.type === "text") {
          pendingItems.push(item);
          return;
        }
        if (item.type === "tool.result") {
          const name = toolNames.get(item.callId);
          toolNames.delete(item.callId);
          session.sendToolResponse({
            functionResponses: [
              { id: item.callId, name, response: toGeminiResponse(item.output) },
            ],
          });
        } else if (item.type === "text") {
          session.sendClientContent({
            turns: [{ role: geminiRole(item.role), parts: [{ text: item.text }] }],
            turnComplete: false,
          });
        }
      };

      return {
        send: sendItem,
        pushAudio(pcm) {
          // Gemini resamples internally — we just tag the client's actual rate.
          // Buffer until `setupComplete`, else the frame is rejected.
          const data = Buffer.from(pcm).toString("base64");
          if (ready) sendAudio(data);
          else pendingAudio.push(data);
        },
        requestResponse() {
          if (ready) sendTrigger();
          else pendingTrigger = true;
        },
        async close() {
          try {
            session.close();
          } catch {
            /* already closed */
          }
        },
      };
    },
  };
}
