// Grok (xAI) realtime provider — server-side, over a raw WebSocket using xAI's
// realtime API (OpenAI-realtime compatible). It implements RealtimeProviderV1
// and does nothing but translate the standardized call/events to and from
// xAI's wire format: the session core owns state, transcripts, and tool
// execution. The browser driver's bulk (buffers, tool dispatch, barge-in
// abort) is gone — that all lives in the core now. The caller-facing types
// (models, voices, options) live in ./types.

import type {
  RealtimeCall,
  RealtimeConnectArgs,
  RealtimeHandle,
  RealtimeModelV1,
  RealtimeOutbound,
  RealtimeToolDef,
} from "../../spec";
import { normalizeKeyterms } from "../../keyterms";
import { describeClose } from "../ws";
import type {
  GrokAudioFormat,
  GrokModelOptions,
  GrokReasoningEffort,
  GrokSampleRate,
  GrokVoice,
  GrokVoiceModel,
  GrokWebSocketFactory,
  WSLike,
} from "./types";

export type {
  GrokAudioFormat,
  GrokModelOptions,
  GrokReasoningEffort,
  GrokSampleRate,
  GrokVoice,
  GrokVoiceModel,
  GrokWebSocketFactory,
} from "./types";

const DEFAULT_URL = "wss://api.x.ai/v1/realtime";
const DEFAULT_VOICE: GrokVoice = "eve";
const DEFAULT_SAMPLE_RATE: GrokSampleRate = 24_000;
const DEFAULT_FORMAT: GrokAudioFormat = "audio/pcm";
const WS_OPEN = 1;

// --- pure wire mappers (exported for tests) ---

export function grokToolDef(t: RealtimeToolDef): Record<string, unknown> {
  return {
    type: "function",
    name: t.name,
    description: t.description,
    parameters: t.parametersJsonSchema,
  };
}

export function grokSessionUpdate(
  call: RealtimeCall,
  opts: {
    voice: GrokVoice;
    sampleRate: GrokSampleRate;
    format?: GrokAudioFormat;
    reasoningEffort?: GrokReasoningEffort;
    language?: string;
    keyterms?: readonly string[];
    turnDetection?: Record<string, unknown>;
  },
): Record<string, unknown> {
  const format = opts.format ?? DEFAULT_FORMAT;
  const keyterms = normalizeKeyterms(opts.keyterms);
  return {
    type: "session.update",
    session: {
      voice: opts.voice,
      instructions: call.instructions,
      tools: call.tools.map(grokToolDef),
      ...(opts.reasoningEffort ? { reasoning_effort: opts.reasoningEffort } : {}),
      // `type` alone turns server VAD on (omitting the block disables turn
      // detection entirely); every threshold is left to xAI's defaults.
      turn_detection: opts.turnDetection ?? { type: "server_vad" },
      audio: {
        input: {
          format: { type: format, rate: opts.sampleRate },
          // Presence of `transcription` enables it; an empty object takes xAI's
          // defaults, with optional biases: a BCP-47 language hint and the key
          // terms the ASR should favour when it half-hears a proper noun.
          transcription: {
            ...(opts.language ? { language_hint: opts.language } : {}),
            ...(keyterms ? { keyterms } : {}),
          },
        },
        output: { format: { type: format, rate: opts.sampleRate } },
      },
    },
  };
}

/** Map a core outbound item (seed replay OR a live tool result) to xAI wire
 *  items. One mapping serves both, exactly as the spec intends. */
export function grokOutboundItems(item: RealtimeOutbound): Record<string, unknown>[] {
  switch (item.type) {
    case "text":
      return [
        {
          type: "conversation.item.create",
          item: {
            type: "message",
            role: item.role,
            content: [
              item.role === "assistant"
                ? { type: "text", text: item.text }
                : { type: "input_text", text: item.text },
            ],
          },
        },
      ];
    case "tool.call":
      return [
        {
          type: "conversation.item.create",
          item: {
            type: "function_call",
            call_id: item.callId,
            name: item.name,
            arguments:
              typeof item.input === "string" ? item.input : JSON.stringify(item.input ?? {}),
          },
        },
      ];
    case "tool.result":
      return [
        {
          type: "conversation.item.create",
          item: {
            type: "function_call_output",
            call_id: item.callId,
            // The wire has no error channel — the output string is the only
            // one — so a failed call is rendered as prose the model can read.
            output: item.isError
              ? errorProse(item.output)
              : typeof item.output === "string"
                ? item.output
                : JSON.stringify(item.output ?? ""),
          },
        },
      ];
  }
}

function errorProse(output: unknown): string {
  const message =
    typeof output === "object" && output !== null && "error" in output
      ? String((output as { error: unknown }).error)
      : JSON.stringify(output ?? "");
  return `Error: ${message}`;
}

function b64ToArrayBuffer(b64: string): ArrayBuffer {
  const buf = Buffer.from(b64, "base64");
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function parseArgs(raw: unknown): unknown {
  if (typeof raw !== "string") return raw ?? {};
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

const defaultFactory: GrokWebSocketFactory = (url, options) =>
  new (WebSocket as unknown as { new (u: string, o: unknown): WSLike })(url, options);

/**
 * Create a Grok (xAI) realtime model — the voice analogue of `openai("gpt-4")`.
 * Pass it as `model` to `createRealtimeSession`. Connects server-to-server over
 * a WebSocket using xAI's realtime API (OpenAI-realtime compatible); pure wire
 * translation, the session core owns state and tools.
 *
 * @param modelId - The xAI realtime model id (e.g. `"grok-voice-think-fast-1.0"`).
 * @param options - API key (defaults to `XAI_API_KEY`), voice, endpoint, etc.
 *
 * @example
 * grok("grok-voice-latest")                          // uses process.env.XAI_API_KEY
 * grok("grok-voice-think-fast-1.0", { apiKey, voice: "ara", reasoningEffort: "high" })
 */
export function grok(modelId: GrokVoiceModel, options: GrokModelOptions = {}): RealtimeModelV1 {
  const voice = options.voice ?? DEFAULT_VOICE;
  const sampleRate = options.sampleRate ?? DEFAULT_SAMPLE_RATE;
  const factory = options.createWebSocket ?? defaultFactory;

  return {
    specificationVersion: "realtime-v1",
    provider: "grok",
    modelId,

    async connect({ call, emit, onAudio, onChunk, signal }: RealtimeConnectArgs): Promise<RealtimeHandle> {
      const apiKey = options.apiKey ?? process.env.XAI_API_KEY;
      if (!apiKey) {
        throw new Error(
          "Missing xAI API key: set XAI_API_KEY or pass grok(model, { apiKey }).",
        );
      }
      const url = `${options.url ?? DEFAULT_URL}?model=${encodeURIComponent(modelId)}`;
      const ws = factory(url, {
        headers: { Authorization: `Bearer ${apiKey}` },
      });

      let speaking = false;
      let ready = false;
      const bufferedAudio: string[] = [];
      // A response is in flight between `response.created` and `response.done`.
      // We only `response.cancel` when one actually exists — otherwise xAI
      // replies "Cancellation failed: no active response found".
      let responseActive = false;

      // Continuation bookkeeping: xAI (like OpenAI realtime) needs an explicit
      // `response.create` after tool outputs are returned. We fire it once
      // every emitted tool call has a result back AND the response is done.
      let expectedToolOutputs = 0;
      let receivedToolOutputs = 0;
      let responseDone = false;

      const sendRaw = (payload: Record<string, unknown>) => {
        if (ws.readyState === WS_OPEN) ws.send(JSON.stringify(payload));
      };

      function maybeContinueAfterTools() {
        if (
          expectedToolOutputs > 0 &&
          receivedToolOutputs >= expectedToolOutputs &&
          responseDone &&
          !speaking
        ) {
          expectedToolOutputs = 0;
          receivedToolOutputs = 0;
          responseDone = false;
          sendRaw({ type: "response.create" });
        }
      }

      function handleEvent(event: Record<string, unknown>) {
        switch (event.type) {
          case "session.updated":
            ready = true;
            for (const audio of bufferedAudio) {
              sendRaw({ type: "input_audio_buffer.append", audio });
            }
            bufferedAudio.length = 0;
            return;
          case "input_audio_buffer.speech_started":
            // Barge-in. Cancel only a response that's actually in flight, then
            // let the core abort pending tools off the one `speech.interrupted`.
            if (responseActive) {
              sendRaw({ type: "response.cancel" });
              responseActive = false;
            }
            responseDone = false;
            expectedToolOutputs = 0;
            receivedToolOutputs = 0;
            speaking = false;
            emit({
              type: "speech.interrupted",
              ...(typeof event.item_id === "string" ? { utterance: event.item_id } : {}),
            });
            return;
          case "response.created":
            responseActive = true;
            responseDone = false;
            emit({ type: "response.start" });
            return;
          case "conversation.item.input_audio_transcription.updated": {
            // xAI streams the whole transcript-so-far each time (and revises it),
            // not incremental deltas — forward it as a replacing snapshot, keyed
            // by the conversation item so the core can tell utterances apart.
            const t = event.transcript;
            const itemId = typeof event.item_id === "string" ? event.item_id : undefined;
            if (typeof t === "string") {
              emit({
                type: "transcript.update",
                role: "user",
                text: t,
                ...(itemId ? { utterance: itemId } : {}),
              });
            }
            return;
          }
          case "conversation.item.input_audio_transcription.completed": {
            // xAI emits `completed` repeatedly per utterance: the partial ones
            // carry status "in_progress", the terminal one "completed". Treat the
            // partials as live snapshots and finalize only on the terminal event.
            // xAI may repeat or reattribute the terminal one; the core dedupes
            // and routes by the utterance key, so it passes through untouched.
            const t = typeof event.transcript === "string" ? event.transcript : "";
            const itemId = typeof event.item_id === "string" ? event.item_id : undefined;
            const utterance = itemId ? { utterance: itemId } : {};
            if (event.status === "in_progress") {
              emit({ type: "transcript.update", role: "user", text: t, ...utterance });
              return;
            }
            emit({ type: "transcript.final", role: "user", text: t, ...utterance });
            return;
          }
          case "conversation.item.input_audio_transcription.failed":
            // The ASR gave up on this utterance: no `completed` is coming. Say
            // it is over anyway — the core holds the utterance open from
            // speech-started and anything waiting on it would wait forever.
            // `done` keeps whatever partial text arrived.
            emit({
              type: "transcript.done",
              role: "user",
              ...(typeof event.item_id === "string" ? { utterance: event.item_id } : {}),
            });
            return;
          case "response.output_audio.delta": {
            const delta = event.delta as string | undefined;
            if (delta) {
              if (!speaking) {
                speaking = true;
                emit({ type: "speech.start" });
              }
              onAudio(b64ToArrayBuffer(delta));
            }
            return;
          }
          case "response.output_audio_transcript.delta": {
            const delta = event.delta as string | undefined;
            if (delta) emit({ type: "transcript.delta", role: "assistant", text: delta });
            return;
          }
          case "response.output_audio_transcript.done":
            // Transcript text is complete — settle it now, before `response.done`
            // (which waits on audio), so the text doesn't sit muted through the tail.
            emit({ type: "transcript.done", role: "assistant" });
            return;
          case "response.function_call_arguments.done": {
            expectedToolOutputs += 1;
            const name = typeof event.name === "string" ? event.name : "";
            emit({
              type: "tool.call",
              callId: typeof event.call_id === "string" ? event.call_id : name,
              name,
              input: parseArgs(event.arguments),
            });
            return;
          }
          case "response.done":
            responseActive = false;
            responseDone = true;
            if (speaking) {
              speaking = false;
              emit({ type: "speech.stop" });
            }
            // After `speech.stop`, so the core settles the status from the more
            // specific of the two.
            emit({ type: "response.done" });
            maybeContinueAfterTools();
            return;
          case "error": {
            const err = event.error as { message?: string } | undefined;
            const message = err?.message ?? "Grok realtime error";
            // Benign: a cancel raced an already-finished/absent response.
            if (/no active response|cancellation failed/i.test(message)) return;
            emit({ type: "error", message, fatal: false });
            return;
          }
        }
      }

      ws.addEventListener("message", (ev) => {
        if (typeof ev.data !== "string") return;
        let event: Record<string, unknown>;
        try {
          event = JSON.parse(ev.data);
        } catch {
          return;
        }
        onChunk?.(event); // raw frame, incl. response.done.usage the mapping drops
        handleEvent(event);
      });
      // Closed by us (the core's abort, or `close()` on the handle) is the
      // expected end; anything else dropped under a live session, and the host
      // needs its cause. A socket `error` says nothing of its own and is always
      // followed by this close, so the close is where a failure is reported.
      let closing = false;
      ws.addEventListener("close", (ev) => {
        const unexpected = !closing && !signal.aborted;
        emit({
          type: "transport",
          status: "disconnected",
          ...(unexpected ? { cause: describeClose("xAI Grok", ev) } : {}),
        });
      });
      signal.addEventListener("abort", () => {
        try {
          ws.close();
        } catch {
          /* already closed */
        }
      });

      emit({ type: "transport", status: "connecting" });
      await new Promise<void>((resolve, reject) => {
        const onOpen = () => {
          cleanup();
          resolve();
        };
        const onError = () => {
          cleanup();
          reject(new Error("xAI Grok connection failed"));
        };
        const cleanup = () => {
          ws.removeEventListener("open", onOpen);
          ws.removeEventListener("error", onError);
        };
        ws.addEventListener("open", onOpen);
        ws.addEventListener("error", onError);
      });
      emit({ type: "transport", status: "connected" });

      sendRaw(
        grokSessionUpdate(call, {
          voice,
          sampleRate,
          format: options.format,
          reasoningEffort: options.reasoningEffort,
          language: options.language,
          keyterms: options.keyterms,
          turnDetection: options.turnDetection,
        }),
      );
      for (const item of call.seed) {
        for (const wire of grokOutboundItems(item)) sendRaw(wire);
      }
      if (call.triggerResponse) sendRaw({ type: "response.create" });

      return {
        send(item) {
          for (const wire of grokOutboundItems(item)) sendRaw(wire);
          if (item.type === "tool.result") {
            receivedToolOutputs += 1;
            maybeContinueAfterTools();
          }
        },
        pushAudio(pcm) {
          const audio = Buffer.from(pcm).toString("base64");
          if (ready) sendRaw({ type: "input_audio_buffer.append", audio });
          else bufferedAudio.push(audio);
        },
        requestResponse() {
          sendRaw({ type: "response.create" });
        },
        async close() {
          closing = true;
          try {
            ws.close();
          } catch {
            /* already closed */
          }
        },
      };
    },
  };
}
