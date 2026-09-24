// Vercel AI Gateway realtime provider — server-side, over a raw WebSocket
// speaking the gateway's normalized realtime dialect. Unlike the direct
// providers, the wire is NOT the underlying model's format: the gateway
// normalizes provider events server-side and the socket carries the AI SDK's
// normalized client/server events (subprotocol `ai-gateway-realtime.v1`),
// with each server event keeping the provider's original payload in `raw`.
// It implements RealtimeModelV1 and does nothing but translate the
// standardized call/events to and from that dialect: the session core owns
// state, transcripts, and tool execution.
//
// Wire facts this module encodes (verified live against the gateway):
//   - auth is the gateway API key in a subprotocol; no ephemeral secret.
//   - after a tool output the model does NOT continue on its own — the same
//     explicit response-create nudge the direct Grok provider sends.
//   - transcript events carry the conversation item id (`itemId`), the
//     utterance key the core aligns turns by.
//   - the gateway forwards the ASR's live snapshots as `completed` events
//     whose raw status is "in_progress" (it once consumed them server-side;
//     forwarding verified live) — only a raw status of "completed" is
//     terminal. The terminal transcript can itself arrive MORE THAN ONCE for
//     the same item: the ASR emits a second `completed` carrying the
//     corrected full text when it revises what it heard, and an utterance
//     resuming after a pause keeps growing the SAME item when barge-in
//     cancelled the pending answer before any audio was produced. The core
//     folds both back into the turn it already settled.
//   - seed replay: text turns pass for both roles; a `function-call` item is
//     silently dropped (not in the normalized item union), while its
//     `function-call-output` lands and carries the context.

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
  GatewayModelOptions,
  GatewayVoiceModel,
  GatewayWebSocketFactory,
  WSLike,
} from "./types";

export type {
  GatewayModelOptions,
  GatewayVoiceModel,
  GatewayWebSocketFactory,
} from "./types";

const DEFAULT_URL = "wss://ai-gateway.vercel.sh/v4/ai/realtime-model";
const REALTIME_SUBPROTOCOL = "ai-gateway-realtime.v1";
const AUTH_SUBPROTOCOL_PREFIX = "ai-gateway-auth.";
const DEFAULT_SAMPLE_RATE = 24_000;
const WS_OPEN = 1;

/** The gateway passes `voice` through to the underlying provider, so the
 *  sensible default depends on the model's creator prefix. */
export function gatewayDefaultVoice(modelId: string): string | undefined {
  if (modelId.startsWith("xai/")) return "eve";
  if (modelId.startsWith("openai/")) return "marin";
  return undefined;
}

// --- pure wire mappers (exported for tests) ---

export function gatewayToolDef(t: RealtimeToolDef): Record<string, unknown> {
  return {
    type: "function",
    name: t.name,
    description: t.description,
    parameters: t.parametersJsonSchema,
  };
}

/** Fold key terms into the provider-native session fields. The normalized
 *  dialect has no place for them, and unknown fields added to
 *  `inputAudioTranscription` are dropped before they reach the provider (card
 *  #449), so they travel in xAI's native shape through the bare merge. */
export function withKeyterms(
  providerOptions: Record<string, unknown> | undefined,
  terms: readonly string[] | undefined,
): Record<string, unknown> | undefined {
  const keyterms = normalizeKeyterms(terms);
  if (!keyterms) return providerOptions;
  const audio = (providerOptions?.audio ?? {}) as Record<string, unknown>;
  const input = (audio.input ?? {}) as Record<string, unknown>;
  const transcription = (input.transcription ?? {}) as Record<string, unknown>;
  return {
    ...providerOptions,
    audio: {
      ...audio,
      input: { ...input, transcription: { ...transcription, keyterms } },
    },
  };
}

export function gatewaySessionUpdate(
  call: RealtimeCall,
  opts: {
    voice?: string;
    sampleRate: number;
    language?: string;
    keyterms?: readonly string[];
    turnDetection?: Record<string, unknown>;
    providerOptions?: Record<string, unknown>;
  },
): Record<string, unknown> {
  const providerOptions = withKeyterms(opts.providerOptions, opts.keyterms);
  return {
    type: "session-update",
    config: {
      instructions: call.instructions,
      ...(opts.voice ? { voice: opts.voice } : {}),
      outputModalities: ["audio"],
      inputAudioFormat: { type: "audio/pcm", rate: opts.sampleRate },
      outputAudioFormat: { type: "audio/pcm", rate: opts.sampleRate },
      // Presence enables transcription, like the native configs it maps to.
      inputAudioTranscription: opts.language ? { language: opts.language } : {},
      outputAudioTranscription: {},
      // `type` alone turns server VAD on (omitting the block disables turn
      // detection entirely); every threshold is left to the provider's defaults.
      turnDetection: opts.turnDetection ?? { type: "server-vad" },
      tools: call.tools.map(gatewayToolDef),
      ...(providerOptions ? { providerOptions } : {}),
    },
  };
}

/** Map a core outbound item (seed replay OR a live tool result) to normalized
 *  client events. A seed `tool.call` maps to nothing: the normalized item
 *  union has no function-call item and the gateway drops one silently, while
 *  the paired `tool.result` lands as a tool-role item the model attributes. */
export function gatewayOutboundItems(item: RealtimeOutbound): Record<string, unknown>[] {
  switch (item.type) {
    case "text":
      return [
        {
          type: "conversation-item-create",
          item: { type: "text-message", role: item.role, text: item.text },
        },
      ];
    case "tool.call":
      return [];
    case "tool.result":
      return [
        {
          type: "conversation-item-create",
          item: {
            type: "function-call-output",
            callId: item.callId,
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

const defaultFactory: GatewayWebSocketFactory = (url, protocols) =>
  new (WebSocket as unknown as { new (u: string, p: string[]): WSLike })(url, protocols);

/**
 * Create a Vercel AI Gateway realtime model — the voice analogue of resolving
 * a text model through the gateway. Pass it as `model` to
 * `createRealtimeSession`. Connects server-to-server over a WebSocket speaking
 * the gateway's normalized realtime dialect; pure wire translation, the
 * session core owns state and tools.
 *
 * @param modelId - The `creator/model` gateway id (e.g. `"xai/grok-voice-think-fast-2.0"`).
 * @param options - API key (defaults to `AI_GATEWAY_API_KEY`), voice, endpoint, etc.
 *
 * @example
 * gateway("xai/grok-voice-think-fast-2.0")   // uses process.env.AI_GATEWAY_API_KEY
 * gateway("openai/gpt-realtime-2", { apiKey, voice: "alloy" })
 */
export function gateway(
  modelId: GatewayVoiceModel,
  options: GatewayModelOptions = {},
): RealtimeModelV1 {
  const voice = options.voice ?? gatewayDefaultVoice(modelId);
  const sampleRate = options.sampleRate ?? DEFAULT_SAMPLE_RATE;
  const factory = options.createWebSocket ?? defaultFactory;

  return {
    specificationVersion: "realtime-v1",
    provider: "gateway",
    modelId,

    async connect({ call, emit, onAudio, onChunk, signal }: RealtimeConnectArgs): Promise<RealtimeHandle> {
      const apiKey = options.apiKey ?? process.env.AI_GATEWAY_API_KEY;
      if (!apiKey) {
        throw new Error(
          "Missing gateway API key: set AI_GATEWAY_API_KEY or pass gateway(model, { apiKey }).",
        );
      }
      const url = `${options.url ?? DEFAULT_URL}?ai-model-id=${encodeURIComponent(modelId)}`;
      const ws = factory(url, [REALTIME_SUBPROTOCOL, `${AUTH_SUBPROTOCOL_PREFIX}${apiKey}`]);

      let speaking = false;
      let ready = false;
      const bufferedAudio: string[] = [];
      // A response is in flight between `response-created` and `response-done`.
      // We only cancel when one actually exists — otherwise the provider
      // replies "Cancellation failed: no active response found".
      let responseActive = false;

      // Continuation bookkeeping: through the gateway, exactly as against xAI
      // directly, the model needs an explicit response after tool outputs are
      // returned. We fire it once every emitted tool call has a result back
      // AND the response is done.
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
          sendRaw({ type: "response-create" });
        }
      }

      function handleEvent(event: Record<string, unknown>) {
        const itemId = typeof event.itemId === "string" ? event.itemId : undefined;
        switch (event.type) {
          case "session-updated":
            ready = true;
            for (const audio of bufferedAudio) {
              sendRaw({ type: "input-audio-append", audio });
            }
            bufferedAudio.length = 0;
            return;
          case "speech-started":
            // Barge-in. Cancel only a response that's actually in flight, then
            // let the core abort pending tools off the one `speech.interrupted`.
            if (responseActive) {
              sendRaw({ type: "response-cancel" });
              responseActive = false;
            }
            responseDone = false;
            expectedToolOutputs = 0;
            receivedToolOutputs = 0;
            speaking = false;
            emit({ type: "speech.interrupted", ...(itemId ? { utterance: itemId } : {}) });
            return;
          case "response-created":
            responseActive = true;
            responseDone = false;
            emit({ type: "response.start" });
            return;
          case "input-transcription-completed": {
            // The gateway forwards the ASR's live snapshots under this event
            // type too — xAI marks them "in_progress", and only the terminal
            // one "completed". A snapshot must not finalize the turn.
            const t = typeof event.transcript === "string" ? event.transcript : "";
            const utterance = itemId ? { utterance: itemId } : {};
            const raw = event.raw as { status?: unknown } | undefined;
            if (raw?.status === "in_progress") {
              emit({ type: "transcript.update", role: "user", text: t, ...utterance });
              return;
            }
            emit({ type: "transcript.final", role: "user", text: t, ...utterance });
            return;
          }
          case "custom":
            // The gateway's own vocabulary has a completion for a transcription
            // and no failure, so a transcription that fails arrives here, under
            // the provider's native name. It matters because no completion
            // follows it: the core holds the utterance open from speech-started
            // and anything waiting on it would wait for the rest of the
            // call. `done` keeps whatever partial text arrived.
            if (event.rawType === "conversation.item.input_audio_transcription.failed") {
              const failed = (event.raw as { item_id?: unknown } | undefined)?.item_id;
              emit({
                type: "transcript.done",
                role: "user",
                ...(typeof failed === "string" ? { utterance: failed } : {}),
              });
            }
            return;
          case "audio-delta": {
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
          case "audio-transcript-delta": {
            const delta = event.delta as string | undefined;
            if (delta) emit({ type: "transcript.delta", role: "assistant", text: delta });
            return;
          }
          case "audio-transcript-done":
            // Transcript text is complete — settle it now, before `response-done`
            // (which waits on audio), so the text doesn't sit muted through the tail.
            emit({ type: "transcript.done", role: "assistant" });
            return;
          case "function-call-arguments-done": {
            expectedToolOutputs += 1;
            const name = typeof event.name === "string" ? event.name : "";
            emit({
              type: "tool.call",
              callId: typeof event.callId === "string" ? event.callId : name,
              name,
              input: parseArgs(event.arguments),
            });
            return;
          }
          case "response-done":
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
            const message =
              typeof event.message === "string" ? event.message : "gateway realtime error";
            // Benign: a cancel raced an already-finished/absent response.
            if (/no active response|cancellation failed/i.test(message)) return;
            emit({ type: "error", message, fatal: false });
            return;
          }
          // Everything else — `custom` passthroughs (pings), lifecycle chatter —
          // maps to nothing; `onChunk` already saw the raw frame.
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
        onChunk?.(event); // raw frame, incl. the provider payload under `raw`
        handleEvent(event);
      });
      ws.addEventListener("error", () =>
        emit({ type: "error", message: "AI Gateway connection failed", fatal: false }),
      );
      // Closed by us (the core's abort, or `close()` on the handle) is the
      // expected end; anything else dropped under a live session, and the host
      // needs its cause.
      let closing = false;
      ws.addEventListener("close", (ev) => {
        if (!closing && !signal.aborted) {
          emit({ type: "error", message: describeClose("AI Gateway", ev), fatal: false });
        }
        emit({ type: "transport", status: "disconnected" });
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
          reject(new Error("AI Gateway connection failed"));
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
        gatewaySessionUpdate(call, {
          voice,
          sampleRate,
          language: options.language,
          keyterms: options.keyterms,
          turnDetection: options.turnDetection,
          providerOptions: options.providerOptions,
        }),
      );
      for (const item of call.seed) {
        for (const wire of gatewayOutboundItems(item)) sendRaw(wire);
      }
      if (call.triggerResponse) sendRaw({ type: "response-create" });

      return {
        send(item) {
          for (const wire of gatewayOutboundItems(item)) sendRaw(wire);
          if (item.type === "tool.result") {
            receivedToolOutputs += 1;
            maybeContinueAfterTools();
          }
        },
        pushAudio(pcm) {
          const audio = Buffer.from(pcm).toString("base64");
          if (ready) sendRaw({ type: "input-audio-append", audio });
          else bufferedAudio.push(audio);
        },
        requestResponse() {
          sendRaw({ type: "response-create" });
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
