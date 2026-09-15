// Convert a stored UIMessage transcript into a realtime seed — the voice
// analogue of the AI SDK's `convertToModelMessages`. Conversations are persisted
// in the AI SDK's `UIMessage` shape (the same one `useVoiceChat` exposes and the
// text chat stores), so resuming one is: load the messages, convert, and pass
// the result as `createRealtimeSession`'s `seed`. Text parts become text turns;
// a completed tool round-trip becomes the `tool.call` paired with its
// `tool.result`, so a resumed session keeps full fidelity — providers without
// native history (Gemini) rebuild these as turns.
//
// Async for the same reason `convertToModelMessages` is: a tool's optional
// `toModelOutput` (which shapes how the model sees a result) may return a
// promise, and — like the AI SDK — we `await` it. Pass the SAME `tools` you pass
// to the session so a result is seeded exactly as the model originally saw it.

import {
  getToolName,
  isTextUIPart,
  isToolUIPart,
  type ToolSet,
  type UIMessage,
} from "ai";
import type { ToolResultOutput } from "@ai-sdk/provider-utils";
import type { RealtimeOutbound, Role } from "./spec";
import type { RealtimeMessage } from "./session";

/** Flatten a `toModelOutput` result to the plain value the realtime providers
 *  serialize — keeping the seed's `output` shape identical to the no-`toModelOutput`
 *  case (every variant but `execution-denied` carries the model-facing value). */
function modelOutputValue(out: ToolResultOutput): unknown {
  return out.type === "execution-denied"
    ? { error: "execution denied", reason: out.reason }
    : out.value;
}

/**
 * Convert UIMessages (the stored/rendered transcript) into a `RealtimeOutbound[]`
 * seed for `createRealtimeSession`. The voice analogue of `convertToModelMessages`.
 *
 * - Text parts → `{ type: "text", role, text }` (empty text and non-user/assistant
 *   roles like `system` are dropped — the system prompt is `instructions`, not seed).
 *   Consecutive text parts of one message join into a single seed turn, so a
 *   decorated turn (e.g. a `<message_metadata>` part prepended to the user's
 *   text) replays as the one utterance it was, not as two.
 * - A settled tool part → `{ type: "tool.call" }` + `{ type: "tool.result" }`. If the
 *   matching tool in `options.tools` defines `toModelOutput`, the result is mapped
 *   through it (and awaited) so the model sees it exactly as it did live; otherwise
 *   the stored output is used as-is. An `output-error` is carried through as
 *   `{ error }`. Tool parts still mid-flight (no settled outcome) are skipped, so a
 *   resume never replays a dangling call.
 */
export async function convertToRealtimeSeed(
  messages: UIMessage[],
  options?: { tools?: ToolSet },
): Promise<RealtimeOutbound[]> {
  const tools = options?.tools;
  const seed: RealtimeOutbound[] = [];
  for (const message of messages) {
    const role: Role | null =
      message.role === "user" ? "user" : message.role === "assistant" ? "assistant" : null;
    let pendingText: string[] = [];
    const flushText = () => {
      if (role && pendingText.length)
        seed.push({ type: "text", role, text: pendingText.join("\n") });
      pendingText = [];
    };
    for (const part of message.parts) {
      if (isTextUIPart(part)) {
        const text = part.text.trim();
        if (text) pendingText.push(text);
        continue;
      }
      if (!isToolUIPart(part)) continue;
      flushText();
      if (part.state === "output-available") {
        const name = getToolName(part);
        seed.push({ type: "tool.call", callId: part.toolCallId, name, input: part.input });
        const toModelOutput = tools?.[name]?.toModelOutput;
        const output = toModelOutput
          ? modelOutputValue(
              await toModelOutput({
                toolCallId: part.toolCallId,
                input: part.input,
                output: part.output,
              }),
            )
          : part.output;
        seed.push({ type: "tool.result", callId: part.toolCallId, output });
      } else if (part.state === "output-error") {
        // Errors short-circuit `toModelOutput` (as in the AI SDK) — surface the text.
        seed.push({
          type: "tool.call",
          callId: part.toolCallId,
          name: getToolName(part),
          input: part.input,
        });
        seed.push({
          type: "tool.result",
          callId: part.toolCallId,
          output: { error: part.errorText },
          isError: true,
        });
      }
    }
    flushText();
  }
  return seed;
}

/**
 * Convert finalized `RealtimeMessage`s (what the session emits via `onMessage`)
 * into the stored/rendered `UIMessage` shape — the inverse of
 * {@link convertToRealtimeSeed}. Shared by `VoiceChat` (live render) and the
 * relay (persistence) so a voice turn renders and persists identically, and in
 * the same shape the text chat stores. The turn's `createdAt` rides in
 * `metadata` as the cross-mode ordering key.
 *
 * - A `text` message → one settled `text` part (`state: "done"`).
 * - A `tool` round-trip → one `dynamic-tool` part (`state: "output-available"`),
 *   the same part a resumed/persisted tool turn carries.
 */
export function toUIMessages(messages: RealtimeMessage[]): UIMessage[] {
  return messages.map((m) => {
    // `timings` rides along with `createdAt` so voice turns persist their
    // latency profile in the same metadata seam text turns use.
    const metadata = {
      createdAt: m.createdAt,
      ...(m.timings ? { timings: m.timings } : {}),
    };
    if (m.type === "text") {
      return {
        id: m.id,
        role: m.role,
        parts: [{ type: "text", text: m.text, state: "done" }],
        metadata,
      } as UIMessage;
    }
    return {
      id: m.id,
      role: "assistant",
      parts: [
        {
          type: "dynamic-tool",
          toolName: m.name,
          toolCallId: m.callId,
          state: "output-available",
          input: m.input,
          output: m.output,
        },
      ],
      metadata,
    } as UIMessage;
  });
}
