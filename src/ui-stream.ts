// The bridge from a turn to a browser. Every route that streams a turn to a
// client needs the same four things wired together, and each one is something
// the harness already knows and the caller would otherwise reassemble from
// outside: the response copy of the teed stream, the id the assistant message
// is persisted under (so a client-held message and its stored row agree, which
// is what per-message feedback keys on), the turn's `createdAt` (server clock,
// the same domain as the voice relay's turns, so text and voice order in one
// transcript) and its timings on the finish part.
//
// Defaults are the union of what hand-written route handlers got right
// separately — one had timings and no id alignment, another the reverse — and
// every one of them is overridable, because `sendReasoning: false` for a public
// widget or a CORS header is genuinely per-route.

import {
  createUIMessageStreamResponse,
  toUIMessageStream,
  type ToolSet,
  type UIMessage,
} from "ai";
import type { TurnTimings } from "./timings";

type StreamArgs<
  TTools extends ToolSet,
  UI_MESSAGE extends UIMessage,
> = Parameters<typeof toUIMessageStream<TTools, UI_MESSAGE>>[0];

/** Everything `toUIMessageStream` takes except the stream itself, which the
 *  turn supplies. */
export type UIMessageStreamOptions<
  TTools extends ToolSet,
  UI_MESSAGE extends UIMessage = UIMessage,
> = Omit<StreamArgs<TTools, UI_MESSAGE>, "stream">;

/** `createUIMessageStreamResponse`'s init, plus the stream options. */
export type UIMessageStreamResponseOptions<
  TTools extends ToolSet,
  UI_MESSAGE extends UIMessage = UIMessage,
> = UIMessageStreamOptions<TTools, UI_MESSAGE> & {
  status?: number;
  statusText?: string;
  headers?: HeadersInit;
};

export interface UIStreamSource<TTools extends ToolSet> {
  stream: StreamArgs<TTools, UIMessage>["stream"];
  /** The id the turn's assistant message is persisted under. */
  responseMessageId: string;
  timings: () => TurnTimings;
  now: () => number;
}

/** The turn's UI-message stream, with the harness's defaults applied. */
export function turnToUIMessageStream<
  TTools extends ToolSet,
  UI_MESSAGE extends UIMessage = UIMessage,
>(
  source: UIStreamSource<TTools>,
  options?: UIMessageStreamOptions<TTools, UI_MESSAGE>,
) {
  const createdAt = source.now();
  return toUIMessageStream<TTools, UI_MESSAGE>({
    stream: source.stream,
    // Persistence mode: without a non-null `originalMessages` the SDK omits
    // the id from the start chunk, and the alignment below silently does
    // nothing. Empty is right here — this copy carries only the new turn.
    originalMessages: [],
    generateMessageId: () => source.responseMessageId,
    // `createdAt` on every part so a client can order the message as soon as
    // it starts; the timings are only final once the turn is.
    messageMetadata: ({ part }) =>
      part.type === "finish"
        ? { createdAt, timings: source.timings() }
        : { createdAt },
    ...options,
  } as StreamArgs<TTools, UI_MESSAGE>);
}

/** The turn as an HTTP response — the one-liner a route handler wants. */
export function turnToUIMessageStreamResponse<
  TTools extends ToolSet,
  UI_MESSAGE extends UIMessage = UIMessage,
>(
  source: UIStreamSource<TTools>,
  options?: UIMessageStreamResponseOptions<TTools, UI_MESSAGE>,
): Response {
  const { status, statusText, headers, ...streamOptions } = options ?? {};
  return createUIMessageStreamResponse({
    stream: turnToUIMessageStream<TTools, UI_MESSAGE>(
      source,
      streamOptions as UIMessageStreamOptions<TTools, UI_MESSAGE>,
    ),
    ...(status === undefined ? {} : { status }),
    ...(statusText === undefined ? {} : { statusText }),
    ...(headers === undefined ? {} : { headers }),
  });
}
