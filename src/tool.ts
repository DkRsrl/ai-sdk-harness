// `harnessTool` is `tool()` with the running session injected into `execute`.
// A tool defined this way receives `{ ...options, session }` as its second
// argument, so it can persist artifacts under `session.id`, spawn a
// `subsession`, or inspect `session.messages` — without the harness threading
// the session through the tool's input schema. The session is resolved at call
// time via `currentSession()` (AsyncLocalStorage), so the same tool object works
// across concurrent sessions.

import { tool, type ToolExecutionOptions } from "ai";
import type {
  Context,
  FlexibleSchema,
  ToolResultOutput,
} from "@ai-sdk/provider-utils";
import { currentSession } from "./session-context";
import type { Session } from "./types";

export type HarnessToolExecuteFunction<INPUT, OUTPUT, CONTEXT extends Context> = (
  input: INPUT,
  options: ToolExecutionOptions<CONTEXT> & { session: Session },
) => AsyncIterable<OUTPUT> | PromiseLike<OUTPUT> | OUTPUT;

// Mirrors the AI SDK's `toModelOutput`, but declared here instead of picked from
// `Parameters<typeof tool>`. Through that indirection the `NoInfer<OUTPUT>`
// linkage is lost, so `output` collapses to `unknown` (and `OUTPUT` stops being
// inferred from `execute`). Declaring it directly keeps `execute`'s return the
// sole inference site for OUTPUT, so `output` follows it — matching `tool()`.
// The `0 extends 1 & OUTPUT` / `[OUTPUT] extends [never]` guards mirror the SDK:
// they widen `output` to `any` when OUTPUT is `any` or `never`.
type HarnessToModelOutput<INPUT, OUTPUT> = (options: {
  toolCallId: string;
  input: [INPUT] extends [never] ? unknown : INPUT;
  output: 0 extends 1 & OUTPUT
    ? any
    : [OUTPUT] extends [never]
      ? any
      : NoInfer<OUTPUT>;
}) => ToolResultOutput | PromiseLike<ToolResultOutput>;

type ToolBaseFields<INPUT, OUTPUT, CONTEXT extends Context> = Pick<
  Extract<
    Parameters<typeof tool<INPUT, OUTPUT, CONTEXT>>[0],
    { execute: unknown }
  >,
  | "title"
  | "description"
  | "inputSchema"
  | "outputSchema"
  | "providerOptions"
  | "metadata"
  | "strict"
  | "inputExamples"
  | "onInputStart"
  | "onInputDelta"
  | "onInputAvailable"
  | "needsApproval"
>;

export interface HarnessToolInput<
  INPUT,
  OUTPUT,
  CONTEXT extends Context = Context,
> extends ToolBaseFields<INPUT, OUTPUT, CONTEXT> {
  inputSchema: FlexibleSchema<INPUT>;
  contextSchema?: FlexibleSchema<CONTEXT>;
  execute: HarnessToolExecuteFunction<INPUT, OUTPUT, CONTEXT>;
  toModelOutput?: HarnessToModelOutput<INPUT, OUTPUT>;
}

export function harnessTool<INPUT, OUTPUT, CONTEXT extends Context = Context>(
  def: HarnessToolInput<INPUT, OUTPUT, CONTEXT>,
): ReturnType<typeof tool<INPUT, OUTPUT, CONTEXT>> {
  const { execute, ...rest } = def;
  return tool({
    ...rest,
    execute: (input: INPUT, opts: ToolExecutionOptions<CONTEXT>) =>
      execute(input, { ...opts, session: currentSession() }),
  } as unknown as Parameters<typeof tool<INPUT, OUTPUT, CONTEXT>>[0]);
}
