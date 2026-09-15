// The Promise face of this package — NOT part of the upstream vendor. Effect
// is an implementation detail of the interpreter; consumers (the AI harness,
// apps, plain Node/Bun code) integrate through these wrappers and never see
// an Effect type. Keep every other file diffable against upstream; additions
// belong here.

import { Effect } from "effect"
import * as CodeMode from "./codemode"
import * as Tool from "./tool"

/** A host tool defined with a plain async execute and render-only JSON
 *  Schemas. A thrown `ToolError` (via `toolError`) is a safe, model-visible
 *  `ToolFailure`; any other thrown value is sanitized. */
export const promiseTool = (options: {
  readonly description: string
  readonly input: Tool.JsonSchema
  readonly output?: Tool.JsonSchema
  readonly execute: (input: unknown) => Promise<unknown>
}): Tool.Tool =>
  Tool.make({
    description: options.description,
    input: options.input,
    ...(options.output === undefined ? {} : { output: options.output }),
    execute: (input) => Effect.tryPromise({ try: () => options.execute(input), catch: (error) => error }),
  })

export type PromiseRuntimeOptions = {
  /** Tools exposed to the program as `tools` — `promiseTool` results and/or
   *  nested namespaces of them. */
  readonly tools?: CodeMode.Options["tools"]
  readonly limits?: CodeMode.ExecutionLimits
  readonly onToolCallStart?: (call: CodeMode.ToolCallStarted) => void | Promise<void>
  readonly onToolCallEnd?: (call: CodeMode.ToolCallEnded) => void | Promise<void>
}

/** A reusable confined runtime with a Promise surface. `execute` never
 *  rejects for program failures — they come back as `Result` diagnostics. */
export type PromiseRuntime = {
  readonly catalog: () => ReadonlyArray<CodeMode.ToolDescription>
  readonly execute: (code: string) => Promise<CodeMode.Result>
}

export const makeRuntime = (options: PromiseRuntimeOptions = {}): PromiseRuntime => {
  const runtime = CodeMode.make({
    tools: options.tools,
    limits: options.limits,
    ...(options.onToolCallStart === undefined
      ? {}
      : {
          onToolCallStart: (call: CodeMode.ToolCallStarted) =>
            Effect.promise(async () => {
              await options.onToolCallStart!(call)
            }),
        }),
    ...(options.onToolCallEnd === undefined
      ? {}
      : {
          onToolCallEnd: (call: CodeMode.ToolCallEnded) =>
            Effect.promise(async () => {
              await options.onToolCallEnd!(call)
            }),
        }),
  })
  return {
    catalog: runtime.catalog,
    execute: (code) => Effect.runPromise(runtime.execute(code)),
  }
}
