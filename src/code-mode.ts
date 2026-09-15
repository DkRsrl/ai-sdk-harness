// Code mode: instead of one direct call per tool, the model writes a program
// and the confined interpreter in ./codemode (vendored opencode
// runtime — pure TypeScript, no eval/VM/workers, runs identically under Node
// and Bun) executes it; the program's only door to the world is the routed
// tools. The *role* owns the routing, in the AI SDK's `experimental_toolCallers`
// vocabulary — but the harness implements it itself, with no experimental SDK
// APIs: `role({ tools, toolCallers })` keeps one `tools` list (which must
// include `codeMode` — the sandbox is a registry tool like any other, wired
// as `createRegistry({ ..., codeMode: codeModeTool })`) and maps a subset of
// those names to their callers. `["code"]` moves a tool into the sandbox
// (the harness drops its direct call from the model-visible set itself),
// adding `DIRECT_TOOL_CALL` keeps both ways open, no entry means a normal
// direct call. Skills merge their own map in when invoked. At each turn the
// text drive swaps the registry's `codeMode` entry for a session-bound tool
// built here: the tool's description is invariant, the routed tools' catalog
// renders into the session's instructions (opencode style — see
// code-mode-catalog.ts), and execute runs the interpreter. The voice drive
// does the same, but binds once at connect — realtime providers advertise
// tools once, and the sandbox runs in this process, so the provider calls it
// like any other tool.

import {
  asSchema,
  jsonSchema,
  tool,
  type Tool as AiTool,
  type ToolSet,
} from "ai"
import { z } from "zod"
import {
  makeRuntime,
  promiseTool,
  toolError,
  type CodeMode,
  type Tool as SandboxTool,
} from "./codemode"
import { renderCatalogInstructions, type CatalogEntry } from "./code-mode-catalog"

/** The registry key `codeModeTool` must be wired under — the caller name that
 *  role/skill `toolCallers` maps reference. */
export const CODE_MODE_TOOL_NAME = "code"

/** The AI SDK's marker for "the model may call this tool directly" — same
 *  value the SDK uses, kept for vocabulary compatibility. */
export const DIRECT_TOOL_CALL = "AI_SDK_DIRECT_TOOL_CALL"

export type ToolCaller = typeof CODE_MODE_TOOL_NAME | typeof DIRECT_TOOL_CALL

/** A role/skill's routing declaration: who may invoke each of its tools.
 *  Keys must come from the declaring role/skill's own `tools` list. */
export type ToolCallers<TTool extends string = string> = {
  readonly [K in TTool]?: readonly ToolCaller[]
}

export function routesToCodeMode(callers: readonly ToolCaller[] | undefined): boolean {
  return callers?.includes(CODE_MODE_TOOL_NAME) ?? false
}

/** The namespace a tool declares for itself, riding in the AI SDK's own
 *  tool `metadata` (which is never sent to the model): a tool built with
 *  `tool({ metadata: { namespace: "erp" }, ... })` — or `harnessTool`
 *  likewise — is called as `tools.erp.<name>(input)` inside programs and
 *  grouped under `erp` in the sandbox catalog (headers, counts, `search`
 *  namespace scoping). Namespaces may nest with dots (`erp.orders`). The
 *  namespace is the tool's identity, not the role's routing: it applies in
 *  every role that sandboxes the tool, and is inert for direct calls. */
export function toolNamespace(registryTool: {
  metadata?: unknown
}): string | undefined {
  const metadata = registryTool.metadata
  if (typeof metadata !== "object" || metadata === null) return undefined
  const namespace = (metadata as Record<string, unknown>)["namespace"]
  return typeof namespace === "string" ? namespace : undefined
}

/** Whether a tool asks to be pinned in the inline catalog: a tool built with
 *  `tool({ metadata: { pinned: true }, ... })` always keeps its full listing
 *  in the rendered instructions, charged to the budget before anything else.
 *  Like the namespace, it is the tool's identity, not the role's routing. */
export function toolPinned(registryTool: { metadata?: unknown }): boolean {
  const metadata = registryTool.metadata
  if (typeof metadata !== "object" || metadata === null) return false
  return (metadata as Record<string, unknown>)["pinned"] === true
}

/** Fail loudly on a namespace the sandbox trie would reject at bind time. */
export function assertValidNamespace(name: string, registryTool: { metadata?: unknown }): void {
  const namespace = toolNamespace(registryTool)
  if (namespace === undefined) return
  if (namespace === "" || namespace.split(".").some((segment) => segment.trim() === "")) {
    throw new Error(
      `tool "${name}": metadata.namespace "${namespace}" has an empty segment`,
    )
  }
}

/** The names a merged map routes into the sandbox. */
export function codeModeToolNames(toolCallers: ToolCallers): string[] {
  return Object.entries(toolCallers)
    .filter(([, callers]) => routesToCodeMode(callers))
    .map(([name]) => name)
}

/** The subset routed into the sandbox with no direct call left — the harness
 *  removes these from the model-visible active set each step. */
export function codeModeOnlyNames(toolCallers: ToolCallers): string[] {
  return Object.entries(toolCallers)
    .filter(([, callers]) => routesToCodeMode(callers) && !callers?.includes(DIRECT_TOOL_CALL))
    .map(([name]) => name)
}

/** Fail loudly on a caller the harness doesn't support. */
export function assertKnownCallers(label: string, toolCallers: ToolCallers): void {
  for (const [name, callers] of Object.entries(toolCallers)) {
    const unknown = (callers ?? []).filter(
      (caller) => caller !== CODE_MODE_TOOL_NAME && caller !== DIRECT_TOOL_CALL,
    )
    if (unknown.length > 0) {
      throw new Error(
        `${label}: toolCallers("${name}") names unknown caller(s) ${unknown.join(", ")} — supported: "${CODE_MODE_TOOL_NAME}", DIRECT_TOOL_CALL`,
      )
    }
  }
}

const INPUT_SCHEMA = z.object({
  description: z
    .string()
    .optional()
    .describe(
      "One short sentence, in the conversation's language, shown to the user while the program runs.",
    ),
  code: z.string().describe("The program."),
})

type CodeModeInput = z.infer<typeof INPUT_SCHEMA>

// Invariant model-facing guidance — upstream opencode's description verbatim
// (core/src/codemode/tool.ts @ 4d22d4e); the changing tool catalog is
// rendered into the session's instructions instead. Every claim is true of
// this runtime: pending promise work is fiber-interrupted at program end
// (interpreter/promises.ts) and bracket paths resolve (tool-paths.test.ts).
// The `search`-is-a-program-global clarification lives in the catalog's
// Search section (code-mode-catalog.ts), not here.
const BASE_DESCRIPTION = [
  "Run JavaScript in a confined Code Mode runtime to orchestrate tool calls and compose their results.",
  "Imports, direct filesystem access, and timers are unavailable. Do not use `fetch`; all external access goes through `tools`.",
  "Within `{ code }`, the only callable tools are those explicitly listed in the Code Mode catalog instructions or returned by `search`. Inside `{ code }`, ignore tools shown outside the Code Mode catalog. They are not available in the Code Mode runtime.",
  'Call tools through `tools` using only exact paths and signatures from the catalog. Do not infer or normalize tool names; preserve bracket notation such as `tools.<namespace>["tool-name"](input)`.',
  "Prefer an explicit `return`; if omitted, the final top-level expression becomes the result.",
  "Await every call whose completion matters; pending calls are interrupted when execution ends. Run independent calls concurrently with `Promise.all`.",
].join("\n")

/** The interpreter's outcome, as the AI SDK output type of the tool — a
 *  permissive schema (failures are data the model reads, never validation
 *  errors) carrying `CodeMode.Result` for inference: the agent's toolset and
 *  everything a frontend derives from it see typed `tool-codeMode` parts. */
const OUTPUT_SCHEMA = jsonSchema<CodeMode.Result>({ type: "object" })

// The interpreter enforces these per program run; a program that busy-loops
// or fans out unreasonably comes back as a diagnostic, not a hung turn.
const DEFAULT_LIMITS: CodeMode.ExecutionLimits = {
  timeoutMs: 30_000,
  maxToolCalls: 32,
  maxOutputBytes: 128 * 1024,
}

/** What the text drive hands the tool's binder each turn: the session-wrapped
 *  registry, the currently routed names, and their execution contexts. */
export type CodeModeBinding = {
  tools: ToolSet
  names: readonly string[]
  toolsContext?: Record<string, unknown>
}

const BIND = Symbol.for("ai-sdk-harness.code-mode.bind")

type Bindable = { [BIND]: (binding: CodeModeBinding) => BoundCodeMode }

/** The sandbox tool, AI SDK style: ONE factory whose tool late-binds to the
 *  session, mirroring the SDK's own code-mode shape (`codeModeTool()` +
 *  `bind(tools)`). Wire the result under the `code` registry key
 *  (`createRegistry({ ..., code: codeModeTool() })`) and list it in a
 *  role's `tools` like any other tool. Unbound it is still a complete,
 *  working tool — a pure-computation sandbox with no host tools; each turn
 *  the text drive rebinds it to the role's routed tools via the internal
 *  binder, which regenerates the description (the routed tools' typed API)
 *  and the execute. */
export function codeModeTool(
  /** Per-program budgets; merged over the defaults (30s, 32 calls, 128KB). */
  options?: CodeMode.ExecutionLimits,
): CodeModeTool {
  const limits = { ...DEFAULT_LIMITS, ...options }

  const bind = (binding: CodeModeBinding): BoundCodeMode => {
    const sandboxTools: Record<string, SandboxTool.Tool> = {}
    const pinnedPaths = new Set<string>()
    for (const name of binding.names) {
      const registryTool = binding.tools[name]
      if (registryTool === undefined) continue
      const inputSchema = asSchema(registryTool.inputSchema)
      const outputSchema = registryTool.outputSchema
        ? asSchema(registryTool.outputSchema)
        : undefined
      // A dotted key nests in the runtime's tool trie: namespace "erp" puts
      // the tool at `tools.erp.<name>` and under `erp` for `search` scoping.
      const namespace = toolNamespace(registryTool)
      const path = namespace === undefined ? name : `${namespace}.${name}`
      if (toolPinned(registryTool)) pinnedPaths.add(path)
      sandboxTools[path] = promiseTool({
        description:
          typeof registryTool.description === "string"
            ? registryTool.description
            : "",
        input: (inputSchema.jsonSchema ?? { type: "object" }) as SandboxTool.JsonSchema,
        // Always declare an output: a sandbox tool without one is `void` and
        // the runtime drops its result. `{}` (any JSON) is the honest default
        // for tools that don't state a schema.
        output: (outputSchema?.jsonSchema ?? {}) as SandboxTool.JsonSchema,
        execute: async (input) => {
          // The SDK loop validates direct calls; the sandbox path validates
          // here, as a safe ToolFailure the model can correct.
          const validated = await inputSchema.validate?.(input)
          if (validated !== undefined && !validated.success) {
            throw toolError(`Invalid input for '${path}': ${validated.error.message}`)
          }
          const args = validated !== undefined && validated.success ? validated.value : input
          const execute = registryTool.execute as (
            input: unknown,
            options: unknown,
          ) => PromiseLike<unknown> | unknown
          // Errors propagate as thrown: a deliberate ToolError carries its
          // message to the model, anything else is masked by the runtime —
          // driver and infrastructure errors must not cross the boundary.
          return await execute(args, {
            toolCallId: crypto.randomUUID(),
            messages: [],
            context: binding.toolsContext?.[name],
          })
        },
      })
    }

    const runtime = makeRuntime({ tools: sandboxTools, limits })
    const entries: CatalogEntry[] = runtime
      .catalog()
      .map((entry) => (pinnedPaths.has(entry.path) ? { ...entry, pinned: true } : entry))
    const catalog = renderCatalogInstructions(entries)
    return {
      tool: tool({
        description: BASE_DESCRIPTION,
        inputSchema: INPUT_SCHEMA,
        outputSchema: OUTPUT_SCHEMA,
        execute: (input) => runtime.execute(input.code),
      }),
      catalogInstructions: catalog,
    }
  }

  const unbound = makeRuntime({ limits })
  return Object.assign(
    tool({
      description: BASE_DESCRIPTION,
      inputSchema: INPUT_SCHEMA,
      outputSchema: OUTPUT_SCHEMA,
      execute: (input) => unbound.execute(input.code),
    }),
    { [BIND]: bind } satisfies Bindable,
  )
}

export type CodeModeTool = AiTool<CodeModeInput, CodeMode.Result>

/** What a bind yields: the session-bound tool (invariant description) and the
 *  routed tools' catalog rendered as an instructions block, for the drive to
 *  append to the session's system instructions. */
export type BoundCodeMode = {
  readonly tool: CodeModeTool
  readonly catalogInstructions: string
}

/** The binder the text drive calls each turn, if the registry entry is a
 *  `codeModeTool()`; undefined for any other tool under the key. */
export function codeModeBinder(
  registryTool: unknown,
): ((binding: CodeModeBinding) => BoundCodeMode) | undefined {
  if (typeof registryTool !== "object" || registryTool === null) return undefined
  return (registryTool as Partial<Bindable>)[BIND]
}

