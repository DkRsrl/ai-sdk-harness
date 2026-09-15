# Code mode — the vendored interpreter

This directory is a port of **opencode's codemode** package, MIT:

<https://github.com/anomalyco/opencode/tree/v2/packages/codemode>

We follow the opencode approach rather than the usual alternatives, and this
file records what that approach is, what we changed, and the rules for keeping
the two in step.

## The approach

The model writes a program in a JavaScript subset; a pure TypeScript
interpreter executes it; the program's only door to the world is the `tools`
object the host provides.

- **Interpretation, not execution.** No `eval`, no `vm`, no worker threads, no
  WASM, no container. Upstream's phrasing is that it "only runs the language
  features we implement" — confinement is a property of the evaluator, so
  there is no sandbox to escape and nothing to configure per host. The same
  code runs under Node, Bun and bundlers without ceremony.
- **Tools are the only capability.** A program cannot reach the network, the
  filesystem or a process. It can call what the host put in `tools`, and
  nothing else. Authorization stays where the host already does it.
- **Failures are data.** `execute` does not reject for program failures. It
  returns a `Result` carrying diagnostics that distinguish parse errors,
  unsupported syntax, tool failures and execution problems — so an agent can
  read what went wrong and retry.
- **REPL-style results.** Without an explicit `return`, the final top-level
  expression is the result. Program values are normalized to JSON-like
  boundaries (`undefined` and non-finite numbers become `null`).
- **Bounded by limits, not by trust.** `timeoutMs`, `maxToolCalls` and
  `maxOutputBytes` (`codemode.ts`). Timeouts are cooperative — deadline checks
  inside the evaluator, so a busy loop is caught too — and are applied in
  `interpreter/execute.ts`.
- **Unfinished work is interrupted at return.** If the program returns while
  background promises are still running, they are interrupted and the result
  carries a warning naming the un-awaited work rather than silently dropping
  it.

### Declare `output`, or the value is dropped

A tool with no `output` schema renders as `Promise<void>` in the catalog and
its return value is nullified on the way back to the program — the program sees
nothing and fails on the next property access. `output` is optional in the
type, but it is what carries the result:

```ts
promiseTool({
  description: "List every order id.",
  input: { type: "object", properties: {} },
  output: { type: "array", items: { type: "string" } },  // required in practice
  execute: async () => ids,
})
```

`interpreter-support.md` is the checkable support matrix for the language and
standard-library surface. It is upstream's, kept current here: when behaviour
changes, update it and the tests in the same change.

## What we changed

- **The Promise face is ours.** `run.ts` (`makeRuntime`, `promiseTool`) is not
  upstream. Effect is still the interpreter's engine — it has *not* been
  removed — but it is an implementation detail: consumers integrate through
  `run.ts` and never see an Effect type. Additions belong there.
- **The OpenAPI module was not ported.** Upstream can turn an OpenAPI spec
  into callable tools; we define tools directly.
- **Everything else stays diffable against upstream.** When touching any other
  file, keep the shape close enough that an upstream diff still reads.

## Layout

| Path | What it is |
| --- | --- |
| `index.ts` | The public surface: `CodeMode`, `Tool`, `ToolError`, plus our `makeRuntime` / `promiseTool` |
| `run.ts` | The Promise facade over Effect — **not upstream** |
| `codemode.ts` | Runtime construction, execution limits, the model-visible tool catalog |
| `interpreter/` | Parser allowlist, evaluator, scope, promises, diagnostics |
| `stdlib/` | The built-ins programs may use |
| `tool*.ts` | Tool definition, schemas, runtime dispatch, error sanitizing |
| `interpreter-support.md` | The support matrix (upstream's) |

Tests live at `test/codemode/` in the repo root, including test262-derived
suites under their own license (`test/codemode/LICENSE.test262`).

## Upstream tracking

The port predates this repository and **the upstream commit it was taken from
was not recorded**. Before the next upstream sync, pin one here — a re-diff
against `v2` is the only way to recover the baseline, and it gets more
expensive the longer it waits.

`typescript` is a runtime dependency of this package because programs are
transpiled before interpretation (`interpreter/execute.ts`), and `acorn` is the
parser. Both are real dependencies of any consumer that imports code mode.
