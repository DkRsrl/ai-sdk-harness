# Examples

Runnable programs, in the order they build on each other. They import from
`../src` rather than the published package, so they exercise the sources in
this repo directly and are typechecked by `bun run lint` along with everything
else.

| File | What it shows | Needs |
| --- | --- | --- |
| [`text-session.ts`](./text-session.ts) | Registry, role, tools, one persisted turn | `AI_GATEWAY_API_KEY` |
| [`voice-session.ts`](./voice-session.ts) | The same registry driven by a realtime model | `XAI_API_KEY` |
| [`code-mode.ts`](./code-mode.ts) | Routing tools into the model's sandboxed programs | `AI_GATEWAY_API_KEY` |
| [`code-mode-runtime.ts`](./code-mode-runtime.ts) | The confined interpreter on its own | nothing — runs offline |

```sh
bun run examples/code-mode-runtime.ts                  # no key needed
AI_GATEWAY_API_KEY=... bun run examples/text-session.ts "where is order 42?"
XAI_API_KEY=... bun run examples/voice-session.ts
```

Start with `code-mode-runtime.ts` if you want to see something work without
reaching for a key: it defines two tools, has the interpreter run a program
across them, and then shows the two failure shapes — a tool error arriving as
data rather than a throw, and `fetch` simply not existing.

`text-session.ts` is the one to read first for the harness itself. Everything
else in the package is built on the `init()` call it makes.
