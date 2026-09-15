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
| [`assistant.ts`](./assistant.ts) | `createAssistant` — text *and* voice over one durable conversation | both keys |
| [`code-mode-runtime.ts`](./code-mode-runtime.ts) | The confined interpreter on its own | nothing — runs offline |

```sh
bun run examples/code-mode-runtime.ts                  # no key needed
AI_GATEWAY_API_KEY=... bun run examples/text-session.ts "where is order 42?"
XAI_API_KEY=... bun run examples/voice-session.ts
AI_GATEWAY_API_KEY=... XAI_API_KEY=... bun run examples/assistant.ts
```

Start with `code-mode-runtime.ts` if you want to see something work without
reaching for a key: it defines two tools, has the interpreter run a program
across them, and then shows the two failure shapes — a tool error arriving as
data rather than a throw, and `fetch` simply not existing.

`text-session.ts` is the one to read first for the harness itself. Everything
else in the package is built on the `init()` call it makes.

`assistant.ts` is the last one to read. It shows the same registry answering on
both drives over one conversation — a text turn, then a voice call on the same
session id that opens with that turn already in its history — and is written
against `createAssistant`, which is `init()` with the per-request shape filled
in. Read it after you can see what it would take to write it with `init()`
directly.
