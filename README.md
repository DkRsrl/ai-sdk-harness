# ai-sdk-harness

Application-agnostic agent harness over the [AI SDK](https://ai-sdk.dev): a
tool registry gated per-step by an injected role and skills, with pluggable
session storage. The model spec picks the drive — a `LanguageModel` yields a
text session (`.prompt`), a `RealtimeModelV1` a voice session (`.voice`).

```sh
bun add ai-sdk-harness
```

## Entrypoints

| Import | What it is |
| --- | --- |
| `ai-sdk-harness` | The harness: `init`, `createAssistant`, registry, roles, skills, code mode, storage |
| `ai-sdk-harness/voice` | The realtime-voice core — `createRealtimeSession` and the `RealtimeModelV1` spec |
| `ai-sdk-harness/voice/grok` | Grok realtime provider |
| `ai-sdk-harness/voice/gemini` | Gemini Live provider |
| `ai-sdk-harness/voice/gateway` | Vercel AI Gateway realtime provider |
| `ai-sdk-harness/voice/protocol` | The client↔relay wire protocol |
| `ai-sdk-harness/voice/ui` | Browser-side transport and voice-chat controller |
| `ai-sdk-harness/voice/react` | `useVoiceChat` (React 19, optional peer) |

`react` is an optional peer dependency — only `ai-sdk-harness/voice/react`
needs it.

## Usage

`init()` is the harness. Everything else in this package is built on it.
Runnable versions of everything below live in [`examples/`](./examples) —
`bun run examples/code-mode-runtime.ts` needs no API key.

A **registry** is every tool the harness can ever expose. A **role** names the
subset that is active by default, plus the system prompt. Tools are referenced
*by name* — the harness owns the registry, the role only declares which names
reach the model on a normal step.

```ts
import { createRegistry, harnessTool, init, type Session } from "ai-sdk-harness";
import { z } from "zod";

const { registry, role } = createRegistry({
  lookupOrder: harnessTool({
    description: "Look up an order by id",
    inputSchema: z.object({ id: z.string() }),
    contextSchema: z.object({ userId: z.string() }),
    // The running session is injected — no threading it through the schema.
    // Annotate both parameters: neither schema drives inference here, and an
    // un-inferred one collapses the whole object to `never`.
    execute: async (
      input: { id: string },
      { context, session }: { context: { userId: string }; session: Session },
    ) => fetchOrder({ id: input.id, userId: context.userId, sessionId: session.id }),
  }),
});

const support = role({
  name: "support",
  systemPrompt: "You help customers with their orders.",
  tools: ["lookupOrder"],
});
```

`createRegistry` binds the factories to the registry, so a typo in `tools`
is a compile error rather than a runtime one.

```ts
const harness = await init({
  registry,
  model: "anthropic/claude-sonnet-5",
  role: support(),
  toolsContext: { lookupOrder: { userId } },
  storage,
});

const session = await harness.session({ sessionId });
const turn = await session.prompt("Where is order 42?");

for await (const part of turn.textStream) process.stdout.write(part);
await turn.committed;
```

`session.prompt()` returns the AI SDK's `StreamTextResult` plus two additions:
`committed`, which resolves once the assistant turn has been assembled and
saved, and `timings()`, the turn's latency profile — the same one that lands on
the persisted message's `metadata.timings`.

Omitting `storage` gives an `InMemorySessionStorage`; supply a `SessionStorage`
to persist. Passing a `sessionId` that storage already knows resumes that
transcript.

### The model spec picks the drive

The same `init()` produces a voice session when the model is a
`RealtimeModelV1` instead of a `LanguageModel` — the type follows the spec, so
`.prompt()` and `.voice()` are never both present:

```ts
import { grok } from "ai-sdk-harness/voice/grok";

const harness = await init({
  registry,
  model: grok("grok-voice-latest"),
  role: support(),
});
const session = await harness.session({ sessionId });

const call = await session.voice({ onStatus, onAudio });
await call.start();
call.pushAudio(pcm);
await call.stop();
```

`voice()` seeds the prior conversation, exposes the same role-gated tools and
persists each finalized turn; it returns the live `RealtimeSession` without
connecting, so you `start()` it yourself — or hand it a duplex wire with
`call.serve(send)`, the voice analogue of `result.toUIMessageStream()`.

Both drives share the registry, the role, the skills and the transcript, so a
session can change drive between turns and carry the history forward. A voice
session can also escalate work to a text worker through `session.subsession()`,
which names its own model.

### Skills

A role is fixed for the session; skills are progressive. A skill carries
instructions and can toggle on tools the role didn't expose. The host injects
one with `session.skill(name)` between turns; the model can load one itself
mid-turn through `skillLoaderTool()`, with `loadableSkills` naming the sources
it may draw from.

### Code mode

`role({ toolCallers })` can route a tool into a sandboxed program instead of a
direct call — the model writes JavaScript against `tools.name(input)` and a
confined interpreter runs it, with the host's tools as its only door to the
world. See `src/codemode/README.md`.

## Assistant — the higher-order usage

`Assistant` is a convenience over `init()`, not a separate system. Use it when
one application has a reusable text-and-voice assistant whose context and
storage should resolve once per durable session rather than per turn. Anything
it does can be done with `init()` directly.


```ts
const assistant = createAssistant<AppScope>()({
  registry,

  async prepare({ sessionId, scope, signal }) {
    return {
      storage: await openConversationStorage({ sessionId, scope, signal }),
      context: await resolveAssistantContext({ scope, signal }),
    };
  },

  async text({ context }) {
    return {
      model: await resolveTextModel(context),
      role: createTextRole(context),
      toolsContext: createToolContext(context),
    };
  },

  async voice({ context }) {
    return {
      model: await resolveVoiceModel(context),
      role: createVoiceRole(context),
      toolsContext: createToolContext(context),
    };
  },
});
```

`scope` is caller-provided identity and authorization input. `context` is the
application-resolved information and resources used to configure both drives.
`storage` is the durable transcript shared by them.

```ts
const session = await assistant.session({
  sessionId,
  scope,
  signal,
});

const turn = await session.prompt("Hello");
await turn.committed;

const call = await session.voice();
call.pushAudio(pcm);
await call.stop();
```

`voice()` resolves after the realtime provider is ready for input. `stop()`
settles finalized turns and waits for their persistence.

One `AssistantSession` admits one response-producing operation at a time. A
text operation releases it when `committed` settles; a voice operation releases
it when `stop()` settles. Each activation reloads durable history, so changing
drives after those barriers carries the complete transcript forward.

Exclusivity is local to one `AssistantSession` object. Applications that can
open the same durable session concurrently across requests or processes still
need storage-level revisions, leases, or another coordinator.

## Provenance

Extracted from the `reco-ai` monorepo, unifying three workspace packages:
`@reco-ai/ai-sdk-harness`, `@reco-ai/ai-sdk-voice` (now `src/voice`, behind the
`/voice` entrypoints) and `@reco-ai/codemode` (vendored into `src/codemode`).
See LICENSE for third-party attribution.
