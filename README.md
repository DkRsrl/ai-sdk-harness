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

## Orchestration

The package exposes two levels of orchestration:

- `init()` configures one text or realtime drive directly. Use it for jobs,
  structured roles, subsessions, and tests that need low-level session control.
- `Assistant` configures one reusable text-and-voice assistant. It resolves an
  application's context and storage once per durable session, then exposes
  `prompt()` and `voice()` on that session.

## Assistant

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
