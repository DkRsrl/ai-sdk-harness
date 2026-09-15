import type { InferToolSetContext } from "@ai-sdk/provider-utils";
import type { LanguageModel, ToolSet } from "ai";
import type {
  RealtimeModelV1,
  RealtimeServeHandle,
  VoiceStatus,
} from "./voice";
import { init } from "./init";
import type { BoundRole } from "./role";
import type { SkillSource } from "./skills";
import type { SessionMessage, SessionStorage } from "./storage";
import type {
  GenerateId,
  HarnessConfig,
  PromptOptions,
  StreamResult,
  VoiceOptions,
} from "./types";

type MaybePromise<T> = T | Promise<T>;

export interface AssistantPrepareArgs<TScope> {
  sessionId: string;
  scope: TScope;
  signal?: AbortSignal;
}

export interface AssistantPreparation<TContext> {
  /** The durable transcript shared by both drives. */
  storage: SessionStorage;
  /** Application-resolved resources used to configure both drives. */
  context: TContext;
}

export interface AssistantChannelArgs<
  TScope,
  TContext,
> extends AssistantPrepareArgs<TScope> {
  context: TContext;
}

export type AssistantTextConfig<TTools extends ToolSet, TOutput = never> = Omit<
  HarnessConfig<TTools, LanguageModel, TOutput>,
  "registry" | "storage"
>;

export interface AssistantVoiceConfig<TTools extends ToolSet> {
  model: RealtimeModelV1;
  role: BoundRole;
  toolsContext?: Partial<InferToolSetContext<TTools>>;
  loadableSkills?: readonly SkillSource[];
  generateId?: GenerateId;
}

export interface AssistantConfig<
  TScope,
  TContext,
  TTools extends ToolSet,
  TOutput = never,
> {
  /** The complete capability registry; each resolved role activates a subset. */
  registry: TTools;
  /** Resolve one conversation's durable storage and application context. */
  prepare(
    args: AssistantPrepareArgs<TScope>,
  ): MaybePromise<AssistantPreparation<TContext>>;
  text(
    args: AssistantChannelArgs<TScope, TContext>,
  ): MaybePromise<AssistantTextConfig<TTools, TOutput>>;
  voice(
    args: AssistantChannelArgs<TScope, TContext>,
  ): MaybePromise<AssistantVoiceConfig<TTools>>;
}

export interface AssistantSessionOptions<TScope> {
  sessionId: string;
  scope: TScope;
  title?: string;
  signal?: AbortSignal;
}

export type AssistantVoiceOptions<TTools extends ToolSet> =
  VoiceOptions<TTools> & {
    abortSignal?: AbortSignal;
  };

export interface AssistantVoiceCall {
  readonly status: VoiceStatus;
  /** Messages loaded and seeded before the realtime provider was started. */
  readonly initialMessages: ReadonlyArray<SessionMessage>;
  pushAudio(pcm: ArrayBuffer): void;
  sendText(text: string): void;
  serve(send: (frame: string | ArrayBuffer) => void): RealtimeServeHandle;
  /** Stop the live drive and wait until its finalized transcript is durable. */
  stop(): Promise<void>;
}

export interface AssistantSession<TTools extends ToolSet, TOutput = never> {
  readonly id: string;
  /** Run one text turn. Another operation may start after `result.committed`. */
  prompt(
    input: string | SessionMessage,
    options?: PromptOptions,
  ): Promise<StreamResult<TTools, TOutput>>;
  /** Start voice and resolve only after the provider is ready for input. */
  voice(options?: AssistantVoiceOptions<TTools>): Promise<AssistantVoiceCall>;
}

function operationSignal(
  sessionSignal: AbortSignal | undefined,
  operationSignal: AbortSignal | undefined,
): AbortSignal | undefined {
  if (!sessionSignal) return operationSignal;
  if (!operationSignal || operationSignal === sessionSignal)
    return sessionSignal;
  return AbortSignal.any([sessionSignal, operationSignal]);
}

export class Assistant<
  TScope,
  TContext,
  TTools extends ToolSet,
  TOutput = never,
> {
  readonly #config: AssistantConfig<TScope, TContext, TTools, TOutput>;

  constructor(config: AssistantConfig<TScope, TContext, TTools, TOutput>) {
    this.#config = config;
  }

  /** Prepare one durable conversation for text and voice operations. */
  async session(
    options: AssistantSessionOptions<TScope>,
  ): Promise<AssistantSession<TTools, TOutput>> {
    const { sessionId, scope, title, signal: sessionSignal } = options;
    sessionSignal?.throwIfAborted();
    const prepared = await this.#config.prepare({
      sessionId,
      scope,
      signal: sessionSignal,
    });
    sessionSignal?.throwIfAborted();

    let active: "text" | "voice" | undefined;
    const begin = (channel: "text" | "voice") => {
      if (active) {
        throw new Error(
          `assistant session "${sessionId}" already has an active ${active} operation`,
        );
      }
      active = channel;
    };
    const release = (channel: "text" | "voice") => {
      if (active === channel) active = undefined;
    };

    return {
      id: sessionId,

      prompt: async (input, promptOptions = {}) => {
        begin("text");
        const signal = operationSignal(
          sessionSignal,
          promptOptions.abortSignal,
        );
        try {
          signal?.throwIfAborted();
          const drive = await this.#config.text({
            sessionId,
            scope,
            context: prepared.context,
            signal,
          });
          signal?.throwIfAborted();
          const harness = await init({
            ...drive,
            registry: this.#config.registry,
            storage: prepared.storage,
          });
          const session = await harness.session({ sessionId, title });
          const result = await session.prompt(input, {
            ...promptOptions,
            abortSignal: signal,
          });
          void result.committed.then(
            () => release("text"),
            () => release("text"),
          );
          return result;
        } catch (error) {
          release("text");
          throw error;
        }
      },

      voice: async (voiceOptions = {}) => {
        begin("voice");
        const { abortSignal, ...options } = voiceOptions;
        const signal = operationSignal(sessionSignal, abortSignal);
        let releaseOnFailure = true;
        try {
          signal?.throwIfAborted();
          const drive = await this.#config.voice({
            sessionId,
            scope,
            context: prepared.context,
            signal,
          });
          signal?.throwIfAborted();
          const harness = await init({
            ...drive,
            registry: this.#config.registry,
            storage: prepared.storage,
          });
          signal?.throwIfAborted();
          const session = await harness.session({ sessionId, title });
          signal?.throwIfAborted();
          const initialMessages = [...session.messages];
          const realtime = await session.voice(options);
          signal?.throwIfAborted();
          let stopPromise: Promise<void> | undefined;
          function stop() {
            signal?.removeEventListener("abort", onAbort);
            return (stopPromise ??= realtime.stop().finally(() => {
              release("voice");
            }));
          }
          function onAbort() {
            void stop().catch(() => {});
          }
          signal?.addEventListener("abort", onAbort, { once: true });
          if (signal?.aborted) {
            await stop();
            signal.throwIfAborted();
          }

          try {
            await realtime.start();
            if (signal?.aborted) {
              await stop();
              signal.throwIfAborted();
            }
          } catch (error) {
            await stop().catch(() => {});
            throw error;
          }

          releaseOnFailure = false;
          return {
            get status() {
              return realtime.status;
            },
            initialMessages,
            pushAudio: (pcm) => realtime.pushAudio(pcm),
            sendText: (text) => realtime.sendText(text),
            serve: (send) => realtime.serve(send),
            stop,
          };
        } finally {
          if (releaseOnFailure) release("voice");
        }
      },
    };
  }
}

/** Bind an application's scope type while inferring context, tools and output. */
export function createAssistant<TScope>() {
  return <TContext, TTools extends ToolSet, TOutput = never>(
    config: AssistantConfig<TScope, TContext, TTools, TOutput>,
  ) => new Assistant(config);
}
