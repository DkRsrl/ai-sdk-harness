import { randomUUID } from "node:crypto";
import type { ToolSet } from "ai";
import { createSession } from "./session";
import {
  assembleInstructions,
  mergeSkillSources,
  SKILL_LOADER_TOOL_NAME,
  skillLoaderBinder,
} from "./skills";
import { InMemorySessionStorage, type SessionMessage } from "./storage";
import type {
  DriveModel,
  Harness,
  HarnessConfig,
  SessionFor,
  SessionOptions,
} from "./types";

/** Build a harness from a config (registry + model + role + optional storage).
 *  The model's spec picks the session shape: a `LanguageModel` gives sessions
 *  with `.prompt()`, a `RealtimeModelV1` gives sessions with `.voice()`. The
 *  role's instructions are resolved once here and reused for every session;
 *  per-session message history is loaded lazily from storage when a sessionId
 *  is supplied. */
export async function init<
  TTools extends ToolSet = ToolSet,
  TModel extends DriveModel = DriveModel,
  TOutput = never,
  TSkillName extends string = string,
>(
  config: HarnessConfig<TTools, TModel, TOutput, TSkillName>,
): Promise<Harness<TTools, TModel, TOutput, TSkillName>> {
  const sources = config.loadableSkills ?? [];
  if (
    sources.length > 0 &&
    skillLoaderBinder(config.registry[SKILL_LOADER_TOOL_NAME]) === undefined
  ) {
    throw new Error(
      `init: loadableSkills sources are configured but the "${SKILL_LOADER_TOOL_NAME}" registry entry is not the harness loader — wire skillLoaderTool() from "ai-sdk-harness" under that key`,
    );
  }
  const skills = mergeSkillSources(sources);
  // A name collision makes the universe ambiguous; surface it here even when
  // no role renders the catalog (merged `list()` refuses duplicates).
  if (skills && sources.length > 1) await skills.list();
  // Initial skills join the system prompt after the role's instructions;
  // the catalog is appended last (only when the loader tool is active — the
  // role's or an initial skill's), so it is resolved once here like the rest.
  const instructions = await assembleInstructions(
    config.role,
    config.initialSkills ?? [],
    skills,
  );
  // A session always has storage; default to in-memory when none is configured,
  // and thread that one instance through every session this harness creates.
  const storage = config.storage ?? new InMemorySessionStorage();
  const resolvedConfig: HarnessConfig<TTools, TModel, TOutput, TSkillName> = {
    ...config,
    storage,
  };

  return {
    config: resolvedConfig,
    instructions,
    async session(
      opts: SessionOptions = {},
    ): Promise<SessionFor<TTools, TModel, TOutput, TSkillName>> {
      const generateId = resolvedConfig.generateId ?? randomUUID;
      const sessionId = opts.sessionId ?? generateId();

      let messages: SessionMessage[] | undefined = opts.messages;
      if (!messages && opts.sessionId) {
        messages = await storage.loadMessages(opts.sessionId);
      }

      await storage.createSession?.(sessionId, {
        title: opts.title ?? "New chat",
      });

      return createSession<TTools, TModel, TOutput, TSkillName>(
        resolvedConfig,
        instructions,
        { sessionId, messages },
      );
    },
  };
}
