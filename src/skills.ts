// Skills are progressive capability injected mid-conversation. A skill carries
// a short description and a body of instructions; invoking it (session.skill())
// appends a `<skill-init>` user message and — crucially — *toggles on* any tool
// names the skill declares, on top of whatever the role already exposed. Skills
// can be defined inline (with arg interpolation) or loaded from a SKILL.md file
// with YAML frontmatter, mirroring the Claude skill format.
//
// Skills reach a session by two routes, both resolving by name through the
// session's `loadableSkills` sources. The HOST calls `session.skill(name)`
// between turns. The MODEL loads one itself through the loader tool at the
// bottom of this file: `init({ loadableSkills })` names `SkillSource`s, the catalog
// (names + descriptions) renders into the instructions of every role that
// lists the loader, and the loaded skill's instructions come back as the tool
// RESULT — the one path that reaches the model mid-turn on both drives (text
// snapshots its messages at turn start; voice advertises tools once at
// connect) and persists with the transcript, so a loaded skill stays in
// force for the rest of the session.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { tool, type Tool as AiTool } from "ai";
import { z } from "zod";
import type { ToolCallers } from "./code-mode";
import { assertPlaceholdersDeclared, resolvePrompt } from "./prompt";

const MAX_DESCRIPTION = 425;

export type ParsedSkill = {
  name: string;
  description: string;
  instructions: string;
  tools: string[];
  /** The skill's routing map, honored on model-load exactly as on binding.
   *  Only a source handing over ready skill values (e.g. a static source over
   *  code-defined skills) can carry one — SKILL.md's flat frontmatter cannot
   *  express it, so `parseSkill` never sets it. */
  toolCallers?: ToolCallers;
};

export interface BoundSkill<TName extends string = string> {
  name: TName;
  description: string;
  /** Tool names (registry keys) this skill turns on when invoked. */
  tools: string[];
  /** This skill's routing map, merged into the session's when invoked (see
   *  `RoleDefinition.toolCallers`); empty for plain direct calls. */
  toolCallers: ToolCallers;
  resolveInstructions(): Promise<string>;
}

export interface SkillDefinition<
  TArgs = void,
  TTool extends string = string,
  TName extends string = string,
> {
  name: TName;
  /** Shown to the model so it knows when to reach for the skill. Kept short. */
  description: string;
  argsSchema?: z.ZodType<TArgs>;
  instructions: string | ((args: TArgs) => string | Promise<string>);
  /** Tool names to activate while this skill is in effect. `TTool` defaults to
   *  `string`; `createRegistry` binds it to the registry's keys so a typo here
   *  becomes a compile error. */
  tools?: TTool[];
  /** Who may invoke each of this skill's tools while it is in effect (see
   *  `RoleDefinition.toolCallers`). Keys must come from `tools`. */
  toolCallers?: ToolCallers<TTool>;
}

export type SkillFactory<
  TArgs,
  TName extends string = string,
> = TArgs extends void ? () => BoundSkill<TName> : (args: TArgs) => BoundSkill<TName>;

export function skill<
  TArgs = void,
  TTool extends string = string,
  TName extends string = string,
>(def: SkillDefinition<TArgs, TTool, TName>): SkillFactory<TArgs, TName> {
  if (def.description.length > MAX_DESCRIPTION) {
    throw new Error(
      `skill("${def.name}"): description must be under ${MAX_DESCRIPTION} characters`,
    );
  }
  if (typeof def.instructions === "string") {
    assertPlaceholdersDeclared(
      `skill("${def.name}")`,
      def.instructions,
      def.argsSchema,
    );
  }

  const factory = (args?: TArgs): BoundSkill<TName> => {
    const validatedArgs = def.argsSchema
      ? def.argsSchema.parse(args)
      : (args as TArgs);

    return {
      name: def.name,
      description: def.description,
      tools: def.tools ?? [],
      toolCallers: def.toolCallers ?? {},
      resolveInstructions: async () => {
        if (typeof def.instructions === "string") {
          return resolvePrompt(
            def.instructions,
            validatedArgs as Record<string, unknown>,
          );
        }
        return def.instructions(validatedArgs);
      },
    };
  };

  return factory as SkillFactory<TArgs, TName>;
}

/** Minimal YAML-ish frontmatter parser: flat `key: value` pairs plus folded
 *  (`>` / `|`) block scalars. Enough for SKILL.md headers without a YAML dep. */
function parseFrontmatter(raw: string): Record<string, string> {
  const result: Record<string, string> = {};
  let currentKey: string | null = null;

  for (const line of raw.split("\n")) {
    const trimmed = line.trim();

    if (currentKey && line.length > 0 && (line[0] === " " || line[0] === "\t")) {
      if (trimmed) {
        result[currentKey] = result[currentKey]
          ? `${result[currentKey]} ${trimmed}`
          : trimmed;
      }
      continue;
    }

    const colon = trimmed.indexOf(":");
    if (colon === -1 || !trimmed) {
      currentKey = null;
      continue;
    }

    const key = trimmed.slice(0, colon).trim();
    const value = trimmed.slice(colon + 1).trim();

    if (value === ">" || value === "|") {
      currentKey = key;
      result[key] = "";
      continue;
    }

    result[key] = value;
    currentKey = key;
  }

  return result;
}

/** Split a frontmatter `tools` field — comma- or whitespace-separated names. */
function parseToolList(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split(/[,\s]+/)
    .map((t) => t.trim())
    .filter(Boolean);
}

export function parseSkill(content: string): ParsedSkill {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match?.[1]) {
    throw new Error("SKILL.md must have YAML frontmatter between --- delimiters");
  }

  const frontmatter = parseFrontmatter(match[1]);
  const body = content.slice(match[0].length);

  const name = frontmatter["name"];
  const description = frontmatter["description"];

  if (!name) throw new Error("SKILL.md frontmatter must include 'name'");
  if (!description) {
    throw new Error("SKILL.md frontmatter must include 'description'");
  }
  if (description.length > MAX_DESCRIPTION) {
    throw new Error(
      `SKILL.md description must be under ${MAX_DESCRIPTION} characters`,
    );
  }

  return {
    name,
    description,
    instructions: body.trim(),
    tools: parseToolList(frontmatter["tools"]),
  };
}

/** Render the message appended to the conversation when a skill is invoked. */
export function formatSkillInit(skill: ParsedSkill): string {
  return [
    `<skill-init name="${skill.name}">`,
    skill.description,
    "",
    skill.instructions,
    `</skill-init>`,
  ].join("\n");
}

/** Load a skill from `<skillsDir>/<name>/SKILL.md`. */
export async function loadSkill(
  skillsDir: string,
  name: string,
): Promise<BoundSkill> {
  const path = join(skillsDir, name, "SKILL.md");
  const raw = await readFile(path, "utf-8");
  const parsed = parseSkill(raw);
  return {
    name: parsed.name,
    description: parsed.description,
    tools: parsed.tools,
    // SKILL.md's flat frontmatter can't express a routing map; a file-loaded
    // skill's tools are all direct calls.
    toolCallers: {},
    resolveInstructions: async () => parsed.instructions,
  };
}

// ── Model-triggered loading ──────────────────────────────────────────────────

/** A skill's name and when-to-use line, as the catalog advertises it. */
export type SkillListing = { name: string; description: string };

/** Where a session's on-demand skills come from. `init({ loadableSkills })`
 *  takes an array of these, merged into one universe: `list()` renders into
 *  the instructions of every role that lists the loader tool; the loader
 *  resolves `read(name)` when the model asks (null for a name the source
 *  doesn't hold). Implementations own their storage — a directory, a
 *  database-backed filesystem — and typically parse with `parseSkill`. */
export interface SkillSource<TName extends string = string> {
  list(): Promise<SkillListing[]>;
  read(name: string): Promise<ParsedSkill | null>;
  /** Type-level only, never set or read at runtime: the statically known
   *  skill names this source declares. A source over code-defined skills
   *  narrows it to their name literals, which is what types
   *  `session.skill(name)`; a source whose names are only known at runtime
   *  (e.g. the knowledge base) declares `SkillSource<never>` so it doesn't
   *  widen the session's name union to `string`. */
  readonly staticNames?: readonly TName[];
}

/** The session's skill universe as one source: listings concatenate in source
 *  order and reads resolve through the sources in order. Two sources declaring
 *  the same name make the universe ambiguous, so `list()` refuses it — `init`
 *  lists once up front, turning a collision into a session-init error. */
export function mergeSkillSources(
  sources: readonly SkillSource[],
): SkillSource | undefined {
  if (sources.length === 0) return undefined;
  const only = sources[0];
  if (sources.length === 1 && only) return only;
  return {
    list: async () => {
      const listings = (
        await Promise.all(sources.map((source) => source.list()))
      ).flat();
      const seen = new Set<string>();
      for (const { name } of listings) {
        if (seen.has(name)) {
          throw new Error(
            `skill "${name}" is declared by more than one loadableSkills source — every skill name must be unique across the session's sources`,
          );
        }
        seen.add(name);
      }
      return listings;
    },
    read: async (name) => {
      for (const source of sources) {
        const skill = await source.read(name);
        if (skill) return skill;
      }
      return null;
    },
  };
}

/** The registry key `skillLoaderTool` must be wired under — the name the
 *  catalog tells the model to call, and the key the drives late-bind. */
export const SKILL_LOADER_TOOL_NAME = "loadSkill";

// Invariant model-facing guidance (opencode-style wording); the changing part
// — which skills exist — renders into the session's instructions instead
// (formatSkillCatalog).
const LOADER_DESCRIPTION = [
  "Load a skill: the instructions for one of the domains listed under <available_skills>.",
  "",
  "Call it as your ONLY call in the step — a tool called alongside runs without the instructions it was supposed to follow. The returned instructions apply for the rest of the session. An unknown name is an error listing the skills that exist.",
].join("\n");

const LOADER_INPUT = z.object({
  name: z.string().describe("The skill's name, exactly as listed."),
});

export type SkillLoaderTool = AiTool<z.infer<typeof LOADER_INPUT>, ParsedSkill>;

/** What a drive hands the loader's binder: the session's skill source and the
 *  hook that unions a loaded skill's declared tools into the active set. The
 *  toggle only reaches the model on the text drive (voice advertises tools
 *  once at connect), so a skill meant for voice should declare no tools. */
export type SkillLoaderBinding = {
  source: SkillSource;
  activate(skill: ParsedSkill): void;
};

const BIND = Symbol.for("ai-sdk-harness.skill-loader.bind");

type Bindable = { [BIND]: (binding: SkillLoaderBinding) => SkillLoaderTool };

/** The loader tool, AI SDK style: ONE factory whose tool late-binds to the
 *  session, like `codeModeTool()`. Wire the result under the `loadSkill`
 *  registry key and list it in a role's `tools` like any other tool. Unbound
 *  (no `loadableSkills` source configured) it is still a complete tool that answers
 *  every call with "no skills available"; each session rebinds it to its
 *  source. The result is rendered with `formatSkillInit`, the same shape
 *  `session.skill()` injects. */
export function skillLoaderTool(): SkillLoaderTool {
  const make = (binding?: SkillLoaderBinding): SkillLoaderTool =>
    tool({
      description: LOADER_DESCRIPTION,
      inputSchema: LOADER_INPUT,
      execute: async ({ name }) => {
        const skill = await binding?.source.read(name);
        if (!skill) {
          const available = (await binding?.source.list()) ?? [];
          throw new Error(
            available.length
              ? `Unknown skill "${name}" — available: ${available.map((s) => s.name).join(", ")}`
              : `Unknown skill "${name}" — no skills are available in this session.`,
          );
        }
        binding?.activate(skill);
        return skill;
      },
      toModelOutput: ({ output }) => ({
        type: "text",
        value: formatSkillInit(output),
      }),
    });
  return Object.assign(make(), { [BIND]: make } satisfies Bindable);
}

/** The binder a drive calls, if the registry entry under
 *  `SKILL_LOADER_TOOL_NAME` is a `skillLoaderTool()`; undefined otherwise. */
export function skillLoaderBinder(
  registryTool: unknown,
): ((binding: SkillLoaderBinding) => SkillLoaderTool) | undefined {
  if (typeof registryTool !== "object" || registryTool === null)
    return undefined;
  return (registryTool as Partial<Bindable>)[BIND];
}

/** The advertisement rendered into the instructions, in opencode's shape: a
 *  short prose preamble carrying the one behavior rule — load before acting in
 *  a covered domain — and an `<available_skills>` block. */
export function formatSkillCatalog(listings: SkillListing[]): string {
  return [
    "Skills carry the instructions for specific domains. Two rules:",
    `- The moment the conversation touches a covered domain, load its skill with ${SKILL_LOADER_TOOL_NAME}. A vague or incomplete request counts: how to respond is in the skill.`,
    `- ${SKILL_LOADER_TOOL_NAME} is your ONLY call in that step. Read what it returns, then decide.`,
    "<available_skills>",
    ...listings.flatMap((s) => [
      "  <skill>",
      `    <name>${s.name}</name>`,
      `    <description>${s.description}</description>`,
      "  </skill>",
    ]),
    "</available_skills>",
  ].join("\n");
}

/** The one assembly path for a session's system prompt: the role's
 *  instructions, any initial skills' after them, and the skill catalog
 *  trailing when the loader tool is active. `init` and `subsession` both
 *  build through here so the two cannot drift. */
export async function assembleInstructions(
  role: {
    resolveInstructions(): Promise<string>;
    tools: readonly string[];
  },
  initialSkills: readonly BoundSkill[],
  skills: SkillSource | undefined,
): Promise<string> {
  const withCatalog = await withSkillCatalog(
    await role.resolveInstructions(),
    { tools: [...role.tools, ...initialSkills.flatMap((s) => s.tools)] },
    skills,
  );
  // An initial skill wears the exact envelope a loaded one arrives in
  // (`formatSkillInit`) and follows the catalog, just as a load off the
  // catalog would — so the model can tell it is already active and doesn't
  // load it again.
  const inits = await Promise.all(
    initialSkills.map(async (s) =>
      formatSkillInit({
        name: s.name,
        description: s.description,
        instructions: await s.resolveInstructions(),
        tools: [...s.tools],
      }),
    ),
  );
  return [withCatalog, ...inits].join("\n\n");
}

export async function withSkillCatalog(
  instructions: string,
  role: { tools: readonly string[] },
  skills: SkillSource | undefined,
): Promise<string> {
  if (!skills || !role.tools.includes(SKILL_LOADER_TOOL_NAME))
    return instructions;
  const listings = await skills.list();
  if (listings.length === 0) return instructions;
  return `${instructions}\n\n${formatSkillCatalog(listings)}`;
}
