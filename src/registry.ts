// Bind the role/skill factories to a concrete tool registry so their `tools`
// declarations are type-checked against the registry's keys. `createRegistry`
// takes the full toolset and returns it as `registry` (feed it straight to
// `init({ registry })`) alongside `role` and `skill` factories whose
// `tools?: (...)[]` field only accepts names that exist in it — turning a typo
// into a compile error, ahead of the harness's runtime `assertToolsRegistered`
// guard. The bare `role` / `skill` exports still accept any `string` for callers
// that don't want to bind a registry.

import type { ToolSet } from "ai";
import { assertValidNamespace } from "./code-mode";
import { role, type RoleDefinition, type RoleFactory } from "./role";
import { skill, type SkillDefinition, type SkillFactory } from "./skills";

export interface Registry<TTools extends ToolSet> {
  /** The complete toolset — pass straight to `init({ registry })`. */
  registry: TTools;
  /** Role factory whose `tools` are constrained to this registry's keys. */
  role<TArgs = void, TOutput = never>(
    def: RoleDefinition<TArgs, keyof TTools & string, TOutput>,
  ): RoleFactory<TArgs, TOutput>;
  /** Skill factory whose `tools` are constrained to this registry's keys. */
  skill<TArgs = void, TName extends string = string>(
    def: SkillDefinition<TArgs, keyof TTools & string, TName>,
  ): SkillFactory<TArgs, TName>;
}

export function createRegistry<TTools extends ToolSet>(
  tools: TTools,
): Registry<TTools> {
  // A tool's sandbox namespace (`metadata.namespace`) is registry-level
  // identity; a malformed one fails here, not on the first bound turn.
  for (const [name, registryTool] of Object.entries(tools)) {
    assertValidNamespace(name, registryTool);
  }
  return {
    registry: tools,
    role<TArgs = void, TOutput = never>(
      def: RoleDefinition<TArgs, keyof TTools & string, TOutput>,
    ) {
      return role<TArgs, keyof TTools & string, TOutput>(def);
    },
    skill<TArgs = void, TName extends string = string>(
      def: SkillDefinition<TArgs, keyof TTools & string, TName>,
    ) {
      return skill<TArgs, keyof TTools & string, TName>(def);
    },
  };
}
