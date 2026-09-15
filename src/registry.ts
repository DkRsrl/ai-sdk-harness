// Bind the role/skill factories to a concrete tool registry so their `tools`
// declarations are type-checked against the registry's keys. `createRegistry`
// takes the full toolset and returns `registry` — a function of the context
// those tools run with (`init({ registry: registry({ userId, fs }) })`) —
// alongside `role` and `skill` factories whose `tools?: (...)[]` field only
// accepts names that exist in it, turning a typo into a compile error ahead of
// the harness's runtime `assertToolsRegistered` guard. The bare `role` /
// `skill` exports still accept any `string` for callers that don't want to bind
// a registry.
//
// Context is supplied ONCE here rather than once per tool. Each tool already
// declares what it needs (`contextSchema`), and the AI SDK validates the value
// through that schema before `execute` runs — a zod object strips down to its
// declared keys — so a tool still only ever sees what it asked for. That
// projection is what `registry(context)` leans on instead of hand-writing a
// name-keyed map whose entries repeat the same value.

import type { InferToolContext } from "@ai-sdk/provider-utils";
import type { ToolSet } from "ai";
import { assertValidNamespace } from "./code-mode";
import { role, type RoleDefinition, type RoleFactory } from "./role";
import { skill, type SkillDefinition, type SkillFactory } from "./skills";

type UnionToIntersection<U> = (U extends unknown ? (k: U) => void : never) extends (
  k: infer I,
) => void
  ? I
  : never;

/** Everything a registry's tools collectively declare, as one object: the
 *  intersection of every tool's `contextSchema` type. Tools that declare no
 *  context contribute nothing. This is the argument `registry()` asks for, so
 *  the shape is derived from the tools themselves and autocompletes. */
export type RegistryContext<TTools extends ToolSet> = UnionToIntersection<
  { [K in keyof TTools]: InferToolContext<TTools[K]> }[keyof TTools]
>;

/** Per-tool context overrides — for a tool whose context genuinely differs
 *  from the shared one rather than being a subset of it (a `bash` reaching a
 *  sandbox filesystem while everything else reads the user's). An override
 *  REPLACES that tool's context; it does not merge over the shared value. A
 *  tool declaring no `contextSchema` has no context to override. */
export type ToolsContextOverrides<TTools extends ToolSet> = {
  [K in keyof TTools]?: InferToolContext<TTools[K]>;
};

const BOUND_REGISTRY = Symbol.for("ai-sdk-harness.boundRegistry");

/** A registry's tools together with the context they run with — what
 *  `registry(context)` returns and `init({ registry })` takes. */
export interface BoundRegistry<TTools extends ToolSet> {
  readonly [BOUND_REGISTRY]: true;
  readonly tools: TTools;
  /** Shared across every tool that declares a `contextSchema`. */
  readonly context?: RegistryContext<TTools>;
  readonly overrides?: ToolsContextOverrides<TTools>;
}

/** What `init({ registry })` accepts: a bound registry, or a bare toolset for
 *  the registries whose tools need no context at all. */
export type RegistrySource<TTools extends ToolSet> =
  | TTools
  | BoundRegistry<TTools>;

export function isBoundRegistry<TTools extends ToolSet>(
  source: RegistrySource<TTools>,
): source is BoundRegistry<TTools> {
  return (
    typeof source === "object" &&
    source !== null &&
    (source as Partial<BoundRegistry<TTools>>)[BOUND_REGISTRY] === true
  );
}

/** The toolset behind either form. */
export function registryTools<TTools extends ToolSet>(
  source: RegistrySource<TTools>,
): TTools {
  return isBoundRegistry(source) ? source.tools : source;
}

/** Resolve the AI SDK's name-keyed `toolsContext` from the shared context and
 *  the two override layers. Precedence, most specific first: the config's own
 *  `toolsContext`, then the registry's `overrides`, then the shared context —
 *  and the shared context reaches ONLY tools that declare a `contextSchema`,
 *  since the SDK passes context through untouched when a tool declares none
 *  and an undeclared tool would otherwise receive everything. */
export function resolveToolsContext<TTools extends ToolSet>(
  source: RegistrySource<TTools>,
  configToolsContext: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  const bound = isBoundRegistry(source) ? source : undefined;
  const shared = bound?.context as Record<string, unknown> | undefined;
  const overrides = bound?.overrides as Record<string, unknown> | undefined;
  if (shared === undefined && overrides === undefined) return configToolsContext;

  const resolved: Record<string, unknown> = {};
  for (const [name, entry] of Object.entries(registryTools(source))) {
    const specific = configToolsContext?.[name] ?? overrides?.[name];
    if (specific !== undefined) {
      resolved[name] = specific;
      continue;
    }
    // Declaring a context is what admits a tool to the shared one.
    if (shared !== undefined && entry?.contextSchema != null) {
      resolved[name] = shared;
    }
  }
  // Entries for tools outside the registry would be dead weight, but dropping
  // them silently would hide a typo; keep them so the SDK's own lookup decides.
  for (const [name, value] of Object.entries(configToolsContext ?? {})) {
    if (!(name in resolved)) resolved[name] = value;
  }
  return resolved;
}

export interface Registry<TTools extends ToolSet> {
  /** Bind this registry's tools to the context they run with. The argument is
   *  the union of what the tools' `contextSchema`s declare, so it is derived,
   *  typed and autocompleted; it disappears when no tool declares any. */
  registry: (
    ...args: RegistryContextArgs<TTools>
  ) => BoundRegistry<TTools>;
  /** Role factory whose `tools` are constrained to this registry's keys. */
  role<TArgs = void, TOutput = never>(
    def: RoleDefinition<TArgs, keyof TTools & string, TOutput>,
  ): RoleFactory<TArgs, TOutput>;
  /** Skill factory whose `tools` are constrained to this registry's keys. */
  skill<TArgs = void, TName extends string = string>(
    def: SkillDefinition<TArgs, keyof TTools & string, TName>,
  ): SkillFactory<TArgs, TName>;
}

/** `registry()` for a registry whose tools declare no context, `registry(ctx)`
 *  (optionally with overrides) for one whose tools do. */
export type RegistryContextArgs<TTools extends ToolSet> = [
  RegistryContext<TTools>,
] extends [never]
  ? []
  : unknown extends RegistryContext<TTools>
    ? []
    : [context: RegistryContext<TTools>, overrides?: ToolsContextOverrides<TTools>];

export function createRegistry<TTools extends ToolSet>(
  tools: TTools,
): Registry<TTools> {
  // A tool's sandbox namespace (`metadata.namespace`) is registry-level
  // identity; a malformed one fails here, not on the first bound turn.
  for (const [name, registryTool] of Object.entries(tools)) {
    assertValidNamespace(name, registryTool);
  }
  return {
    registry: (...args: RegistryContextArgs<TTools>) =>
      ({
        [BOUND_REGISTRY]: true,
        tools,
        context: args[0] as RegistryContext<TTools> | undefined,
        overrides: args[1] as ToolsContextOverrides<TTools> | undefined,
      }) satisfies BoundRegistry<TTools>,
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
