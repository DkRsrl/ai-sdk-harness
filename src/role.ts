// A role is the agent's "who am I" — a named system prompt plus the subset of
// the harness tool registry that is active by default. Tools are referenced *by
// name*, not by value: the harness owns the full registry; the role merely
// declares which of those names are exposed to the model on a normal step.
// Skills can later toggle on additional names (see skills.ts / session.ts).

import type { z } from "zod";
import type { ToolCallers } from "./code-mode";
import { assertPlaceholdersDeclared, resolvePrompt } from "./prompt";

export interface RoleDefinition<
  TArgs = void,
  TTool extends string = string,
  TOutput = never,
> {
  name: string;
  argsSchema?: z.ZodType<TArgs>;
  /** A `{{placeholder}}` template (filled from validated args) or a function. */
  systemPrompt: string | ((args: TArgs) => string | Promise<string>);
  /** Tool names (registry keys) active by default for this role. Omit for a
   *  pure conversational role with no tools active until a skill turns some on.
   *  `TTool` defaults to `string` for the bare factory; `createRegistry` binds
   *  it to the registry's keys so a typo here becomes a compile error. */
  tools?: TTool[];
  /** Who may invoke each of this role's tools, in the AI SDK's
   *  `experimental_toolCallers` vocabulary. `["code"]` moves a tool into
   *  the model's sandboxed programs (`await tools.name(input)`) instead of a
   *  direct call; add `DIRECT_TOOL_CALL` to keep both ways open; no entry
   *  means a normal direct call. Keys must come from `tools`. The role owns
   *  the routing — the same registry tool can be direct in one role and
   *  sandboxed in another. The text drive rebinds the sandbox every turn;
   *  the voice drive binds it once at connect. */
  toolCallers?: ToolCallers<TTool>;
  /** When set, the role yields a structured object matching this schema instead
   *  of free prose: the text drive wires it to the model's `output` spec
   *  (`Output.object`), so the model is constrained to the shape and
   *  `prompt()`'s result exposes the validated value as `result.output` (typed
   *  as `TOutput`). Text drive only — realtime/voice has no structured output,
   *  and a voice session built from such a role simply ignores it. */
  outputSchema?: z.ZodType<TOutput>;
}

export interface BoundRole<TOutput = never> {
  name: string;
  /** Tool names this role activates by default (validated against the registry
   *  when the session starts). */
  tools: string[];
  /** This role's routing map (see `RoleDefinition.toolCallers`); empty when
   *  every tool is a plain direct call. */
  toolCallers: ToolCallers;
  /** The role's structured-output schema, if any — the harness wraps it in the
   *  SDK's `Output.object(...)` for the text drive. */
  outputSchema?: z.ZodType<TOutput>;
  resolveInstructions(): Promise<string>;
}

export type RoleFactory<TArgs, TOutput = never> = TArgs extends void
  ? () => BoundRole<TOutput>
  : (args: TArgs) => BoundRole<TOutput>;

export function role<
  TArgs = void,
  TTool extends string = string,
  TOutput = never,
>(def: RoleDefinition<TArgs, TTool, TOutput>): RoleFactory<TArgs, TOutput> {
  if (typeof def.systemPrompt === "string") {
    assertPlaceholdersDeclared(
      `role("${def.name}")`,
      def.systemPrompt,
      def.argsSchema,
    );
  }

  const factory = (args?: TArgs): BoundRole<TOutput> => {
    const validatedArgs = def.argsSchema
      ? def.argsSchema.parse(args)
      : (args as TArgs);

    return {
      name: def.name,
      tools: def.tools ?? [],
      toolCallers: def.toolCallers ?? {},
      outputSchema: def.outputSchema,
      resolveInstructions: async () => {
        if (typeof def.systemPrompt === "string") {
          return resolvePrompt(
            def.systemPrompt,
            validatedArgs as Record<string, unknown>,
          );
        }
        return def.systemPrompt(validatedArgs);
      },
    };
  };

  return factory as RoleFactory<TArgs, TOutput>;
}
