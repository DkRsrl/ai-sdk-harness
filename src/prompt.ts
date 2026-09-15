// Prompt-template helpers shared by roles and skills. A template may embed
// `{{placeholder}}` tokens that are filled from a validated args object. Keeping
// the resolver and the placeholder/schema checks here avoids duplicating them in
// `role.ts` and `skills.ts`.

import type { z } from "zod";

const PLACEHOLDER_RE = /\{\{(\w+)\}\}/g;

/** Substitute every `{{key}}` in `template` with `String(vars[key])`. Unknown
 *  placeholders are left intact (so a typo is visible rather than silently
 *  blanked). The result is trimmed. */
export function resolvePrompt(
  template: string,
  vars: Record<string, unknown>,
): string {
  return template
    .replace(PLACEHOLDER_RE, (match, key: string) => {
      const value = vars[key];
      return value === undefined ? match : String(value);
    })
    .trim();
}

/** The distinct placeholder names referenced by a template. */
export function extractPlaceholders(template: string): string[] {
  const seen = new Set<string>();
  for (const match of template.matchAll(PLACEHOLDER_RE)) {
    seen.add(match[1]!);
  }
  return [...seen];
}

/** The top-level keys of a zod object schema, or null if not introspectable. */
function getSchemaKeys(
  schema: z.ZodType<unknown> | undefined,
): Set<string> | null {
  if (!schema) return null;
  const shape = (schema as unknown as { shape?: Record<string, unknown> })
    .shape;
  if (shape && typeof shape === "object") return new Set(Object.keys(shape));
  return null;
}

/** Throw at definition time if a template references a placeholder the args
 *  schema does not declare — a static guard against drift between the two. */
export function assertPlaceholdersDeclared(
  label: string,
  template: string,
  schema: z.ZodType<unknown> | undefined,
): void {
  const placeholders = extractPlaceholders(template);
  const schemaKeys = getSchemaKeys(schema);
  if (placeholders.length === 0 || !schemaKeys) return;
  const missing = placeholders.filter((p) => !schemaKeys.has(p));
  if (missing.length > 0) {
    throw new Error(
      `${label}: template has placeholders not declared in argsSchema: ${missing
        .map((m) => `{{${m}}}`)
        .join(", ")}`,
    );
  }
}
