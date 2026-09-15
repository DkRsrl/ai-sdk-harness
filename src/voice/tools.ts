// Bridge between an AI SDK ToolSet — the exact `tool()`-built tools the text
// harness uses — and what a realtime provider advertises to the model. The
// definitions are derived here; execution is the SDK's own `executeTool`,
// invoked from the session core. Same tools, two consumers (streamText + voice).

import { asSchema, type ToolSet } from "ai";
import type { RealtimeToolDef } from "./spec";

/** Derive the wire tool definitions a provider advertises from a ToolSet. Only
 *  executable tools are advertised (provider-defined/no-execute tools skipped). */
export async function toRealtimeToolDefs(
  tools: ToolSet,
): Promise<RealtimeToolDef[]> {
  const defs: RealtimeToolDef[] = [];
  for (const [name, t] of Object.entries(tools)) {
    if (typeof t.execute !== "function") continue;
    const parametersJsonSchema = (await asSchema(t.inputSchema)
      .jsonSchema) as Record<string, unknown>;
    defs.push({
      name,
      description: typeof t.description === "string" ? t.description : name,
      parametersJsonSchema,
    });
  }
  return defs;
}
