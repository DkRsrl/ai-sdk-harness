// A minimal text session: one registry, one role, one persisted turn.
//
//   AI_GATEWAY_API_KEY=... bun run examples/text-session.ts "where is order 42?"
//
// Storage is left at the default in-memory store, so nothing outlives the
// process — swap in a `SessionStorage` to persist.

import { createRegistry, harnessTool, init, type Session } from "../src/index";
import { z } from "zod";

type Order = { id: string; status: string; eta: string };

const ORDERS: Record<string, Order> = {
  "42": { id: "42", status: "in transit", eta: "Thursday" },
  "43": { id: "43", status: "delivered", eta: "—" },
};

const { registry, role } = createRegistry({
  // `harnessTool` is the SDK's `tool()` with the running session injected into
  // `execute`. Both `execute` parameters need annotating: neither `inputSchema`
  // nor `contextSchema` drives inference through the harness's execute
  // signature, and an un-inferred one collapses the whole object to `never`.
  lookupOrder: harnessTool({
    description: "Look up an order by its id.",
    inputSchema: z.object({ id: z.string() }),
    contextSchema: z.object({ userId: z.string() }),
    execute: async (
      input: { id: string },
      { context, session }: { context: { userId: string }; session: Session },
    ) => {
      console.log(`  [tool] user=${context.userId} session=${session.id}`);
      return ORDERS[input.id] ?? { error: "no such order" };
    },
  }),
});

// A role is the agent's "who am I": a system prompt plus the registry keys
// active by default. Tools are named, never passed by value.
const support = role({
  name: "support",
  systemPrompt: [
    "You are a support agent. Look orders up before answering.",
    "Answer in one short sentence.",
  ].join(" "),
  tools: ["lookupOrder"],
});

const harness = await init({
  registry,
  model: "anthropic/claude-sonnet-5",
  role: support(),
  // Per-tool execution context, keyed by tool name. Partial on purpose: you
  // feed the whole registry, but only the tools a role activates are executed.
  toolsContext: { lookupOrder: { userId: "user_1" } },
});

const session = await harness.session();
const turn = await session.prompt(process.argv[2] ?? "Where is order 42?");

for await (const part of turn.textStream) process.stdout.write(part);
process.stdout.write("\n");

// `committed` resolves once the assistant turn has been assembled and saved.
// Await it before starting another operation on the same session.
await turn.committed;

console.log("\ntimings:", turn.timings());
console.log("messages persisted:", session.messages.length);
