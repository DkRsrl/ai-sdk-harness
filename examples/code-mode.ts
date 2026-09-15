// Code mode: instead of calling tools one at a time, the model writes a small
// JavaScript program and a confined interpreter runs it, with the host's tools
// as its only door to the world. Useful when a task is a loop, a join or a
// filter over several tool calls — one program replaces N round-trips.
//
//   AI_GATEWAY_API_KEY=... bun run examples/code-mode.ts
//
// See src/codemode/README.md for what the interpreter does and does not allow.

import {
  CODE_MODE_TOOL_NAME,
  codeModeTool,
  createRegistry,
  harnessTool,
  init,
  type Session,
} from "../src/index";
import { z } from "zod";

type Order = { id: string; status: string; total: number };

const ORDERS: Order[] = [
  { id: "40", status: "delivered", total: 12 },
  { id: "41", status: "in transit", total: 140 },
  { id: "42", status: "in transit", total: 38 },
  { id: "43", status: "delivered", total: 9 },
];

const { registry, role } = createRegistry({
  listOrders: harnessTool({
    description: "List every order id.",
    inputSchema: z.object({}),
    execute: async (_input: Record<string, never>, _opts: { session: Session }) =>
      ORDERS.map((order) => order.id),
  }),
  lookupOrder: harnessTool({
    description: "Look up one order by its id.",
    inputSchema: z.object({ id: z.string() }),
    execute: async (input: { id: string }, _opts: { session: Session }) =>
      ORDERS.find((order) => order.id === input.id) ?? null,
  }),
  // The sandbox itself is a registry entry, wired under a well-known key.
  [CODE_MODE_TOOL_NAME]: codeModeTool(),
});

const analyst = role({
  name: "analyst",
  systemPrompt: "Answer questions about orders. Prefer one program over many calls.",
  tools: ["listOrders", "lookupOrder", CODE_MODE_TOOL_NAME],
  // Routing: these two are reachable only from inside a program, as
  // `await tools.listOrders({})`. Add DIRECT_TOOL_CALL to keep both routes
  // open. The role owns this — the same tool can be direct in another role.
  toolCallers: { listOrders: ["code"], lookupOrder: ["code"] },
});

const harness = await init({
  registry,
  model: "anthropic/claude-sonnet-5",
  role: analyst(),
});

const session = await harness.session();

console.log("tools reachable only from code:", session.activeCodeTools);

const turn = await session.prompt(
  "What is the total value of the orders still in transit?",
);

for await (const part of turn.textStream) process.stdout.write(part);
process.stdout.write("\n");
await turn.committed;
