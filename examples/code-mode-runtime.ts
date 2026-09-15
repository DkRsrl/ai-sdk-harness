// The confined interpreter on its own, with no model involved — the thing the
// harness's code-mode tool wraps. Runs offline:
//
//   bun run examples/code-mode-runtime.ts
//
// The program below is what a model would have written. It loops, joins and
// filters across several tool calls, and the interpreter is the only thing
// that ever executes it: no eval, no vm, no worker, no network.

import { makeRuntime, promiseTool, toolError } from "../src/codemode/index";

type Order = { id: string; status: string; total: number };

const ORDERS: Order[] = [
  { id: "40", status: "delivered", total: 12 },
  { id: "41", status: "in transit", total: 140 },
  { id: "42", status: "in transit", total: 38 },
  { id: "43", status: "delivered", total: 9 },
];

const runtime = makeRuntime({
  tools: {
    orders: {
      list: promiseTool({
        description: "List every order id.",
        input: { type: "object", properties: {} },
        // Declare `output` or the value never reaches the program: an
        // undeclared output renders as `Promise<void>` in the catalog and is
        // nullified on the way back.
        output: { type: "array", items: { type: "string" } },
        execute: async () => ORDERS.map((order) => order.id),
      }),
      lookup: promiseTool({
        description: "Look up one order by id.",
        input: {
          type: "object",
          properties: { id: { type: "string" } },
          required: ["id"],
        },
        output: {
          type: "object",
          properties: {
            id: { type: "string" },
            status: { type: "string" },
            total: { type: "number" },
          },
          required: ["id", "status", "total"],
        },
        execute: async (input) => {
          const { id } = input as { id: string };
          const order = ORDERS.find((o) => o.id === id);
          // A thrown `toolError` is a safe, model-visible failure. Anything
          // else thrown is sanitized before the program can see it.
          if (!order) throw toolError(`no order ${id}`);
          return order;
        },
      }),
    },
  },
  limits: { timeoutMs: 5_000, maxToolCalls: 32 },
  onToolCallEnd: (call) => console.log(`  [tool] ${call.name} ${call.outcome} (${call.durationMs}ms)`),
});

// What the model sees when deciding what to write.
console.log("catalog:");
for (const entry of runtime.catalog()) console.log(" ", entry.signature);

const result = await runtime.execute(`
  const ids = await tools.orders.list({})
  const orders = await Promise.all(ids.map((id) => tools.orders.lookup({ id })))
  const pending = orders.filter((o) => o.status === "in transit")
  return { count: pending.length, total: pending.reduce((sum, o) => sum + o.total, 0) }
`);

console.log("\nresult:", result);

// Failures are data, never thrown — an agent reads the diagnostics and retries.
const bad = await runtime.execute(`return await tools.orders.lookup({ id: "99" })`);
console.log("\nfailure is a value:", bad.ok, JSON.stringify(bad.ok ? null : bad.error));

// And the confinement is real: there is no `fetch` to reach for.
const denied = await runtime.execute(`return await fetch("https://example.com")`);
console.log("no ambient capabilities:", denied.ok, JSON.stringify(denied.ok ? null : denied.error));
