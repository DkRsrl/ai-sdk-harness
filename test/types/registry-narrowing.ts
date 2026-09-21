// Type-level assertions for `registry.only`. Nothing here runs: each
// `@ts-expect-error` fails the build if that line ever stops being an error,
// which is how the narrowing is pinned. Context narrowing is a type guarantee,
// so this is where it is tested.

import { z } from "zod";
import { createRegistry, harnessTool } from "../../src/index";

const r = createRegistry({
  scoped: harnessTool({
    description: "Declares userId only.",
    inputSchema: z.object({}),
    contextSchema: z.object({ userId: z.string() }),
    execute: async () => "ok",
  }),
  filed: harnessTool({
    description: "Declares fs only.",
    inputSchema: z.object({}),
    contextSchema: z.object({ fs: z.string() }),
    execute: async () => "ok",
  }),
});

// The whole registry asks for every tool's context at once.
r.registry({ userId: "u1", fs: "/home" });

// @ts-expect-error `filed` is in this registry, so its `fs` is required.
r.registry({ userId: "u1" });

// Narrowed, the context narrows with it: `fs` is neither needed nor accepted.
r.registry.only("scoped")({ userId: "u1" });

// @ts-expect-error no tool in this narrowing declares `fs`.
r.registry.only("scoped")({ userId: "u1", fs: "/home" });

// @ts-expect-error narrowing to `filed` makes its `fs` required.
r.registry.only("filed")({});

// Closed under narrowing, and the context follows each step.
r.registry.only("scoped", "filed").only("filed")({ fs: "/home" });

// @ts-expect-error a name the registry does not hold.
r.registry.only("missing");
