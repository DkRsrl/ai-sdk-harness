import assert from "node:assert/strict";
import { test } from "node:test";
import { renderCatalogInstructions, summarize, type CatalogEntry } from "./code-mode-catalog";

const entry = (path: string, over: Partial<CatalogEntry> = {}): CatalogEntry => ({
  path,
  description: `What ${path} does.\nSecond line never shown inline.`,
  signature: `tools.${path}(input: { id: string }): Promise<unknown>`,
  ...over,
});

test("an empty catalog renders the do-not-call notice", () => {
  assert.match(renderCatalogInstructions([]), /No Code Mode tools are currently available/);
});

test("a small catalog renders complete: every entry, first-line blurbs, no Search section", () => {
  const rendered = renderCatalogInstructions([entry("erp.one"), entry("erp.two"), entry("crm.three")]);
  assert.match(rendered, /The Code Mode tool catalog below is complete\./);
  assert.doesNotMatch(rendered, /## Search/);
  assert.match(rendered, /- crm \(1 tool\)/);
  assert.match(rendered, /- erp \(2 tools\)/);
  assert.match(rendered, /tools\.erp\.one\(input: \{ id: string \}\): Promise<unknown> \/\/ What erp\.one does\./);
  assert.doesNotMatch(rendered, /Second line never shown inline/);
});

test("a long first line is cut to a 120-char blurb", () => {
  const rendered = renderCatalogInstructions([
    entry("erp.chatty", { description: `${"a".repeat(200)}\nrest` }),
  ]);
  assert.match(rendered, new RegExp(` // ${"a".repeat(117)}\\.\\.\\.$`, "m"));
});

test("over budget the catalog turns partial: headers keep every namespace, pinned survives the cut", () => {
  // Identical costs, so the unpinned cut is decided by path order — zz.z99
  // sorts last and would be dropped without its pin.
  const entries = Array.from({ length: 200 }, (_, n) =>
    entry(`zz.z${String(n).padStart(2, "0")}`, {
      description: "x".repeat(119),
      signature: `tools.zz.z${String(n).padStart(2, "0")}(input: { a: string, b: string, c: string }): Promise<{ out: string }>`,
    }),
  );
  entries.push(entry("aa.only", { description: "short" }));
  entries[99] = { ...entries[99]!, pinned: true };

  const summary = summarize(entries);
  assert.equal(summary.total, 201);
  assert.ok(summary.shown < summary.total);
  const rendered = renderCatalogInstructions(entries);
  assert.match(rendered, /The Code Mode tool catalog below is partial\./);
  assert.match(rendered, /## Search/);
  assert.match(rendered, /- aa \(1 tool\)/);
  assert.match(rendered, /- zz \(200 tools, \d+ shown\)/);
  assert.match(rendered, /tools\.zz\.z99\(/);
});
