import { test } from "node:test";
import assert from "node:assert/strict";

import { MAX_KEYTERM_LENGTH, MAX_KEYTERMS, normalizeKeyterms } from "./keyterms";

test("normalizeKeyterms returns undefined for an empty or missing list", () => {
  assert.equal(normalizeKeyterms(undefined), undefined);
  assert.equal(normalizeKeyterms([]), undefined);
  assert.equal(normalizeKeyterms(["  ", ""]), undefined);
});

test("normalizeKeyterms trims, de-duplicates and keeps order", () => {
  assert.deepEqual(normalizeKeyterms([" Acme ", "Northwind", "Acme"]), ["Acme", "Northwind"]);
});

test("normalizeKeyterms truncates a term to the wire limit", () => {
  const [term] = normalizeKeyterms(["x".repeat(MAX_KEYTERM_LENGTH + 20)]) ?? [];
  assert.equal(term?.length, MAX_KEYTERM_LENGTH);
});

test("normalizeKeyterms caps the list at the wire limit", () => {
  const many = Array.from({ length: MAX_KEYTERMS + 50 }, (_, i) => `term-${i}`);
  const terms = normalizeKeyterms(many) ?? [];
  assert.equal(terms.length, MAX_KEYTERMS);
  assert.equal(terms[0], "term-0");
});
