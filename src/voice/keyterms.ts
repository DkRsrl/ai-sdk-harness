// Key terms that bias input transcription toward proper nouns the model would
// otherwise mis-hear (product names, client names). The wire limits below are
// xAI's; every provider that accepts key terms goes through this normalizer so
// a caller can hand over an unbounded list without knowing them.

export const MAX_KEYTERMS = 100;
export const MAX_KEYTERM_LENGTH = 50;

/** Trim, drop empties, truncate over-long terms, de-duplicate and cap the list
 *  at the API's limits. Returns undefined when nothing survives, so callers can
 *  omit the field entirely. */
export function normalizeKeyterms(terms: readonly string[] | undefined): string[] | undefined {
  if (!terms?.length) return undefined;
  const seen = new Set<string>();
  for (const term of terms) {
    const value = term.trim().slice(0, MAX_KEYTERM_LENGTH).trim();
    if (value) seen.add(value);
    if (seen.size >= MAX_KEYTERMS) break;
  }
  return seen.size ? [...seen] : undefined;
}
