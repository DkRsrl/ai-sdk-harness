// The code-mode catalog as system instructions, ported from upstream opencode
// (packages/core/src/codemode/catalog.ts + instructions.ts, v2 @ 4d22d4e):
// the `code` tool's description stays invariant and the changing catalog
// rides in the prompt instead. Namespace headers always appear — existence is
// never hidden, only detail — and full listings are selected pinned-first,
// then one per namespace per round, shortest first, until the inline token
// budget runs out. `search` is advertised only when the catalog is partial;
// full tool descriptions are always search-served (inline entries carry only
// the first line, cut to a blurb). Upstream's delta rendering (`update`) is
// deliberately not ported: the text drive re-renders the instructions every
// turn and the voice drive binds once at connect, so a mid-session catalog
// diff has no seam to fire through. One wording divergence from upstream: the
// Search section spells out that `search` is a program global — probed 21/08,
// the realtime voice model went hunting for undisclosed tools through the
// knowledge-base search tool instead of the program's own `search`.

import { searchSignature, type CodeMode } from "./codemode"
// Cycle with code-mode.ts (which imports the renderer): safe because the name
// is only read inside functions, never at module evaluation.
import { CODE_MODE_TOOL_NAME } from "./code-mode"

/** One catalog entry as the binder collects it: the runtime's rendered
 *  description plus the tool's own `metadata.pinned` flag. */
export type CatalogEntry = CodeMode.ToolDescription & { readonly pinned?: boolean }

const DESCRIPTION_LIMIT = 120
const CHARACTERS_PER_TOKEN = 4
const INLINE_BUDGET = 2_000

type Listing = { readonly path: string; readonly line: string }

type NamespaceSummary = {
  readonly name: string
  readonly count: number
  readonly entries: ReadonlyArray<Listing>
}

export type CatalogSummary = {
  readonly total: number
  readonly shown: number
  readonly namespaces: ReadonlyArray<NamespaceSummary>
}

const byPath = (left: { path: string }, right: { path: string }): number =>
  left.path < right.path ? -1 : left.path > right.path ? 1 : 0

const cost = (listing: Listing): number => Math.round(listing.line.length / CHARACTERS_PER_TOKEN)

/** Keep every namespace visible, then select full listings pinned-first, one
 *  per namespace per round, shortest first, until the budget is exhausted. */
export function summarize(entries: ReadonlyArray<CatalogEntry>, budget = INLINE_BUDGET): CatalogSummary {
  const groups = new Map<string, CatalogEntry[]>()
  for (const entry of entries) {
    const name = entry.path.split(".", 1)[0] ?? entry.path
    const group = groups.get(name)
    if (group === undefined) groups.set(name, [entry])
    else group.push(entry)
  }
  const namespaces = [...groups]
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
    .map(([name, namespaceEntries]) => {
      const listings = namespaceEntries
        .map((entry) => {
          const firstLine = entry.description.split("\n", 1)[0]?.trim() ?? ""
          const description =
            firstLine.length > DESCRIPTION_LIMIT ? `${firstLine.slice(0, DESCRIPTION_LIMIT - 3)}...` : firstLine
          const suffix = description.length === 0 ? "" : ` // ${description}`
          return { path: entry.path, line: `  - ${entry.signature}${suffix}` }
        })
        .sort(byPath)
      const ranked = listings
        .map((listing) => ({ listing, cost: cost(listing) }))
        .sort((left, right) => left.cost - right.cost || byPath(left.listing, right.listing))
      const pinned = new Set(
        namespaceEntries
          .filter((entry) => entry.pinned)
          .flatMap((entry) => listings.filter((listing) => listing.path === entry.path)),
      )
      return {
        name,
        listings,
        selectionOrder: ranked.filter((candidate) => !pinned.has(candidate.listing)),
        selected: pinned,
        selectionIndex: 0,
      }
    })

  const active = new Set(namespaces)
  let remaining =
    budget -
    namespaces
      .flatMap((namespace) => namespace.listings.filter((listing) => namespace.selected.has(listing)))
      .reduce((total, listing) => total + cost(listing), 0)
  while (active.size > 0) {
    for (const namespace of active) {
      const candidate = namespace.selectionOrder[namespace.selectionIndex]
      if (candidate === undefined || candidate.cost > remaining) {
        active.delete(namespace)
        continue
      }
      namespace.selected.add(candidate.listing)
      namespace.selectionIndex += 1
      remaining -= candidate.cost
      if (namespace.selectionIndex === namespace.selectionOrder.length) active.delete(namespace)
    }
  }

  const summaries = namespaces.map((namespace) => ({
    name: namespace.name,
    count: namespace.listings.length,
    entries: namespace.listings.filter((listing) => namespace.selected.has(listing)),
  }))
  return {
    total: entries.length,
    shown: summaries.reduce((total, namespace) => total + namespace.entries.length, 0),
    namespaces: summaries,
  }
}

// prettier-ignore
const header = (hasMoreTools: boolean) => `The Code Mode tool catalog below is ${hasMoreTools ? "partial" : "complete"}.

${hasMoreTools ? "The Code Mode catalog and `search` results are" : "This catalog is"} the complete set of tools available within Code Mode. Tools presented elsewhere are not available in this runtime.${hasMoreTools ? `

## Search

To discover exact paths and signatures for additional tools, call the \`search\` global from inside a \`${CODE_MODE_TOOL_NAME}\` program — it is a program global, not a session tool, and no other search reads this catalog. Query it in English: it matches the tools' own names and descriptions, which are English.

- ${searchSignature}` : ""}

## Available tools`

export function renderCatalogInstructions(entries: ReadonlyArray<CatalogEntry>): string {
  const catalog = summarize(entries)
  if (catalog.total === 0)
    return `No Code Mode tools are currently available. Later Code Mode catalog updates may add or remove tools. Do not call \`${CODE_MODE_TOOL_NAME}\` unless there is at least one available Code Mode tool.`

  const tools = catalog.namespaces.flatMap((namespace) => {
    const count = namespace.count === 1 ? "1 tool" : `${namespace.count} tools`
    const label =
      namespace.entries.length === namespace.count
        ? count
        : namespace.entries.length === 0
          ? `${count}, none shown`
          : `${count}, ${namespace.entries.length} shown`
    return [`- ${namespace.name} (${label})`, ...namespace.entries.map((entry) => entry.line)]
  })

  return `${header(catalog.shown < catalog.total)}

${tools.join("\n")}`
}
