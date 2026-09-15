// Per-turn metadata the harness injects into what the model sees. Each user
// turn is prefixed with a `<message_metadata>` block carrying the time it was
// sent (and, when known, the user's timezone), so the model has temporal
// context ("how long ago did I ask X", "is it evening for the user"). The block
// is added only to the model-facing copy of the conversation — persisted and
// UI-rendered messages keep the raw user text untouched.
//
// Both the time and the timezone are frozen per message: the time comes from
// `createdAt`, the zone from `metadata.timezone` (stamped when the turn is
// created), never from the current clock or the latest request. That keeps each
// turn's decoration byte-identical across replays — both fields are persisted
// and restored — so the prompt prefix stays stable and provider-side prefix
// caching is preserved even if the user's timezone later changes. A turn missing
// `createdAt` is left undecorated rather than stamped with `now`, which would
// otherwise be a fresh value on every replay and bust the cache.

import type { SessionMessage } from "./storage";

/** Render the `<message_metadata>` block prepended to a user turn. When
 *  `timeZone` is a valid IANA name the time is rendered as local wall-clock with
 *  offset (e.g. `2026-06-16T16:30:00+02:00`) plus the zone; otherwise it falls
 *  back to UTC ISO 8601. */
export function formatMessageMetadata(at: Date, timeZone?: string): string {
  const zoned = timeZone ? isoWithOffset(at, timeZone) : null;
  const lines = ["<message_metadata>"];
  if (zoned) {
    lines.push(`time: ${zoned}`, `timezone: ${timeZone}`);
  } else {
    lines.push(`time: ${at.toISOString()}`);
  }
  lines.push("</message_metadata>");
  return lines.join("\n");
}

/** Return a model-facing copy of `message` with a `<message_metadata>` text
 *  part prepended for user turns (time from `createdAt`, zone from
 *  `metadata.timezone`). Non-user turns, and user turns missing `createdAt`,
 *  pass through unchanged so the decoration stays a pure function of stable
 *  persisted data. */
export function withMessageMetadata(message: SessionMessage): SessionMessage {
  if (message.role !== "user" || !message.createdAt) return message;
  const metaPart = {
    type: "text" as const,
    text: formatMessageMetadata(message.createdAt, readTimeZone(message.metadata)),
  };
  return { ...message, parts: [metaPart, ...message.parts] };
}

/** Render `at` as ISO 8601 in `timeZone` with a numeric offset, or null if the
 *  zone is unusable (so the caller falls back to UTC). */
function isoWithOffset(at: Date, timeZone: string): string | null {
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone,
      hour12: false,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      timeZoneName: "longOffset",
    }).formatToParts(at);
    const get = (type: string) =>
      parts.find((p) => p.type === type)?.value ?? "";
    const hour = get("hour") === "24" ? "00" : get("hour");
    // `longOffset` renders "GMT+02:00" (or "GMT" at UTC); normalize to "+02:00".
    const offset = get("timeZoneName").replace(/^GMT/, "") || "+00:00";
    return `${get("year")}-${get("month")}-${get("day")}T${hour}:${get("minute")}:${get("second")}${offset}`;
  } catch {
    return null;
  }
}

function readTimeZone(metadata: unknown): string | undefined {
  if (metadata && typeof metadata === "object" && "timezone" in metadata) {
    const tz = (metadata as { timezone?: unknown }).timezone;
    if (typeof tz === "string" && tz) return tz;
  }
  return undefined;
}
