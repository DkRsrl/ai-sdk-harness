// Session persistence is a *moving part* of the harness: inject any
// implementation of `SessionStorage` and the harness will load/save messages
// through it. The interface is deliberately minimal and ecosystem-agnostic —
// it is keyed purely by `sessionId` and knows nothing about tenants, users, or
// any particular database. Consumers (e.g. apps/web) implement it against their
// own store; `InMemorySessionStorage` ships here for tests and ephemeral use.

import type { UIMessage } from "ai";

/** A persisted message. Mirrors the AI SDK `UIMessage` (so it round-trips
 *  through `convertToModelMessages` / `toUIMessageStream`) plus an optional
 *  creation timestamp for ordering. Metadata is left open via the UIMessage
 *  generic at the call site. */
export type SessionMessage = UIMessage & { createdAt?: Date };

export interface SessionMeta {
  title?: string;
  /** The chat that spawned this one (via `subsession`); `null`/absent for a
   *  top-level session. Lets a store keep subsession transcripts for cost/usage
   *  tracking while filtering them out of user-facing chat lists. */
  parentId?: string | null;
}

/** A session as seen when listing them: identity plus whatever metadata the
 *  store happens to record. */
export interface SessionInfo {
  id: string;
  /** Display title, if the store records one. */
  title: string | null;
  /** Creation time, if the store records one (also drives ordering). */
  createdAt?: Date;
  /** The chat that spawned this one, if it's a subsession; `null` for a
   *  top-level session. Absent when the store doesn't track parentage. */
  parentId?: string | null;
}

/** A `SessionInfo` with its messages — what `listSessions({ populate: true })`
 *  returns (the overload narrows to this automatically). */
export interface PopulatedSessionInfo extends SessionInfo {
  /** The session's messages, oldest first. */
  messages: SessionMessage[];
}

export interface ListSessionsOptions {
  /** Max sessions to return; omit for all. */
  limit?: number;
  /** Creation-time order. `"newest"` (the default) lists the most recent first. */
  order?: "newest" | "oldest";
  /** Load each session's messages in the same pass (the result is then
   *  `PopulatedSessionInfo`). Lets a store fetch them in one shot (e.g. a single
   *  batched query) instead of the caller reading each session separately. Off
   *  by default. */
  populate?: boolean;
  /** Filter by parentage: `null` returns only top-level sessions (no parent), a
   *  chat id returns only that chat's subsessions, omit for all. */
  parentId?: string | null;
}

export interface SessionStorage {
  /** Load a session's messages, oldest first. Return `[]` for unknown ids. */
  loadMessages(sessionId: string): Promise<SessionMessage[]>;
  /** Upsert messages by `id` (append new, replace existing). */
  saveMessages(sessionId: string, messages: SessionMessage[]): Promise<void>;
  /** List the sessions this store holds, newest first by default. The store
   *  instance already fixes the scope (e.g. one user's `DatabaseSessionStorage`),
   *  so this stays agnostic: it takes no user/tenant and never special-cases a
   *  "current" session — callers filter that out themselves. With `populate: true`
   *  each session carries its `messages`, and the return type narrows to
   *  `PopulatedSessionInfo` automatically. */
  listSessions(
    opts: ListSessionsOptions & { populate: true },
  ): Promise<PopulatedSessionInfo[]>;
  listSessions(opts?: ListSessionsOptions): Promise<SessionInfo[]>;
  /** Register a session before its first message is saved. Optional: a store
   *  that creates rows lazily on `saveMessages` can omit this. */
  createSession?(sessionId: string, meta?: SessionMeta): Promise<void>;
}

/** Process-memory storage. Useful for tests and throwaway sessions; nothing is
 *  persisted beyond the lifetime of the instance. */
export class InMemorySessionStorage implements SessionStorage {
  private readonly sessions = new Map<string, Map<string, SessionMessage>>();
  private readonly meta = new Map<string, SessionMeta>();

  async createSession(sessionId: string, meta: SessionMeta = {}): Promise<void> {
    if (!this.sessions.has(sessionId)) this.sessions.set(sessionId, new Map());
    if (!this.meta.has(sessionId)) this.meta.set(sessionId, meta);
  }

  async loadMessages(sessionId: string): Promise<SessionMessage[]> {
    return [...(this.sessions.get(sessionId)?.values() ?? [])];
  }

  listSessions(
    opts: ListSessionsOptions & { populate: true },
  ): Promise<PopulatedSessionInfo[]>;
  listSessions(opts?: ListSessionsOptions): Promise<SessionInfo[]>;
  async listSessions({
    limit,
    order,
    populate,
    parentId,
  }: ListSessionsOptions = {}): Promise<SessionInfo[]> {
    // Map iteration preserves insertion order, which tracks creation order;
    // reverse it for newest-first.
    let ids = [...this.sessions.keys()];
    if (order !== "oldest") ids.reverse();
    if (parentId !== undefined)
      ids = ids.filter((id) => (this.meta.get(id)?.parentId ?? null) === parentId);
    const limited = limit != null ? ids.slice(0, limit) : ids;
    return limited.map((id) => ({
      id,
      title: this.meta.get(id)?.title ?? null,
      parentId: this.meta.get(id)?.parentId ?? null,
      ...(populate
        ? { messages: [...(this.sessions.get(id)?.values() ?? [])] }
        : {}),
    }));
  }

  async saveMessages(
    sessionId: string,
    messages: SessionMessage[],
  ): Promise<void> {
    let bucket = this.sessions.get(sessionId);
    if (!bucket) {
      bucket = new Map();
      this.sessions.set(sessionId, bucket);
    }
    for (const message of messages) bucket.set(message.id, message);
  }
}
