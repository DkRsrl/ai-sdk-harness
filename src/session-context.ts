// Ambient access to the running session. Tools created with `harnessTool` (and
// any tool wrapped by the session) execute inside `runWithSession`, so they can
// call `currentSession()` to reach the session that invoked them — without the
// session having to thread itself through every tool's input. Backed by
// AsyncLocalStorage, so concurrent sessions never see each other's context.

import { AsyncLocalStorage } from "node:async_hooks";
import type { Session } from "./types";

type AnySession = Session<never>;

const sessionStore = new AsyncLocalStorage<AnySession>();

export function currentSession(): Session {
  const session = sessionStore.getStore();
  if (!session) {
    throw new Error(
      "currentSession() called outside a harness session. Tools that use currentSession() must be invoked through session.prompt() / .skill() / .subsession().",
    );
  }
  return session as unknown as Session;
}

export function runWithSession<T>(session: AnySession, fn: () => T): T {
  return sessionStore.run(session, fn);
}
