// A minimal WebSocket surface shared by the server-side providers, so none of
// them depends on a particular runtime's WebSocket typings (Bun's constructor
// accepts options the DOM/undici lib types don't model) and so tests can
// inject a fake socket.
export interface WSLike {
  readyState: number;
  send(data: string): void;
  close(): void;
  addEventListener(type: "message", listener: (ev: { data: unknown }) => void): void;
  addEventListener(type: "close", listener: (ev?: WSCloseInfo) => void): void;
  addEventListener(type: "open" | "error", listener: () => void): void;
  removeEventListener(type: string, listener: (...args: never[]) => void): void;
}

/** The part of a CloseEvent a provider reports. */
export interface WSCloseInfo {
  code?: number;
  reason?: string;
}

/** The cause of a socket the host did not close: its code and reason, so an
 *  upstream drop (e.g. a gateway's 1011 "Upstream connection closed") reaches
 *  the host explained instead of as a bare disconnect. `fallback` stands in
 *  when the close says nothing (an earlier socket error's message). */
export function describeClose(
  label: string,
  ev: WSCloseInfo | undefined,
  fallback?: string,
): string {
  const parts = [
    ...(ev?.code !== undefined ? [`code=${ev.code}`] : []),
    ...(ev?.reason ? [`reason=${ev.reason}`] : []),
  ];
  if (parts.length > 0) return `${label} closed: ${parts.join(" ")}`;
  return fallback ? `${label} closed: ${fallback}` : `${label} closed unexpectedly`;
}
