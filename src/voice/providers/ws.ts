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

/** The message for a socket the host did not close: its code and reason, so
 *  an upstream drop (e.g. a gateway's 1011 "Upstream connection closed")
 *  reaches the host with its cause instead of as a bare disconnect. */
export function describeClose(label: string, ev: WSCloseInfo | undefined): string {
  const parts = [
    ...(ev?.code !== undefined ? [`code=${ev.code}`] : []),
    ...(ev?.reason ? [`reason=${ev.reason}`] : []),
  ];
  return parts.length > 0 ? `${label} closed: ${parts.join(" ")}` : `${label} closed unexpectedly`;
}
