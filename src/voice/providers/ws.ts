// A minimal WebSocket surface shared by the server-side providers, so none of
// them depends on a particular runtime's WebSocket typings (Bun's constructor
// accepts options the DOM/undici lib types don't model) and so tests can
// inject a fake socket.
export interface WSLike {
  readyState: number;
  send(data: string): void;
  close(): void;
  addEventListener(type: "message", listener: (ev: { data: unknown }) => void): void;
  addEventListener(type: "open" | "close" | "error", listener: () => void): void;
  removeEventListener(type: string, listener: (...args: never[]) => void): void;
}
