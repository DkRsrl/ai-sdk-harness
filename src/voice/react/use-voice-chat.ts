import type { UIMessage } from "ai";
import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";
import type { VoiceStatus } from "../spec";
import { VoiceChat, type VoiceChatOptions } from "../ui";

export type UseVoiceChatOptions = VoiceChatOptions & {
  /** Throttle message-list re-renders to at most once per N ms. Streaming
   *  transcripts update very frequently; this coalesces them. */
  experimental_throttle?: number;
};

export interface UseVoiceChatResult<UI_MESSAGE extends UIMessage = UIMessage> {
  /** The session/chat id this conversation is bound to. */
  id: string;
  /** The single derived status — the conversation lifecycle with transport
   *  health folded in (`idle` → `connecting` → `listening`/`speaking`/`thinking`
   *  → `idle`/`error`). */
  status: VoiceStatus;
  /** The live transcript as AI-SDK `UIMessage`s — same shape `useChat` returns. */
  messages: UI_MESSAGE[];
  /** Whether the mic is muted. */
  muted: boolean;
  /** The last error, if any. */
  error: string | null;
  /** Open the connection and go live. */
  start: () => Promise<void>;
  /** End the call. */
  stop: () => void;
  setMuted: (muted: boolean) => void;
  toggleMute: () => void;
  /** Read the mic level (0..1) imperatively — call from a rAF visualizer. It is
   *  deliberately not reactive state, which would re-render every frame. */
  inputLevel: () => number;
  /** Read the assistant output level (0..1) imperatively. */
  outputLevel: () => number;
  /** The underlying instance, for escape hatches. */
  chat: VoiceChat<UI_MESSAGE>;
}

/**
 * React binding for {@link VoiceChat} — the voice analogue of `useChat`. Pass
 * options (a `VoiceChat` is created once and owned by the hook) or an existing
 * `VoiceChat` you manage yourself. State is read with per-field
 * `useSyncExternalStore` (messages separate from status) so renders are granular;
 * audio levels are read imperatively. Type tool messages by asserting `TMessage`
 * as `RealtimeMessage<typeof tools>` (the wire is untyped JSON).
 */
export function useVoiceChat<UI_MESSAGE extends UIMessage = UIMessage>(
  options: VoiceChat<UI_MESSAGE> | UseVoiceChatOptions,
): UseVoiceChatResult<UI_MESSAGE> {
  const chatRef = useRef<VoiceChat<UI_MESSAGE> | null>(null);
  if (chatRef.current === null) {
    chatRef.current =
      options instanceof VoiceChat ? options : new VoiceChat<UI_MESSAGE>(options);
  } else if (options instanceof VoiceChat && options !== chatRef.current) {
    chatRef.current = options;
  }
  const chat = chatRef.current;
  const owned = !(options instanceof VoiceChat);
  const throttleMs = options instanceof VoiceChat ? undefined : options.experimental_throttle;

  const subscribeMessages = useCallback(
    (update: () => void) => chat.subscribeMessages(update, throttleMs),
    [chat, throttleMs],
  );
  const messages = useSyncExternalStore(
    subscribeMessages,
    () => chat.messages,
    () => chat.messages,
  );
  const snapshot = useSyncExternalStore(
    chat.subscribeStatus,
    () => chat.snapshot,
    () => chat.snapshot,
  );

  useEffect(() => {
    // The hook owns the instance it created — tear the call down on unmount so
    // the mic doesn't stay open. A caller-supplied instance is theirs to manage.
    if (!owned) return;
    return () => chat.stop();
  }, [chat, owned]);

  const start = useCallback(() => chat.start(), [chat]);
  const stop = useCallback(() => chat.stop(), [chat]);
  const setMuted = useCallback((m: boolean) => chat.setMuted(m), [chat]);
  const toggleMute = useCallback(() => chat.setMuted(!chat.muted), [chat]);
  const inputLevel = useCallback(() => chat.inputLevel(), [chat]);
  const outputLevel = useCallback(() => chat.outputLevel(), [chat]);

  return {
    id: chat.id,
    status: snapshot.status,
    muted: snapshot.muted,
    error: snapshot.error,
    messages,
    start,
    stop,
    setMuted,
    toggleMute,
    inputLevel,
    outputLevel,
    chat,
  };
}
