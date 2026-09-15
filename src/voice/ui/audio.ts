// Browser audio I/O for VoiceChat: AudioWorklet mic capture and PCM playback,
// raw little-endian int16 ArrayBuffers in and out (the wire protocol's binary
// frames). Hidden behind the VoiceAudio interface so VoiceChat stays framework-
// and platform-agnostic — and unit-testable with a stub.

export interface VoiceAudio {
  /** Begin mic capture; `onFrame` receives raw little-endian int16 PCM. */
  start(onFrame: (pcm: ArrayBuffer) => void): Promise<void>;
  /** Enqueue an assistant PCM frame for playback. */
  play(pcm: ArrayBuffer): void;
  /** Drop any assistant audio still scheduled for playback (barge-in). Unlike
   *  `stop()`, the mic and output context stay live for the next turn. */
  clearPlayback?(): void;
  /** Stop capture + playback and release the mic. */
  stop(): void;
  /** Mute the mic (keeps the worklet running so server VAD still sees silence). */
  setMuted(muted: boolean): void;
  /** Mic input level, 0..1 (0 when muted/inactive). */
  inputLevel(): number;
  /** Assistant output level, 0..1 (0 when silent). */
  outputLevel(): number;
  /** Whether assistant audio is still scheduled/playing (buffer not drained).
   *  Lets the caller keep "speaking" until playback truly ends, even after the
   *  provider has stopped sending. */
  isPlaying?(): boolean;
  /** Register a handler fired whenever the playback queue drains to empty. */
  onPlaybackEnd?(handler: () => void): void;
}

export interface BrowserAudioOptions {
  /** Capture + playback sample rate (Hz). Must match the server model. Default 24000. */
  sampleRate?: number;
  /** getUserMedia audio constraints. */
  constraints?: MediaStreamConstraints["audio"];
}

const WORKLET_BUFFER_SIZE = 2048;

const workletSource = (name: string) => `
class P extends AudioWorkletProcessor {
  buf = new Int16Array(${WORKLET_BUFFER_SIZE});
  i = 0;
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    for (let j = 0; j < ch.length; j++) {
      this.buf[this.i++] = Math.max(-32768, Math.min(32767, ch[j] * 32767));
      if (this.i >= this.buf.length) {
        this.port.postMessage(this.buf.slice(0, this.i).buffer);
        this.i = 0;
      }
    }
    return true;
  }
}
registerProcessor("${name}", P);
`;

function pcm16ToFloat32(pcm: ArrayBuffer): Float32Array {
  const view = new DataView(pcm);
  const out = new Float32Array(pcm.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

function rms(analyser: AnalyserNode, buf: Uint8Array<ArrayBuffer>): number {
  analyser.getByteFrequencyData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] ?? 0;
  return sum / (buf.length * 255);
}

export function createBrowserAudio(options: BrowserAudioOptions = {}): VoiceAudio {
  const sampleRate = options.sampleRate ?? 24_000;
  const constraints = options.constraints ?? {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  };

  let stream: MediaStream | null = null;
  let micCtx: AudioContext | null = null;
  let worklet: AudioWorkletNode | null = null;
  let micAnalyser: AnalyserNode | null = null;
  let muted = false;
  const micBuf = new Uint8Array(128);

  let outCtx: AudioContext | null = null;
  let outAnalyser: AnalyserNode | null = null;
  let nextStart = 0;
  const live: AudioBufferSourceNode[] = [];
  const outBuf = new Uint8Array(128);
  let playbackEndHandler: (() => void) | null = null;

  function ensureOut() {
    if (!outCtx || outCtx.state === "closed") {
      outCtx = new AudioContext();
      outAnalyser = outCtx.createAnalyser();
      outAnalyser.fftSize = 256;
      outAnalyser.connect(outCtx.destination);
      void outCtx.resume().catch(() => {});
      nextStart = outCtx.currentTime;
    }
    return { ctx: outCtx, analyser: outAnalyser as AnalyserNode };
  }

  return {
    async start(onFrame) {
      stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
      stream.getAudioTracks().forEach((t) => (t.enabled = !muted));
      micCtx = new AudioContext({ sampleRate });
      const source = micCtx.createMediaStreamSource(stream);
      micAnalyser = micCtx.createAnalyser();
      micAnalyser.fftSize = 256;
      const blob = new Blob([workletSource("voice-capture")], {
        type: "application/javascript",
      });
      const url = URL.createObjectURL(blob);
      await micCtx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      worklet = new AudioWorkletNode(micCtx, "voice-capture");
      worklet.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        if (e.data) onFrame(e.data);
      };
      source.connect(micAnalyser);
      micAnalyser.connect(worklet);
    },

    play(pcm) {
      const { ctx, analyser } = ensureOut();
      const float32 = pcm16ToFloat32(pcm);
      const buffer = ctx.createBuffer(1, float32.length, sampleRate);
      buffer.getChannelData(0).set(float32);
      const node = ctx.createBufferSource();
      node.buffer = buffer;
      node.connect(analyser);
      if (nextStart < ctx.currentTime) nextStart = ctx.currentTime;
      node.start(nextStart);
      nextStart += buffer.duration;
      live.push(node);
      node.onended = () => {
        const i = live.indexOf(node);
        if (i !== -1) live.splice(i, 1);
        if (live.length === 0) playbackEndHandler?.();
      };
    },

    clearPlayback() {
      for (const n of live) {
        try {
          n.stop();
        } catch {
          /* already stopped */
        }
      }
      live.length = 0;
      nextStart = outCtx?.currentTime ?? 0;
    },

    stop() {
      if (worklet) worklet.port.onmessage = null;
      worklet?.disconnect();
      micAnalyser?.disconnect();
      micCtx?.close().catch(() => {});
      worklet = micAnalyser = micCtx = null;
      stream?.getTracks().forEach((t) => t.stop());
      stream = null;
      for (const n of live) {
        try {
          n.stop();
        } catch {
          /* already stopped */
        }
      }
      live.length = 0;
      outCtx?.close().catch(() => {});
      outCtx = outAnalyser = null;
      nextStart = 0;
    },

    setMuted(next) {
      muted = next;
      stream?.getAudioTracks().forEach((t) => (t.enabled = !next));
    },

    inputLevel() {
      if (!micAnalyser || muted) return 0;
      return rms(micAnalyser, micBuf);
    },

    outputLevel() {
      if (!outAnalyser || live.length === 0) return 0;
      return rms(outAnalyser, outBuf);
    },

    isPlaying() {
      return live.length > 0;
    },

    onPlaybackEnd(handler) {
      playbackEndHandler = handler;
    },
  };
}
