// Per-turn latency, measured by the AI SDK itself: `agent.stream`'s telemetry
// callbacks (`onStepEnd`, `onToolExecutionEnd`) already time every model call
// and tool execution — the recorder only aggregates them into one `TurnTimings`
// attached to the assistant message's `metadata.timings`. Riding in metadata
// (jsonb, alongside `createdAt`) means the numbers persist with the transcript
// and reach every reader — web UI, testing harness — with no extra plumbing.
//
// The voice core stamps the same shape on its turns (see ./voice),
// so text and voice turns are measured identically: what prompt dispatch is to a
// text turn, the end of the user's spoken turn is to a voice one. Voice answers
// arrive as several messages (a filler, a tool, the reply) and each measures its
// `totalMs` from that same start, so the last message of an answer is the one to
// compare against a text turn's.

/** One model call (step) inside a turn. */
export interface StepTiming {
  /** Time waiting for the language model response (SDK `responseTimeMs`). */
  responseMs: number;
  /** Model call start → first output chunk (SDK `timeToFirstOutputMs`). */
  ttftMs?: number;
}

/** One tool execution inside a turn (SDK `toolExecutionMs`). */
export interface ToolTiming {
  callId: string;
  name: string;
  ms: number;
}

/** Latency profile of one assistant turn. All durations are integer ms. */
export interface TurnTimings {
  /** First model call start → its first output chunk. Absent for non-streaming
   *  runs or when the model produced no output. */
  ttftMs?: number;
  /** Prompt dispatch → end of the run (all steps and tools included). */
  totalMs: number;
  /** Per model call, in order. Absent when no step settled (e.g. abort). */
  steps?: StepTiming[];
  /** Per tool execution, in completion order. Absent when no tool ran. */
  tools?: ToolTiming[];
}

export interface TimingsRecorder {
  /** Feed each step's SDK performance record (from `onStepEnd`). */
  step(performance: {
    responseTimeMs: number;
    timeToFirstOutputMs: number | undefined;
  }): void;
  /** Feed each tool execution (from `onToolExecutionEnd`). */
  tool(timing: { callId: string; name: string; ms: number }): void;
  /** Settle the recording — call when the run has finished. */
  finish(): TurnTimings;
}

/** Start recording a turn. The clock starts now (i.e. create the recorder at
 *  prompt dispatch); inject `now` for deterministic tests. */
export function createTimingsRecorder(
  now: () => number = () => performance.now(),
): TimingsRecorder {
  const startedAt = now();
  const steps: StepTiming[] = [];
  const tools: ToolTiming[] = [];

  return {
    step(performance) {
      steps.push({
        responseMs: Math.round(performance.responseTimeMs),
        ...(performance.timeToFirstOutputMs != null
          ? { ttftMs: Math.round(performance.timeToFirstOutputMs) }
          : {}),
      });
    },

    tool(timing) {
      tools.push({ ...timing, ms: Math.round(timing.ms) });
    },

    finish() {
      const firstStep = steps[0];
      return {
        ...(firstStep?.ttftMs != null ? { ttftMs: firstStep.ttftMs } : {}),
        totalMs: Math.round(now() - startedAt),
        ...(steps.length > 0 ? { steps } : {}),
        ...(tools.length > 0 ? { tools } : {}),
      };
    },
  };
}

/** `metadata` with `timings` merged in, preserving whatever else it carries. */
export function withTimings(
  metadata: unknown,
  timings: TurnTimings,
): Record<string, unknown> {
  return {
    ...(metadata && typeof metadata === "object"
      ? (metadata as Record<string, unknown>)
      : {}),
    timings,
  };
}
