import assert from "node:assert/strict";
import { test } from "node:test";
import { createTimingsRecorder, withTimings } from "./timings";

test("recorder aggregates SDK step/tool telemetry into TurnTimings", () => {
  let t = 1000;
  const recorder = createTimingsRecorder(() => t);

  recorder.step({ responseTimeMs: 120.4, timeToFirstOutputMs: 45.6 });
  recorder.tool({ callId: "call-1", name: "search", ms: 300.2 });
  recorder.step({ responseTimeMs: 80, timeToFirstOutputMs: 30 });
  t = 1000 + 512;

  assert.deepEqual(recorder.finish(), {
    ttftMs: 46, // first step's time-to-first-output
    totalMs: 512,
    steps: [
      { responseMs: 120, ttftMs: 46 },
      { responseMs: 80, ttftMs: 30 },
    ],
    tools: [{ callId: "call-1", name: "search", ms: 300 }],
  });
});

test("a run with no settled steps or tools yields only totalMs", () => {
  let t = 0;
  const recorder = createTimingsRecorder(() => t);
  t = 42;
  assert.deepEqual(recorder.finish(), { totalMs: 42 });
});

test("a non-streaming step (no time-to-first-output) omits ttftMs", () => {
  const recorder = createTimingsRecorder(() => 0);
  recorder.step({ responseTimeMs: 200, timeToFirstOutputMs: undefined });
  const timings = recorder.finish();
  assert.equal(timings.ttftMs, undefined);
  assert.deepEqual(timings.steps, [{ responseMs: 200 }]);
});

test("withTimings merges into existing metadata without clobbering it", () => {
  const merged = withTimings({ createdAt: 123, custom: "x" }, { totalMs: 9 });
  assert.deepEqual(merged, {
    createdAt: 123,
    custom: "x",
    timings: { totalMs: 9 },
  });
});

test("withTimings tolerates absent or non-object metadata", () => {
  assert.deepEqual(withTimings(undefined, { totalMs: 1 }), {
    timings: { totalMs: 1 },
  });
  assert.deepEqual(withTimings("junk", { totalMs: 1 }), {
    timings: { totalMs: 1 },
  });
});
