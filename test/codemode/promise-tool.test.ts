import { describe, expect, test } from "bun:test"
import { makeRuntime, promiseTool, toolError } from "../../src/codemode/index"

// The Promise face is what a host app integrates through, and `toolError` is
// the one thing it asks that app to import. Pin the refusal contract here so
// the published `@dkrdev/ai-sdk-harness/codemode` entrypoint keeps it.

const run = (execute: () => Promise<unknown>) =>
  makeRuntime({
    tools: {
      host: {
        call: promiseTool({
          description: "Call the host",
          input: { type: "object", properties: {}, additionalProperties: false },
          execute,
        }),
      },
    },
  }).execute("return await tools.host.call({})")

describe("promiseTool host failure boundary", () => {
  test("a thrown toolError reaches the model verbatim", async () => {
    const result = await run(async () => {
      throw toolError("No registry data found for client A123.")
    })

    expect(result.ok ? undefined : result.error).toStrictEqual({
      kind: "ToolFailure",
      message: "No registry data found for client A123.",
    })
  })

  test("any other thrown value is sanitized", async () => {
    const result = await run(async () => {
      throw new Error("connect ECONNREFUSED 10.0.0.4:5432")
    })

    const error = result.ok ? undefined : result.error
    expect(error?.kind).toBe("ToolFailure")
    expect(error?.message).not.toContain("10.0.0.4")
  })
})
