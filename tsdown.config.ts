import { defineConfig } from "tsdown";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/codemode/index.ts",
    "src/voice/index.ts",
    "src/voice/protocol.ts",
    "src/voice/ui/index.ts",
    "src/voice/react/index.ts",
    "src/voice/providers/grok/index.ts",
    "src/voice/providers/gemini/index.ts",
    "src/voice/providers/gateway/index.ts",
  ],
  format: "esm",
  dts: true,
  clean: true,
  // Keep the published tree mirroring src/ so the exports map reads plainly.
  unbundle: true,
});
