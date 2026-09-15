import type { Tool } from "./tool"

export type Tools<R = never> = {
  readonly [name: string]: Tool<R> | Tools<R>
}
