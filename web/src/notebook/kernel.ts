// A notebook widget's data (ADR 0016): requests to its kernel over the widget's
// own channel, answered with the JSON the server would send for the same
// question. Branches and optics never change for a widget, so each is asked
// for once.

import type { BranchData, BranchOptics, CaptureEntry, EngineInfo, OrbitFrames } from "../api/client";
import type { ViewerData } from "../api/data";

/** What a widget's model offers (anywidget's): its values shared with the kernel, and messages. */
export interface WidgetModel {
  get(name: string): unknown;
  set(name: string, value: unknown): void;
  save_changes(): void;
  on(event: string, callback: (...args: unknown[]) => void): void;
  off(event: string, callback: (...args: unknown[]) => void): void;
  send(content: unknown): void;
}

/** A request the kernel could not answer, with its reason. */
export class KernelError extends Error {}

interface Answer {
  response: number;
  json?: string;
  error?: string;
}

const aborted = () => new DOMException("The request was cancelled", "AbortError");

/** The widget's data from its kernel; ``close`` stops listening and fails what is still waiting. */
export function kernelData(model: WidgetModel): ViewerData & { close(): void } {
  let next = 0;
  const waiting = new Map<number, { resolve(value: unknown): void; reject(error: Error): void }>();
  const kept = new Map<string, Promise<unknown>>();

  const listener = (...args: unknown[]) => {
    const answer = args[0] as Answer | undefined;
    if (typeof answer?.response !== "number") return;
    const found = waiting.get(answer.response);
    if (!found) return;
    waiting.delete(answer.response);
    if (answer.error !== undefined) found.reject(new KernelError(answer.error));
    else found.resolve(JSON.parse(answer.json ?? "null"));
  };
  model.on("msg:custom", listener);

  const ask = <T>(method: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      if (signal?.aborted) return reject(aborted());
      const id = ++next;
      waiting.set(id, { resolve: resolve as (value: unknown) => void, reject });
      signal?.addEventListener("abort", () => waiting.delete(id) && reject(aborted()), { once: true });
      model.send({ request: id, method, args });
    });

  /** Asked once; a failure is asked again next time. */
  const once = <T>(key: string, request: () => Promise<T>): Promise<T> => {
    let found = kept.get(key);
    if (!found) {
      found = request();
      found.catch(() => kept.delete(key));
      kept.set(key, found);
    }
    return found as Promise<T>;
  };

  return {
    engines: () => once("engines", () => ask<EngineInfo[]>("engines", {})),
    branch: (branch) => once(`branch/${branch}`, () => ask<BranchData>("branch", { branch })),
    optics: (branch, engine, fidelity, mode) =>
      once(`optics/${branch}/${engine}/${fidelity}/${mode}`, () =>
        ask<BranchOptics>("optics", { branch, engine, fidelity, mode }),
      ),
    orbits: (run, every, count, end, signal, source) =>
      ask<OrbitFrames>("orbits", { run, every, count, end: end ?? null, source }, signal),
    captures: (run, signal) => ask<CaptureEntry[]>("captures", { run }, signal),
    close() {
      model.off("msg:custom", listener);
      for (const request of waiting.values()) request.reject(new KernelError("The widget was closed"));
      waiting.clear();
    },
  };
}
