import { describe, expect, it } from "vitest";

import { KernelError, kernelData, type WidgetModel } from "./kernel";

/** A widget's model without a kernel: what the widget sends is kept, and answers are given by hand. */
function fakeModel() {
  const listeners = new Map<string, Set<(...args: unknown[]) => void>>();
  const sent: { request: number; method: string; args: Record<string, unknown> }[] = [];
  const model: WidgetModel = {
    get: () => undefined,
    set: () => undefined,
    save_changes: () => undefined,
    on: (event, callback) => void listeners.set(event, (listeners.get(event) ?? new Set()).add(callback)),
    off: (event, callback) => void listeners.get(event)?.delete(callback),
    send: (content) => void sent.push(content as (typeof sent)[number]),
  };
  const answer = (content: unknown) => listeners.get("msg:custom")?.forEach((listener) => listener(content));
  return { model, sent, answer, listening: () => listeners.get("msg:custom")?.size ?? 0 };
}

describe("a widget's data from its kernel", () => {
  it("asks the kernel and reads its JSON answer", async () => {
    const { model, sent, answer } = fakeModel();
    const data = kernelData(model);
    const optics = data.optics("br_1", "xsuite", "standard", "");
    expect(sent).toEqual([{ request: 1, method: "optics", args: { branch: "br_1", engine: "xsuite", fidelity: "standard", mode: "" } }]);
    answer({ response: 1, json: '{"branch": "br_1", "tune_x": 2.21}' });
    expect(await optics).toEqual({ branch: "br_1", tune_x: 2.21 });
  });

  it("asks once for a branch or optics, again after a failure", async () => {
    const { model, sent, answer } = fakeModel();
    const data = kernelData(model);
    const first = data.branch("br_1");
    expect(data.branch("br_1")).toBe(first);
    answer({ response: 1, error: "no branch br_1" });
    await expect(first).rejects.toThrow(KernelError);
    await expect(first).rejects.toThrow("no branch br_1");
    void data.branch("br_1");
    expect(sent.map((m) => m.request)).toEqual([1, 2]);
  });

  it("asks for orbits every time, and a cancelled request ends as fetch's do", async () => {
    const { model, sent, answer } = fakeModel();
    const data = kernelData(model);
    const controller = new AbortController();
    const orbits = data.orbits("run-1", 5, 5, undefined, controller.signal, "buffer");
    expect(sent[0]!.args).toEqual({ run: "run-1", every: 5, count: 5, end: null, source: "buffer" });
    controller.abort();
    await expect(orbits).rejects.toMatchObject({ name: "AbortError" });
    answer({ response: 1, json: "{}" }); // late: nobody waits for it
    void data.orbits("run-1", 5, 5, 100, undefined, "buffer");
    expect(sent.length).toBe(2);
  });

  it("fails what still waits when the widget closes, and stops listening", async () => {
    const { model, answer, listening } = fakeModel();
    const data = kernelData(model);
    const engines = data.engines();
    expect(listening()).toBe(1);
    data.close();
    await expect(engines).rejects.toThrow("The widget was closed");
    expect(listening()).toBe(0);
    answer({ response: 1, json: "[]" });
  });

  it("ignores messages that answer nothing it asked", async () => {
    const { model, answer } = fakeModel();
    const data = kernelData(model);
    const captures = data.captures("run-1");
    answer({ something: "else" });
    answer({ response: 99, json: "[]" });
    answer({ response: 1, json: '[{"id": "c1"}]' });
    expect(await captures).toEqual([{ id: "c1" }]);
  });
});
