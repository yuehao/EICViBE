// The optional local model (ADR 0009): its status, and answers streamed as it writes them.

import type { components } from "./schema";

export type AiStatus = components["schemas"]["AiStatus"];
export type AskRequest = components["schemas"]["AskRequest"];
export type ExplainCheck = components["schemas"]["ExplainCheck"];

const BASE = "/api/v1";

export const ai = {
  status: (signal?: AbortSignal): Promise<AiStatus> =>
    fetch(`${BASE}/ai`, { signal }).then((r) => (r.ok ? (r.json() as Promise<AiStatus>) : Promise.reject(new Error(r.statusText)))),

  /** POST ``body`` and call ``onText`` with the answer so far, piece by piece; resolves with the whole answer. */
  async stream(path: "ask" | "explain-check", body: AskRequest | ExplainCheck, onText: (text: string) => void, signal?: AbortSignal): Promise<string> {
    const response = await fetch(`${BASE}/ai/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!response.ok || !response.body) {
      let detail = response.statusText;
      try {
        detail = ((await response.json()) as { detail?: string }).detail ?? detail;
      } catch {
        // not JSON
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let text = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      onText(text);
    }
    return text;
  },
};
