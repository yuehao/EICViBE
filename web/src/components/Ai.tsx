// The viewer's AI features (ADR 0009): a model's short answers, streamed as it
// writes them, always labelled with the model and where it runs. They are on by
// default (a local model); when no model answers they say why, and when an
// administrator turns them off they are gone. Nothing else depends on them.

import { useCallback, useEffect, useRef, useState } from "react";

import { ai, type AiStatus, type AskRequest, type ExplainCheck } from "../api/ai";

/** The AI features' status while they are on (null when off): checked on load and once a minute. */
export function useAi(): AiStatus | null {
  const [status, setStatus] = useState<AiStatus | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    const check = () => ai.status(controller.signal).then(setStatus, () => setStatus(null));
    check();
    const timer = window.setInterval(check, 60_000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);
  return status?.enabled ? status : null;
}

/** "qwen3.8:27b-mlx, on this server" or "claude-opus-5-5, by Anthropic": who writes the answers. */
export const aiName = (status: AiStatus) => [status.model ?? "the model", status.where].filter(Boolean).join(", ");

/** Ask once and show the answer as it comes, with a Stop while it does. */
export function useAnswer() {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const stop = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    setBusy(false);
  }, []);
  const start = useCallback(
    (path: "ask" | "explain-check", body: AskRequest | ExplainCheck) => {
      controller.current?.abort();
      const own = new AbortController();
      controller.current = own;
      setText("");
      setError(null);
      setBusy(true);
      ai.stream(path, body, setText, own.signal).then(
        () => {
          if (controller.current === own) setBusy(false);
        },
        (e: Error) => {
          if (e.name === "AbortError") return;
          setError(e.message);
          setBusy(false);
        },
      );
    },
    [],
  );
  useEffect(() => () => controller.current?.abort(), []);
  const clear = useCallback(() => {
    stop();
    setText("");
    setError(null);
  }, [stop]);
  return { text, error, busy, start, stop, clear };
}

/** An answer from the model: its words, what wrote them, and that they may be wrong. */
export function AiAnswer(props: { status: AiStatus; text: string; error: string | null; busy: boolean; onStop(): void; testId?: string }) {
  if (!props.text && !props.error && !props.busy) return null;
  const sent = props.status.provider !== "local";
  return (
    <div className="ai-answer" data-testid={props.testId} aria-live="polite">
      <div className="ai-label">
        <span
          title={`A language model wrote this from EICViBE's numbers${sent ? ", which were sent to it with the question" : ""}. It can be wrong: check it against them.`}
        >
          ✦ {aiName(props.status)}: may be wrong
        </span>
        {props.busy ? (
          <button type="button" className="link" onClick={props.onStop}>
            Stop
          </button>
        ) : null}
      </div>
      {props.error ? <p className="admin-error">{props.error}</p> : <p>{props.text || "…"}</p>}
    </div>
  );
}

/** The top bar's Ask: a question about what the viewer shows, answered in a few seconds. */
export function AskAi(props: { status: AiStatus; context(): Omit<AskRequest, "question"> | null }) {
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const answer = useAnswer();
  const ask = () => {
    const context = props.context();
    if (!context || !question.trim()) return;
    answer.start("ask", { ...context, question: question.trim() });
  };
  return (
    <div className="ai-ask">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        title={props.status.available ? `Ask ${aiName(props.status)} about what the viewer shows` : `No model answers: ${props.status.reason}`}
      >
        ✦ Ask
      </button>
      {open ? (
        <div className="ai-popover" role="dialog" aria-label="Ask the model">
          <textarea
            aria-label="Question"
            rows={2}
            maxLength={500}
            placeholder="e.g. What is the selected element? Is this ring's chromaticity corrected?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask();
              }
            }}
          />
          <div className="ai-row">
            <span className="muted">
              {props.status.available
                ? `About the branch shown, the selected element and the optics computed; answered by ${aiName(props.status)}.`
                : `No model answers now (${props.status.reason}). An administrator can set one up.`}
            </span>
            <button type="button" onClick={ask} disabled={!question.trim() || answer.busy || !props.status.available}>
              Ask
            </button>
          </div>
          <AiAnswer status={props.status} {...answer} onStop={answer.stop} testId="ai-ask-answer" />
        </div>
      ) : null}
    </div>
  );
}
