// The administration page's AI section (ADR 0009): which model answers the
// viewer's AI features. A local model (Ollama) by default; OpenAI or Anthropic
// with an API key. A key is sent to the server, kept there for its owner only,
// and never shown again; every change goes into the action log.

import { useCallback, useEffect, useState } from "react";

import { ai, type AiStatus } from "../api/ai";
import { va, type AiChange, type AiSettings, type AiTest } from "../api/va";

const PROVIDERS = ["local", "openai", "anthropic"] as const;
type Provider = (typeof PROVIDERS)[number];

export function AiPanel({ onError }: { onError(e: unknown): void }) {
  const [settings, setSettings] = useState<AiSettings | null>(null);
  const [status, setStatus] = useState<AiStatus | null>(null);
  const [enabled, setEnabled] = useState(true);
  const [provider, setProvider] = useState<Provider>("local");
  const [model, setModel] = useState("");
  const [url, setUrl] = useState("");
  const [key, setKey] = useState("");
  const [clearKey, setClearKey] = useState(false);
  const [reason, setReason] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [test, setTest] = useState<AiTest | null>(null);
  const [testing, setTesting] = useState(false);
  const [saved, setSaved] = useState(false);

  const load = useCallback(() => {
    va.ai().then((s) => {
      setSettings(s);
      setEnabled(s.enabled);
      setProvider(s.provider);
      setModel(s.model);
      setUrl(s.url);
      setKey("");
      setClearKey(false);
    }, onError);
    ai.status().then(setStatus, () => setStatus(null));
  }, [onError]);
  useEffect(load, [load]);

  if (!settings) return <section>Loading…</section>;
  const info = settings.providers[provider]!;
  const remote = provider !== "local";
  const change = (): AiChange => ({ enabled, provider, model, url, api_key: key || null, clear_key: clearKey, reason });
  const choose = (next: Provider) => {
    setProvider(next);
    setModel(next === settings.provider ? settings.model : "");
    setUrl(next === settings.provider ? settings.url : "");
    setKey("");
    setClearKey(false);
    setModels([]);
    setTest(null);
  };
  const keyNow =
    provider !== settings.provider
      ? "none stored yet for this provider, unless its environment variable is set"
      : settings.key === "stored"
        ? `stored (${settings.key_hint ?? "set"})`
        : settings.key
          ? `from the server's environment variable ${settings.key}`
          : "none";

  return (
    <section data-testid="ai-settings">
      <h2>AI</h2>
      <p className="muted">
        The viewer's ✦ Ask and the Check tab's ✦ Explain: a model writes a few sentences from EICViBE's own numbers, in seconds. It never changes
        anything. Answers are labelled with the model and where it runs.
      </p>
      <p data-testid="ai-now">
        Now:{" "}
        {!status
          ? "unknown"
          : !status.enabled
            ? "off"
            : `${status.model ?? "a model"} ${status.where ?? ""}: ${status.available ? "answers" : status.reason}`}
        {settings.saved ? "" : " (the server's defaults; nothing chosen here yet)"}
      </p>
      <form
        className="admin-form ai-form"
        aria-label="AI model"
        onSubmit={(e) => {
          e.preventDefault();
          va.setAi(change()).then(() => {
            setSaved(true);
            setReason("");
            load();
          }, onError);
        }}
      >
        <label>
          <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} /> AI features on
        </label>
        <fieldset disabled={!enabled}>
          <legend>Model</legend>
          {PROVIDERS.map((p) => (
            <label key={p}>
              <input type="radio" name="provider" value={p} checked={provider === p} onChange={() => choose(p)} />{" "}
              {settings.providers[p]!.label}
              {p === "local" ? ", the default" : ""}
            </label>
          ))}
          <label>
            URL{" "}
            <input aria-label="URL" value={url} placeholder={info.url} onChange={(e) => setUrl(e.target.value)} />
          </label>
          {remote ? (
            <>
              <label>
                API key{" "}
                <input
                  type="password"
                  aria-label="API key"
                  autoComplete="off"
                  value={key}
                  placeholder={settings.key === "stored" && provider === settings.provider ? "type to replace the stored key" : "paste the key"}
                  onChange={(e) => setKey(e.target.value)}
                />
              </label>
              <span className="muted">Key: {keyNow}.</span>
              {settings.key === "stored" && provider === settings.provider ? (
                <label>
                  <input type="checkbox" checked={clearKey} onChange={(e) => setClearKey(e.target.checked)} /> remove the stored key
                </label>
              ) : null}
            </>
          ) : null}
          <label>
            Model name{" "}
            <input
              aria-label="Model name"
              list="ai-models"
              value={model}
              placeholder={info.model || (provider === "local" ? "the server's first model" : "choose one: Test lists them")}
              onChange={(e) => setModel(e.target.value)}
            />
            <datalist id="ai-models">
              {models.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </label>
          <p className="muted ai-note" data-testid="ai-note">
            {remote
              ? `The viewer's questions, and the facts EICViBE gathers for them (lattice, settings, run results), are sent to ${
                  provider === "openai" ? "OpenAI" : "Anthropic"
                }. A ChatGPT or Claude subscription does not let other applications use its models: use an API key from ${
                  provider === "openai" ? "platform.openai.com" : "console.anthropic.com"
                }${info.key_variable ? `, here or in the server's ${info.key_variable}` : ""}.`
              : "A model served by Ollama, on this server or one of the lab's: nothing leaves the host the URL names. Answers take seconds; one question at a time."}
          </p>
        </fieldset>
        <div className="admin-row">
          <button
            type="button"
            disabled={testing || !enabled}
            onClick={() => {
              setTesting(true);
              setTest(null);
              va.testAi(change()).then(
                (t) => {
                  setTest(t);
                  setModels(t.models);
                  setTesting(false);
                },
                (e) => {
                  setTesting(false);
                  onError(e);
                },
              );
            }}
          >
            {testing ? "Testing…" : "Test"}
          </button>
          <input aria-label="Reason" placeholder="reason (goes into the log)" value={reason} onChange={(e) => setReason(e.target.value)} />
          <button type="submit">Save</button>
          {saved ? <span className="muted">Saved.</span> : null}
        </div>
        {test ? (
          <p className={test.available ? "admin-outcome ok" : "admin-error"} data-testid="ai-test">
            {test.available
              ? `Answers: "${test.answer}" in ${test.seconds} s. ${test.reason}.`
              : `Does not answer: ${test.reason}.`}
            {test.models.length ? ` It offers ${test.models.length} model${test.models.length === 1 ? "" : "s"}; choose one under Model name.` : ""}
          </p>
        ) : null}
      </form>
    </section>
  );
}
