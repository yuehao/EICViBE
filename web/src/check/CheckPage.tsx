// The tracking check on a page of its own (check.html#run=<id>): every ring
// run on the server to choose from, and the room its charts need. The viewer's
// Check tab links here.

import { useEffect, useState } from "react";

import { api, type RunEntry } from "../api/client";
import { useAi } from "../components/Ai";
import { CheckView } from "../components/CheckView";
import { useColorScheme } from "../components/useCanvas";

const linked = () => new URLSearchParams(window.location.hash.slice(1)).get("run") ?? undefined;

export function CheckPage() {
  const scheme = useColorScheme();
  const aiStatus = useAi();
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    const load = () =>
      api.runs().then(
        (all) => {
          setRuns(all.filter((r) => r.kind === "ring"));
          setLoaded(true);
        },
        () => setLoaded(true),
      );
    load();
    const timer = window.setInterval(load, 10_000);
    return () => window.clearInterval(timer);
  }, []);
  return (
    <div className="check-page">
      <header className="topbar">
        <strong>EICViBE</strong>
        <span>Tracking check: a ring run's BPM data against the model</span>
        <a className="button" href="./">
          Viewer
        </a>
      </header>
      <main className="check-page-main">
        <CheckView
          runs={runs}
          empty={loaded ? "No ring runs on the server: start a machine, kick its beam, and check it here." : "Reading the runs…"}
          preferRun={linked()}
          scheme={scheme}
          ai={aiStatus}
        />
      </main>
    </div>
  );
}
