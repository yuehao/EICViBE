// The colour scheme of the notebook a widget is in: JupyterLab's theme, VS Code's,
// or else the system's. A notebook may be dark on a light system, or the other way.

import { useEffect, useState, useSyncExternalStore } from "react";

import type { WidgetModel } from "./kernel";

const DARK = "(prefers-color-scheme: dark)";

export function notebookScheme(): "light" | "dark" {
  const body = document.body;
  const lab = body.dataset.jpThemeLight;
  if (lab === "true" || lab === "false") return lab === "true" ? "light" : "dark";
  if (body.classList.contains("vscode-dark") || body.classList.contains("vscode-high-contrast")) return "dark";
  if (body.classList.contains("vscode-light")) return "light";
  return window.matchMedia(DARK).matches ? "dark" : "light";
}

/** The notebook's scheme, following its theme as it changes. */
export function useNotebookScheme(): "light" | "dark" {
  const [scheme, setScheme] = useState(notebookScheme);
  useEffect(() => {
    const update = () => setScheme(notebookScheme());
    const observer = new MutationObserver(update);
    observer.observe(document.body, { attributes: true, attributeFilter: ["class", "data-jp-theme-light"] });
    const query = window.matchMedia(DARK);
    query.addEventListener("change", update);
    return () => {
      observer.disconnect();
      query.removeEventListener("change", update);
    };
  }, []);
  return scheme;
}

/** A value the widget shares with its kernel, re-rendering as either side changes it. */
export function useTrait<T>(model: WidgetModel, name: string): T {
  return useSyncExternalStore(
    (listener) => {
      model.on(`change:${name}`, listener);
      return () => model.off(`change:${name}`, listener);
    },
    () => model.get(name) as T,
  );
}
