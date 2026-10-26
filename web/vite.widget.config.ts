import react from "@vitejs/plugin-react";
import type { AtRule, Plugin, Rule } from "postcss";
import { defineConfig } from "vite";

// The notebook widgets' module (ADR 0016): one ES module and one stylesheet, served by the Python
// package (eicvibe.notebook) to anywidget. React is bundled in: a notebook has none to share.

const SCOPE = ".eicvibe-widget";

/**
 * The page's stylesheet, made the widget's: every rule only within a widget, the page's root
 * (``:root``, ``html``, ``body``) the widget's own element, and the dark scheme the widget's
 * ``data-scheme="dark"`` (its notebook's theme) rather than the system's.
 */
function scopeToWidget(): Plugin {
  const scoped = (selector: string, root: string) =>
    selector
      .split(",")
      .map((part) => part.trim())
      .map((part) =>
        /^(:root|html|body|#root)$/.test(part) ? root : part === "*" ? `${root} *` : `${root} ${part}`,
      )
      .join(", ");
  return {
    postcssPlugin: "eicvibe-scope-to-widget",
    Once(root) {
      if (!root.source?.input.file?.endsWith("styles.css")) return; // the widget's own rules are scoped
      root.walkAtRules("media", (media: AtRule) => {
        if (!media.params.includes("prefers-color-scheme: dark")) return;
        media.walkRules((rule: Rule) => {
          rule.selector = scoped(rule.selector, `${SCOPE}[data-scheme="dark"]`);
        });
        media.replaceWith(media.nodes ?? []);
      });
      root.walkRules((rule: Rule) => {
        if (rule.parent?.type === "atrule" && /keyframes/.test((rule.parent as AtRule).name)) return;
        if (rule.selector.startsWith(SCOPE)) return;
        rule.selector = scoped(rule.selector, SCOPE);
      });
    },
  };
}

export default defineConfig({
  plugins: [react()],
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  css: { postcss: { plugins: [scopeToWidget()] } },
  build: {
    outDir: "../src/eicvibe/notebook/static",
    emptyOutDir: true,
    lib: { entry: "src/notebook/widget.tsx", formats: ["es"], fileName: () => "widget.js", cssFileName: "widget" },
  },
});
