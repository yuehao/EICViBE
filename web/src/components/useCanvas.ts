import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

export interface CanvasSize {
  width: number;
  height: number;
}

/**
 * A canvas that tracks its CSS size at device-pixel resolution and redraws at
 * most once per animation frame, however often ``requestDraw`` is called.
 * ``draw`` receives a context scaled to CSS pixels.
 */
export function useCanvas(draw: (ctx: CanvasRenderingContext2D, size: CanvasSize) => void) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawRef = useRef(draw);
  const frame = useRef<number | null>(null);
  const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0 });
  const sizeRef = useRef(size);
  useLayoutEffect(() => {
    drawRef.current = draw;
  });

  const paint = useCallback(() => {
    frame.current = null;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const { width, height } = sizeRef.current;
    if (!canvas || !ctx || width === 0 || height === 0) return;
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    drawRef.current(ctx, sizeRef.current);
  }, []);

  const requestDraw = useCallback(() => {
    if (frame.current === null) frame.current = requestAnimationFrame(paint);
  }, [paint]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const width = Math.round(entry.contentRect.width);
      const height = Math.round(entry.contentRect.height);
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(width * ratio));
      canvas.height = Math.max(1, Math.round(height * ratio));
      sizeRef.current = { width, height };
      setSize(sizeRef.current);
      paint(); // resizing cleared the canvas; repaint before the browser shows it
    });
    observer.observe(canvas);
    return () => {
      observer.disconnect();
      if (frame.current !== null) cancelAnimationFrame(frame.current);
    };
  }, [paint]);

  return { canvasRef, size, sizeRef, requestDraw };
}

const DARK = "(prefers-color-scheme: dark)";

/**
 * Theme colours from CSS custom properties, re-read when the colour scheme changes. The scheme is
 * the system's, unless an element around this one says otherwise (``data-scheme``: a notebook's
 * widget follows its notebook's theme).
 */
export function readTheme(element: Element | null) {
  const style = element ? getComputedStyle(element) : null;
  const get = (name: string, fallback: string) => style?.getPropertyValue(name).trim() || fallback;
  const scheme = element?.closest<HTMLElement>("[data-scheme]")?.dataset.scheme;
  return {
    dark: scheme ? scheme === "dark" : window.matchMedia(DARK).matches,
    background: get("--canvas-bg", "#ffffff"),
    text: get("--text", "#1d2330"),
    muted: get("--muted", "#6b7280"),
    grid: get("--grid", "#e7e9ee"),
    axis: get("--axis", "#9aa0a6"),
    highlight: get("--highlight", "#ff9800"),
    selection: get("--selection", "#111111"),
    danger: get("--danger", "#d32f2f"),
  };
}

export type Theme = ReturnType<typeof readTheme>;

export function useColorScheme(): string {
  const [scheme, setScheme] = useState(() => (window.matchMedia(DARK).matches ? "dark" : "light"));
  useEffect(() => {
    const query = window.matchMedia(DARK);
    const listener = (event: MediaQueryListEvent) => setScheme(event.matches ? "dark" : "light");
    query.addEventListener("change", listener);
    return () => query.removeEventListener("change", listener);
  }, []);
  return scheme;
}
