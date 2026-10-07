import { useEffect, useRef } from "react";

interface Props {
  /** CSS custom property on the app root that holds the size, e.g. "--side-width". */
  variable: string;
  orientation: "vertical" | "horizontal";
  /** +1 when dragging right/down grows the size, -1 when it shrinks it. */
  sign: 1 | -1;
  min: number;
  max: number;
  initial: number;
}

/** A drag handle that resizes panes through a CSS variable, remembered per browser. */
export function Splitter({ variable, orientation, sign, min, max, initial }: Props) {
  const handle = useRef<HTMLDivElement | null>(null);
  const storageKey = `eicvibe.layout${variable}`;

  useEffect(() => {
    const root = handle.current?.closest<HTMLElement>(".app");
    if (!root) return;
    let size = initial;
    try {
      size = Number(window.localStorage.getItem(storageKey)) || initial;
    } catch {
      // storage unavailable
    }
    root.style.setProperty(variable, `${Math.min(max, Math.max(min, size))}px`);
  }, [variable, storageKey, initial, min, max]);

  const start = (event: React.PointerEvent<HTMLDivElement>) => {
    const root = handle.current?.closest<HTMLElement>(".app");
    if (!root) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const origin = orientation === "vertical" ? event.clientX : event.clientY;
    const begin = parseFloat(getComputedStyle(root).getPropertyValue(variable)) || initial;
    let size = begin;
    const move = (e: PointerEvent) => {
      const delta = (orientation === "vertical" ? e.clientX : e.clientY) - origin;
      size = Math.min(max, Math.max(min, begin + sign * delta));
      root.style.setProperty(variable, `${size}px`);
    };
    const end = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", end);
      try {
        window.localStorage.setItem(storageKey, String(Math.round(size)));
      } catch {
        // storage unavailable
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
  };

  return <div ref={handle} className={`splitter ${orientation}`} onPointerDown={start} role="separator" aria-orientation={orientation} />;
}
