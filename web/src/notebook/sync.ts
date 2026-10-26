// What a notebook view shares with its kernel, and through it or a link with
// other views (ADR 0016): the branch shown, the occurrence selected and the s
// range. Hover stays within a view: it changes on every pointer move.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { BranchModel } from "../model/branch";
import { targetIn } from "../model/route";
import { createStore, sameTarget, type Target } from "../model/store";
import { clampRange, createZoomHistory, type SRange } from "../model/viewport";
import type { WidgetModel } from "./kernel";
import type { Lattice } from "./lattice";
import { useTrait } from "./scheme";

const sameRange = (a: SRange | null, b: SRange | null) => a === b || (!!a && !!b && a.s0 === b.s0 && a.s1 === b.s1);
const wholeRange = (b: BranchModel): SRange => ({ s0: 0, s1: Math.max(b.length, 1e-3) });

/**
 * The selection, as an occurrence id both ways. The user's is told to the kernel; one of another
 * branch moves the view there. The kernel's, or a linked view's, is shown once its branch has
 * loaded and brought into view (``reveal``); a view along s takes only those on what it shows.
 */
export function useSelection(model: WidgetModel, lattice: Lattice, every: boolean, reveal: (target: Target) => void) {
  const [selection, setSelection] = useState<Target | null>(null);
  const live = useRef(lattice);
  live.current = lattice;
  const shown = useRef(reveal);
  shown.current = reveal;
  /** The id this view chose itself: not brought into view again when it comes back. */
  const own = useRef<string | null>(null);

  const select = useCallback(
    (wanted: Target | null) => {
      const { models, routes, currentModel, current } = live.current;
      const target =
        wanted && currentModel && wanted.branch !== currentModel.id ? (targetIn(currentModel, wanted, models, routes) ?? wanted) : wanted;
      setSelection(target);
      const view = target ? (models.get(target.branch) ?? routes.get(target.branch)) : undefined;
      const id = target && view ? view.ids[target.index]! : null;
      own.current = id;
      let changed = false;
      if (id !== model.get("selection")) {
        model.set("selection", id);
        changed = true;
      }
      if (target && target.branch !== current) {
        model.set("branch", target.branch);
        changed = true;
      }
      if (changed) model.save_changes();
    },
    [model],
  );

  const asked = useTrait<string | null>(model, "selection");
  useEffect(() => {
    if (!asked) {
      setSelection(null);
      return;
    }
    const { models, currentModel } = live.current;
    const views = every ? [currentModel, ...models.values()] : [currentModel];
    for (const view of views) {
      const index = view?.index.get(asked);
      if (!view || index === undefined) continue;
      const target = { branch: view.id, index };
      setSelection((now) => (sameTarget(now, target) ? now : target));
      if (asked !== own.current) requestAnimationFrame(() => shown.current(target));
      own.current = asked;
      return;
    }
    setSelection(null); // on a branch this view does not show (or not yet)
  }, [asked, lattice.models, lattice.routes, lattice.currentModel, every]);

  return { selection, select };
}

/** Follow a fork the strip shows: the view moves to the branch it leads to, at the occurrence it lands on. */
export function useFollowFork(model: WidgetModel) {
  return useCallback(
    (fork: { branch: string; occurrence: string | null }) => {
      model.set("branch", fork.branch);
      model.set("selection", fork.occurrence);
      model.save_changes();
    },
    [model],
  );
}

/**
 * The s range of a view along s, both ways: the kernel's (or a linked view's) zoom shows here, and
 * where the user zooms is told to it. A branch shown for the first time spans the whole of it, or
 * the range the kernel asked for first.
 */
export function useRange(model: WidgetModel, currentModel: BranchModel | null) {
  const range = useMemo(() => createStore<SRange | null>(null, sameRange), []);
  const band = useMemo(() => createStore<SRange | null>(null, sameRange), []);
  const history = useMemo(() => createZoomHistory(), []);
  const shownId = useRef<string | null>(null);
  const live = useRef(currentModel);
  live.current = currentModel;

  useEffect(() => {
    if (!currentModel || shownId.current === currentModel.id) return;
    const asked = model.get("s_range") as number[] | null;
    const first =
      shownId.current === null && asked?.length === 2
        ? clampRange({ s0: asked[0]!, s1: asked[1]! }, wholeRange(currentModel))
        : wholeRange(currentModel);
    shownId.current = currentModel.id;
    history.clear();
    range.set(first);
  }, [currentModel, model, range, history]);

  useEffect(() => {
    const listener = () => {
      const asked = model.get("s_range") as number[] | null;
      const b = live.current;
      const now = range.get();
      if (!b || asked?.length !== 2) return;
      const next = clampRange({ s0: asked[0]!, s1: asked[1]! }, wholeRange(b));
      if (now && Math.abs(now.s0 - next.s0) < 1e-9 && Math.abs(now.s1 - next.s1) < 1e-9) return;
      if (now) history.push(now);
      range.set(next);
    };
    model.on("change:s_range", listener);
    let timer: number | undefined;
    const unsubscribe = range.subscribe(() => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        const now = range.get();
        const asked = model.get("s_range") as number[] | null;
        if (!now || (asked?.length === 2 && asked[0] === now.s0 && asked[1] === now.s1)) return;
        model.set("s_range", [now.s0, now.s1]);
        model.save_changes();
      }, 250);
    });
    return () => {
      model.off("change:s_range", listener);
      unsubscribe();
      window.clearTimeout(timer);
    };
  }, [model, range, history]);

  return { range, band, history };
}
