// Small observable values for state that changes on every pointer move
// (hover, visible range). Canvases subscribe and redraw directly, so moving
// the mouse never re-renders the React tree.

import { useSyncExternalStore } from "react";

export interface Store<T> {
  get(): T;
  set(value: T): void;
  subscribe(listener: () => void): () => void;
}

export function createStore<T>(initial: T, equal: (a: T, b: T) => boolean = Object.is): Store<T> {
  let value = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => value,
    set(next) {
      if (equal(value, next)) return;
      value = next;
      listeners.forEach((listener) => listener());
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export function useStore<T>(store: Store<T>): T {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}

/** A primitive derived from a store; the component re-renders only when it changes. */
export function useStoreSelect<T, R extends string | number | boolean | null>(store: Store<T>, select: (value: T) => R): R {
  const snapshot = () => select(store.get());
  return useSyncExternalStore(store.subscribe, snapshot, snapshot);
}

/** An occurrence by branch id and index in the branch. */
export interface Target {
  branch: string;
  index: number;
}

export function sameTarget(a: Target | null, b: Target | null): boolean {
  return a === b || (a !== null && b !== null && a.branch === b.branch && a.index === b.index);
}
