import { useSyncExternalStore } from "react";

// Tiny auth store for the platform API key. The key lives in sessionStorage
// (cleared when the tab closes) and components subscribe via usePlatformKey.

const STORAGE_KEY = "mm_platform_key";

type Listener = () => void;
const listeners = new Set<Listener>();

function emit() {
  listeners.forEach((l) => l());
}

export function getPlatformKey(): string | null {
  return sessionStorage.getItem(STORAGE_KEY);
}

export function setPlatformKey(key: string): void {
  sessionStorage.setItem(STORAGE_KEY, key);
  emit();
}

export function clearPlatformKey(): void {
  sessionStorage.removeItem(STORAGE_KEY);
  emit();
}

export function subscribePlatformKey(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Reactive view of the stored key; null when signed out. */
export function usePlatformKey(): string | null {
  return useSyncExternalStore(subscribePlatformKey, getPlatformKey);
}
