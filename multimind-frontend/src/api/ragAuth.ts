import { useSyncExternalStore } from "react";

// Tiny auth store for the RAG API's optional JWT token. Auth is only enforced
// server-side when JWT_SECRET/API_KEYS are configured (see rag_api.py
// _auth_configured); when it isn't, requests work unauthenticated and this
// store just stays empty. The token lives in sessionStorage (cleared when the
// tab closes) and components subscribe via useRagToken.

const STORAGE_KEY = "mm_rag_token";

type Listener = () => void;
const listeners = new Set<Listener>();

function emit() {
  listeners.forEach((l) => l());
}

export function getRagToken(): string | null {
  return sessionStorage.getItem(STORAGE_KEY);
}

export function setRagToken(token: string): void {
  sessionStorage.setItem(STORAGE_KEY, token);
  emit();
}

export function clearRagToken(): void {
  sessionStorage.removeItem(STORAGE_KEY);
  emit();
}

export function subscribeRagToken(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Reactive view of the stored token; null when signed out / auth not needed. */
export function useRagToken(): string | null {
  return useSyncExternalStore(subscribeRagToken, getRagToken);
}
