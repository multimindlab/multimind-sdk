import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Dev-server proxy so the SPA talks to all backends without CORS:
//   /api/*             -> governance dashboard (multimind dashboard, port 8501)
//   /db-health         -> dashboard /health
//   /v1/*              -> API Gateway (chat/sessions/compare/compliance), via
//                         the backend aggregator (multimind backend, port
//                         8080, mounted at /gateway)
//   /gw-health         -> gateway /health (through the aggregator)
//   /rag/*             -> RAG API, via the backend aggregator (mounted at
//                         /rag; rag_api's own routes are unprefixed, so no
//                         rewrite is needed — /rag/query on the wire already
//                         matches the aggregator's mount path)
//   /rag-health        -> rag /health (through the aggregator)
//   /proxy/*           -> OpenAI-compatible guard proxy (multimind serve, port 8400)
//   /platform/*        -> observability platform (port 8600), prefix stripped
//   /platform-health   -> platform /health
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8501", changeOrigin: true },
      "/db-health": {
        target: "http://127.0.0.1:8501",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/db-health/, "/health"),
      },
      "/v1": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
        rewrite: (path) => `/gateway${path}`,
      },
      "/gw-health": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
        rewrite: () => "/gateway/health",
      },
      // /rag-health must be registered before /rag: Vite's proxy matches by
      // string prefix, so a /rag entry declared first would also swallow
      // /rag-health requests (same reason /platform-health precedes /platform).
      "/rag-health": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
        rewrite: () => "/rag/health",
      },
      "/rag": { target: "http://127.0.0.1:8080", changeOrigin: true },
      "/proxy": {
        target: "http://127.0.0.1:8400",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/proxy/, ""),
      },
      "/platform-health": {
        target: "http://127.0.0.1:8600",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/platform-health/, "/health"),
      },
      "/platform": {
        target: "http://127.0.0.1:8600",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/platform/, ""),
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});
