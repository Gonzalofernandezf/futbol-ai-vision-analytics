// Config aparte para generar un snapshot HTML autocontenido (sin servidor,
// sin TanStack Start/nitro) con datos de ejemplo embebidos — ver static-demo/.
// No usar para el build real del dashboard: usar `bun run build`.
import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  root: resolve(__dirname, "static-demo"),
  plugins: [
    tailwindcss(),
    tsconfigPaths({ projects: [resolve(__dirname, "tsconfig.json")] }),
    react(),
    viteSingleFile(),
  ],
  resolve: {
    alias: { "@": resolve(__dirname, "src") },
  },
  build: {
    outDir: resolve(__dirname, "dist-static"),
    emptyOutDir: true,
    cssCodeSplit: false,
    assetsInlineLimit: 100_000_000,
  },
});
