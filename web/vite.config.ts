import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true, proxy: { "/api": "http://127.0.0.1:8000" } },
  preview: { port: 5173 },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return;
          if (id.includes("/katex/")) return "vendor-katex";
          if (id.includes("/react-markdown/") || id.includes("/remark-") || id.includes("/rehype-") || id.includes("/unified/") || id.includes("/micromark")) return "vendor-markdown";
        },
      },
    },
  },
});
