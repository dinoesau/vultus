import { defineConfig } from "astro/config";

// Cloudflare Pages en prod (static). En dev `npm run dev` sirve :4321.
export default defineConfig({
  output: "static",
  server: { port: 4321, host: "0.0.0.0" },
});
