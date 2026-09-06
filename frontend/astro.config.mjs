import { defineConfig } from "astro/config";

// Cloudflare Pages en prod (static). En dev `npm run dev` sirve :4321.
// `site` canonico prod: el build estatico apunta al dominio publico.
// La API prod via `VITE_API_URL=https://api.vultus.esau.com.mx` en Pages.
export default defineConfig({
  output: "static",
  site: "https://vultus.esau.com.mx",
  server: { port: 4321, host: "0.0.0.0" },
});
