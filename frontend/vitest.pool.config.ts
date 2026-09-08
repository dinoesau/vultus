import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

// Suite HTTP del gateway contra el runtime worker con bindings emulados.
// Root en frontend (ahi vive node_modules); el test y la config dev viven
// en el repo, permitidos via fs.allow. Usa la entrada dev: R2/Queue/DO locales.
// Sin isolatedStorage: el stream WS del DO retiene storage mas alla del
// test y rompe el pop aislado; cada test usa job_ids frescos.
export default defineWorkersConfig({
  server: {
    fs: {
      allow: [".."],
    },
  },
  test: {
    include: ["../edge/worker.http.test.ts"],
    poolOptions: {
      workers: {
        wrangler: { configPath: "../wrangler.dev.toml" },
        isolatedStorage: false,
        singleWorker: true,
      },
    },
  },
});
