import JSZip from "jszip";

// Por qué jszip y no un parse manual del zip en el navegador: el resultado
// es un zip real (deflate + directorio central) generado por el worker.
// Un parser a mano es frágil ante cualquier cambio de compresión;
// jszip es el coste mínimo (~100KB) por robustez en Fase 1.

// Sin Magic Strings para rutas: un solo objeto con los paths del Seam HTTP.
export const API_PATHS = {
  compare: "/v1/compare",
  job: (id: string) => `/v1/jobs/${id}`,
  events: (id: string) => `/v1/jobs/${id}/events`,
  result: (id: string) => `/v1/jobs/${id}/result`,
} as const;

// Nombres exactos del bundle (contrato con el worker, Fase 1 UV canónico).
export const RESULT_FILES = {
  uvA: "uv_a.png",
  uvB: "uv_b.png",
  heat: "heatmap.png",
} as const;

export interface JobEvent {
  job_id: string;
  status: string;
  progress: number;
  stage: string;
}

const TERMINAL = new Set(["done", "failed", "expired"]);

export function isTerminal(status: string): boolean {
  return TERMINAL.has(status);
}

// Deriva la URL WS desde la HTTP cambiando el esquema (http->ws, https->wss).
export function toWsUrl(apiUrl: string, jobId: string): string {
  const base = apiUrl.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  return `${base}${API_PATHS.events(jobId)}`;
}

export function statusMessage(
  status: string,
  stage: string,
  progress: number,
  jobId: string,
): string {
  const pct = Math.round(progress * 100);
  switch (status) {
    case "queued":
      return `en cola (etapa ${stage}, ${pct}%)...`;
    case "processing":
      return `procesando: ${stage} ${pct}%`;
    case "done":
      return `listo (${stage}, 100%). Descargando resultado...`;
    case "failed":
      return `falló en etapa ${stage}. Revisa las imágenes e inténtalo de nuevo.`;
    case "expired":
      return `expiró (TTL 60s en servidor). Si ya descargaste el zip, sigue disponible abajo desde memoria.`;
    default:
      return `${status} ${stage} ${pct}%`;
  }
}

export interface ResultImages {
  uvA: Blob;
  uvB: Blob;
  heat: Blob;
}

// Desempaqueta el zip en memoria; falla con mensaje claro si falta un PNG.
export async function extractResultImages(zipBlob: Blob): Promise<ResultImages> {
  const zip = await JSZip.loadAsync(zipBlob);
  async function pick(name: string): Promise<Blob> {
    const entry = zip.file(name);
    if (!entry) throw new Error(`el zip no contiene ${name}`);
    return entry.async("blob");
  }
  const [uvA, uvB, heat] = await Promise.all([
    pick(RESULT_FILES.uvA),
    pick(RESULT_FILES.uvB),
    pick(RESULT_FILES.heat),
  ]);
  return { uvA, uvB, heat };
}
