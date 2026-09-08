/**
 * Globales minimos del runtime Cloudflare para `tsc --noEmit`.
 * Solo lo que usa el gateway; sin @cloudflare/workers-types como dep.
 * No cruza al dominio: el contrato (`contract.ts`) no usa estos tipos.
 */

interface WebSocket {
  accept(): void;
}

declare class WebSocketPair {
  0: WebSocket;
  1: WebSocket;
}

// Respuesta con upgrade a websocket (101) del runtime Cloudflare.
interface ResponseInit {
  webSocket?: WebSocket | null;
}

interface R2ObjectBody {
  readonly body: ReadableStream;
}

interface R2Bucket {
  put(key: string, value: ArrayBuffer | Uint8Array): Promise<unknown>;
  get(key: string): Promise<(R2ObjectBody & { readonly body: ReadableStream | null }) | null>;
}

interface Queue {
  send(message: unknown): Promise<void>;
}

interface DurableObjectId {
  toString(): string;
}

interface DurableObjectStub {
  fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response>;
}

interface DurableObjectNamespace {
  idFromName(name: string): DurableObjectId;
  get(id: DurableObjectId): DurableObjectStub;
}

interface DurableObjectStorage {
  get<T = unknown>(keys: string[]): Promise<Map<string, T> | Record<string, T>>;
  put(entries: Record<string, unknown>): Promise<void>;
  setAlarm(scheduledTime: number | Date): Promise<void>;
  deleteAll(): Promise<void>;
}

interface DurableObjectState {
  readonly storage: DurableObjectStorage;
}
