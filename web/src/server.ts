/**
 * server.ts
 * ---------
 * Serveur web Bun + Hono pour Boeuf Tracker.
 *
 * Rôle :
 *  1. Sert l'UI statique (HTML/CSS/JS) depuis public/
 *  2. Proxy transparent vers le worker Python (port 8100) pour /api/* et /video_feed
 *  3. Logique applicative : noms, dashboard, timeline (phases ultérieures)
 *
 * L'IA (YOLO + DINOv2 + race) reste dans le worker Python — Bun ne fait que
 * router et présenter.
 */
import { Hono } from "hono";
import { serveStatic } from "hono/bun";

const PYTHON_WORKER = process.env.PYTHON_WORKER ?? "http://localhost:8100";
const PORT = Number(process.env.PORT ?? 8000);

const app = new Hono();

// ─── Proxy vers le worker Python ────────────────────────────────────────
// Tout /api/* et /video_feed est forwardé tel quel (JSON, binaire, MJPEG stream…).
// IMPORTANT : on STREAME la réponse (resp.body) — surtout pas arrayBuffer(),
// sinon un flux MJPEG infini (video_feed) buffere pour toujours et l'image
// ne s'affiche jamais côté navigateur.
const HOP_BY_HOP_REQ = new Set([
  "host", "connection", "content-length", "transfer-encoding",
  "accept-encoding", "keep-alive", "proxy-connection", "upgrade",
]);
const HOP_BY_HOP_RESP = new Set([
  "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
  "te", "trailers", "transfer-encoding", "upgrade",
]);

function filterHeaders(src: Headers, deny: Set<string>): Headers {
  const out = new Headers();
  src.forEach((v, k) => { if (!deny.has(k.toLowerCase())) out.set(k, v); });
  return out;
}

async function proxyToPython(path: string, c: any, opts: { forwardBody: boolean }): Promise<Response> {
  const url = `${PYTHON_WORKER}${path}`;
  const method = c.req.method;
  const hasBody = opts.forwardBody && method !== "GET" && method !== "HEAD";
  const init: RequestInit & { duplex?: "half" } = {
    method,
    headers: filterHeaders(c.req.raw.headers, HOP_BY_HOP_REQ),
    body: hasBody ? c.req.raw.body : undefined,
    // Requis par undici/Bun quand body est un ReadableStream (upload multipart)
    ...(hasBody ? { duplex: "half" as const } : {}),
  };
  try {
    const resp = await fetch(url, init);
    return new Response(resp.body, {
      status: resp.status,
      headers: filterHeaders(resp.headers, HOP_BY_HOP_RESP),
    });
  } catch (e) {
    return Response.json(
      { ok: false, error: `Worker Python injoignable (${PYTHON_WORKER}). Lancez: python app.py --mlx` },
      { status: 502 }
    );
  }
}

// Proxy /api/* (JSON + upload multipart)
app.all("/api/*", (c) => {
  const qs = c.req.raw.url.split("?")[1] ?? "";
  const path = c.req.path + (qs ? `?${qs}` : "");
  return proxyToPython(path, c, { forwardBody: true });
});

// Proxy /video_feed (flux MJPEG — pas de body côté requête)
app.all("/video_feed", (c) => {
  const qs = c.req.raw.url.split("?")[1] ?? "";
  const path = "/video_feed" + (qs ? `?${qs}` : "");
  return proxyToPython(path, c, { forwardBody: false });
});

// ─── Fichiers statiques (UI) ────────────────────────────────────────────
app.use("/*", serveStatic({ root: "./public" }));

// Fallback : index.html pour la route racine
app.get("/", (c) => {
  return serveStatic({ root: "./public" })(
    new Request("http://localhost/index.html"),
    c.env
  );
});

console.log(`╔══════════════════════════════════════════════╗`);
console.log(`║  BOEUF TRACKER — Serveur web Bun + Hono     ║`);
console.log(`║  URL       : http://localhost:${PORT}          ║`);
console.log(`║  Worker PY : ${PYTHON_WORKER.padEnd(28)} ║`);
console.log(`╚══════════════════════════════════════════════╝`);

export default {
  port: PORT,
  fetch: app.fetch,
};
