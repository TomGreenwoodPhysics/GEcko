import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the complete GEcko research dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>GEcko — Grand Exchange Transformation Research \| GEcko<\/title>/i);
  assert.match(html, /Grand Exchange transformation research/);
  assert.match(html, /Most relationships did not survive rolling analysis\./);
  assert.match(html, /The OU baseline outperformed the ML alternatives\./);
  assert.match(html, /0 structural violations across 3,328 hide rows/);
  assert.doesNotMatch(html, /Your site is taking shape|react-loading-skeleton/);
});

test("keeps research evidence, interactions, and social metadata assets in source", async () => {
  const [dashboard, page, packageJson] = await Promise.all([
    readFile(new URL("../app/research-dashboard.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    access(new URL("../public/og-v2.png", import.meta.url)),
    access(new URL("../public/favicon-v2.png", import.meta.url)),
    access(new URL("../public/research/rolling-cointegration.png", import.meta.url)),
  ]);

  assert.match(page, /ResearchDashboard/);
  assert.match(dashboard, /useState\("Hide"\)/);
  assert.match(dashboard, /useState<keyof typeof costModels>\("realistic"\)/);
  assert.match(dashboard, /type="range"/);
  assert.match(dashboard, /Deflated Sharpe probability/);
  assert.match(dashboard, /Estimated safe capital/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
