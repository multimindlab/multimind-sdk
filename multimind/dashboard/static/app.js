// MultiMind governance dashboard and playground: hash-routed views over the local
// /api endpoints (dashboard) and the /v1 and RAG routes (multimind backend).

const view = document.getElementById("view");

const fmtUsd = (v) =>
  "$" + (v >= 100 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(4) : v.toFixed(6));
const fmtInt = (v) => Number(v || 0).toLocaleString();
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const jsonBlock = (obj) => `<pre class="redacted">${esc(JSON.stringify(obj, null, 2))}</pre>`;
const okBadge = (ok, yes = "yes", no = "no") =>
  ok ? `<span class="badge ok">${esc(yes)}</span>` : `<span class="badge danger">${esc(no)}</span>`;

const API_BASE = String(window.MULTIMIND_API_BASE || "").replace(/\/$/, "");

/* ---------- storage, API key, connection mode ---------- */

function remembered(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}
function remember(key, value) {
  try {
    if (value) localStorage.setItem(key, value); else localStorage.removeItem(key);
  } catch { /* private mode: ignore */ }
}

// The API key lives only in this browser (localStorage "multimind.apiKey"); the server
// never embeds one in config.js.
function readApiKey() {
  try { return localStorage.getItem("multimind.apiKey") || ""; } catch { return ""; }
}
function authHeaders(extra) {
  const headers = { ...(extra || {}) };
  const key = readApiKey();
  if (key) headers["X-API-Key"] = key;
  return headers;
}

// Other local services the UI talks to directly (not via the dashboard server).
// Built from the page's scheme so the static bundle carries no absolute URLs.
const localUrl = (port) => `${location.protocol}//127.0.0.1:${port}`;
const DEFAULT_PROXY_BASE = localUrl(8400);
const DEFAULT_BACKEND_BASE = API_BASE || localUrl(8080);

// "same-origin": the page is served by `multimind backend`, so /v1/* and RAG routes are
// on this origin. "standalone": `multimind dashboard`; /v1/* goes to the stored backend URL.
const conn = {
  mode: "unknown",
  backendBase: remembered("multimind.backendBase", DEFAULT_BACKEND_BASE),
  proxyBase: remembered("multimind.proxyBase", DEFAULT_PROXY_BASE),
};
const sameOrigin = () => conn.mode === "same-origin";
const backendBase = () => (sameOrigin() ? API_BASE : conn.backendBase);
const proxyBase = () => (sameOrigin() ? API_BASE : conn.proxyBase);

async function detectMode() {
  try {
    const resp = await fetch(API_BASE + "/v1/features", { headers: authHeaders() });
    conn.mode = resp.status === 404 ? "standalone" : "same-origin";
  } catch {
    conn.mode = "standalone";
  }
  document.body.dataset.mode = conn.mode;
}

const RAG_PATHS = ["/documents", "/files", "/query", "/generate", "/models/switch", "/token"];
const isRagPath = (p) => RAG_PATHS.some((r) => p === r || p.startsWith(r + "/") || p.startsWith(r + "?"));

class ApiError extends Error {
  constructor(message, info) {
    super(message);
    Object.assign(this, info);
  }
}

async function request(base, path, options = {}) {
  const headers = authHeaders(options.headers);
  let init = { ...options, headers };
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    init = { ...init, body: JSON.stringify(options.json) };
    delete init.json;
  }
  const ragToken = remembered("multimind.ragToken", "");
  if (ragToken && isRagPath(path) && !headers.Authorization) headers.Authorization = `Bearer ${ragToken}`;
  let resp;
  try {
    resp = await fetch(base.replace(/\/$/, "") + path, init);
  } catch (err) {
    throw new ApiError(`Could not reach ${base || location.origin}: ${err.message || err}`,
      { status: 0, path, base });
  }
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    let detail = body.detail ?? body.error?.message ?? `HTTP ${resp.status}`;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    if (typeof detail !== "string") detail = JSON.stringify(detail);
    throw new ApiError(`${path} -> ${detail}`, { status: resp.status, path, base, detail });
  }
  if (resp.status === 204) return null;
  const text = await resp.text();
  try { return text ? JSON.parse(text) : null; } catch { return text; }
}

const api = (path, options) => request(API_BASE, path, options);
const externalApi = (base, path, options) => request(base, path, options);
const backendApi = (path, options) => request(backendBase(), path, options);

function explainError(err) {
  const status = err.status;
  const path = err.path || "";
  if (status === 401) {
    return `Unauthorized (401). Set the API key in <a href="#/settings">Settings</a>` +
      (isRagPath(path) ? ", or log in on the RAG view to get a JWT." : ".");
  }
  if (status === 404 && isRagPath(path)) {
    return `RAG routes are not mounted (404): the RAG extra is not installed. Install with ` +
      `<code>pip install "multimind-sdk[rag]"</code> and restart <code>multimind backend</code>.`;
  }
  if (status === 404 && path.startsWith("/v1/")) {
    return `Route <code>${esc(path)}</code> is missing (404): it requires <code>multimind backend</code>` +
      (sameOrigin() ? "." : ` at <code>${esc(backendBase())}</code>. Check the URL in <a href="#/settings">Settings</a>.`);
  }
  if (status === 501) return `Not installed (501): ${esc(err.detail || err.message)}`;
  if (status === 0) {
    return `${esc(err.message)}. Start <code>multimind backend</code> (default port 8080)` +
      (sameOrigin() ? "" : ` and allow this origin with <code>MULTIMIND_CORS_ORIGINS=${esc(location.origin)}</code>`) + ".";
  }
  return esc(err.message || err);
}

function errorCard(err) {
  return `<div class="card"><span class="badge danger">error</span> ${explainError(err)}</div>`;
}

function setMsg(el, kind, text) {
  el.className = "form-msg " + kind;
  el.textContent = text;
}
function setErr(el, err) {
  el.className = "form-msg error";
  el.innerHTML = explainError(err);
}

// Wires a submit/click handler that reports progress and errors in a .form-msg element.
function bindAction(el, msgEl, fn, { event = "click", busy = "Working…" } = {}) {
  el[`on${event}`] = async (e) => {
    if (e && e.preventDefault) e.preventDefault();
    setMsg(msgEl, "", busy);
    try {
      await fn();
    } catch (err) {
      setErr(msgEl, err);
    }
  };
}

const splitLines = (text) => text.split("\n").map((s) => s.trim()).filter(Boolean);
const splitParagraphs = (text) => text.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);

function modeBanner() {
  return sameOrigin()
    ? `<div class="banner ok">Served by <code>multimind backend</code> at <code>${esc(location.origin)}</code>: every route is same-origin.</div>`
    : `<div class="banner ok">Standalone dashboard. <code>/v1/*</code> and RAG routes go to <code>${esc(backendBase())}</code>
        (<code>multimind backend</code>; change it in <a href="#/settings">Settings</a>).</div>`;
}

async function loadModels() {
  // Enabled providers from /v1/models; failures are returned, not thrown, so forms still render.
  try {
    const data = await backendApi("/v1/models");
    return { models: data.models || {}, error: null };
  } catch (err) {
    return { models: {}, error: err };
  }
}
function modelOptions(models, selected) {
  const names = Object.keys(models);
  if (!names.length) return `<option value="${esc(selected || "openai")}">${esc(selected || "openai")} (list unavailable)</option>`;
  return names.map((m) => `<option value="${esc(m)}" ${m === selected ? "selected" : ""}>${esc(m)}${models[m].status === "available" ? "" : " (unavailable)"}</option>`).join("");
}
function modelsNote(error) {
  return error ? `<div class="banner">Model list failed: ${explainError(error)}</div>` : "";
}

/* ---------- charts (hand-rolled SVG, single accent series) ---------- */

function lineChart(points, { valueKey = "cost", labelKey = "date", fmt = fmtUsd, label = "Daily spend, last 30 days" } = {}) {
  if (!points.length) return `<div class="empty">No spend recorded yet.</div>`;
  const W = 640, H = 220, m = { t: 14, r: 16, b: 26, l: 52 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const max = Math.max(...points.map((p) => p[valueKey]), 0) || 1;
  const x = (i) => m.l + (points.length === 1 ? iw / 2 : (i / (points.length - 1)) * iw);
  const y = (v) => m.t + ih - (v / max) * ih;
  const coords = points.map((p, i) => [x(i), y(p[valueKey])]);
  const path = coords.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
  const area = `${path} L${coords[coords.length - 1][0].toFixed(1)},${m.t + ih} L${coords[0][0].toFixed(1)},${m.t + ih} Z`;
  const ticks = [0, 0.5, 1].map((f) => {
    const v = max * f, ty = y(v);
    return `<line class="gridline" x1="${m.l}" x2="${W - m.r}" y1="${ty}" y2="${ty}"></line>
      <text class="axis-label" x="${m.l - 6}" y="${ty + 3}" text-anchor="end">${esc(fmt(v))}</text>`;
  }).join("");
  const first = points[0][labelKey], last = points[points.length - 1][labelKey];
  const dots = points.map((p, i) => {
    const [px, py] = coords[i];
    const tip = `${p[labelKey]}: ${fmt(p[valueKey])}` + (p.calls === undefined ? "" : ` (${fmtInt(p.calls)} calls)`);
    return `<g><circle class="hover-target" cx="${px}" cy="${py}" r="12"><title>${esc(tip)}</title></circle>
      <circle class="series-dot" cx="${px}" cy="${py}" r="3.5" pointer-events="none"></circle></g>`;
  }).join("");
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">
    ${ticks}
    <path class="series-area" d="${area}"></path>
    <path class="series-line" d="${path}"></path>
    ${dots}
    <text class="axis-label" x="${m.l}" y="${H - 8}">${esc(first)}</text>
    <text class="axis-label" x="${W - m.r}" y="${H - 8}" text-anchor="end">${esc(last)}</text>
  </svg></div>`;
}

function barChart(items) {
  // Horizontal bars: identity on the axis, magnitude by length, one hue.
  if (!items.length) return `<div class="empty">No PII detected yet.</div>`;
  const W = 640, rowH = 28, m = { t: 6, r: 60, b: 6, l: 110 };
  const H = m.t + m.b + items.length * rowH;
  const max = Math.max(...items.map((d) => d.value)) || 1;
  const rows = items.map((d, i) => {
    const y = m.t + i * rowH;
    const w = Math.max(2, ((W - m.l - m.r) * d.value) / max);
    return `<g>
      <text class="axis-label" x="${m.l - 8}" y="${y + rowH / 2 + 3}" text-anchor="end">${esc(d.label)}</text>
      <rect class="bar" x="${m.l}" y="${y + 5}" width="${w.toFixed(1)}" height="${rowH - 12}" rx="3">
        <title>${esc(d.label)}: ${esc(fmtInt(d.value))}</title></rect>
      <text class="value-label" x="${m.l + w + 6}" y="${y + rowH / 2 + 3}">${esc(fmtInt(d.value))}</text>
    </g>`;
  }).join("");
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="PII detections by type">${rows}</svg></div>`;
}

/* ---------- governance views ---------- */

async function renderOverview() {
  const [summary, costs] = await Promise.all([api("/api/summary"), api("/api/costs")]);
  const a = summary.audit, day = a.last_24h, c = summary.costs, budget = summary.budget;
  const byType = Object.entries(a.all_time.by_type)
    .sort((x, y) => y[1] - x[1])
    .map(([label, value]) => ({ label, value }));
  const cutoff = new Date(Date.now() - 30 * 86400e3).toISOString().slice(0, 10);
  const daily = costs.daily.filter((d) => d.date >= cutoff);
  const budgetTile = budget
    ? `<div class="tile ${budget.exceeded ? "danger" : ""}">
        <div class="label">Budget</div>
        <div class="value">${esc(fmtUsd(budget.spent))} / ${esc(fmtUsd(budget.max_cost))}</div>
        <div class="delta">${budget.exceeded ? "ceiling reached" : esc(fmtUsd(budget.remaining)) + " remaining"}</div>
      </div>`
    : "";
  view.innerHTML = `
    <h1>Overview</h1>
    <p class="page-sub">AI governance at a glance: PII exposure, spend, and models in use.</p>
    <div class="tiles">
      <div class="tile"><div class="label">PII detections</div>
        <div class="value">${esc(fmtInt(a.all_time.pii_detections))}</div>
        <div class="delta">${esc(fmtInt(day.pii_detections))} in last 24h</div></div>
      <div class="tile ${a.all_time.blocked ? "danger" : ""}"><div class="label">Blocked requests</div>
        <div class="value">${esc(fmtInt(a.all_time.blocked))}</div>
        <div class="delta">${esc(fmtInt(day.blocked))} in last 24h</div></div>
      <div class="tile"><div class="label">Total spend</div>
        <div class="value">${esc(fmtUsd(c.total_cost))}</div>
        <div class="delta">${esc(fmtUsd(c.last_24h_cost))} in last 24h</div></div>
      <div class="tile"><div class="label">Models in use</div>
        <div class="value">${esc(fmtInt(c.models.length))}</div>
        <div class="delta">${esc(fmtInt(c.calls))} tracked calls</div></div>
      ${budgetTile}
    </div>
    <div class="grid-2">
      <div class="card"><h2>Spend, last 30 days</h2>${lineChart(daily)}</div>
      <div class="card"><h2>PII detections by type</h2>${barChart(byType)}</div>
    </div>
    <div class="toolbar">
      <span class="muted">Need an auditor-ready snapshot of these artifacts?</span>
      <a href="#/evidence"><button type="button">Evidence report</button></a>
    </div>`;
}

function evidenceLinks() {
  // Plain links: the report endpoint is a download, so no fetch/JSON here.
  return ["html", "md"].map((f) =>
    `<a id="ev-${f}" href="${esc(API_BASE)}/api/evidence-report?format=${f}" target="_blank" rel="noopener">
      <button type="button" class="${f === "html" ? "primary" : ""}">Download ${f === "html" ? "HTML" : "Markdown"}</button></a>`
  ).join(" ");
}

async function renderEvidence() {
  view.innerHTML = `
    <h1>Evidence Report</h1>
    <p class="page-sub">A compliance evidence report built from the audit log, cost log, and the cached
      inventory scan. Artifacts that are missing are stated as missing; it is not a legal determination.</p>
    <div class="card">
      <h2>Download</h2>
      <div class="toolbar">${evidenceLinks()}</div>
      <div class="note">Served by <code>GET /api/evidence-report?format=html|md</code> on this dashboard.
        Opens in a new tab; most browsers save it as <code>evidence-report.html</code> / <code>.md</code>.</div>
    </div>`;
}

async function renderScan() {
  const strategies = ["", "mask", "hash", "remove"].map((s) =>
    `<option value="${s}">${s || "detect only"}</option>`).join("");
  view.innerHTML = `
    <h1>Scan Text</h1>
    <p class="page-sub">Ad-hoc PII check. The text is scanned in memory by this dashboard and never logged.</p>
    <div class="card"><form id="scan-form">
      <div class="form-row"><label class="row-label" for="scan-text">Text to scan</label>
        <textarea id="scan-text" rows="8" required placeholder="Paste a prompt, log line, or document&hellip;"></textarea></div>
      <div class="form-row"><label class="row-label" for="scan-strategy">Redaction strategy</label>
        <select id="scan-strategy">${strategies}</select></div>
      <button type="submit" class="primary">Scan</button>
      <span id="scan-msg" class="form-msg" role="status"></span>
    </form></div>
    <div id="scan-result"></div>`;
  document.getElementById("scan-form").onsubmit = async (e) => {
    e.preventDefault();
    const msg = document.getElementById("scan-msg");
    const out = document.getElementById("scan-result");
    const strategy = document.getElementById("scan-strategy").value;
    const body = { text: document.getElementById("scan-text").value };
    if (strategy) body.strategy = strategy;
    setMsg(msg, "", "Scanning…");
    try {
      const data = await externalApi(API_BASE, "/api/scan-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setMsg(msg, "ok", `${fmtInt(data.total)} finding(s)`);
      const rows = data.findings.map((f) => `<tr>
          <td>${esc(f.type)}</td><td class="num">${esc(f.start)}</td><td class="num">${esc(f.end)}</td>
          <td><code>${esc(f.text)}</code></td></tr>`).join("");
      const counts = Object.entries(data.counts)
        .map(([t, n]) => `<span class="chip">${esc(t)}&times;${esc(n)}</span>`).join("") || "&mdash;";
      out.innerHTML = `
        <div class="card"><h2>Findings</h2>
          <p class="muted">By type: ${counts}</p>
          <div class="table-wrap"><table>
            <thead><tr><th>Type</th><th class="num">Start</th><th class="num">End</th><th>Text</th></tr></thead>
            <tbody>${rows || `<tr><td colspan="4"><div class="empty">No PII detected.</div></td></tr>`}</tbody>
          </table></div></div>
        ${data.redacted === null ? "" : `<div class="card"><h2>Redacted (${esc(strategy)})</h2>
          <pre class="redacted">${esc(data.redacted)}</pre></div>`}`;
    } catch (err) {
      setErr(msg, err);
      out.innerHTML = "";
    }
  };
}

const auditState = { offset: 0, limit: 25, type: "", blocked: "" };

async function renderAudit() {
  const params = new URLSearchParams({ limit: auditState.limit, offset: auditState.offset });
  if (auditState.type) params.set("type", auditState.type);
  if (auditState.blocked) params.set("blocked", auditState.blocked);
  const data = await api(`/api/audit?${params}`);
  const rows = data.records.map((r) => {
    const pii = Object.entries(r.pii_types || {})
      .map(([t, n]) => `<span class="chip">${esc(t)}&times;${esc(n)}</span>`).join("") || "&mdash;";
    return `<tr class="${r.blocked ? "blocked-row" : ""}">
      <td>${esc((r.timestamp || "").replace("T", " ").slice(0, 19))}</td>
      <td>${esc(r.direction || "")}</td>
      <td>${esc(r.method || r.endpoint || "")}</td>
      <td>${pii}</td>
      <td class="num">${esc(fmtInt(r.count))}</td>
      <td>${esc(r.strategy || "")}</td>
      <td>${r.blocked ? '<span class="badge danger">blocked</span>' : '<span class="badge ok">allowed</span>'}</td>
    </tr>`;
  }).join("");
  const typeOpts = data.pii_types
    .map((t) => `<option value="${esc(t)}" ${t === auditState.type ? "selected" : ""}>${esc(t)}</option>`).join("");
  const from = data.total ? data.offset + 1 : 0;
  const to = Math.min(data.offset + data.limit, data.total);
  view.innerHTML = `
    <h1>Audit Trail</h1>
    <p class="page-sub">PII events recorded by the guard and proxy. Types, counts, and hash tags only; never raw values.</p>
    <div class="filters">
      <label>PII type
        <select id="f-type"><option value="">all types</option>${typeOpts}</select></label>
      <label>Status
        <select id="f-blocked">
          <option value="">all</option>
          <option value="true" ${auditState.blocked === "true" ? "selected" : ""}>blocked only</option>
          <option value="false" ${auditState.blocked === "false" ? "selected" : ""}>allowed only</option>
        </select></label>
    </div>
    <div class="card"><div class="table-wrap"><table>
      <thead><tr><th>Time (UTC)</th><th>Direction</th><th>Method</th><th>PII types</th>
        <th class="num">Count</th><th>Strategy</th><th>Status</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="7"><div class="empty">No audit events match.</div></td></tr>`}</tbody>
    </table></div>
    <div class="pager">
      <button id="pg-prev" ${data.offset === 0 ? "disabled" : ""}>Previous</button>
      <button id="pg-next" ${to >= data.total ? "disabled" : ""}>Next</button>
      <span>${esc(fmtInt(from))}&ndash;${esc(fmtInt(to))} of ${esc(fmtInt(data.total))}</span>
    </div></div>`;
  document.getElementById("f-type").onchange = (e) => {
    auditState.type = e.target.value; auditState.offset = 0; renderAudit().catch(showError);
  };
  document.getElementById("f-blocked").onchange = (e) => {
    auditState.blocked = e.target.value; auditState.offset = 0; renderAudit().catch(showError);
  };
  document.getElementById("pg-prev").onclick = () => {
    auditState.offset = Math.max(0, auditState.offset - auditState.limit); renderAudit().catch(showError);
  };
  document.getElementById("pg-next").onclick = () => {
    auditState.offset += auditState.limit; renderAudit().catch(showError);
  };
}

async function renderCosts() {
  const data = await api("/api/costs");
  const cb = data.chargeback;
  const tags = Object.entries(cb.by_tag).sort((a, b) => b[1].cost - a[1].cost);
  const maxShare = Math.max(...tags.map(([, g]) => g.share_pct), 1);
  const tagRows = tags.map(([tag, g]) => `<tr>
      <td>${esc(tag)}</td>
      <td class="num">${esc(fmtInt(g.calls))}</td>
      <td class="num">${esc(fmtInt(g.total_tokens))}</td>
      <td class="num">${esc(fmtUsd(g.cost))}</td>
      <td class="share-cell"><span class="share-bar" style="width:${((g.share_pct / maxShare) * 140).toFixed(0)}px"></span>
        <span class="muted"> ${g.share_pct.toFixed(1)}%</span></td>
    </tr>`).join("");
  const modelRows = Object.entries(data.by_model).sort((a, b) => b[1].cost - a[1].cost)
    .map(([model, g]) => `<tr>
      <td>${esc(model)}</td>
      <td class="num">${esc(fmtInt(g.calls))}</td>
      <td class="num">${esc(fmtInt(g.input_tokens))}</td>
      <td class="num">${esc(fmtInt(g.output_tokens))}</td>
      <td class="num">${esc(fmtUsd(g.cost))}</td>
    </tr>`).join("");
  view.innerHTML = `
    <h1>Costs</h1>
    <p class="page-sub">Chargeback by tag and per-model totals from the cost log.
      Total: <strong>${esc(fmtUsd(cb.total_cost))}</strong> across ${esc(fmtInt(cb.calls))} calls.</p>
    <div class="card"><h2>Chargeback by tag</h2><div class="table-wrap"><table>
      <thead><tr><th>Tag</th><th class="num">Calls</th><th class="num">Tokens</th>
        <th class="num">Cost</th><th>Share</th></tr></thead>
      <tbody>${tagRows || `<tr><td colspan="5"><div class="empty">No cost records yet.</div></td></tr>`}</tbody>
    </table></div></div>
    <div class="card"><h2>By model</h2><div class="table-wrap"><table>
      <thead><tr><th>Model</th><th class="num">Calls</th><th class="num">Input tokens</th>
        <th class="num">Output tokens</th><th class="num">Cost</th></tr></thead>
      <tbody>${modelRows || `<tr><td colspan="5"><div class="empty">No cost records yet.</div></td></tr>`}</tbody>
    </table></div></div>`;
}

async function renderInventory(refresh = false) {
  view.innerHTML = `<h1>Inventory</h1><p class="page-sub">Scanning project&hellip;</p>`;
  const data = await api(`/api/inventory${refresh ? "?refresh=true" : ""}`);
  const keys = data.risks.hardcoded_keys;
  const banner = keys.length
    ? `<div class="banner">${keys.length} hardcoded API key(s) found in source. Rotate them and move to environment variables.</div>`
    : `<div class="banner ok">No hardcoded API keys detected.</div>`;
  const byProvider = {};
  for (const f of data.findings) (byProvider[f.provider] ??= []).push(f);
  const sections = Object.entries(byProvider).sort((a, b) => b[1].length - a[1].length)
    .map(([provider, findings]) => {
      const rows = findings.map((f) => `<tr class="${f.kind === "hardcoded_key" ? "blocked-row" : ""}">
          <td>${esc(f.kind)}</td>
          <td>${esc(f.evidence)}</td>
          <td>${esc(f.file || "")}${f.line ? ":" + esc(f.line) : ""}</td>
          <td>${esc(f.data_flow || "unknown")}</td>
        </tr>`).join("");
      return `<div class="provider-head"><h2>${esc(provider)}</h2>
          <span class="count">${findings.length} finding(s)</span></div>
        <div class="table-wrap"><table>
          <thead><tr><th>Kind</th><th>Evidence</th><th>Location</th><th>Data flow</th></tr></thead>
          <tbody>${rows}</tbody></table></div>`;
    }).join("");
  view.innerHTML = `
    <h1>Inventory</h1>
    <p class="page-sub">Static shadow-AI scan of <code>${esc(data.root)}</code>: SDKs, API endpoints, credentials.</p>
    ${banner}
    <div class="toolbar">
      <span class="muted">Scanned ${esc(fmtInt(data.summary.scanned_files))} file(s), skipped ${esc(fmtInt(data.summary.skipped_files))}. Last scan: ${esc((data.scanned_at || "").replace("T", " ").slice(0, 19))} UTC</span>
      <button id="inv-refresh">Rescan project</button>
    </div>
    <div class="card">${sections || `<div class="empty">No AI usage found in this project.</div>`}</div>`;
  document.getElementById("inv-refresh").onclick = () => renderInventory(true).catch(showError);
}

async function renderGuardrails() {
  const data = await api("/api/guardrails");
  const cfg = data.config;
  const checks = data.pii_types.map((t) => `<label>
      <input type="checkbox" name="block_on" value="${esc(t)}" ${cfg.block_on.includes(t) ? "checked" : ""}> ${esc(t)}
    </label>`).join("");
  const strategies = ["mask", "hash", "remove"].map((s) =>
    `<option value="${s}" ${cfg.strategy === s ? "selected" : ""}>${s}</option>`).join("");
  view.innerHTML = `
    <h1>Guardrails</h1>
    <p class="page-sub">No-code guardrail authoring. Saves to <code>${esc(data.path)}</code>${data.exists ? "" : " (not created yet)"}.</p>
    <div class="card"><form id="gr-form">
      <div class="form-row"><label class="row-label" for="gr-strategy">Redaction strategy</label>
        <select id="gr-strategy">${strategies}</select></div>
      <div class="form-row"><span class="row-label">Block requests containing</span>
        <div class="check-grid">${checks}</div></div>
      <div class="form-row"><label class="row-label" for="gr-budget">Budget ceiling (USD, empty for unlimited)</label>
        <input id="gr-budget" type="number" min="0" step="0.01" value="${cfg.budget_max_cost ?? ""}"></div>
      <div class="form-row"><label>
        <input id="gr-scan-output" type="checkbox" ${cfg.scan_output ? "checked" : ""}> Also scan and redact model output</label></div>
      <button type="submit" class="primary">Save guardrails</button>
      <span id="gr-msg" class="form-msg" role="status"></span>
    </form></div>
    <div class="card" id="gr-apply">
      <h2>Apply to the running proxy</h2>
      <div class="note" id="gr-hint">Save writes the guardrails file. Then call
        <code>POST /v1/guardrails/reload</code> on the proxy, or restart
        <code>multimind serve --config ${esc(data.path)}</code>.</div>
      <div class="form-row" style="margin-top:1rem" ${sameOrigin() ? "hidden" : ""}><label class="row-label" for="gr-proxy-url">Proxy URL (<code>multimind serve</code>)</label>
        <input id="gr-proxy-url" type="text" value="${esc(conn.proxyBase)}"></div>
      <button type="button" id="gr-reload">Reload proxy</button>
      <span id="gr-reload-msg" class="form-msg" role="status"></span>
    </div>`;
  const proxyInput = document.getElementById("gr-proxy-url");
  proxyInput.onchange = () => { conn.proxyBase = proxyInput.value.trim() || DEFAULT_PROXY_BASE; remember("multimind.proxyBase", conn.proxyBase); };
  document.getElementById("gr-form").onsubmit = async (e) => {
    e.preventDefault();
    const msg = document.getElementById("gr-msg");
    const budgetRaw = document.getElementById("gr-budget").value;
    const body = {
      strategy: document.getElementById("gr-strategy").value,
      block_on: [...document.querySelectorAll('input[name="block_on"]:checked')].map((i) => i.value),
      budget_max_cost: budgetRaw === "" ? null : Number(budgetRaw),
      scan_output: document.getElementById("gr-scan-output").checked,
    };
    try {
      const saved = await externalApi(API_BASE, "/api/guardrails", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setMsg(msg, "ok", "Saved.");
      if (saved.apply_hint) {
        document.getElementById("gr-hint").innerHTML =
          `Saved to <code>${esc(saved.path)}</code>. To apply: ${esc(saved.apply_hint)}`;
      }
    } catch (err) {
      setErr(msg, err);
    }
  };
  document.getElementById("gr-reload").onclick = async () => {
    const msg = document.getElementById("gr-reload-msg");
    const base = proxyBase();
    setMsg(msg, "", "Reloading…");
    try {
      const r = await externalApi(base, "/v1/guardrails/reload", { method: "POST" });
      setMsg(msg, "ok", `Proxy reloaded: strategy=${r.strategy}, block_on=[${(r.block_on || []).join(", ")}], budget=${r.budget ?? "none"}`);
    } catch (err) {
      msg.className = "form-msg error";
      msg.innerHTML = `Reload failed: ${explainError(err)} ` + (sameOrigin() ? "" :
        `Is multimind serve running at ${esc(base)}? A browser-blocked (CORS) call also fails here; then run: <code>curl -X POST ${esc(base)}/v1/guardrails/reload</code>`);
    }
  };
}

/* ---------- compliance (served by `multimind backend`) ---------- */

const complianceState = {
  org: remembered("multimind.orgId", "org-123"),
};

function pct(v) { return (Number(v || 0) * 100).toFixed(1) + "%"; }

function severityBadge(sev) {
  const s = String(sev || "").toLowerCase();
  const cls = s === "critical" || s === "high" ? "danger" : "";
  return `<span class="badge ${cls}">${esc(s || "unknown")}</span>`;
}

function complianceUnreachable(err) {
  return `<div class="banner">Compliance API call failed: ${explainError(err)}</div>`;
}

async function loadComplianceMetrics() {
  const out = document.getElementById("cp-metrics");
  out.innerHTML = `<p class="page-sub">Loading&hellip;</p>`;
  const q = `organization_id=${encodeURIComponent(complianceState.org)}`;
  let m, alerts;
  try {
    [m, alerts] = await Promise.all([
      backendApi(`/v1/compliance/dashboard?${q}`),
      backendApi(`/v1/compliance/alerts?${q}`),
    ]);
  } catch (err) {
    out.innerHTML = complianceUnreachable(err);
    return;
  }
  const trend = (m.trend_data && m.trend_data.compliance_score) || [];
  const trendPoints = trend.map((v, i) => ({ label: `check ${i + 1}`, score: v }));
  const issueRows = (m.recent_issues || []).map((i) => `<tr class="blocked-row">
      <td>${esc((i.timestamp || "").replace("T", " ").slice(0, 19))}</td>
      <td>${severityBadge(i.severity)}</td><td>${esc(i.description || JSON.stringify(i))}</td></tr>`).join("");
  const alertList = Array.isArray(alerts) ? alerts : (alerts && alerts.alerts) || m.alerts || [];
  const alertRows = alertList.map((a) => `<tr>
      <td>${esc(a.id || "")}</td><td>${severityBadge(a.severity)}</td>
      <td>${esc(a.status || "")}</td><td>${esc((a.created_at || "").replace("T", " ").slice(0, 19))}</td>
      <td>${esc(a.description || "")}</td></tr>`).join("");
  out.innerHTML = `
    <div class="tiles">
      <div class="tile"><div class="label">Compliance score</div>
        <div class="value">${esc(pct(m.compliance_score))}</div>
        <div class="delta">passed / (passed + failed)</div></div>
      <div class="tile"><div class="label">Checks</div>
        <div class="value">${esc(fmtInt(m.total_checks))}</div>
        <div class="delta">${esc(fmtInt(m.passed_checks))} passed, ${esc(fmtInt(m.failed_checks))} failed</div></div>
      <div class="tile ${m.unverifiable_checks ? "danger" : ""}"><div class="label">Unverifiable</div>
        <div class="value">${esc(fmtInt(m.unverifiable_checks))}</div>
        <div class="delta">no runtime evidence</div></div>
      <div class="tile ${alertList.length ? "danger" : ""}"><div class="label">Active alerts</div>
        <div class="value">${esc(fmtInt(alertList.length))}</div>
        <div class="delta">org ${esc(complianceState.org)}</div></div>
    </div>
    <div class="grid-2">
      <div class="card"><h2>Score trend (1 = passed, 0 = failed)</h2>
        ${trendPoints.length
          ? lineChart(trendPoints, { valueKey: "score", labelKey: "label", fmt: (v) => v.toFixed(1), label: "Compliance score per check" })
          : `<div class="empty">No pass/fail checks recorded yet.</div>`}</div>
      <div class="card"><h2>Recent issues</h2><div class="table-wrap"><table>
        <thead><tr><th>Time</th><th>Severity</th><th>Description</th></tr></thead>
        <tbody>${issueRows || `<tr><td colspan="3"><div class="empty">No failed checks.</div></td></tr>`}</tbody>
      </table></div></div>
    </div>
    <div class="card"><h2>Alerts</h2><div class="table-wrap"><table>
      <thead><tr><th>ID</th><th>Severity</th><th>Status</th><th>Created</th><th>Description</th></tr></thead>
      <tbody>${alertRows || `<tr><td colspan="5"><div class="empty">No active alerts.</div></td></tr>`}</tbody>
    </table></div></div>`;
}

async function loadComplianceCatalog() {
  const out = document.getElementById("cp-catalog");
  try {
    const [regs, useCases] = await Promise.all([
      backendApi("/v1/compliance/regulations"),
      backendApi("/v1/compliance/healthcare/use-cases"),
    ]);
    out.innerHTML = `<div class="grid-2">
      <div class="card"><h2>Regulations (<code>GET /v1/compliance/regulations</code>)</h2>
        ${(regs || []).map((r) => `<span class="chip">${esc(r)}</span>`).join("") || `<div class="empty">none</div>`}</div>
      <div class="card"><h2>Healthcare use cases (<code>GET /v1/compliance/healthcare/use-cases</code>)</h2>
        ${(useCases || []).map((u) => `<span class="chip">${esc(u)}</span>`).join("") || `<div class="empty">none</div>`}</div>
    </div>`;
  } catch (err) {
    out.innerHTML = complianceUnreachable(err);
  }
}

async function renderCompliance() {
  view.innerHTML = `
    <h1>Compliance</h1>
    <p class="page-sub">Regulation checks recorded by the MultiMind API (<code>multimind backend</code>),
      read live from its <code>/v1/compliance</code> routes.</p>
    <div class="filters">
      <label ${sameOrigin() ? "hidden" : ""}>Backend URL <input id="cp-base" type="text" value="${esc(conn.backendBase)}"></label>
      <label>Organization id <input id="cp-org" type="text" value="${esc(complianceState.org)}"></label>
      <button id="cp-load" type="button">Load</button>
    </div>
    <div id="cp-metrics"></div>
    <div id="cp-catalog"></div>
    <div class="card"><h2>Record a check</h2>
      <p class="muted">Posts to <code>POST /v1/compliance/monitor</code>; the text is PII-scanned and the
        outcome is appended to the organization's history above. <strong>Generate report</strong> sends the
        same body to <code>POST /v1/compliance/report</code>.</p>
      <form id="cp-form">
        <div class="filters">
          <label>Organization name <input id="cp-org-name" type="text" value="${esc(remembered("multimind.orgName", "Acme Corp"))}"></label>
          <label>DPO email <input id="cp-dpo" type="text" value="${esc(remembered("multimind.dpoEmail", "dpo@example.com"))}"></label>
          <label>Regulations (comma-separated) <input id="cp-regs" type="text" value="GDPR"></label>
        </div>
        <div class="form-row"><label class="row-label" for="cp-text">Text to check (metadata.text)</label>
          <textarea id="cp-text" rows="5" placeholder="Prompt or document to evaluate&hellip;"></textarea></div>
        <button type="submit" class="primary">Run check</button>
        <button type="button" id="cp-report">Generate report</button>
        <span id="cp-msg" class="form-msg" role="status"></span>
      </form>
      <div id="cp-result"></div>
    </div>
    <div class="card"><h2>Alert rules</h2>
      <p class="muted"><code>POST /v1/compliance/alerts/configure?organization_id=…</code> stores the rules for this organization in the backend process.</p>
      <form id="cp-alert-form">
        <div class="filters">
          <label><input id="cp-alert-enabled" type="checkbox" checked> Alerts enabled</label>
          <label>Minimum severity
            <select id="cp-alert-sev"><option>low</option><option>medium</option><option selected>high</option><option>critical</option></select></label>
          <label>Notify email <input id="cp-alert-email" type="text" value="${esc(remembered("multimind.dpoEmail", "dpo@example.com"))}"></label>
        </div>
        <button type="submit">Save alert rules</button>
        <span id="cp-alert-msg" class="form-msg" role="status"></span>
      </form>
    </div>`;
  const syncInputs = () => {
    const baseInput = document.getElementById("cp-base");
    conn.backendBase = baseInput.value.trim() || DEFAULT_BACKEND_BASE;
    complianceState.org = document.getElementById("cp-org").value.trim() || "org-123";
    remember("multimind.backendBase", conn.backendBase);
    remember("multimind.orgId", complianceState.org);
  };
  const checkBody = () => {
    const orgName = document.getElementById("cp-org-name").value.trim();
    const dpo = document.getElementById("cp-dpo").value.trim();
    remember("multimind.orgName", orgName);
    remember("multimind.dpoEmail", dpo);
    return {
      organization_id: complianceState.org,
      organization_name: orgName,
      dpo_email: dpo,
      enabled_regulations: document.getElementById("cp-regs").value.split(",").map((s) => s.trim()).filter(Boolean),
      compliance_rules: {},
      metadata: { text: document.getElementById("cp-text").value },
    };
  };
  document.getElementById("cp-load").onclick = () => { syncInputs(); loadComplianceMetrics(); loadComplianceCatalog(); };
  const msg = document.getElementById("cp-msg");
  const out = document.getElementById("cp-result");
  bindAction(document.getElementById("cp-form"), msg, async () => {
    syncInputs();
    const r = await backendApi("/v1/compliance/monitor", { method: "POST", json: checkBody() });
    const fe = r.final_evaluation || {};
    // Backend statuses: checked (all clear), attention (findings), unverifiable (no evidence).
    setMsg(msg, fe.status === "checked" ? "ok" : "error", `Result: ${fe.status || "recorded"}`);
    out.innerHTML = jsonBlock(r);
    loadComplianceMetrics();
  }, { event: "submit", busy: "Running…" });
  bindAction(document.getElementById("cp-report"), msg, async () => {
    syncInputs();
    const r = await backendApi("/v1/compliance/report", { method: "POST", json: checkBody() });
    setMsg(msg, "ok", "Report generated.");
    out.innerHTML = jsonBlock(r);
  }, { busy: "Generating report…" });
  const alertMsg = document.getElementById("cp-alert-msg");
  bindAction(document.getElementById("cp-alert-form"), alertMsg, async () => {
    syncInputs();
    const rules = {
      enabled: document.getElementById("cp-alert-enabled").checked,
      min_severity: document.getElementById("cp-alert-sev").value,
      notify_email: document.getElementById("cp-alert-email").value.trim(),
    };
    const r = await backendApi(`/v1/compliance/alerts/configure?organization_id=${encodeURIComponent(complianceState.org)}`,
      { method: "POST", json: rules });
    setMsg(alertMsg, "ok", r.message || r.status || "Saved.");
  }, { event: "submit", busy: "Saving…" });
  loadComplianceMetrics();
  loadComplianceCatalog();
}

/* ---------- playground: features overview ---------- */

const FEATURE_VIEWS = {
  chat: "chat", compare: "compare", models: "models", sessions: "chat", "session-switch": "chat",
  "compliance-monitor": "compliance", "guard-proxy": "chat", "scan-text": "scan", audit: "audit",
  evidence: "evidence", inventory: "inventory", costs: "costs", guardrails: "guardrails", rag: "rag",
  hallucination: "hallucination", "context-transfer": "context", agents: "agents", memory: "memory",
  "vector-stores": "vectorstores", "synthetic-qa": "finetune", "fine-tuning": "finetune",
  "model-conversion": "conversion", "regulatory-watch": "regulatory", "differential-privacy": "dp",
  mcp: "mcp", integrations: "integrations", tracing: "tracing",
};

async function renderFeatures() {
  let data;
  try {
    data = await backendApi("/v1/features");
  } catch (err) {
    view.innerHTML = `<h1>Playground</h1>${modeBanner()}${errorCard(err)}`;
    return;
  }
  const rows = data.features.map((f) => {
    const target = FEATURE_VIEWS[f.name];
    const routes = (f.routes || []).map((r) => `<code class="chip">${esc(r)}</code>`).join(" ");
    return `<tr>
      <td><strong>${esc(f.name)}</strong></td>
      <td><code>${esc(f.module)}</code></td>
      <td>${okBadge(f.available, "available", "unavailable")}</td>
      <td class="wrap">${f.available ? "" : esc(f.reason || "")}${f.install ? `<br><code>${esc(f.install)}</code>` : ""}</td>
      <td class="wrap">${routes}</td>
      <td>${target ? `<a href="#/${target}">Open</a>` : "&mdash;"}</td>
    </tr>`;
  }).join("");
  const available = data.features.filter((f) => f.available).length;
  view.innerHTML = `
    <h1>Playground</h1>
    <p class="page-sub">Every feature in <code>GET /v1/features</code>, whether its module imports in the
      server process, the routes it exposes, and the view that exercises them end to end.</p>
    ${modeBanner()}
    <div class="tiles">
      <div class="tile"><div class="label">Features</div><div class="value">${esc(fmtInt(data.count))}</div></div>
      <div class="tile"><div class="label">Available</div><div class="value">${esc(fmtInt(available))}</div>
        <div class="delta">${esc(fmtInt(data.count - available))} need an extra</div></div>
      <div class="tile"><div class="label">API key</div><div class="value">${readApiKey() ? "set" : "none"}</div>
        <div class="delta"><a href="#/settings">Settings</a></div></div>
    </div>
    <div class="card"><div class="table-wrap"><table>
      <thead><tr><th>Feature</th><th>Module</th><th>Available</th><th>Reason / install</th><th>Routes</th><th>View</th></tr></thead>
      <tbody>${rows}</tbody></table></div></div>`;
}

/* ---------- settings ---------- */

function flattenChanges(before, after) {
  // Flat "a.b.c" keys for values that differ; arrays compared by JSON.
  const out = {};
  const same = (x, y) => JSON.stringify(x) === JSON.stringify(y);
  for (const key of ["default_model", "log_level", "cors_origins"]) {
    if (!same(before[key], after[key])) out[key] = after[key];
  }
  for (const key of Object.keys(after.proxy || {})) {
    if (!same(before.proxy[key], after.proxy[key])) out[`proxy.${key}`] = after.proxy[key];
  }
  for (const [name, p] of Object.entries(after.providers || {})) {
    for (const key of ["enabled", "model_name", "base_url"]) {
      if (!same(before.providers[name][key], p[key])) out[`providers.${name}.${key}`] = p[key];
    }
  }
  return out;
}

async function renderSettings() {
  const hasKey = Boolean(readApiKey());
  view.innerHTML = `
    <h1>Settings</h1>
    <p class="page-sub">Connection, API key, and the runtime config store (<code>GET/PUT /api/config</code>).</p>
    ${modeBanner()}
    <div class="card"><h2>Connection</h2>
      <form id="st-conn">
        <div class="form-row"><label class="row-label" for="st-key">API key (sent as <code>X-API-Key</code> on every call; stored only in this browser)</label>
          <input id="st-key" type="password" autocomplete="off" placeholder="${hasKey ? "key set" : "none"}" value="${esc(readApiKey())}">
          <button type="button" id="st-key-clear">Clear</button></div>
        <div ${sameOrigin() ? "hidden" : ""}>
          <div class="form-row"><label class="row-label" for="st-backend">Backend URL (<code>multimind backend</code>, for <code>/v1/*</code> and RAG routes)</label>
            <input id="st-backend" type="text" value="${esc(conn.backendBase)}"></div>
          <div class="form-row"><label class="row-label" for="st-proxy">Proxy URL (<code>multimind serve</code>)</label>
            <input id="st-proxy" type="text" value="${esc(conn.proxyBase)}"></div>
        </div>
        <button type="submit" class="primary">Save connection</button>
        <span id="st-conn-msg" class="form-msg" role="status"></span>
      </form>
    </div>
    <div id="st-config"><p class="page-sub">Loading config&hellip;</p></div>`;
  const connMsg = document.getElementById("st-conn-msg");
  document.getElementById("st-key-clear").onclick = () => { document.getElementById("st-key").value = ""; };
  bindAction(document.getElementById("st-conn"), connMsg, async () => {
    remember("multimind.apiKey", document.getElementById("st-key").value.trim());
    conn.backendBase = document.getElementById("st-backend").value.trim() || DEFAULT_BACKEND_BASE;
    conn.proxyBase = document.getElementById("st-proxy").value.trim() || DEFAULT_PROXY_BASE;
    remember("multimind.backendBase", conn.backendBase);
    remember("multimind.proxyBase", conn.proxyBase);
    await detectMode();
    setMsg(connMsg, "ok", `Saved. Mode: ${conn.mode}. API key ${readApiKey() ? "set" : "cleared"}.`);
  }, { event: "submit", busy: "Saving…" });
  await renderConfigForm();
}

async function renderConfigForm() {
  const box = document.getElementById("st-config");
  let data;
  try {
    data = await api("/api/config");
  } catch (err) {
    box.innerHTML = errorCard(err);
    return;
  }
  const eff = data.effective;
  const providers = Object.keys(eff.providers);
  const levels = ["DEBUG", "INFO", "WARNING", "ERROR"];
  const providerRows = providers.map((name) => {
    const p = eff.providers[name];
    return `<tr data-provider="${esc(name)}">
      <td><strong>${esc(name)}</strong></td>
      <td><input type="checkbox" class="pv-enabled" ${p.enabled ? "checked" : ""}></td>
      <td><input type="text" class="pv-model" value="${esc(p.model_name ?? "")}"></td>
      <td><input type="text" class="pv-base" value="${esc(p.base_url ?? "")}"></td>
      <td>${okBadge(p.has_api_key, "yes", "no")}</td>
      <td><button type="button" class="pv-test">Test</button> <span class="form-msg pv-test-msg" role="status"></span></td>
    </tr>`;
  }).join("");
  box.innerHTML = `
    <div class="card"><h2>Runtime config</h2>
      <p class="muted">Store: <code>${esc(data.path)}</code>${data.exists ? "" : " (not created yet)"}. Keys are read from the
        environment and never shown or edited here.</p>
      <form id="cfg-form">
        <div class="filters">
          <label>Default model <select id="cfg-default">${providers.map((n) => `<option ${n === eff.default_model ? "selected" : ""}>${esc(n)}</option>`).join("")}</select></label>
          <label>Log level <select id="cfg-log">${levels.map((l) => `<option ${l === String(eff.log_level).toUpperCase() ? "selected" : ""}>${l}</option>`).join("")}</select></label>
          <label>CORS origins (comma-separated) <input id="cfg-cors" type="text" value="${esc((eff.cors_origins || []).join(", "))}"></label>
        </div>
        <h2>Proxy guardrails</h2>
        <div class="filters">
          <label>Strategy <select id="cfg-strategy">${["mask", "hash", "remove"].map((s) => `<option ${s === eff.proxy.strategy ? "selected" : ""}>${s}</option>`).join("")}</select></label>
          <label>Block on (comma-separated PII types) <input id="cfg-block" type="text" value="${esc((eff.proxy.block_on || []).join(", "))}"></label>
          <label>Budget (USD, empty = none) <input id="cfg-budget" type="number" min="0" step="0.01" value="${eff.proxy.budget ?? ""}"></label>
          <label><input id="cfg-scan-output" type="checkbox" ${eff.proxy.scan_output ? "checked" : ""}> Scan output</label>
        </div>
        <h2>Providers</h2>
        <div class="table-wrap"><table id="cfg-providers">
          <thead><tr><th>Provider</th><th>Enabled</th><th>Model name</th><th>Base URL</th><th>Key set in env</th><th>Test</th></tr></thead>
          <tbody>${providerRows}</tbody></table></div>
        <div class="toolbar" style="margin-top:1rem">
          <button type="submit" class="primary">Save changed keys</button>
          <span id="cfg-msg" class="form-msg" role="status"></span>
        </div>
        <div class="note" id="cfg-hint" hidden></div>
      </form>
    </div>`;
  const msg = document.getElementById("cfg-msg");
  const collect = () => {
    const budgetRaw = document.getElementById("cfg-budget").value;
    const after = {
      default_model: document.getElementById("cfg-default").value,
      log_level: document.getElementById("cfg-log").value,
      cors_origins: document.getElementById("cfg-cors").value.split(",").map((s) => s.trim()).filter(Boolean),
      proxy: {
        strategy: document.getElementById("cfg-strategy").value,
        block_on: document.getElementById("cfg-block").value.split(",").map((s) => s.trim()).filter(Boolean),
        budget: budgetRaw === "" ? null : Number(budgetRaw),
        scan_output: document.getElementById("cfg-scan-output").checked,
      },
      providers: {},
    };
    for (const tr of document.querySelectorAll("#cfg-providers tbody tr")) {
      after.providers[tr.dataset.provider] = {
        enabled: tr.querySelector(".pv-enabled").checked,
        model_name: tr.querySelector(".pv-model").value.trim(),
        base_url: tr.querySelector(".pv-base").value.trim() || null,
      };
    }
    return after;
  };
  const normalizedBefore = {
    ...eff,
    log_level: String(eff.log_level).toUpperCase(),
    providers: Object.fromEntries(Object.entries(eff.providers).map(([n, p]) =>
      [n, { enabled: p.enabled, model_name: p.model_name ?? "", base_url: p.base_url || null }])),
  };
  bindAction(document.getElementById("cfg-form"), msg, async () => {
    const changes = flattenChanges(normalizedBefore, collect());
    if (!Object.keys(changes).length) { setMsg(msg, "", "Nothing changed."); return; }
    const saved = await api("/api/config", { method: "PUT", json: changes });
    setMsg(msg, "ok", `Saved ${Object.keys(changes).length} key(s).`);
    const hint = document.getElementById("cfg-hint");
    hint.hidden = false;
    hint.textContent = saved.apply_hint || "Saved.";
    await renderConfigForm();
    const again = document.getElementById("cfg-hint");
    again.hidden = false;
    again.textContent = saved.apply_hint || "Saved.";
  }, { event: "submit", busy: "Saving…" });
  for (const tr of document.querySelectorAll("#cfg-providers tbody tr")) {
    const testMsg = tr.querySelector(".pv-test-msg");
    bindAction(tr.querySelector(".pv-test"), testMsg, async () => {
      const r = await api("/api/config/test-provider", { method: "POST", json: { provider: tr.dataset.provider } });
      if (r.ok) setMsg(testMsg, "ok", `ok, ${r.latency_ms} ms`);
      else setMsg(testMsg, "error", `${r.latency_ms} ms: ${r.error}`);
    }, { busy: "Testing…" });
  }
}

/* ---------- models ---------- */

async function renderModels() {
  const { models, error } = await loadModels();
  let defaultModel = "";
  try { defaultModel = (await api("/api/config")).effective.default_model; } catch { /* dashboard config optional */ }
  const rows = Object.entries(models).map(([name, m]) => `<tr>
      <td><strong>${esc(name)}</strong>${name === defaultModel ? ' <span class="badge">default</span>' : ""}</td>
      <td>${okBadge(m.status === "available", "available", "unavailable")}</td>
      <td>${esc(m.config?.model_name ?? "")}</td>
      <td class="num">${esc(m.config?.temperature ?? "")}</td>
      <td class="num">${esc(m.config?.max_tokens ?? "")}</td>
    </tr>`).join("");
  view.innerHTML = `
    <h1>Models</h1>
    <p class="page-sub"><code>GET /v1/models</code>: enabled providers and whether a key or base URL makes them usable.
      Change models, keys, and enabled flags in <a href="#/settings">Settings</a>.</p>
    ${modelsNote(error)}
    <div class="card"><div class="table-wrap"><table>
      <thead><tr><th>Provider</th><th>Status</th><th>Model name</th><th class="num">Temperature</th><th class="num">Max tokens</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="5"><div class="empty">No providers listed.</div></td></tr>`}</tbody></table></div></div>
    <div class="grid-2">
      <div class="card"><h2>Metrics (<code>GET /v1/metrics</code>)</h2>
        <div class="toolbar"><button id="md-metrics">Load metrics</button><span id="md-metrics-msg" class="form-msg" role="status"></span></div>
        <div id="md-metrics-out"></div></div>
      <div class="card"><h2>Health (<code>POST /v1/health/check</code>)</h2>
        <div class="filters"><label>Model (optional) <select id="md-health-model"><option value="">all available</option>${modelOptions(models, "")}</select></label>
          <button id="md-health">Check health</button><span id="md-health-msg" class="form-msg" role="status"></span></div>
        <div id="md-health-out"></div></div>
    </div>`;
  const mMsg = document.getElementById("md-metrics-msg");
  bindAction(document.getElementById("md-metrics"), mMsg, async () => {
    const r = await backendApi("/v1/metrics");
    const entries = Object.entries(r.metrics || {});
    document.getElementById("md-metrics-out").innerHTML = entries.length
      ? `<div class="table-wrap"><table><thead><tr><th>Model</th><th>Metrics</th></tr></thead><tbody>${entries.map(([m, v]) =>
          `<tr><td>${esc(m)}</td><td class="wrap"><code>${esc(JSON.stringify(v))}</code></td></tr>`).join("")}</tbody></table></div>`
      : `<div class="empty">No requests tracked in this backend process yet.</div>`;
    setMsg(mMsg, "ok", `${entries.length} model(s)`);
  }, { busy: "Loading…" });
  const hMsg = document.getElementById("md-health-msg");
  bindAction(document.getElementById("md-health"), hMsg, async () => {
    const model = document.getElementById("md-health-model").value;
    const r = await backendApi(`/v1/health/check${model ? `?model=${encodeURIComponent(model)}` : ""}`, { method: "POST" });
    document.getElementById("md-health-out").innerHTML = jsonBlock(r);
    setMsg(hMsg, "ok", "Done.");
  }, { busy: "Checking…" });
}

/* ---------- chat playground ---------- */

const chatState = { messages: [], model: remembered("multimind.chatModel", ""), session: null };

function chatMessagesHtml() {
  if (!chatState.messages.length) return `<div class="empty">No messages yet.</div>`;
  return chatState.messages.map((m) => `<div class="msg ${esc(m.role)}">
      <div class="msg-role">${esc(m.role)}${m.meta ? ` <span class="muted">${esc(m.meta)}</span>` : ""}</div>
      <div class="msg-body">${esc(m.content)}</div></div>`).join("");
}

async function renderChat() {
  const { models, error } = await loadModels();
  if (!chatState.model || !models[chatState.model]) chatState.model = Object.keys(models)[0] || chatState.model || "openai";
  view.innerHTML = `
    <h1>Chat</h1>
    <p class="page-sub"><code>POST /v1/chat</code> with the message history, or <code>POST /v1/generate</code> for a single prompt.
      Guarded mode sends the same prompt through the proxy (<code>POST /v1/chat/completions</code>) so redaction is visible.</p>
    ${modelsNote(error)}
    <div class="card">
      <div class="filters">
        <label>Model <select id="ch-model">${modelOptions(models, chatState.model)}</select></label>
        <label>Mode <select id="ch-mode">
          <option value="chat">chat (history)</option>
          <option value="generate">generate (no chat)</option></select></label>
        <label><input id="ch-guarded" type="checkbox"> Guarded (via proxy)</label>
        <label id="ch-proxy-model-row" hidden>Upstream model name <input id="ch-proxy-model" type="text" value="${esc(remembered("multimind.proxyModel", "gpt-4o-mini"))}"></label>
        <button type="button" id="ch-clear">Clear history</button>
      </div>
      <div id="ch-messages" class="messages">${chatMessagesHtml()}</div>
      <form id="ch-form">
        <div class="form-row"><textarea id="ch-input" rows="3" required placeholder="Say something&hellip;"></textarea></div>
        <button type="submit" class="primary">Send</button>
        <button type="button" id="ch-embed" hidden>Embed via proxy</button>
        <span id="ch-msg" class="form-msg" role="status"></span>
      </form>
      <div id="ch-guard-out"></div>
    </div>
    <div class="card" id="ch-sessions"><h2>Sessions</h2><p class="muted">Loading&hellip;</p></div>`;
  const modelSel = document.getElementById("ch-model");
  modelSel.onchange = () => { chatState.model = modelSel.value; remember("multimind.chatModel", chatState.model); };
  const guarded = document.getElementById("ch-guarded");
  const syncGuard = () => {
    document.getElementById("ch-proxy-model-row").hidden = !guarded.checked;
    document.getElementById("ch-embed").hidden = !guarded.checked;
  };
  guarded.onchange = syncGuard;
  const list = document.getElementById("ch-messages");
  const msg = document.getElementById("ch-msg");
  const guardOut = document.getElementById("ch-guard-out");
  document.getElementById("ch-clear").onclick = () => { chatState.messages = []; list.innerHTML = chatMessagesHtml(); guardOut.innerHTML = ""; };
  const proxyModel = () => {
    const v = document.getElementById("ch-proxy-model").value.trim() || "gpt-4o-mini";
    remember("multimind.proxyModel", v);
    return v;
  };
  bindAction(document.getElementById("ch-form"), msg, async () => {
    const input = document.getElementById("ch-input");
    const text = input.value.trim();
    if (!text) return;
    const mode = document.getElementById("ch-mode").value;
    const started = performance.now();
    chatState.messages.push({ role: "user", content: text });
    list.innerHTML = chatMessagesHtml();
    if (guarded.checked) {
      // Show what the proxy will redact, then call it with the same prompt.
      let scan = null;
      try {
        scan = await api("/api/scan-text", { method: "POST", json: { text, strategy: "mask" } });
      } catch { /* scan is illustrative; the proxy call is the real test */ }
      const r = await externalApi(proxyBase(), "/v1/chat/completions", {
        method: "POST",
        json: { model: proxyModel(), messages: chatState.messages.map((m) => ({ role: m.role, content: m.content })) },
      });
      const reply = r.choices?.[0]?.message?.content ?? "";
      const ms = Math.round(performance.now() - started);
      chatState.messages.push({ role: "assistant", content: reply, meta: `proxy, ${ms} ms${r.usage ? `, ${fmtInt(r.usage.total_tokens)} tokens` : ""}` });
      guardOut.innerHTML = `<h2>Redaction effects</h2>
        ${scan ? `<p class="muted">PII in your prompt: ${Object.entries(scan.counts).map(([t, n]) => `<span class="chip">${esc(t)}&times;${esc(n)}</span>`).join("") || "none"}</p>
          <pre class="redacted">${esc(scan.redacted ?? text)}</pre>` : ""}
        <p class="muted">Proxy reply (output scanned when <code>scan_output</code> is on) and usage:</p>${jsonBlock({ usage: r.usage ?? null, model: r.model })}`;
    } else if (mode === "generate") {
      const r = await backendApi("/v1/generate", { method: "POST", json: { prompt: text, model: chatState.model } });
      const ms = Math.round(performance.now() - started);
      chatState.messages.push({ role: "assistant", content: r.content, meta: `${r.model}, generate, ${ms} ms${r.usage ? `, ${fmtInt(r.usage.total_tokens)} tokens` : ""}` });
    } else {
      const r = await backendApi("/v1/chat", {
        method: "POST",
        json: { model: chatState.model, messages: chatState.messages.map((m) => ({ role: m.role, content: m.content })) },
      });
      const ms = Math.round(performance.now() - started);
      chatState.messages.push({ role: "assistant", content: r.content, meta: `${r.model}, ${ms} ms${r.usage ? `, ${fmtInt(r.usage.total_tokens)} tokens` : ""}` });
    }
    input.value = "";
    list.innerHTML = chatMessagesHtml();
    setMsg(msg, "ok", "Reply received.");
  }, { event: "submit", busy: "Sending…" });
  bindAction(document.getElementById("ch-embed"), msg, async () => {
    const text = document.getElementById("ch-input").value.trim();
    if (!text) { setMsg(msg, "error", "Type the text to embed first."); return; }
    const r = await externalApi(proxyBase(), "/v1/embeddings", {
      method: "POST", json: { model: "text-embedding-3-small", input: text },
    });
    const vec = r.data?.[0]?.embedding || [];
    guardOut.innerHTML = `<h2>Embedding via proxy</h2><p class="muted">${esc(vec.length)} dimensions; input was PII-redacted by the proxy before upstream.</p>
      ${jsonBlock({ model: r.model, usage: r.usage ?? null, first_values: vec.slice(0, 8) })}`;
    setMsg(msg, "ok", "Embedded.");
  }, { busy: "Embedding…" });
  await renderSessionsPanel(models);
}

async function renderSessionsPanel(models) {
  const box = document.getElementById("ch-sessions");
  let sessions = [];
  try {
    sessions = await backendApi("/v1/sessions");
  } catch (err) {
    box.innerHTML = `<h2>Sessions</h2>${errorCard(err)}`;
    return;
  }
  const rows = sessions.map((s) => `<tr class="${s.session_id === chatState.session ? "active-row" : ""}">
      <td><code>${esc(s.session_id)}</code></td><td>${esc(s.model)}</td><td class="num">${esc(fmtInt(s.message_count))}</td>
      <td>${esc((s.updated_at || "").replace("T", " ").slice(0, 19))}</td>
      <td><button type="button" class="ss-open" data-id="${esc(s.session_id)}">Open</button>
        <button type="button" class="ss-save" data-id="${esc(s.session_id)}">Save</button>
        <button type="button" class="ss-delete" data-id="${esc(s.session_id)}">Delete</button></td></tr>`).join("");
  box.innerHTML = `<h2>Sessions</h2>
    <p class="muted"><code>GET/POST /v1/sessions</code>, <code>/{id}/messages</code>, <code>/{id}/save</code>, <code>/load</code>,
      <code>/{id}/switch</code> (ModelSession strategies), <code>DELETE /{id}</code>.</p>
    <div class="filters">
      <label>New session model <select id="ss-model">${modelOptions(models, chatState.model)}</select></label>
      <label>System prompt <input id="ss-system" type="text" placeholder="optional"></label>
      <button type="button" id="ss-create">Create</button>
      <label>Load by id <input id="ss-load-id" type="text" placeholder="chat_… id"></label>
      <button type="button" id="ss-load">Load</button>
      <span id="ss-msg" class="form-msg" role="status"></span>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>Session</th><th>Model</th><th class="num">Messages</th><th>Updated</th><th></th></tr></thead>
      <tbody>${rows || `<tr><td colspan="5"><div class="empty">No sessions in this backend process.</div></td></tr>`}</tbody></table></div>
    <div id="ss-detail"></div>`;
  const msg = document.getElementById("ss-msg");
  bindAction(document.getElementById("ss-create"), msg, async () => {
    const r = await backendApi("/v1/sessions", {
      method: "POST",
      json: { model: document.getElementById("ss-model").value, system_prompt: document.getElementById("ss-system").value.trim() || null, metadata: {} },
    });
    chatState.session = r.session_id;
    await renderSessionsPanel(models);
  }, { busy: "Creating…" });
  bindAction(document.getElementById("ss-load"), msg, async () => {
    const r = await backendApi("/v1/sessions/load", { method: "POST", json: { session_id: document.getElementById("ss-load-id").value.trim() } });
    chatState.session = r.session_id;
    await renderSessionsPanel(models);
  }, { busy: "Loading…" });
  for (const b of box.querySelectorAll(".ss-open")) b.onclick = () => { chatState.session = b.dataset.id; renderSessionDetail(models); };
  for (const b of box.querySelectorAll(".ss-save")) {
    bindAction(b, msg, async () => {
      const r = await backendApi(`/v1/sessions/${encodeURIComponent(b.dataset.id)}/save`, { method: "POST" });
      setMsg(msg, "ok", `Saved to ${r.path}`);
    }, { busy: "Saving…" });
  }
  for (const b of box.querySelectorAll(".ss-delete")) {
    bindAction(b, msg, async () => {
      await backendApi(`/v1/sessions/${encodeURIComponent(b.dataset.id)}`, { method: "DELETE" });
      if (chatState.session === b.dataset.id) chatState.session = null;
      await renderSessionsPanel(models);
    }, { busy: "Deleting…" });
  }
  if (chatState.session) await renderSessionDetail(models);
}

async function renderSessionDetail(models) {
  const box = document.getElementById("ss-detail");
  const id = chatState.session;
  let s;
  try {
    s = await backendApi(`/v1/sessions/${encodeURIComponent(id)}`);
  } catch (err) {
    box.innerHTML = errorCard(err);
    return;
  }
  const msgs = (s.messages || []).map((m) => `<div class="msg ${esc(m.role)}"><div class="msg-role">${esc(m.role)} <span class="muted">${esc(m.model || "")}</span></div>
      <div class="msg-body">${esc(m.content)}</div></div>`).join("") || `<div class="empty">Empty session.</div>`;
  box.innerHTML = `<div class="provider-head"><h2>Session <code>${esc(id)}</code></h2><span class="count">model ${esc(s.model)}</span></div>
    <div class="messages">${msgs}</div>
    <form id="sd-form"><div class="form-row"><textarea id="sd-input" rows="2" required placeholder="Message for this session&hellip;"></textarea></div>
      <button type="submit" class="primary">Send to session</button>
      <span id="sd-msg" class="form-msg" role="status"></span></form>
    <div class="filters" style="margin-top:1rem">
      <label>Switch to model <select id="sd-switch-model">${modelOptions(models, s.model)}</select></label>
      <label>Strategy <select id="sd-strategy"><option>full</option><option>summary</option><option>adapted</option></select></label>
      <button type="button" id="sd-switch">Switch model</button>
      <span id="sd-switch-msg" class="form-msg" role="status"></span>
    </div>
    <div id="sd-switch-out"></div>`;
  const msg = document.getElementById("sd-msg");
  bindAction(document.getElementById("sd-form"), msg, async () => {
    const content = document.getElementById("sd-input").value.trim();
    await backendApi(`/v1/sessions/${encodeURIComponent(id)}/messages`, { method: "POST", json: { role: "user", content } });
    await renderSessionDetail(models);
  }, { event: "submit", busy: "Sending…" });
  const swMsg = document.getElementById("sd-switch-msg");
  bindAction(document.getElementById("sd-switch"), swMsg, async () => {
    const r = await backendApi(`/v1/sessions/${encodeURIComponent(id)}/switch`, {
      method: "POST",
      json: { model: document.getElementById("sd-switch-model").value, strategy: document.getElementById("sd-strategy").value },
    });
    document.getElementById("sd-switch-out").innerHTML = jsonBlock(r);
    setMsg(swMsg, "ok", `Now on ${r.model} (${r.strategy_applied}), ${r.message_count} message(s).`);
  }, { busy: "Switching…" });
}

/* ---------- compare ---------- */

async function renderCompare() {
  const { models, error } = await loadModels();
  const checks = Object.keys(models).map((m) => `<label><input type="checkbox" name="cmp-model" value="${esc(m)}" ${models[m].status === "available" ? "checked" : ""}> ${esc(m)}</label>`).join("")
    || `<label><input type="checkbox" name="cmp-model" value="openai" checked> openai</label>`;
  view.innerHTML = `
    <h1>Compare</h1>
    <p class="page-sub"><code>POST /v1/compare</code>: one prompt to several providers, side by side. Unavailable models are skipped by the server.</p>
    ${modelsNote(error)}
    <div class="card"><form id="cmp-form">
      <div class="form-row"><label class="row-label" for="cmp-prompt">Prompt</label><textarea id="cmp-prompt" rows="4" required></textarea></div>
      <div class="form-row"><span class="row-label">Models</span><div class="check-grid">${checks}</div></div>
      <button type="submit" class="primary">Compare</button>
      <span id="cmp-msg" class="form-msg" role="status"></span>
    </form></div>
    <div id="cmp-out" class="grid-2"></div>`;
  const msg = document.getElementById("cmp-msg");
  bindAction(document.getElementById("cmp-form"), msg, async () => {
    const chosen = [...document.querySelectorAll('input[name="cmp-model"]:checked')].map((i) => i.value);
    const r = await backendApi("/v1/compare", { method: "POST", json: { prompt: document.getElementById("cmp-prompt").value, models: chosen } });
    const got = Object.entries(r.responses || {});
    const skipped = chosen.filter((m) => !(m in (r.responses || {})));
    document.getElementById("cmp-out").innerHTML = got.map(([m, resp]) => `<div class="card"><h2>${esc(m)}</h2>
        <div class="msg-body">${esc(resp.content)}</div>
        <p class="muted">${resp.usage ? esc(JSON.stringify(resp.usage)) : "no usage reported"}</p></div>`).join("")
      || `<div class="card"><div class="empty">No model answered.</div></div>`;
    setMsg(msg, got.length ? "ok" : "error", `${got.length} response(s)` + (skipped.length ? `; skipped (unavailable): ${skipped.join(", ")}` : ""));
  }, { event: "submit", busy: "Comparing…" });
}

/* ---------- RAG ---------- */

function docsTable(docs) {
  return `<div class="table-wrap"><table><thead><tr><th>Score</th><th>Text</th><th>Metadata</th></tr></thead><tbody>${
    (docs || []).map((d) => `<tr><td class="num">${d.score == null ? "" : esc(Number(d.score).toFixed(3))}</td>
      <td class="wrap">${esc(d.text)}</td><td class="wrap"><code>${esc(JSON.stringify(d.metadata || {}))}</code></td></tr>`).join("")
    || `<tr><td colspan="3"><div class="empty">No documents returned.</div></td></tr>`}</tbody></table></div>`;
}

async function renderRag() {
  const hasToken = Boolean(remembered("multimind.ragToken", ""));
  view.innerHTML = `
    <h1>RAG</h1>
    <p class="page-sub">The RAG routes mounted by <code>multimind backend</code> when the RAG extra is installed:
      <code>/documents</code>, <code>/files</code>, <code>/query</code>, <code>/generate</code>, <code>/models/switch</code>, <code>/token</code>.</p>
    <div id="rag-status"></div>
    <div class="grid-2">
      <div class="card"><h2>Login (<code>POST /token</code>)</h2>
        <p class="muted">Only needed when the server has JWT users configured; the token is kept in this browser and sent as <code>Authorization: Bearer</code> on RAG calls. ${hasToken ? '<span class="badge ok">token stored</span>' : ""}</p>
        <form id="rag-login"><div class="filters">
          <label>Username <input id="rag-user" type="text"></label>
          <label>Password <input id="rag-pass" type="password"></label>
          <button type="submit">Get token</button><button type="button" id="rag-logout">Forget token</button></div>
          <span id="rag-login-msg" class="form-msg" role="status"></span></form></div>
      <div class="card"><h2>Switch generation model (<code>POST /models/switch</code>)</h2>
        <form id="rag-switch"><div class="filters">
          <label>Type <select id="rag-mtype"><option>openai</option><option>anthropic</option></select></label>
          <label>Model name <input id="rag-mname" type="text" value="gpt-4o-mini"></label>
          <button type="submit">Switch</button></div>
          <span id="rag-switch-msg" class="form-msg" role="status"></span></form></div>
    </div>
    <div class="card"><h2>Add documents (<code>POST /documents</code>)</h2>
      <form id="rag-add">
        <div class="form-row"><label class="row-label" for="rag-docs">Documents, one per blank-line-separated paragraph</label>
          <textarea id="rag-docs" rows="5" placeholder="MultiMind is an AI SDK.&#10;&#10;It ships a PII guard."></textarea></div>
        <div class="form-row"><label class="row-label" for="rag-meta">Metadata (JSON, applied to each)</label>
          <input id="rag-meta" type="text" value='{"source": "dashboard"}'></div>
        <button type="submit" class="primary">Add</button>
        <span id="rag-add-msg" class="form-msg" role="status"></span>
      </form>
      <form id="rag-upload" style="margin-top:1rem"><div class="filters">
        <label>Upload a file (<code>POST /files</code>) <input id="rag-file" type="file"></label>
        <button type="submit">Upload</button><span id="rag-upload-msg" class="form-msg" role="status"></span></div></form>
      <div class="toolbar" style="margin-top:1rem">
        <span><button type="button" id="rag-count">Count documents</button> <span id="rag-count-msg" class="form-msg" role="status"></span></span>
        <span><button type="button" id="rag-clear">Delete all documents</button> <span id="rag-clear-msg" class="form-msg" role="status"></span></span>
      </div>
    </div>
    <div class="card"><h2>Query and generate</h2>
      <form id="rag-query">
        <div class="form-row"><label class="row-label" for="rag-q">Question</label><input id="rag-q" type="text" required style="width:100%;max-width:720px"></div>
        <div class="filters"><label>Top k <input id="rag-topk" type="number" min="1" value="3"></label>
          <button type="submit">Query (<code>POST /query</code>)</button>
          <button type="button" id="rag-generate">Generate (<code>POST /generate</code>)</button>
          <span id="rag-query-msg" class="form-msg" role="status"></span></div>
      </form>
      <div id="rag-out"></div>
    </div>`;
  const status = document.getElementById("rag-status");
  try {
    const c = await backendApi("/documents/count");
    status.innerHTML = `<div class="banner ok">RAG routes are mounted. ${esc(fmtInt(c.count))} document(s) indexed.</div>`;
  } catch (err) {
    status.innerHTML = errorCard(err);
  }
  const loginMsg = document.getElementById("rag-login-msg");
  bindAction(document.getElementById("rag-login"), loginMsg, async () => {
    const form = new URLSearchParams({ username: document.getElementById("rag-user").value, password: document.getElementById("rag-pass").value });
    const r = await backendApi("/token", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: form.toString() });
    remember("multimind.ragToken", r.access_token);
    setMsg(loginMsg, "ok", "Token stored for RAG calls.");
  }, { event: "submit", busy: "Logging in…" });
  document.getElementById("rag-logout").onclick = () => { remember("multimind.ragToken", ""); setMsg(loginMsg, "ok", "Token forgotten."); };
  const switchMsg = document.getElementById("rag-switch-msg");
  bindAction(document.getElementById("rag-switch"), switchMsg, async () => {
    const form = new URLSearchParams({ model_type: document.getElementById("rag-mtype").value, model_name: document.getElementById("rag-mname").value });
    const r = await backendApi("/models/switch", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: form.toString() });
    setMsg(switchMsg, "ok", r.message || "Switched.");
  }, { event: "submit", busy: "Switching…" });
  const addMsg = document.getElementById("rag-add-msg");
  bindAction(document.getElementById("rag-add"), addMsg, async () => {
    let metadata = {};
    try { metadata = JSON.parse(document.getElementById("rag-meta").value || "{}"); } catch { throw new Error("Metadata is not valid JSON"); }
    const documents = splitParagraphs(document.getElementById("rag-docs").value).map((text) => ({ text, metadata }));
    if (!documents.length) throw new Error("Enter at least one document");
    const r = await backendApi("/documents", { method: "POST", json: { documents } });
    setMsg(addMsg, "ok", `Added ${r.total} document(s).`);
  }, { event: "submit", busy: "Adding…" });
  const upMsg = document.getElementById("rag-upload-msg");
  bindAction(document.getElementById("rag-upload"), upMsg, async () => {
    const file = document.getElementById("rag-file").files[0];
    if (!file) throw new Error("Choose a file first");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("metadata", document.getElementById("rag-meta").value || "{}");
    const r = await backendApi("/files", { method: "POST", body: fd });
    setMsg(upMsg, "ok", r.message || `Uploaded ${file.name}.`);
  }, { event: "submit", busy: "Uploading…" });
  const countMsg = document.getElementById("rag-count-msg");
  bindAction(document.getElementById("rag-count"), countMsg, async () => {
    const r = await backendApi("/documents/count");
    setMsg(countMsg, "ok", `${fmtInt(r.count)} document(s)`);
  }, { busy: "Counting…" });
  const clearMsg = document.getElementById("rag-clear-msg");
  bindAction(document.getElementById("rag-clear"), clearMsg, async () => {
    if (!window.confirm("Delete every document in the RAG index?")) { setMsg(clearMsg, "", "Cancelled."); return; }
    const r = await backendApi("/documents", { method: "DELETE" });
    setMsg(clearMsg, "ok", r.message || "Cleared.");
  }, { busy: "Deleting…" });
  const qMsg = document.getElementById("rag-query-msg");
  const out = document.getElementById("rag-out");
  const qBody = () => ({ query: document.getElementById("rag-q").value, top_k: Number(document.getElementById("rag-topk").value) || 3 });
  bindAction(document.getElementById("rag-query"), qMsg, async () => {
    const r = await backendApi("/query", { method: "POST", json: qBody() });
    out.innerHTML = `<h2>${esc(fmtInt(r.total))} match(es)</h2>${docsTable(r.documents)}`;
    setMsg(qMsg, "ok", "Queried.");
  }, { event: "submit", busy: "Querying…" });
  bindAction(document.getElementById("rag-generate"), qMsg, async () => {
    const r = await backendApi("/generate", { method: "POST", json: qBody() });
    out.innerHTML = `<h2>Answer</h2><div class="msg-body">${esc(r.text)}</div><h2 style="margin-top:1rem">Sources</h2>${docsTable(r.documents)}`;
    setMsg(qMsg, "ok", "Generated.");
  }, { busy: "Generating…" });
}

/* ---------- feature views ---------- */

function verdictBadge(v) {
  const s = String(v || "").toLowerCase();
  const cls = s === "supported" ? "ok" : s === "unsupported" || s === "contradicted" ? "danger" : "";
  return `<span class="badge ${cls}">${esc(s)}</span>`;
}

async function renderHallucination() {
  const { models } = await loadModels();
  view.innerHTML = `
    <h1>Hallucination</h1>
    <p class="page-sub"><code>POST /v1/evaluation/hallucination</code>: lexical grounding of each sentence against the sources
      (<code>HallucinationDetector.check_grounding</code>); an optional judge model adds claim verdicts.</p>
    <div class="card"><form id="hl-form">
      <div class="form-row"><label class="row-label" for="hl-response">Model response to check</label>
        <textarea id="hl-response" rows="4" required>Paris is the capital of France. It has 80 million people.</textarea></div>
      <div class="form-row"><label class="row-label" for="hl-sources">Sources, one per line</label>
        <textarea id="hl-sources" rows="4" required>Paris is the capital and largest city of France.</textarea></div>
      <div class="filters"><label>Judge model (optional) <select id="hl-judge"><option value="">none (lexical only)</option>${modelOptions(models, "")}</select></label></div>
      <button type="submit" class="primary">Check grounding</button>
      <span id="hl-msg" class="form-msg" role="status"></span>
    </form></div>
    <div id="hl-out"></div>`;
  const msg = document.getElementById("hl-msg");
  bindAction(document.getElementById("hl-form"), msg, async () => {
    const body = { response: document.getElementById("hl-response").value, sources: splitLines(document.getElementById("hl-sources").value) };
    const judge = document.getElementById("hl-judge").value;
    if (judge) body.judge_model = judge;
    const r = await backendApi("/v1/evaluation/hallucination", { method: "POST", json: body });
    const rows = r.sentences.map((s) => `<tr><td class="wrap">${esc(s.sentence)}</td><td>${verdictBadge(s.verdict)}</td>
        <td class="num">${esc(s.score.toFixed(2))}</td><td class="wrap muted">${esc(s.best_source_snippet || "")}</td></tr>`).join("");
    document.getElementById("hl-out").innerHTML = `
      <div class="tiles">
        <div class="tile ${r.score < 0.5 ? "danger" : ""}"><div class="label">Grounding score</div><div class="value">${esc(r.score.toFixed(2))}</div></div>
        <div class="tile ${r.unsupported_count ? "danger" : ""}"><div class="label">Unsupported sentences</div><div class="value">${esc(r.unsupported_count)}</div></div>
      </div>
      <div class="card"><p class="muted">${esc(r.summary)}</p><div class="table-wrap"><table>
        <thead><tr><th>Sentence</th><th>Verdict</th><th class="num">Score</th><th>Best source snippet</th></tr></thead>
        <tbody>${rows}</tbody></table></div></div>
      ${r.claims ? `<div class="card"><h2>Judge claims (score ${esc(r.claims.score)})</h2><p class="muted">${esc(r.claims.summary)}</p>
        ${r.claims.claims.map((c) => `<div>${verdictBadge(c.verdict)} ${esc(c.claim)}</div>`).join("")}</div>` : ""}`;
    setMsg(msg, "ok", `Score ${r.score.toFixed(2)}`);
  }, { event: "submit", busy: "Checking…" });
}

async function renderAgents() {
  const { models, error } = await loadModels();
  view.innerHTML = `
    <h1>Agents</h1>
    <p class="page-sub"><code>POST /v1/agents/run</code>: <code>multimind.agents.Agent</code> with the calculator tool; the answer and the tool-call trace come back.</p>
    ${modelsNote(error)}
    <div class="card"><form id="ag-form">
      <div class="filters">
        <label>Model <select id="ag-model">${modelOptions(models, chatState.model)}</select></label>
        <label>Max turns <input id="ag-turns" type="number" min="1" max="20" value="5"></label>
        <label><input id="ag-calc" type="checkbox" checked> calculator tool</label>
      </div>
      <div class="form-row"><label class="row-label" for="ag-task">Task</label><textarea id="ag-task" rows="3" required>What is 17 * 23?</textarea></div>
      <button type="submit" class="primary">Run agent</button>
      <span id="ag-msg" class="form-msg" role="status"></span>
    </form></div>
    <div id="ag-out"></div>`;
  const msg = document.getElementById("ag-msg");
  bindAction(document.getElementById("ag-form"), msg, async () => {
    const r = await backendApi("/v1/agents/run", {
      method: "POST",
      json: {
        model: document.getElementById("ag-model").value,
        task: document.getElementById("ag-task").value,
        tools: document.getElementById("ag-calc").checked ? ["calculator"] : [],
        max_turns: Number(document.getElementById("ag-turns").value) || 5,
      },
    });
    const calls = (r.tool_calls || []).map((c) => `<tr><td>${esc(c.tool || c.name || "")}</td><td class="wrap"><code>${esc(JSON.stringify(c.args ?? c.arguments ?? c.input ?? ""))}</code></td>
        <td class="wrap"><code>${esc(JSON.stringify(c.result ?? c.output ?? ""))}</code></td></tr>`).join("");
    document.getElementById("ag-out").innerHTML = `
      <div class="card"><h2>Answer</h2><div class="msg-body">${esc(r.answer ?? "")}</div>
        ${r.error ? `<p class="form-msg error">${esc(r.error)}</p>` : ""}
        <p class="muted">native tool calling: ${okBadge(r.native_tools)} tools: ${(r.tools || []).map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</p></div>
      <div class="card"><h2>Tool-call trace</h2><div class="table-wrap"><table><thead><tr><th>Tool</th><th>Arguments</th><th>Result</th></tr></thead>
        <tbody>${calls || `<tr><td colspan="3"><div class="empty">No tool calls.</div></td></tr>`}</tbody></table></div></div>`;
    setMsg(msg, r.error ? "error" : "ok", r.error ? "Agent reported an error." : "Done.");
  }, { event: "submit", busy: "Running…" });
}

async function renderMemory() {
  const [{ models }, data] = await Promise.all([loadModels(), backendApi("/v1/memory")]);
  const rows = data.kinds.map((k) => `<tr><td><strong>${esc(k.kind)}</strong></td><td><code>${esc(k.class)}</code></td>
      <td>${k.needs_model ? "yes" : "no"}</td><td>${okBadge(k.constructible)}${k.error ? ` <span class="muted">${esc(k.error)}</span>` : ""}</td>
      <td>${k.active ? '<span class="badge">active</span>' : ""}</td></tr>`).join("");
  view.innerHTML = `
    <h1>Memory</h1>
    <p class="page-sub"><code>GET /v1/memory</code> lists the demo kinds; messages go through the real class's
      <code>add_message</code>, <code>get_messages</code>, and <code>clear</code>. One instance per kind lives in the backend process.</p>
    <div class="card"><div class="table-wrap"><table>
      <thead><tr><th>Kind</th><th>Class</th><th>Needs model</th><th>Constructible</th><th></th></tr></thead><tbody>${rows}</tbody></table></div></div>
    <div class="card"><form id="mm-form">
      <div class="filters">
        <label>Kind <select id="mm-kind">${data.kinds.map((k) => `<option>${esc(k.kind)}</option>`).join("")}</select></label>
        <label>Role <select id="mm-role"><option>user</option><option>assistant</option><option>system</option></select></label>
        <label>Model (model-backed kinds) <select id="mm-model"><option value="">none</option>${modelOptions(models, "")}</select></label>
      </div>
      <div class="form-row"><textarea id="mm-content" rows="2" required>Remember that my meeting is at 3pm.</textarea></div>
      <button type="submit" class="primary">Add message</button>
      <button type="button" id="mm-list">List messages</button>
      <button type="button" id="mm-clear">Clear kind</button>
      <span id="mm-msg" class="form-msg" role="status"></span>
    </form><div id="mm-out"></div></div>`;
  const msg = document.getElementById("mm-msg");
  const out = document.getElementById("mm-out");
  const kind = () => encodeURIComponent(document.getElementById("mm-kind").value);
  const listMessages = async () => {
    const r = await backendApi(`/v1/memory/${kind()}/messages`);
    out.innerHTML = `<p class="muted">${r.active ? "active instance" : "no instance yet"}, ${r.messages.length} message(s)</p>${jsonBlock(r.messages)}`;
  };
  bindAction(document.getElementById("mm-form"), msg, async () => {
    const body = { role: document.getElementById("mm-role").value, content: document.getElementById("mm-content").value };
    const model = document.getElementById("mm-model").value;
    if (model) body.model = model;
    const r = await backendApi(`/v1/memory/${kind()}/messages`, { method: "POST", json: body });
    setMsg(msg, "ok", `Stored; ${r.message_count} message(s) in ${r.kind}.`);
    await listMessages();
  }, { event: "submit", busy: "Adding…" });
  bindAction(document.getElementById("mm-list"), msg, async () => { await listMessages(); setMsg(msg, "ok", "Listed."); }, { busy: "Listing…" });
  bindAction(document.getElementById("mm-clear"), msg, async () => {
    const r = await backendApi(`/v1/memory/${kind()}`, { method: "DELETE" });
    setMsg(msg, "ok", r.cleared ? "Cleared." : "Nothing to clear.");
    out.innerHTML = "";
  }, { busy: "Clearing…" });
}

async function renderVectorStores(refresh = false) {
  view.innerHTML = `<h1>Vector stores</h1><p class="page-sub">Probing backends&hellip;</p>`;
  const data = await backendApi(`/v1/vector-stores${refresh ? "?refresh=true" : ""}`);
  const rows = data.backends.map((b) => `<tr>
      <td><strong>${esc(b.name)}</strong></td>
      <td>${b.local ? '<span class="badge">local</span>' : ""} ${b.experimental ? '<span class="badge danger">experimental</span>' : ""}</td>
      <td>${okBadge(b.constructible)}</td>
      <td>${b.client_module ? `<code>${esc(b.client_module)}</code> ${okBadge(b.client_installed, "installed", "missing")}` : "&mdash;"}</td>
      <td class="wrap muted">${esc(b.reason || "")}</td></tr>`).join("");
  view.innerHTML = `
    <h1>Vector stores</h1>
    <p class="page-sub"><code>GET /v1/vector-stores</code>: every backend in the registry, constructed once with a dummy config (nothing connects).</p>
    <div class="tiles">
      <div class="tile"><div class="label">Backends</div><div class="value">${esc(fmtInt(data.total))}</div></div>
      <div class="tile"><div class="label">Constructible</div><div class="value">${esc(fmtInt(data.constructible))}</div></div>
    </div>
    <div class="toolbar"><span class="muted">Cached per backend process.</span><button id="vs-refresh">Re-probe</button></div>
    <div class="card"><div class="table-wrap"><table>
      <thead><tr><th>Backend</th><th>Kind</th><th>Constructible</th><th>Client</th><th>Reason</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  document.getElementById("vs-refresh").onclick = () => renderVectorStores(true).catch(showError);
}

async function renderFinetune() {
  const [{ models }, caps] = await Promise.all([loadModels(), backendApi("/v1/finetune/capabilities")]);
  const cards = Object.entries(caps.methods).map(([name, m]) => `<div class="tile ${m.available ? "" : "danger"}">
      <div class="label">${esc(name)}</div><div class="value" style="font-size:16px">${okBadge(m.available, "available", "unavailable")}</div>
      <div class="delta"><code>${esc(m.class)}</code>${m.reason ? `<br>${esc(m.reason)}` : ""}</div></div>`).join("");
  view.innerHTML = `
    <h1>Fine-tuning</h1>
    <p class="page-sub"><code>GET /v1/finetune/capabilities</code> says which trainers import here (torch ${caps.torch_installed ? "installed" : "not installed"};
      <code>${esc(caps.install)}</code>). <code>POST /v1/finetune/synthetic-qa</code> runs <code>SyntheticQAGenerator</code>.</p>
    <div class="tiles">${cards}</div>
    <div class="card"><h2>Synthetic Q/A</h2><form id="ft-form">
      <div class="form-row"><label class="row-label" for="ft-chunks">Document chunks, blank-line separated</label>
        <textarea id="ft-chunks" rows="4" required>MultiMind SDK ships a PII guard that masks emails and phone numbers.</textarea></div>
      <div class="filters">
        <label>Model <select id="ft-model">${modelOptions(models, chatState.model)}</select></label>
        <label>Questions per chunk <input id="ft-qpc" type="number" min="1" max="20" value="3"></label>
        <label>Max pairs <input id="ft-max" type="number" min="1" value="5"></label>
      </div>
      <button type="submit" class="primary">Generate pairs</button>
      <span id="ft-msg" class="form-msg" role="status"></span>
    </form><div id="ft-out"></div></div>`;
  const msg = document.getElementById("ft-msg");
  bindAction(document.getElementById("ft-form"), msg, async () => {
    const r = await backendApi("/v1/finetune/synthetic-qa", {
      method: "POST",
      json: {
        chunks: splitParagraphs(document.getElementById("ft-chunks").value),
        model: document.getElementById("ft-model").value,
        questions_per_chunk: Number(document.getElementById("ft-qpc").value) || 3,
        max_pairs: Number(document.getElementById("ft-max").value) || null,
      },
    });
    document.getElementById("ft-out").innerHTML = `<div class="table-wrap"><table><thead><tr><th>Question</th><th>Answer</th><th class="num">Grounding</th></tr></thead>
      <tbody>${r.pairs.map((p) => `<tr><td class="wrap">${esc(p.question)}</td><td class="wrap">${esc(p.answer)}</td><td class="num">${esc(Number(p.grounding_score).toFixed(2))}</td></tr>`).join("")}</tbody></table></div>
      <h2 style="margin-top:1rem">Instruction records</h2>${jsonBlock(r.records)}`;
    setMsg(msg, "ok", `${r.count} pair(s)`);
  }, { event: "submit", busy: "Generating…" });
}

async function renderConversion() {
  const r = await backendApi("/v1/convert/formats");
  view.innerHTML = `
    <h1>Model conversion</h1>
    <p class="page-sub"><code>GET /v1/convert/formats</code>: what <code>multimind.model_conversion</code> can do in this process.</p>
    ${r.available
      ? `<div class="grid-2">
          <div class="card"><h2>Sources</h2>${r.sources.map((s) => `<span class="chip">${esc(s)}</span>`).join("")}</div>
          <div class="card"><h2>Targets</h2>${r.targets.map((s) => `<span class="chip">${esc(s)}</span>`).join("")}</div>
        </div><div class="card"><h2>Converters</h2>${jsonBlock(r.converters)}</div>`
      : `<div class="banner">Not available: ${esc(r.reason)}<br>Install with <code>${esc(r.install)}</code>. Conversion itself runs from the CLI (<code>multimind convert</code>).</div>`}`;
}

async function renderRegulatory() {
  view.innerHTML = `
    <h1>Regulatory watch</h1>
    <p class="page-sub"><code>POST /v1/compliance/regulatory-check</code>: <code>RegulatoryWatcher</code> hashes each source and reports changes
      since the last check. Offline mode uses the content you paste; run it twice with an edit to see a change event.</p>
    <div class="card"><form id="rw-form">
      <div id="rw-rows"></div>
      <div class="toolbar"><button type="button" id="rw-add">Add source</button>
        <label><input id="rw-offline" type="checkbox" checked> Offline (use pasted content; unchecked fetches each URL)</label></div>
      <button type="submit" class="primary">Check</button>
      <span id="rw-msg" class="form-msg" role="status"></span>
    </form></div>
    <div id="rw-out"></div>`;
  const rows = document.getElementById("rw-rows");
  const addRow = (name = "", content = "") => {
    const div = document.createElement("div");
    div.className = "rw-row";
    div.innerHTML = `<div class="filters"><label>Name <input type="text" class="rw-name" value="${esc(name)}" required></label>
        <label>URL (online mode) <input type="text" class="rw-url" placeholder="optional"></label>
        <button type="button" class="rw-remove">Remove</button></div>
      <div class="form-row"><textarea class="rw-content" rows="3" placeholder="Page text (offline mode)">${esc(content)}</textarea></div>`;
    div.querySelector(".rw-remove").onclick = () => div.remove();
    rows.appendChild(div);
  };
  addRow("eu-ai-act", "Article 1. This Regulation lays down harmonised rules on artificial intelligence.");
  document.getElementById("rw-add").onclick = () => addRow();
  const msg = document.getElementById("rw-msg");
  bindAction(document.getElementById("rw-form"), msg, async () => {
    const sources = [...rows.querySelectorAll(".rw-row")].map((row) => ({
      name: row.querySelector(".rw-name").value.trim(),
      url: row.querySelector(".rw-url").value.trim() || null,
      content: row.querySelector(".rw-content").value,
    }));
    const r = await backendApi("/v1/compliance/regulatory-check", {
      method: "POST", json: { sources, offline: document.getElementById("rw-offline").checked },
    });
    const evRows = r.events.map((e) => `<tr class="${e.changed ? "blocked-row" : ""}"><td>${esc(e.name)}</td><td>${okBadge(!e.changed, "unchanged", "changed")}</td>
        <td><code>${esc(String(e.previous_hash || "").slice(0, 12))}</code> &rarr; <code>${esc(String(e.current_hash || "").slice(0, 12))}</code></td>
        <td class="wrap muted">${esc(e.diff || "")}</td></tr>`).join("");
    document.getElementById("rw-out").innerHTML = `<div class="card"><p class="muted">${esc(r.mode)} mode, state at <code>${esc(r.state_path)}</code></p>
      <div class="table-wrap"><table><thead><tr><th>Source</th><th>Status</th><th>Hash</th><th>Diff</th></tr></thead><tbody>${evRows}</tbody></table></div></div>`;
    setMsg(msg, "ok", `${r.events.filter((e) => e.changed).length} change(s) of ${r.events.length}`);
  }, { event: "submit", busy: "Checking…" });
}

async function renderDp() {
  view.innerHTML = `
    <h1>Differential privacy</h1>
    <p class="page-sub"><code>POST /v1/compliance/dp/privatize</code>: <code>DPMechanism.privatize</code> adds Laplace noise and, with a named
      budget, <code>PrivacyBudget</code> tracks epsilon across calls until it is exhausted (HTTP 400).</p>
    <div class="card"><form id="dp-form">
      <div class="filters">
        <label>Value <input id="dp-value" type="number" step="any" value="42"></label>
        <label>Epsilon per call <input id="dp-eps" type="number" step="any" min="0.0001" value="0.5"></label>
        <label>Sensitivity <input id="dp-sens" type="number" step="any" min="0.0001" value="1"></label>
        <label>Total budget (empty = none) <input id="dp-budget" type="number" step="any" min="0.0001" value="1"></label>
        <label>Budget id <input id="dp-budget-id" type="text" value="demo"></label>
      </div>
      <button type="submit" class="primary">Privatize once</button>
      <button type="button" id="dp-repeat">Repeat until exhausted</button>
      <span id="dp-msg" class="form-msg" role="status"></span>
    </form></div>
    <div id="dp-out"></div>`;
  const msg = document.getElementById("dp-msg");
  const out = document.getElementById("dp-out");
  const results = [];
  const body = () => {
    const b = {
      value: Number(document.getElementById("dp-value").value),
      epsilon: Number(document.getElementById("dp-eps").value),
      sensitivity: Number(document.getElementById("dp-sens").value),
    };
    const budget = document.getElementById("dp-budget").value;
    if (budget !== "") { b.budget = Number(budget); b.budget_id = document.getElementById("dp-budget-id").value.trim() || null; }
    return b;
  };
  const render = (exhausted) => {
    const rows = results.map((r, i) => `<tr><td class="num">${i + 1}</td><td class="num">${esc(Number(r.value).toFixed(4))}</td>
        <td class="num">${esc(r.epsilon)}</td><td class="num">${r.budget ? esc(Number(r.budget.spent_epsilon).toFixed(3)) : "&mdash;"}</td>
        <td class="num">${r.budget ? esc(Number(r.budget.remaining_epsilon).toFixed(3)) : "&mdash;"}</td></tr>`).join("");
    out.innerHTML = `${exhausted ? `<div class="banner">Budget exhausted: ${esc(exhausted)}</div>` : ""}
      <div class="card"><div class="table-wrap"><table><thead><tr><th class="num">#</th><th class="num">Noised value</th><th class="num">Epsilon</th>
        <th class="num">Spent</th><th class="num">Remaining</th></tr></thead><tbody>${rows || `<tr><td colspan="5"><div class="empty">No calls yet.</div></td></tr>`}</tbody></table></div></div>`;
  };
  bindAction(document.getElementById("dp-form"), msg, async () => {
    const r = await backendApi("/v1/compliance/dp/privatize", { method: "POST", json: body() });
    results.push(r);
    render(null);
    setMsg(msg, "ok", `Noised value ${Number(r.value).toFixed(4)}`);
  }, { event: "submit", busy: "Privatizing…" });
  bindAction(document.getElementById("dp-repeat"), msg, async () => {
    for (let i = 0; i < 50; i++) {
      try {
        results.push(await backendApi("/v1/compliance/dp/privatize", { method: "POST", json: body() }));
      } catch (err) {
        if (err.status === 400) { render(err.detail); setMsg(msg, "ok", `Exhausted after ${results.length} call(s).`); return; }
        throw err;
      }
    }
    render(null);
    setMsg(msg, "ok", "Stopped after 50 calls (no budget set?).");
  }, { busy: "Repeating…" });
}

async function renderMcp() {
  const data = await backendApi("/v1/mcp/tools");
  const tools = data.tools.map((t) => {
    const params = Object.entries(t.parameters || {}).map(([p, info]) => `<label>${esc(p)}${info.required ? "*" : ""} <span class="muted">${esc(info.annotation)}</span>
        <input type="text" data-param="${esc(p)}" ${p === "text" ? 'value="Mail me at jane@example.com"' : ""}></label>`).join("");
    return `<div class="card" data-tool="${esc(t.name)}"><h2>${esc(t.name)}</h2><p class="muted">${esc(t.description)}</p>
      <form><div class="filters">${params}<button type="submit">Invoke</button><span class="form-msg mcp-msg" role="status"></span></div></form>
      <div class="mcp-out"></div></div>`;
  }).join("");
  view.innerHTML = `
    <h1>MCP tools</h1>
    <p class="page-sub"><code>GET /v1/mcp/tools</code> lists the tools <code>${esc(data.server)}</code> registers; each form posts to
      <code>POST /v1/mcp/tools/{name}</code> and runs the tool in-process. Server startable (mcp package): ${okBadge(data.server_startable)}
      ${data.server_startable ? "" : `<code>${esc(data.install)}</code>`}</p>
    ${tools || `<div class="card"><div class="empty">No tools registered.</div></div>`}`;
  for (const card of view.querySelectorAll("[data-tool]")) {
    const msg = card.querySelector(".mcp-msg");
    bindAction(card.querySelector("form"), msg, async () => {
      const args = {};
      for (const input of card.querySelectorAll("[data-param]")) {
        const raw = input.value;
        if (raw === "") continue;
        try { args[input.dataset.param] = JSON.parse(raw); } catch { args[input.dataset.param] = raw; }
      }
      const r = await backendApi(`/v1/mcp/tools/${encodeURIComponent(card.dataset.tool)}`, { method: "POST", json: { arguments: args } });
      card.querySelector(".mcp-out").innerHTML = jsonBlock(r.result);
      setMsg(msg, "ok", "Done.");
    }, { event: "submit", busy: "Invoking…" });
  }
}

async function renderIntegrations() {
  const data = await backendApi("/v1/integrations");
  const rows = data.frameworks.map((f) => `<tr><td><strong>${esc(f.framework)}</strong></td><td><code>${esc(f.client_module)}</code></td>
      <td>${okBadge(f.installed, "installed", "missing")}</td><td class="wrap">${f.entry_points.map((e) => `<code class="chip">${esc(e)}</code>`).join(" ")}</td>
      <td><code>${esc(f.install)}</code></td></tr>`).join("");
  view.innerHTML = `
    <h1>Integrations</h1>
    <p class="page-sub"><code>GET /v1/integrations</code>: framework adapters in <code>multimind.integrations.frameworks</code> and whether each framework is installed.</p>
    <div class="card"><div class="table-wrap"><table>
      <thead><tr><th>Framework</th><th>Client module</th><th>Installed</th><th>Entry points</th><th>Install</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

async function renderTracing() {
  const data = await backendApi("/v1/traces?limit=20");
  const runs = (data.runs || []).map((r) => `<tr><td><code>${esc(String(r.run_id || r.id || "").slice(0, 12))}</code></td><td>${esc(r.name || r.kind || "")}</td>
      <td>${esc(r.model || "")}</td><td>${esc(String(r.start_time || "").replace("T", " ").slice(0, 19))}</td><td>${esc(r.status || "")}</td></tr>`).join("");
  view.innerHTML = `
    <h1>Tracing</h1>
    <p class="page-sub"><code>GET /v1/traces</code> reads root runs from the <code>RunTracer</code> SQLite sink; <code>POST /v1/traces/demo</code> records one traced call on a stub model.</p>
    ${data.available ? `<div class="banner ok">Sink: <code>${esc(data.db)}</code>${data.reason ? ` (${esc(data.reason)})` : ""}</div>`
      : `<div class="banner">${esc(data.reason)}. Demo traces still run but are not persisted; set <code>MULTIMIND_TRACING_DB=traces.db</code> and restart to keep them.</div>`}
    <div class="card"><form id="tr-form"><div class="filters">
      <label>Prompt <input id="tr-prompt" type="text" value="Hello from the trace demo"></label>
      <button type="submit" class="primary">Record demo trace</button><span id="tr-msg" class="form-msg" role="status"></span></div></form>
      <div id="tr-out"></div></div>
    <div class="card"><h2>Recent runs</h2><div class="table-wrap"><table>
      <thead><tr><th>Run</th><th>Name</th><th>Model</th><th>Start</th><th>Status</th></tr></thead>
      <tbody>${runs || `<tr><td colspan="5"><div class="empty">No runs recorded.</div></td></tr>`}</tbody></table></div></div>`;
  const msg = document.getElementById("tr-msg");
  bindAction(document.getElementById("tr-form"), msg, async () => {
    const r = await backendApi("/v1/traces/demo", { method: "POST", json: { prompt: document.getElementById("tr-prompt").value } });
    document.getElementById("tr-out").innerHTML = `<p class="muted">persisted: ${okBadge(r.persisted)} ${r.db ? `<code>${esc(r.db)}</code>` : ""}; ${r.spans.length} span(s)</p>${jsonBlock({ run: r.run, spans: r.spans })}`;
    setMsg(msg, "ok", "Recorded.");
  }, { event: "submit", busy: "Recording…" });
}

async function renderContextTransfer() {
  view.innerHTML = `
    <h1>Context transfer</h1>
    <p class="page-sub"><code>POST /v1/context-transfer</code>: summarize the last N turns and format them as a prompt for another provider
      (<code>ContextTransferManager</code>, same as <code>multimind context-transfer</code>).</p>
    <div class="card"><form id="ct-form">
      <div class="form-row"><label class="row-label" for="ct-messages">Conversation, one <code>role: content</code> per line (or a JSON array)</label>
        <textarea id="ct-messages" rows="5" required>user: Plan a 3-day trip to Rome.
assistant: Day 1: Colosseum and Forum. Day 2: Vatican. Day 3: Trastevere.</textarea></div>
      <div class="filters">
        <label>Source model <input id="ct-source" type="text" value="chatgpt"></label>
        <label>Target model <input id="ct-target" type="text" value="claude"></label>
        <label>Strategy <select id="ct-strategy"><option>concise</option><option>detailed</option><option>structured</option></select></label>
        <label>Last N turns <input id="ct-lastn" type="number" min="1" value="5"></label>
      </div>
      <button type="submit" class="primary">Transfer</button>
      <span id="ct-msg" class="form-msg" role="status"></span>
    </form></div>
    <div id="ct-out"></div>`;
  const msg = document.getElementById("ct-msg");
  bindAction(document.getElementById("ct-form"), msg, async () => {
    const raw = document.getElementById("ct-messages").value.trim();
    let messages;
    if (raw.startsWith("[")) {
      messages = JSON.parse(raw);
    } else {
      messages = splitLines(raw).map((line) => {
        const i = line.indexOf(":");
        return i > 0 ? { role: line.slice(0, i).trim(), content: line.slice(i + 1).trim() } : { role: "user", content: line };
      });
    }
    const r = await backendApi("/v1/context-transfer", {
      method: "POST",
      json: {
        messages,
        source_model: document.getElementById("ct-source").value.trim(),
        target_model: document.getElementById("ct-target").value.trim(),
        strategy: document.getElementById("ct-strategy").value,
        last_n: Number(document.getElementById("ct-lastn").value) || 5,
      },
    });
    document.getElementById("ct-out").innerHTML = `<div class="card"><h2>Formatted prompt</h2><pre class="redacted">${esc(r.formatted_prompt || "")}</pre></div>
      <div class="card"><h2>Metadata</h2>${jsonBlock(r.metadata || {})}</div>`;
    setMsg(msg, "ok", "Transferred.");
  }, { event: "submit", busy: "Transferring…" });
}

/* ---------- router ---------- */

const routes = {
  features: renderFeatures,
  overview: renderOverview,
  audit: renderAudit,
  costs: renderCosts,
  inventory: renderInventory,
  guardrails: renderGuardrails,
  scan: renderScan,
  evidence: renderEvidence,
  compliance: renderCompliance,
  models: renderModels,
  chat: renderChat,
  compare: renderCompare,
  rag: renderRag,
  hallucination: renderHallucination,
  agents: renderAgents,
  memory: renderMemory,
  vectorstores: renderVectorStores,
  finetune: renderFinetune,
  conversion: renderConversion,
  regulatory: renderRegulatory,
  dp: renderDp,
  mcp: renderMcp,
  integrations: renderIntegrations,
  tracing: renderTracing,
  context: renderContextTransfer,
  settings: renderSettings,
};

const DEFAULT_VIEW = "features";

function showError(err) {
  view.innerHTML = errorCard(err);
}

function route() {
  const name = (location.hash.replace(/^#\//, "") || DEFAULT_VIEW).split("?")[0];
  const render = routes[name] || routes[DEFAULT_VIEW];
  for (const a of document.querySelectorAll(".sidebar nav a")) {
    a.classList.toggle("active", a.dataset.view === (routes[name] ? name : DEFAULT_VIEW));
  }
  view.innerHTML = `<p class="page-sub">Loading&hellip;</p>`;
  render().catch(showError);
}

window.addEventListener("hashchange", route);
detectMode().then(route);
