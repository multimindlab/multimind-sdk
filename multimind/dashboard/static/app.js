// MultiMind governance dashboard: hash-routed views over the local /api endpoints.

const view = document.getElementById("view");

const fmtUsd = (v) =>
  "$" + (v >= 100 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(4) : v.toFixed(6));
const fmtInt = (v) => Number(v || 0).toLocaleString();
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const API_BASE = String(window.MULTIMIND_API_BASE || "").replace(/\/$/, "");

async function api(path, options) {
  const headers = { ...(options && options.headers) };
  if (window.MULTIMIND_API_KEY) headers["X-API-Key"] = window.MULTIMIND_API_KEY;
  const resp = await fetch(API_BASE + path, { ...options, headers });
  if (!resp.ok) throw new Error(`${path} -> HTTP ${resp.status}`);
  return resp.json();
}

function errorCard(err) {
  return `<div class="card"><span class="badge danger">error</span> ${esc(err.message || err)}</div>`;
}

// Other local services the UI talks to directly (not via the dashboard server).
// Built from the page's scheme so the static bundle carries no absolute URLs.
const localUrl = (port) => `${location.protocol}//127.0.0.1:${port}`;
const DEFAULT_PROXY_BASE = localUrl(8400);
const DEFAULT_BACKEND_BASE = API_BASE || localUrl(8080);

function remembered(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}
function remember(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode: ignore */ }
}

async function externalApi(base, path, options) {
  // Direct browser call to another origin; a CORS or connection failure surfaces as TypeError.
  const headers = { ...(options && options.headers) };
  if (window.MULTIMIND_API_KEY) headers["X-API-Key"] = window.MULTIMIND_API_KEY;
  let resp;
  try {
    resp = await fetch(base.replace(/\/$/, "") + path, { ...options, headers });
  } catch (err) {
    throw new Error(`Could not reach ${base}: ${err.message || err}`);
  }
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    const msg = detail.detail?.[0]?.msg || detail.detail || `HTTP ${resp.status}`;
    throw new Error(`${path} -> ${typeof msg === "string" ? msg : JSON.stringify(msg)}`);
  }
  return resp.json();
}

function setMsg(el, kind, text) {
  el.className = "form-msg " + kind;
  el.textContent = text;
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

/* ---------- views ---------- */

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
      setMsg(msg, "error", "Scan failed: " + (err.message || err));
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
      <div class="form-row" style="margin-top:1rem"><label class="row-label" for="gr-proxy-url">Proxy URL (<code>multimind serve</code>)</label>
        <input id="gr-proxy-url" type="text" value="${esc(remembered("multimind.proxyBase", DEFAULT_PROXY_BASE))}"></div>
      <button type="button" id="gr-reload">Reload proxy</button>
      <span id="gr-reload-msg" class="form-msg" role="status"></span>
    </div>`;
  const proxyInput = document.getElementById("gr-proxy-url");
  proxyInput.onchange = () => remember("multimind.proxyBase", proxyInput.value.trim());
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
      setMsg(msg, "error", "Not saved: " + (err.message || err));
    }
  };
  document.getElementById("gr-reload").onclick = async () => {
    const msg = document.getElementById("gr-reload-msg");
    const base = proxyInput.value.trim() || DEFAULT_PROXY_BASE;
    remember("multimind.proxyBase", base);
    setMsg(msg, "", "Reloading…");
    try {
      const r = await externalApi(base, "/v1/guardrails/reload", { method: "POST" });
      setMsg(msg, "ok", `Proxy reloaded: strategy=${r.strategy}, block_on=[${(r.block_on || []).join(", ")}], budget=${r.budget ?? "none"}`);
    } catch (err) {
      setMsg(msg, "error", `Reload failed: ${err.message || err}. Is multimind serve running at ${base}? ` +
        "A browser-blocked (CORS) call also fails here; then run: curl -X POST " + base + "/v1/guardrails/reload");
    }
  };
}

/* ---------- compliance (lives on `multimind backend`, not this server) ---------- */

const complianceState = {
  base: remembered("multimind.backendBase", DEFAULT_BACKEND_BASE),
  org: remembered("multimind.orgId", "org-123"),
};

function pct(v) { return (Number(v || 0) * 100).toFixed(1) + "%"; }

function severityBadge(sev) {
  const s = String(sev || "").toLowerCase();
  const cls = s === "critical" || s === "high" ? "danger" : "";
  return `<span class="badge ${cls}">${esc(s || "unknown")}</span>`;
}

function complianceUnreachable(err) {
  return `<div class="banner">Compliance API unreachable at <code>${esc(complianceState.base)}</code>:
      ${esc(err.message || err)}.<br>
      Start it with <code>multimind backend</code> (default port 8080) and allow this page's origin with
      <code>MULTIMIND_CORS_ORIGINS=${esc(location.origin)}</code>, then reload.</div>`;
}

async function loadComplianceMetrics() {
  const out = document.getElementById("cp-metrics");
  out.innerHTML = `<p class="page-sub">Loading&hellip;</p>`;
  const q = `organization_id=${encodeURIComponent(complianceState.org)}`;
  let m, alerts;
  try {
    [m, alerts] = await Promise.all([
      externalApi(complianceState.base, `/v1/compliance/dashboard?${q}`),
      externalApi(complianceState.base, `/v1/compliance/alerts?${q}`),
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

async function renderCompliance() {
  view.innerHTML = `
    <h1>Compliance</h1>
    <p class="page-sub">Regulation checks recorded by the MultiMind API (<code>multimind backend</code>),
      read live from its <code>/v1/compliance</code> routes. This is a separate process from the dashboard.</p>
    <div class="filters">
      <label>Backend URL <input id="cp-base" type="text" value="${esc(complianceState.base)}"></label>
      <label>Organization id <input id="cp-org" type="text" value="${esc(complianceState.org)}"></label>
      <button id="cp-load" type="button">Load</button>
    </div>
    <div id="cp-metrics"></div>
    <div class="card"><h2>Record a check</h2>
      <p class="muted">Posts to <code>POST /v1/compliance/monitor</code>; the text is PII-scanned and the
        outcome is appended to the organization's history above.</p>
      <form id="cp-form">
        <div class="filters">
          <label>Organization name <input id="cp-org-name" type="text" value="${esc(remembered("multimind.orgName", "Acme Corp"))}"></label>
          <label>DPO email <input id="cp-dpo" type="text" value="${esc(remembered("multimind.dpoEmail", "dpo@example.com"))}"></label>
          <label>Regulations (comma-separated) <input id="cp-regs" type="text" value="GDPR"></label>
        </div>
        <div class="form-row"><label class="row-label" for="cp-text">Text to check (metadata.text)</label>
          <textarea id="cp-text" rows="5" placeholder="Prompt or document to evaluate&hellip;"></textarea></div>
        <button type="submit" class="primary">Run check</button>
        <span id="cp-msg" class="form-msg" role="status"></span>
      </form>
      <div id="cp-result"></div>
    </div>`;
  const syncInputs = () => {
    complianceState.base = document.getElementById("cp-base").value.trim() || DEFAULT_BACKEND_BASE;
    complianceState.org = document.getElementById("cp-org").value.trim() || "org-123";
    remember("multimind.backendBase", complianceState.base);
    remember("multimind.orgId", complianceState.org);
  };
  document.getElementById("cp-load").onclick = () => { syncInputs(); loadComplianceMetrics(); };
  document.getElementById("cp-form").onsubmit = async (e) => {
    e.preventDefault();
    syncInputs();
    const msg = document.getElementById("cp-msg");
    const out = document.getElementById("cp-result");
    const orgName = document.getElementById("cp-org-name").value.trim();
    const dpo = document.getElementById("cp-dpo").value.trim();
    remember("multimind.orgName", orgName);
    remember("multimind.dpoEmail", dpo);
    const body = {
      organization_id: complianceState.org,
      organization_name: orgName,
      dpo_email: dpo,
      enabled_regulations: document.getElementById("cp-regs").value.split(",").map((s) => s.trim()).filter(Boolean),
      compliance_rules: {},
      metadata: { text: document.getElementById("cp-text").value },
    };
    setMsg(msg, "", "Running…");
    try {
      const r = await externalApi(complianceState.base, "/v1/compliance/monitor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const fe = r.final_evaluation || {};
      // Backend statuses: checked (all clear), attention (findings), unverifiable (no evidence).
      setMsg(msg, fe.status === "checked" ? "ok" : "error", `Result: ${fe.status || "recorded"}`);
      out.innerHTML = `<pre class="redacted">${esc(JSON.stringify(r, null, 2))}</pre>`;
      loadComplianceMetrics();
    } catch (err) {
      setMsg(msg, "error", "Check failed: " + (err.message || err));
      out.innerHTML = "";
    }
  };
  loadComplianceMetrics();
}

/* ---------- router ---------- */

const routes = {
  overview: renderOverview,
  audit: renderAudit,
  costs: renderCosts,
  inventory: renderInventory,
  guardrails: renderGuardrails,
  scan: renderScan,
  evidence: renderEvidence,
  compliance: renderCompliance,
};

function showError(err) {
  view.innerHTML = errorCard(err);
}

function route() {
  const name = (location.hash.replace(/^#\//, "") || "overview").split("?")[0];
  const render = routes[name] || renderOverview;
  for (const a of document.querySelectorAll(".sidebar nav a")) {
    a.classList.toggle("active", a.dataset.view === (routes[name] ? name : "overview"));
  }
  view.innerHTML = `<p class="page-sub">Loading&hellip;</p>`;
  render().catch(showError);
}

window.addEventListener("hashchange", route);
route();
