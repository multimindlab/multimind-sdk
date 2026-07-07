// MultiMind governance dashboard: hash-routed views over the local /api endpoints.

const view = document.getElementById("view");

const fmtUsd = (v) =>
  "$" + (v >= 100 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(4) : v.toFixed(6));
const fmtInt = (v) => Number(v || 0).toLocaleString();
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${path} -> HTTP ${resp.status}`);
  return resp.json();
}

function errorCard(err) {
  return `<div class="card"><span class="badge danger">error</span> ${esc(err.message || err)}</div>`;
}

/* ---------- charts (hand-rolled SVG, single accent series) ---------- */

function lineChart(points, { valueKey = "cost", labelKey = "date" } = {}) {
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
      <text class="axis-label" x="${m.l - 6}" y="${ty + 3}" text-anchor="end">${esc(fmtUsd(v))}</text>`;
  }).join("");
  const first = points[0][labelKey], last = points[points.length - 1][labelKey];
  const dots = points.map((p, i) => {
    const [px, py] = coords[i];
    const tip = `${p[labelKey]}: ${fmtUsd(p[valueKey])} (${fmtInt(p.calls)} calls)`;
    return `<g><circle class="hover-target" cx="${px}" cy="${py}" r="12"><title>${esc(tip)}</title></circle>
      <circle class="series-dot" cx="${px}" cy="${py}" r="3.5" pointer-events="none"></circle></g>`;
  }).join("");
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Daily spend, last 30 days">
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
    </div>`;
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
    <div class="note">Applying changes: the proxy reads this file at startup, so restart it with
      <code>multimind serve --config ${esc(data.path)}</code> after saving.</div>`;
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
      const resp = await fetch("/api/guardrails", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail?.[0]?.msg || detail.detail || `HTTP ${resp.status}`);
      }
      msg.className = "form-msg ok";
      msg.textContent = "Saved. Restart the proxy with --config to apply.";
    } catch (err) {
      msg.className = "form-msg error";
      msg.textContent = "Not saved: " + (err.message || err);
    }
  };
}

/* ---------- router ---------- */

const routes = {
  overview: renderOverview,
  audit: renderAudit,
  costs: renderCosts,
  inventory: renderInventory,
  guardrails: renderGuardrails,
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
