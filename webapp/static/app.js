/* ProcGen RL control room - dashboard logic (no build step, Chart.js only). */
"use strict";

const STEPS_PER_SIM_HOUR = 54000;
const DEFAULT_SPS = 2600;
const S = { state: null, runs: [], run: null, charts: {}, lastSps: null, logJob: null, presUpdated: null };

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 3500);
}

function fmtHours(h) {
  if (h == null || isNaN(h)) return "-";
  if (h < 1) return `${Math.round(h * 60)} min`;
  return `${h < 10 ? h.toFixed(2) : h.toFixed(1)} h`;
}
function ghostLabel(h) {
  return h < 1 ? `${Math.round(h * 60)}min` : `${Number(h.toFixed(4))}h`;
}
function fmtDur(min) {
  if (min == null || !isFinite(min)) return "-";
  if (min < 1) return "< 1 min";
  const h = Math.floor(min / 60), m = Math.round(min % 60);
  return h ? `${h} h ${m} min` : `${m} min`;
}
function fmtTime(t) {
  return t ? new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "-";
}
const pct = (v) => (v == null || isNaN(v) ? "-" : `${Math.round(v * 100)}%`);
const sps = () => S.lastSps || DEFAULT_SPS;
const estMinutes = (hours) => (hours * STEPS_PER_SIM_HOUR) / sps() / 60;

function store(key, val) {
  try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; }
}

// ------------------------------------------------------------------ tabs + theme
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  store("tab", name);
  if (name === "presentation") loadPresentation();
  if (name === "newrun") loadPlan();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

function applyTheme(t) {
  if (t) document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
  Object.values(S.charts).forEach((c) => c.destroy());
  S.charts = {};
  refreshCharts();
}
$("themeBtn").addEventListener("click", () => {
  const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
    (!document.documentElement.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
  const next = dark ? "light" : "dark";
  store("theme", next);
  applyTheme(next);
});

// ------------------------------------------------------------------ charts
function baseOptions({ yPct = false, yMin, yMax, legend = false, stepped = false } = {}) {
  const grid = css("--grid"), muted = css("--muted"), text2 = css("--text-2");
  return {
    responsive: true, maintainAspectRatio: false, animation: false, parsing: false, normalized: true,
    interaction: { mode: "index", intersect: false },
    layout: { padding: { right: 8 } },
    plugins: {
      legend: { display: legend, position: "top", align: "start", labels: { color: text2, boxWidth: 14, boxHeight: 2, usePointStyle: false } },
      tooltip: {
        callbacks: {
          title: (items) => (items.length ? `sim ${fmtHours(items[0].parsed.x)}` : ""),
          label: (it) => `${it.dataset.label}: ${yPct ? pct(it.parsed.y) : Number(it.parsed.y).toLocaleString(undefined, { maximumFractionDigits: 3 })}`,
        },
      },
    },
    elements: { line: { borderWidth: 2, tension: stepped ? 0 : 0.15, stepped }, point: { radius: 0, hoverRadius: 4 } },
    scales: {
      x: {
        type: "linear", min: 0, grid: { color: grid }, border: { color: css("--axis") },
        ticks: { color: muted, callback: (v) => `${v}h` }, title: { display: true, text: "simulated training hours", color: text2 },
      },
      y: {
        min: yMin, max: yMax, grid: { color: grid }, border: { display: false },
        ticks: { color: muted, callback: yPct ? (v) => `${Math.round(v * 100)}%` : undefined },
      },
    },
  };
}

function ensureChart(id, opts) {
  if (S.charts[id]) return S.charts[id];
  const ctx = $(id);
  S.charts[id] = new Chart(ctx, { type: "line", data: { datasets: [] }, options: baseOptions(opts) });
  return S.charts[id];
}

function setEmpty(id, empty, msg = "No data yet") {
  const box = $(id).parentElement;
  let note = box.querySelector(".empty-note");
  if (empty && !note) {
    note = document.createElement("div");
    note.className = "empty-note";
    box.appendChild(note);
  }
  if (note) {
    note.textContent = msg;
    note.hidden = !empty;
  }
}

function lineDs(label, points, color, extra = {}) {
  return { label, data: points, borderColor: color, backgroundColor: color, pointBackgroundColor: color, ...extra };
}

function pts(rows, key) {
  return rows.filter((r) => r[key] != null && !isNaN(r[key])).map((r) => ({ x: r.sim_hours, y: Number(r[key]) }));
}

function updateMetricCharts(rows) {
  const s1 = css("--series-1"), s2 = css("--series-2");
  const learn = ensureChart("chLearn", { yPct: true, yMin: 0, yMax: 1, legend: true });
  learn.data.datasets = [lineDs("mean progress", pts(rows, "progress"), s1), lineDs("levels completed", pts(rows, "success_rate"), s2)];
  learn.update("none");
  setEmpty("chLearn", rows.length < 2, "Waiting for the first training episodes");

  const diff = ensureChart("chDiff", { yMin: 0, yMax: 1, stepped: true });
  diff.data.datasets = [lineDs("difficulty", pts(rows, "difficulty"), s1, { stepped: true, fill: { target: "origin" }, backgroundColor: s1 + "22" })];
  diff.update("none");
  setEmpty("chDiff", rows.length < 2);

  for (const [id, key, label] of [["chReward", "episode_reward", "episode reward"], ["chEntropy", "entropy_loss", "entropy loss"], ["chSpeed", "steps_per_sec", "steps/s"]]) {
    const c = ensureChart(id, {});
    c.data.datasets = [lineDs(label, pts(rows, key), s1)];
    c.update("none");
    setEmpty(id, pts(rows, key).length < 2);
  }
}

function updateEvalChart(evalRows) {
  const ord = [css("--ord-1"), css("--ord-2"), css("--ord-3"), css("--ord-4")];
  const unseen = evalRows.filter((r) => r.eval_set === "unseen");
  const diffs = [...new Set(unseen.map((r) => Number(r.difficulty)))].sort((a, b) => a - b).slice(-4);
  const ch = ensureChart("chEval", { yPct: true, yMin: 0, yMax: 1, legend: true });
  ch.data.datasets = diffs.map((d, i) => {
    const data = unseen.filter((r) => Number(r.difficulty) === d)
      .map((r) => ({ x: Number(r.sim_hours), y: Number(r.success_rate) })).sort((a, b) => a.x - b.x);
    const color = ord[diffs.length === 4 ? i : i + (4 - diffs.length)];
    return lineDs(`difficulty ${d}`, data, color, { pointRadius: 4, pointBorderColor: css("--surface"), pointBorderWidth: 2 });
  });
  ch.update("none");
  setEmpty("chEval", unseen.length === 0, "Ghosts are evaluated as they are saved");
}

function updateGhostTable(run, evalRows) {
  const files = run ? run.ghosts : [];
  const hoursList = files.map((f) => parseFloat(f.replace("ghost_", "").replace("h.npz", ""))).sort((a, b) => a - b);
  const unseen = evalRows.filter((r) => r.eval_set === "unseen");
  const diffs = [...new Set(unseen.map((r) => Number(r.difficulty)))].sort((a, b) => a - b);
  const hasTrain = evalRows.some((r) => r.eval_set === "train");
  const table = $("ghostTable");
  if (!hoursList.length) {
    table.innerHTML = `<tr><td class="dim">No ghosts saved yet for this run.</td></tr>`;
    return;
  }
  let html = `<tr><th>Ghost</th><th class="num">Sim hours</th>${diffs.map((d) => `<th class="num">d = ${d}</th>`).join("")}${hasTrain ? `<th class="num">own training levels</th>` : ""}<th>Files</th></tr>`;
  for (const h of hoursList.slice().reverse()) {
    const label = ghostLabel(h);
    const rows = evalRows.filter((r) => r.label === label);
    const cell = (d) => {
      const r = rows.find((x) => x.eval_set === "unseen" && Number(x.difficulty) === d);
      return `<td class="num">${r ? pct(Number(r.success_rate)) : '<span class="dim">pending</span>'}</td>`;
    };
    const tr = rows.find((x) => x.eval_set === "train");
    const fname = `ghost_${h.toFixed(2).padStart(7, "0")}h.npz`;
    html += `<tr><td><b>${esc(label)}</b></td><td class="num">${h}</td>${diffs.map(cell).join("")}${hasTrain ? `<td class="num">${tr ? pct(Number(tr.success_rate)) : "-"}</td>` : ""}<td class="small"><a href="/files/${esc(run.name)}/ghosts/${fname}" download>ghost</a> · <a href="/files/${esc(run.name)}/checkpoints/ppo_${esc(label)}.zip" download>checkpoint</a></td></tr>`;
  }
  table.innerHTML = html;
}

async function refreshCharts() {
  if (!S.run) return;
  try {
    const [metrics, evalRows] = await Promise.all([api(`/api/metrics?run=${encodeURIComponent(S.run)}`), api(`/api/eval?run=${encodeURIComponent(S.run)}`)]);
    updateMetricCharts(metrics);
    updateEvalChart(evalRows);
    updateGhostTable(S.runs.find((r) => r.name === S.run), evalRows);
    const run = S.runs.find((r) => r.name === S.run);
    if (run) {
      const c = run.config || {};
      const mode = c.train_levels ? `${c.train_levels} fixed levels at difficulty ${c.fixed_difficulty}` : c.fixed_difficulty != null ? `unlimited levels, fixed difficulty ${c.fixed_difficulty}` : "unlimited levels, curriculum";
      $("runInfo").textContent = `${mode} · target ${c.hours ?? "?"} sim h · ${run.ghosts.length} ghosts · launched ${c.launched || "-"}`;
    }
  } catch (e) {
    console.warn(e);
  }
}

// ------------------------------------------------------------------ state polling
function renderState(st) {
  S.state = st;
  const cur = st.current;
  const status = st.status;
  if (status && status.steps_per_sec > 50) S.lastSps = status.steps_per_sec;

  const pill = $("queuePill");
  pill.className = "pill " + (cur ? "running" : st.active ? "" : "paused");
  pill.textContent = cur ? `running: ${cur.run}` : st.active ? "queue idle" : "queue paused";
  $("awake").textContent = `keep awake: ${st.keep_awake ? "on" : "off"}`;
  $("awake").className = "pill " + (st.keep_awake ? "running" : "muted");

  $("curTitle").textContent = cur ? cur.title : "No job running";
  const queued = st.jobs.filter((j) => j.status === "queued");
  $("curSub").textContent = cur
    ? `Job ${cur.id} · started ${fmtTime(cur.started)} · ${queued.length} more queued${cur.note ? " · " + cur.note : ""}`
    : queued.length ? `${queued.length} job(s) queued${st.active ? "" : ", queue paused: press Start queue"}` : 'Queue a run from "New run", or start the overnight plan.';

  const show = cur && cur.type === "train" && status;
  $("progWrap").hidden = !show;
  if (show) {
    const frac = Math.min(1, status.sim_hours / status.target_hours);
    $("progBar").style.width = `${(frac * 100).toFixed(1)}%`;
    $("progLeft").textContent = `${(frac * 100).toFixed(1)}% · phase: ${status.phase}`;
    const later = queued.filter((j) => j.type === "train").reduce((a, j) => a + estMinutes(Number(j.args.hours || 100)), 0);
    $("progRight").textContent = later ? `queue finishes in about ${fmtDur(status.eta_minutes + later)}` : "";
    $("tHours").textContent = fmtHours(status.sim_hours);
    $("tHoursSub").textContent = `of ${status.target_hours} h target`;
    $("tEta").textContent = fmtDur(status.eta_minutes);
    $("tEtaSub").textContent = status.eta_minutes ? `done around ${fmtTime(Date.now() / 1000 + status.eta_minutes * 60)}` : "";
    $("tSpeed").textContent = Math.round(status.steps_per_sec).toLocaleString();
    $("tDiff").textContent = Number(status.difficulty).toFixed(1);
    $("tSucc").textContent = pct(status.success_rate);
    if (!S.userPickedRun && S.run !== cur.run) {
      S.run = cur.run;
      syncRunSelects();
      refreshCharts();
    }
  }
  $("liveLog").textContent = (st.log || []).join("\n") || "No output yet.";
  const ll = $("liveLog");
  ll.scrollTop = ll.scrollHeight;

  renderQueue(st.jobs);
}

function renderQueue(jobs) {
  const now = Date.now() / 1000;
  const rows = jobs.slice().reverse().map((j) => {
    const dur = j.started ? ((j.finished || now) - j.started) / 60 : null;
    const acts = [];
    if (j.log || j.status !== "queued") acts.push(`<button class="linkish" data-log="${j.id}">log</button>`);
    if (j.status === "queued") acts.push(`<button class="linkish" data-job="${j.id}" data-a="cancel">cancel</button>`);
    if (j.status === "running") acts.push(`<button class="linkish" data-job="${j.id}" data-a="cancel">stop</button>`);
    if (["stopped", "failed", "cancelled", "done"].includes(j.status)) {
      acts.push(`<button class="linkish" data-job="${j.id}" data-a="requeue">${j.type === "train" && j.status !== "done" ? "resume" : "re-run"}</button>`);
      acts.push(`<button class="linkish" data-job="${j.id}" data-a="remove">remove</button>`);
    }
    const est = j.type === "train" && j.status === "queued" ? ` <span class="dim small">~${fmtDur(estMinutes(Number(j.args.hours || 100)))}</span>` : "";
    return `<tr><td class="num">${j.id}</td><td>${esc(j.title)}${est}${j.note ? `<div class="small dim">${esc(j.note)}</div>` : ""}</td><td>${j.type}</td><td><span class="status ${j.status}">${j.status}</span></td><td>${fmtTime(j.started)}</td><td class="num">${dur == null ? "-" : fmtDur(dur)}</td><td>${acts.join(" ")}</td></tr>`;
  });
  $("queueTable").innerHTML = `<tr><th class="num">#</th><th>Job</th><th>Type</th><th>Status</th><th>Started</th><th class="num">Duration</th><th></th></tr>` +
    (rows.join("") || `<tr><td colspan="7" class="dim">Queue is empty.</td></tr>`);
}

$("queueTable").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.log) {
    S.logJob = Number(b.dataset.log);
    loadJobLog();
    return;
  }
  const a = b.dataset.a;
  if (a === "remove" && !confirm("Remove this job from the list? (Its run folder is kept.)")) return;
  if (a === "cancel" && b.textContent === "stop" && !confirm("Stop the running job? Training saves its model first.")) return;
  await api(`/api/jobs/${b.dataset.job}/${a}`, {});
  poll();
});

async function loadJobLog() {
  if (S.logJob == null) return;
  const r = await api(`/api/log?job=${S.logJob}&lines=400`);
  $("logTitle").textContent = `Log of job ${S.logJob}`;
  const el = $("jobLog");
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
  el.textContent = r.lines.join("\n") || "No output.";
  if (atBottom) el.scrollTop = el.scrollHeight;
}

async function poll() {
  try {
    renderState(await api("/api/state"));
    if (S.logJob != null && document.getElementById("tab-queue").classList.contains("active")) loadJobLog();
  } catch (e) {
    $("queuePill").className = "pill failed";
    $("queuePill").textContent = "server not reachable";
  }
}

// ------------------------------------------------------------------ runs
function syncRunSelects() {
  for (const id of ["runSelect", "demoRun"]) {
    const sel = $(id);
    const names = S.runs.map((r) => r.name);
    if (S.run && !names.includes(S.run)) names.unshift(S.run);
    const want = id === "runSelect" ? S.run : sel.value || "main";
    sel.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
    if (names.includes(want)) sel.value = want;
  }
}
async function refreshRuns() {
  try {
    S.runs = await api("/api/runs");
    if (!S.run) {
      const byRecent = S.runs.slice().sort((a, b) => b.modified - a.modified);
      S.run = (S.runs.find((r) => r.name === "main") || byRecent[0] || {}).name || null;
    }
    syncRunSelects();
  } catch (e) { /* server down: poll() shows it */ }
}
$("runSelect").addEventListener("change", (e) => {
  S.run = e.target.value;
  S.userPickedRun = true;
  refreshCharts();
});

// ------------------------------------------------------------------ queue controls
async function queueAction(act) {
  if (act === "stop" && !confirm("Stop the current job? Training saves its model first, and the queue pauses.")) return;
  await api({ start: "/api/queue/start", pause: "/api/queue/pause", stop: "/api/stop" }[act], {});
  toast({ start: "Queue started", pause: "Queue paused (current job keeps running)", stop: "Stop requested: saving and stopping" }[act]);
  poll();
}
$("btnStart").onclick = () => queueAction("start");
$("btnPause").onclick = () => queueAction("pause");
$("btnStop").onclick = () => queueAction("stop");
document.querySelectorAll("[data-act]").forEach((b) => (b.onclick = () => queueAction(b.dataset.act)));

// ------------------------------------------------------------------ new run
async function loadPlan() {
  const plan = await api("/api/plan");
  let total = 0;
  $("planTable").innerHTML = `<tr><th>Run folder</th><th>What</th><th class="num">Sim hours</th><th class="num">Est. real time</th></tr>` +
    plan.map((p) => {
      const m = estMinutes(p.args.hours);
      total += m;
      return `<tr><td><code>runs/${esc(p.run)}</code></td><td>${esc(p.title)}</td><td class="num">${p.args.hours}</td><td class="num">${fmtDur(m)}</td></tr>`;
    }).join("");
  $("planEta").textContent = `Total about ${fmtDur(total)} at ${Math.round(sps()).toLocaleString()} steps/s, plus a few minutes of evaluation per run. Ghost files, checkpoints and latest.zip are saved continuously.`;
}
$("btnPlan").onclick = async () => {
  const busy = S.state && S.state.jobs.some((j) => j.status === "queued" || j.status === "running");
  if (busy && !confirm("Jobs are already queued or running. Add the overnight plan after them?")) return;
  await api("/api/plan/overnight", {});
  toast("Overnight plan queued and started");
  showTab("dashboard");
  poll();
};

const PRESETS = {
  main: { run: "main", hours: 300, milestones: "0,0.5,1,2,5,10,20,30,50,75,100,150,200,300", train_levels: 0, fixed_difficulty: "", eval_milestones: 100 },
  gen: { run: "gen_10", hours: 100, milestones: "0,1,2,5,10,20,50,100", train_levels: 10, fixed_difficulty: 0.5, eval_milestones: 100 },
  quick: { run: "quick", hours: 5, milestones: "0,0.5,1,2,5", train_levels: 0, fixed_difficulty: "", eval_milestones: 30 },
};
document.querySelectorAll("[data-preset]").forEach((b) => (b.onclick = () => {
  const f = $("runForm");
  for (const [k, v] of Object.entries(PRESETS[b.dataset.preset])) f.elements[k].value = v;
  updateFormEta();
}));
function updateFormEta() {
  const h = Number($("runForm").elements.hours.value || 0);
  $("formEta").textContent = h ? `Estimated real time: ${fmtDur(estMinutes(h))} (${(h * STEPS_PER_SIM_HOUR).toLocaleString()} agent steps)` : "";
}
$("runForm").addEventListener("input", updateFormEta);
$("runForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const start = e.submitter && e.submitter.dataset.start === "1";
  const args = {};
  for (const el of f.elements) {
    if (!el.name || el.name === "run" || el.name === "post" || el.value === "") continue;
    args[el.name] = el.value;
  }
  const run = f.elements.run.value.trim();
  if (S.runs.some((r) => r.name === run) && !confirm(`runs/${run} already exists. Continue it from its latest autosave?`)) return;
  try {
    await api("/api/jobs", { type: "train", run, args, post: f.elements.post.checked, start });
    toast(start ? `Queued and started: ${run}` : `Queued: ${run}`);
    showTab(start ? "dashboard" : "queue");
    poll();
  } catch (err) {
    toast(`Error: ${err.message}`);
  }
});

// ------------------------------------------------------------------ presentation
function renderMarkdown(md) {
  const lines = md.split("\n");
  let html = "", inList = false, table = [];
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !/^\|\s*-/.test(r)).map((r) => r.split("|").slice(1, -1).map((c) => c.trim()));
    html += `<table class="data compact">${rows.map((r, i) => `<tr>${r.map((c) => (i ? `<td>${esc(c)}</td>` : `<th>${esc(c)}</th>`)).join("")}</tr>`).join("")}</table>`;
    table = [];
  };
  for (const ln of lines) {
    if (ln.startsWith("|")) { table.push(ln); continue; }
    flushTable();
    if (ln.startsWith("- ")) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${esc(ln.slice(2))}</li>`;
      continue;
    }
    if (inList) { html += "</ul>"; inList = false; }
    if (ln.startsWith("## ")) html += `<h2>${esc(ln.slice(3))}</h2>`;
    else if (ln.startsWith("# ")) html += `<h1>${esc(ln.slice(2))}</h1>`;
    else if (ln.trim()) html += `<p class="dim">${esc(ln)}</p>`;
  }
  flushTable();
  if (inList) html += "</ul>";
  return html;
}

async function loadPresentation() {
  const p = await api("/api/presentation");
  const fs = S.state && S.state.figures;
  $("presUpdated").textContent = (p.updated ? `Last generated ${new Date(p.updated * 1000).toLocaleString()} · ${p.figures.length} figures in runs/presentation` : "Not generated yet.") +
    (fs && fs.running ? " · regenerating now..." : "") + (fs && fs.error ? ` · last error: ${fs.error}` : "");
  if (p.updated === S.presUpdated) return;
  S.presUpdated = p.updated;
  $("summaryCard").hidden = !p.summary;
  $("summary").innerHTML = renderMarkdown(p.summary || "");
  $("gallery").innerHTML = p.figures.map((f) => {
    const src = `/files/presentation/${encodeURIComponent(f.file)}?v=${p.updated}`;
    return `<figure class="fig" style="margin:0"><a class="img" href="${src}" target="_blank" rel="noopener"><img src="${src}" alt="${esc(f.title)}" loading="lazy"></a>
      <figcaption class="meta"><h3>${esc(f.title)}</h3><p>${esc(f.caption)}</p><div class="file">${esc(f.file)} · <a href="${src}" download="${esc(f.file)}">download</a></div></figcaption></figure>`;
  }).join("") || `<p class="dim">Press "Refresh figures now" to generate the illustrations; training charts appear once a run has data.</p>`;
}
$("btnFigures").onclick = async () => {
  const r = await api("/api/figures", {});
  toast(r.started ? "Generating figures (about a minute)..." : "Already generating");
  const t = setInterval(async () => {
    await poll();
    await loadPresentation();
    if (!(S.state && S.state.figures.running)) clearInterval(t);
  }, 4000);
};

// ------------------------------------------------------------------ demo
$("demoDiff").oninput = (e) => ($("demoDiffVal").textContent = e.target.value);
async function launchDemo(spectate) {
  await api("/api/demo", { run: $("demoRun").value, difficulty: Number($("demoDiff").value), spectate, only: $("demoOnly").value.trim() });
  $("demoMsg").textContent = "Demo window opening on this laptop (it can take a few seconds).";
}
$("btnPlay").onclick = () => launchDemo(false);
$("btnSpectate").onclick = () => launchDemo(true);
$("btnInstall").onclick = async () => {
  try {
    const r = await api("/api/ghosts/install", { run: $("demoRun").value });
    $("demoMsg").textContent = `Copied ${r.installed} ghosts into ./ghosts: "python demo.py" now uses them.`;
  } catch (e) {
    $("demoMsg").textContent = e.message;
  }
};

// ------------------------------------------------------------------ boot
(async function boot() {
  const theme = store("theme");
  if (theme) document.documentElement.setAttribute("data-theme", theme);
  await refreshRuns();
  await poll();
  const fromHash = location.hash.slice(1);
  showTab($(`tab-${fromHash}`) ? fromHash : store("tab") || "dashboard");
  updateFormEta();
  refreshCharts();
  setInterval(poll, 3000);
  setInterval(refreshRuns, 15000);
  setInterval(refreshCharts, 10000);
  setInterval(() => {
    if (document.getElementById("tab-presentation").classList.contains("active")) loadPresentation();
  }, 30000);
})();
