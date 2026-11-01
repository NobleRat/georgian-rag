/* ქართული RAG — ბრაუზერის მხარე. API გასაღები აქ არასდროს ჩანს: ყველა OpenAI მოთხოვნა სერვერზე სრულდება. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const HEADERS = { "Content-Type": "application/json", "X-Rag-App": "1" };
const LAYER_NAMES = {
  native: "ტექსტი",
  visual_note: "ვიზუალური აღწერა",
  formula: "ფორმულა",
  ocr_unverified: "OCR (გადაუმოწმებელი)",
};
const FLAG_NAMES = {
  visual_content: "გვერდზე ვიზუალური მასალაა",
  embedded_visual_object: "ჩაშენებული ობიექტი/სურათი",
  legacy_font_unmapped_characters: "ძველი შრიფტის გაურკვეველი სიმბოლოები",
  visual_navigation_only: "მხოლოდ სანავიგაციო",
  assistant_visual_transcription: "ასისტენტის ტრანსკრიფცია",
  unverified_ocr: "გადაუმოწმებელი OCR",
};
const EXAMPLES = [
  "როდის იქნა მიღებული გამყინვარების თეორია?",
  "რა ტიპებად იყოფა ნალექები არქეოლოგიურ ძეგლებზე?",
  "რა არის ნაჩვენები გუდაუთის კერნის პალინოლოგიურ დიაგრამაზე?",
];

const state = {
  docs: [],
  docById: {},
  selected: new Set(),
  conversationId: null,
  busy: false,
  status: null,
  estimate: null,
  pollTimer: null,
};

// ------------------------------------------------------------ helpers

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: HEADERS, ...opts });
  let data = null;
  try { data = await res.json(); } catch (_) { /* ignore */ }
  if (!res.ok) throw new Error((data && data.detail) || `შეცდომა (HTTP ${res.status})`);
  return data;
}

function fileUrl(documentId, path) {
  const doc = state.docById[documentId];
  const lib = doc ? doc.library_id : "main";
  return `/files/${encodeURIComponent(lib)}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

function pageNum(path) {
  const m = /(\d+)\.webp$/.exec(path || "");
  return m ? parseInt(m[1], 10) : null;
}

function money(v) {
  if (v === null || v === undefined) return "უცნობია";
  return "$" + Number(v).toFixed(v < 0.01 ? 5 : 4);
}

function showBanner(html) {
  const b = $("#banner");
  if (!html) { b.classList.add("hidden"); return; }
  b.innerHTML = html;
  b.classList.remove("hidden");
}

// Allow only a safe subset of HTML (tables, sup/sub, basic emphasis) from documents and answers.
const ALLOWED_TAGS = new Set(["TABLE", "THEAD", "TBODY", "TR", "TD", "TH", "SUP", "SUB", "BR", "B", "I", "STRONG", "EM", "P", "UL", "OL", "LI", "CODE"]);
function sanitizeHtml(html) {
  const doc = new DOMParser().parseFromString(`<div>${html}</div>`, "text/html");
  const walk = (node) => {
    for (const child of [...node.childNodes]) {
      if (child.nodeType === Node.ELEMENT_NODE) {
        walk(child);
        if (!ALLOWED_TAGS.has(child.tagName)) {
          child.replaceWith(...child.childNodes);
        } else {
          for (const attr of [...child.attributes]) {
            if (!["colspan", "rowspan"].includes(attr.name)) child.removeAttribute(attr.name);
          }
        }
      } else if (child.nodeType !== Node.TEXT_NODE) {
        child.remove();
      }
    }
  };
  const root = doc.body.firstChild;
  walk(root);
  return root.innerHTML;
}

// Document excerpt: markdown text + embedded HTML tables.
function renderExcerpt(md) {
  const blocks = String(md || "").split(/\n{2,}/);
  return blocks.map((block) => {
    if (/<table/i.test(block)) return sanitizeHtml(block);
    const withLinks = block.replace(/\[([^\]]+)\]\((#[^)]+)\)/g, "[$1]");
    const safe = sanitizeHtml(esc(withLinks).replace(/&lt;(\/?)(sup|sub)&gt;/g, "<$1$2>"));
    return `<p>${safe.replace(/\n/g, "<br>")}</p>`;
  }).join("");
}

// Minimal Markdown for answers + citation chips.
function inlineMd(text) {
  let t = esc(text).replace(/&lt;(\/?)(sup|sub|br)\s*\/?&gt;/g, "<$1$2>");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  return t;
}

function renderAnswer(text, offered) {
  const lines = String(text || "").split("\n");
  const out = [];
  let list = null;
  let table = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const closeTable = () => {
    if (table) {
      const rows = table.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r));
      out.push("<table>" + rows.map((r) => "<tr>" + r.replace(/^\s*\||\|\s*$/g, "").split("|").map((c) => `<td>${inlineMd(c.trim())}</td>`).join("") + "</tr>").join("") + "</table>");
      table = null;
    }
  };
  for (const line of lines) {
    if (/^\s*\|.*\|\s*$/.test(line)) { closeList(); (table = table || []).push(line); continue; }
    closeTable();
    let m;
    if ((m = /^\s*#{1,6}\s+(.*)$/.exec(line))) { closeList(); out.push(`<p><strong>${inlineMd(m[1])}</strong></p>`); }
    else if ((m = /^\s*[-*•]\s+(.*)$/.exec(line))) { if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inlineMd(m[1])}</li>`); }
    else if ((m = /^\s*\d+[.)]\s+(.*)$/.exec(line))) { if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inlineMd(m[1])}</li>`); }
    else if (!line.trim()) { closeList(); }
    else { closeList(); out.push(`<p>${inlineMd(line)}</p>`); }
  }
  closeList(); closeTable();
  let html = out.join("");
  // [07-c0058, 09-c0020] -> chips; validity checked against ids actually sent to the model.
  html = html.replace(/\[([^\[\]<>]{2,300})\]/g, (whole, inner) => {
    const parts = inner.split(/[,;]\s*|\s+/).filter(Boolean);
    if (!parts.length || !parts.every((p) => /^[A-Za-z0-9][\w.]*(?:-[\w.]+)+$/.test(p) && /\d/.test(p))) return whole;
    return parts.map((p) => offered.has(p)
      ? `<button class="cite" data-cite="${esc(p)}" title="წყაროს ჩვენება">${esc(p)}</button>`
      : `<span class="cite invalid" title="ეს იდენტიფიკატორი მოძიებულ წყაროებს შორის არ არის — მითითება ვერ დადასტურდა">⚠ ${esc(p)}</span>`).join("");
  });
  return html;
}

// ------------------------------------------------------------ status & documents

async function loadStatus() {
  try {
    const s = await api("/api/status");
    state.status = s;
    renderStatus(s);
  } catch (e) {
    showBanner(`სერვერთან კავშირი ვერ დამყარდა: ${esc(e.message)}`);
  }
}

function renderStatus(s) {
  const st = s.settings;
  const cov = s.embedding_coverage;
  const pills = [];
  pills.push(st.api_key_configured
    ? `<span class="pill ok">OpenAI გასაღები: მითითებულია</span>`
    : `<span class="pill bad">OpenAI გასაღები: არ არის</span>`);
  pills.push(`<span class="pill">პასუხები: ${esc(st.chat_model)}</span>`);
  pills.push(`<span class="pill">embeddings: ${esc(st.embedding_model)}</span>`);
  const hybrid = cov.done > 0;
  pills.push(hybrid
    ? `<span class="pill ${cov.pending ? "warn" : "ok"}">ძიება: ჰიბრიდული (${cov.done}/${cov.total})</span>`
    : `<span class="pill warn">ძიება: მხოლოდ საკვანძო სიტყვები</span>`);
  $("#pills").innerHTML = pills.join("");

  if (!st.api_key_configured) {
    showBanner(`<strong>OpenAI API გასაღები არ არის მითითებული.</strong> ძიება და წყაროები მუშაობს, მაგრამ AI პასუხი არ შეიქმნება.
      ${st.env_path
        ? `გახსენით ფაილი <code>${esc(st.env_path)}</code> Notepad-ით, ჩაწერეთ <code>OPENAI_API_KEY=თქვენი_გასაღები</code>,
      შეინახეთ და „ინდექსი“ ჩანართში დააჭირეთ „OpenAI-ის შემოწმებას“. სერვერის გადატვირთვა საჭირო არ არის.`
        : `ჰოსტინგის სერვისის Environment პარამეტრებში დაამატეთ ცვლადი <code>OPENAI_API_KEY</code> და გადატვირთეთ სერვისი.`}`);
  } else if (!hybrid) {
    showBanner(`სემანტიკური ინდექსი ჯერ არ შექმნილა. „ინდექსი“ ჩანართში დააჭირეთ „ხარჯის შეფასებას“ და შემდეგ „ინდექსირების დაწყებას“.`);
  } else {
    showBanner("");
  }

  const L = s.chunks_by_layer;
  const imp = s.last_import;
  $("#importInfo").innerHTML = `
    <span class="k">დოკუმენტები</span><span>${s.documents}</span>
    <span class="k">ძირითადი ჩანაწერები</span><span>${L.native}</span>
    <span class="k">ვიზუალური აღწერები</span><span>${L.visual_note}</span>
    <span class="k">შემოწმებული ფორმულები</span><span>${L.formula}</span>
    <span class="k">OCR (ცალკე, გადაუმოწმებელი)</span><span>${L.ocr_unverified}</span>
    <span class="k">ორიგინალების ჰეში</span><span>${imp ? `${imp.sources_verified}/${imp.documents} დადასტურდა` : "—"}</span>`;
  $("#coverage").innerHTML = `
    <span class="k">მოდელი/სივრცე</span><span>${esc(cov.space)}</span>
    <span class="k">ვექტორები</span><span>${cov.done} / ${cov.total}</span>
    ${cov.other_spaces.length ? `<span class="k">სხვა სივრცეები</span><span>${cov.other_spaces.map((o) => `${esc(o.space)} (${o.n})`).join(", ")} — არ გამოიყენება</span>` : ""}`;
  $("#openaiInfo").innerHTML = `
    <span class="k">გასაღები</span><span>${st.api_key_configured ? "მითითებულია (სერვერზე)" : "არ არის"}</span>
    <span class="k">პასუხების მოდელი</span><span>${esc(st.chat_model)}</span>
    <span class="k">reasoning</span><span>${esc(st.reasoning_effort)}</span>
    <span class="k">ვებძიება</span><span>${st.allow_web_search ? "ჩართულია" : "გამორთულია"}</span>
    <span class="k">პარამეტრები</span><span style="word-break:break-all">${st.env_path ? esc(st.env_path) : "ჰოსტინგის Environment ცვლადები"}</span>
    <span class="k">პაროლით დაცვა</span><span>${st.password_protected ? "ჩართულია" : "გამორთულია"}</span>`;
  renderJob(s.index_job, cov);
  renderUsage(s);
}

function renderUsage(s) {
  if (!s.usage.length) {
    $("#usage").innerHTML = `<p class="hint">OpenAI API ჯერ არ გამოყენებულა.</p>`;
  } else {
    const names = { embedding: "დოკუმენტების embeddings", query_embedding: "კითხვების embeddings", answer: "პასუხები", query_rewrite: "კითხვის გადაფორმულირება" };
    $("#usage").innerHTML = `<table class="usage"><tr><th>ტიპი</th><th>მოდელი</th><th>გამოძახება</th><th>ტოკენები (შემ./გამ.)</th><th>ხარჯი</th></tr>
      ${s.usage.map((u) => `<tr><td>${esc(names[u.kind] || u.kind)}${u.errors ? ` <span class="badge warn">${u.errors} შეცდ.</span>` : ""}</td><td>${esc(u.model)}</td><td>${u.calls}</td>
      <td>${(u.input_tokens || 0).toLocaleString()} / ${(u.output_tokens || 0).toLocaleString()}</td><td>${money(u.cost_usd)}</td></tr>`).join("")}
      <tr><th colspan="4">სულ</th><th>${money(s.usage_total_usd)}</th></tr></table>`;
  }
  $("#priceNote").textContent = `ხარჯი გამოითვლება OpenAI-ის დაბრუნებული ტოკენებით და ფასების ცხრილით (შემოწმდა ${s.prices_checked_on}). ზუსტი თანხა იხილეთ OpenAI-ის Usage/Billing გვერდზე.`;
}

async function loadDocuments() {
  state.docs = await api("/api/documents");
  state.docById = Object.fromEntries(state.docs.map((d) => [d.document_id, d]));
  renderDocs();
}

function renderDocs() {
  const f = $("#docFilter").value.trim().toLowerCase();
  $("#docList").innerHTML = state.docs.filter((d) => !f || d.title.toLowerCase().includes(f) || d.document_id.includes(f)).map((d) => {
    const n = d.chunk_counts.native || 0;
    const extra = [];
    if (d.chunk_counts.visual_note) extra.push(`<span class="badge">ვიზ. აღწერა ${d.chunk_counts.visual_note}</span>`);
    if (d.chunk_counts.formula) extra.push(`<span class="badge">ფორმულა ${d.chunk_counts.formula}</span>`);
    return `<li>
      <input type="checkbox" data-doc="${esc(d.document_id)}" ${state.selected.has(d.document_id) ? "checked" : ""} aria-label="ძიებაში ჩართვა">
      <div>
        <button class="dtitle" data-open-doc="${esc(d.document_id)}">${esc(d.document_id)}. ${esc(d.title)}</button>
        <div class="dmeta"><span class="badge ${d.kind}">${d.kind === "pdf" ? "PDF" : "Word"}</span>${d.page_count} გვ. · ${n} ჩანაწერი ${extra.join("")}</div>
      </div></li>`;
  }).join("");
  updateScopeHint();
}

function updateScopeHint() {
  const n = state.selected.size;
  $("#scopeHint").textContent = n ? `ძიება შეზღუდულია ${n} მონიშნული დოკუმენტით.` : "არცერთი არ არის მონიშნული — ძიება ყველა დოკუმენტში.";
}

// ------------------------------------------------------------ indexing

function renderJob(job, cov) {
  const running = job && (job.status === "running" || job.status === "starting");
  const total = cov ? cov.total : 0;
  const done = cov ? cov.done : 0;
  let pct = total ? (100 * done) / total : 0;
  if (running && job.total_items) pct = (100 * job.done_items) / job.total_items;
  $("#progressBar").style.width = `${pct.toFixed(1)}%`;
  const msgs = { idle: "", starting: "იწყება…", running: "მიმდინარეობს…", done: "დასრულდა.", stopped: "შეჩერდა (შეგიძლიათ გააგრძელოთ — უკვე შექმნილი ვექტორები შენახულია).", error: "შეცდომა." };
  let info = job ? (msgs[job.status] || job.status) : "";
  if (job && job.total_items) info += ` ${job.done_items}/${job.total_items} ჩანაწერი · ${job.requests || 0} მოთხოვნა · ${money(job.cost_usd)}`;
  if (job && job.retries) info += ` · განმეორებითი მცდელობა: ${job.retries}`;
  if (job && job.error) info += `<div class="error">${esc(job.error)}</div>`;
  $("#jobInfo").innerHTML = info;
  $("#btnStop").classList.toggle("hidden", !running);
  $("#btnEstimate").disabled = running;
  $("#btnStart").disabled = running || !state.estimate || !state.estimate.items;
  if (running && !state.pollTimer) state.pollTimer = setInterval(pollJob, 1000);
  if (!running && state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

async function pollJob() {
  try {
    const r = await api("/api/index/status");
    renderJob(r.job, r.coverage);
    if (!["running", "starting"].includes(r.job.status)) { state.estimate = null; $("#estimate").classList.add("hidden"); loadStatus(); }
  } catch (_) { /* keep polling */ }
}

async function estimate() {
  const e = await api("/api/index/estimate");
  state.estimate = e;
  const box = $("#estimate");
  box.classList.remove("hidden");
  box.innerHTML = e.items
    ? `დასამუშავებელია <strong>${e.items}</strong> ჩანაწერი, ~<strong>${e.tokens.toLocaleString()}</strong> ტოკენი (${e.requests} მოთხოვნა).<br>
       მიახლოებითი ხარჯი: <strong>${money(e.estimated_cost_usd)}</strong> (${esc(e.model)}).<br>
       <span class="hint">ერთხელ შექმნილი ვექტორები ინახება დისკზე და ხელახლა აღარ გამოითვლება.</span>`
    : "ყველა ჩანაწერს უკვე აქვს ვექტორი — დამატებითი ხარჯი არ არის.";
  $("#btnStart").disabled = !e.items;
}

async function startIndex() {
  if (!state.estimate) return;
  try {
    await api("/api/index/start", { method: "POST", body: JSON.stringify({ confirm: true }) });
    renderJob({ status: "starting" }, null);
    if (!state.pollTimer) state.pollTimer = setInterval(pollJob, 1000);
  } catch (e) {
    $("#jobInfo").innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
}

async function checkOpenAI() {
  $("#checkResult").textContent = "მოწმდება…";
  try {
    const r = await api("/api/openai/check", { method: "POST", body: "{}" });
    $("#checkResult").innerHTML = r.checks.map((c) => `<div class="${c.ok ? "" : "error"}">${c.ok ? "✓" : "✗"} ${esc(c.what)}${c.model ? ` (${esc(c.model)})` : ""}: ${esc(c.message)}</div>`).join("");
  } catch (e) {
    $("#checkResult").innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
  loadStatus();
}

async function reimport() {
  const btn = $("#btnReimport");
  btn.disabled = true;
  try {
    const r = await api("/api/import", { method: "POST", body: "{}" });
    const rep = r.reports[0];
    const L = rep.layers;
    alertInPanel(`იმპორტი დასრულდა: ${rep.documents} დოკუმენტი. ძირითადი ჩანაწერები — დამატებული ${L.native.added}, შეცვლილი ${L.native.changed}, უცვლელი ${L.native.unchanged}, წაშლილი ${L.native.removed}.`);
    await loadDocuments();
    await loadStatus();
  } catch (e) {
    alertInPanel(e.message, true);
  } finally {
    btn.disabled = false;
  }
}

function alertInPanel(text, isError) {
  const el = document.createElement("div");
  el.className = isError ? "error" : "estimate";
  el.textContent = text;
  $("#btnReimport").after(el);
  setTimeout(() => el.remove(), 15000);
}

// ------------------------------------------------------------ chat

function addUserMessage(text) {
  $("#welcome")?.remove();
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  $("#messages").appendChild(div);
}

function addAssistantMessage() {
  const div = document.createElement("div");
  div.className = "msg assistant";
  div.innerHTML = `<div class="bubble">
      <div class="statusline">მუშავდება…</div>
      <div class="answer"></div>
      <div class="warnings"></div>
      <div class="sources-wrap hidden"><div class="sources-title"><span>წყაროები</span><span class="smode"></span></div><div class="sources"></div></div>
      <div class="meta-line hidden"></div>
    </div>`;
  $("#messages").appendChild(div);
  return div;
}

function sourceCard(s) {
  const flags = (s.review_flags || []).map((f) => `<span class="badge warn">${esc(FLAG_NAMES[f] || f)}</span>`).join("");
  const layerBadge = s.layer !== "native" ? `<span class="badge warn">${esc(LAYER_NAMES[s.layer])}</span>` : "";
  const img = s.image_attached ? `<span class="badge ok">გვერდის სურათი გადაეცა მოდელს</span>` : "";
  const notSent = s.sent_to_model ? "" : `<span class="badge">მოდელს არ გადაეცა</span>`;
  const hasPage = (s.page_images || []).length > 0;
  return `<div class="card" data-card="${esc(s.chunk_id)}">
    <div class="cid">[${esc(s.chunk_id)}]</div>
    <div class="ctitle">${esc(s.document_title)}</div>
    <div class="cloc">${esc(s.location.label)}</div>
    <div>${layerBadge}${flags}${img}${notSent}</div>
    <div class="csnip">${esc(s.snippet)}</div>
    <div class="cbtns">
      <button data-excerpt="${esc(s.chunk_id)}">ამონარიდი</button>
      ${hasPage ? `<button data-page="${esc(s.chunk_id)}">გვერდი${s.page_images_derived ? " (≈)" : ""}</button>` : ""}
    </div></div>`;
}

const sourceCache = {};

async function ask(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  $("#btnSend").disabled = true;
  addUserMessage(question);
  const box = addAssistantMessage();
  const answerEl = box.querySelector(".answer");
  const statusEl = box.querySelector(".statusline");
  const warnEl = box.querySelector(".warnings");
  let text = "";
  let offered = new Set();
  let pending = false;
  const scroll = () => { const m = $("#messages"); m.scrollTop = m.scrollHeight; };
  const rerender = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { answerEl.innerHTML = renderAnswer(text, offered); pending = false; scroll(); });
  };
  const addWarning = (w) => { const d = document.createElement("div"); d.className = "warning"; d.textContent = w; warnEl.appendChild(d); };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        question,
        conversation_id: state.conversationId,
        document_ids: [...state.selected],
        image_mode: $("#imageMode").value,
        include_ocr: $("#includeOcr").checked,
      }),
    });
    if (!res.ok || !res.body) throw new Error(`სერვერის შეცდომა (HTTP ${res.status})`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const line = raw.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "conversation") state.conversationId = ev.conversation_id;
        else if (ev.type === "status") statusEl.textContent = ev.message;
        else if (ev.type === "sources") {
          offered = new Set(ev.sources.filter((s) => s.sent_to_model).map((s) => s.chunk_id));
          ev.sources.forEach((s) => { sourceCache[s.chunk_id] = s; });
          const wrap = box.querySelector(".sources-wrap");
          wrap.classList.toggle("hidden", !ev.sources.length);
          wrap.querySelector(".sources").innerHTML = ev.sources.map(sourceCard).join("");
          wrap.querySelector(".smode").textContent = ev.search_mode === "hybrid" ? "ჰიბრიდული ძიება" : "საკვანძო სიტყვებით ძიება";
          (ev.warnings || []).forEach(addWarning);
          if (ev.search_query && ev.search_query !== question) addWarning(`საძიებო ფორმულირება: „${ev.search_query}“`);
          statusEl.textContent = ev.sources.length ? `ნაპოვნია ${ev.sources.length} წყარო${ev.images.length ? `, გვერდის სურათი: ${ev.images.length}` : ""}.` : "წყაროები ვერ მოიძებნა.";
        } else if (ev.type === "delta") {
          text += ev.text;
          statusEl.textContent = "";
          rerender();
        } else if (ev.type === "done") {
          statusEl.textContent = "";
          answerEl.innerHTML = renderAnswer(text, offered);
          finishMessage(box, ev);
          (ev.warnings || []).forEach(addWarning);
        } else if (ev.type === "error") {
          statusEl.textContent = "";
          const d = document.createElement("div");
          d.className = "error";
          d.textContent = ev.message;
          box.querySelector(".answer").after(d);
          if (text) answerEl.innerHTML = renderAnswer(text, offered);
        }
      }
    }
  } catch (e) {
    statusEl.textContent = "";
    const d = document.createElement("div");
    d.className = "error";
    d.textContent = `კავშირი შეწყდა: ${e.message}`;
    box.querySelector(".answer").after(d);
  } finally {
    state.busy = false;
    $("#btnSend").disabled = false;
    loadStatus();
  }
}

function finishMessage(box, ev) {
  const c = ev.citations || {};
  (c.valid || []).forEach((id) => box.querySelector(`[data-card="${CSS.escape(id)}"]`)?.classList.add("cited"));
  const parts = [];
  if (ev.model) parts.push(`მოდელი: ${esc(ev.model)}`);
  if (ev.usage) parts.push(`ტოკენები: ${ev.usage.input_tokens.toLocaleString()} შემ. / ${ev.usage.output_tokens.toLocaleString()} გამ.`);
  if (ev.cost_usd !== undefined && ev.cost_usd !== null) parts.push(`ხარჯი: ${money(ev.cost_usd)}`);
  if (ev.images_sent) parts.push(`სურათები: ${ev.images_sent}`);
  if (c.valid) parts.push(`მითითებები: ${c.valid.length} დადასტურებული${c.invalid && c.invalid.length ? `, <span style="color:var(--bad)">${c.invalid.length} ვერ დადასტურდა</span>` : ""}`);
  if (c.uncited_answer) parts.push(`<span style="color:var(--warn)">პასუხს წყაროს მითითება არ ახლავს</span>`);
  if (ev.note) parts.push(esc(ev.note));
  const meta = box.querySelector(".meta-line");
  meta.innerHTML = parts.join(" · ");
  meta.classList.remove("hidden");
  if (ev.web_sources && ev.web_sources.length) {
    const d = document.createElement("div");
    d.className = "warning";
    d.innerHTML = "ვებ-წყაროები (არა ბიბლიოთეკიდან): " + ev.web_sources.map((w) => esc(w.title || w.url)).join("; ");
    meta.after(d);
  }
}

// ------------------------------------------------------------ modals

function openModal(title, html) {
  $("#modalTitle").textContent = title;
  $("#modalBody").innerHTML = html;
  $("#modal").classList.remove("hidden");
}
function closeModal() { $("#modal").classList.add("hidden"); $("#modalBody").innerHTML = ""; }

async function openExcerpt(chunkId) {
  try {
    const c = await api(`/api/chunks/${encodeURIComponent(chunkId)}`);
    const flags = (c.review_flags || []).map((f) => FLAG_NAMES[f] || f).join(", ");
    const latex = c.extra && c.extra.latex ? `<p class="hint">LaTeX: <code>${esc(c.extra.latex)}</code></p>` : "";
    const pages = (c.page_images || []).length ? `<button class="btn secondary" data-page="${esc(c.chunk_id)}">გვერდის გამოსახულება</button>` : "";
    openModal(`${c.document ? c.document.title : c.document_id} — [${c.chunk_id}]`, `
      <div class="status-box"><strong>${esc(LAYER_NAMES[c.layer])}:</strong> ${esc(c.status_note)}${flags ? `<br>შენიშვნები: ${esc(flags)}` : ""}</div>
      <p class="hint">${esc(c.location.label)}${c.page_images_derived ? " · გვერდი განსაზღვრულია მეზობელი აბზაცებით (მიახლოებით)" : ""}</p>
      <div class="excerpt">${c.layer === "ocr_unverified" ? `<p class="hint">OCR ტექსტი შეიძლება შეიცავდეს შეცდომებს:</p><pre style="white-space:pre-wrap">${esc(c.text)}</pre>` : renderExcerpt(c.markdown_clean)}</div>
      ${latex}
      <p class="hint">წყარო: ${esc(c.source_file)} · SHA-256: ${esc((c.source_sha256 || "").slice(0, 16))}…</p>
      ${pages}`);
  } catch (e) {
    openModal("შეცდომა", `<div class="error">${esc(e.message)}</div>`);
  }
}

async function openPageForChunk(chunkId) {
  let s = sourceCache[chunkId];
  if (!s || !s.page_images) s = await api(`/api/chunks/${encodeURIComponent(chunkId)}`);
  const first = (s.page_images || [])[0];
  if (first) openPage(s.document_id, first, s.page_images_derived);
}

function openPage(documentId, path, approximate) {
  const doc = state.docById[documentId];
  const pages = doc ? doc.page_images : [path];
  let i = Math.max(0, pages.indexOf(path));
  const kindLabel = doc && doc.kind === "word" ? "Word-ის საჩვენებელი ვერსიის გვერდი" : "PDF-ის გვერდი";
  const render = () => {
    const p = pages[i];
    const original = doc ? `<a href="${fileUrl(documentId, doc.source_file)}" target="_blank" rel="noopener">ორიგინალი ფაილი</a>` : "";
    openModal(`${doc ? doc.title : documentId} — ${kindLabel} ${pageNum(p)}`, `
      <div class="page-nav">
        <button class="btn secondary" id="pgPrev" ${i === 0 ? "disabled" : ""}>← წინა</button>
        <span>${i + 1} / ${pages.length}</span>
        <button class="btn secondary" id="pgNext" ${i === pages.length - 1 ? "disabled" : ""}>შემდეგი →</button>
        <button class="btn secondary" id="pgZoom">მასშტაბი</button>
        <a href="${fileUrl(documentId, p)}" target="_blank" rel="noopener">ცალკე გახსნა</a> ${original}
      </div>
      ${doc && doc.kind === "word" ? `<p class="hint">Word-ის გვერდები მიღებულია საჩვენებელი რენდერით; ავტორის Word-ში გვერდების დაყოფა შეიძლება განსხვავდებოდეს. ზუსტი მისამართია აბზაცის ID.</p>` : ""}
      ${approximate ? `<p class="hint">ამ ფრაგმენტის გვერდი განსაზღვრულია მიახლოებით (მეზობელი აბზაცებით).</p>` : ""}
      <div class="page-view" id="pgView"><img src="${fileUrl(documentId, p)}" alt="გვერდი ${pageNum(p)}"></div>`);
    $("#pgPrev").onclick = () => { i--; render(); };
    $("#pgNext").onclick = () => { i++; render(); };
    $("#pgZoom").onclick = () => $("#pgView").classList.toggle("zoom");
  };
  render();
}

async function openDocument(documentId) {
  const d = state.docById[documentId];
  if (!d) return;
  const chunks = await api(`/api/documents/${encodeURIComponent(documentId)}/chunks`);
  openModal(`${d.document_id}. ${d.title}`, `
    <div class="kv">
      <span class="k">ორიგინალი</span><span>${esc(d.original_filename)} · <a href="${fileUrl(documentId, d.source_file)}" target="_blank" rel="noopener">გახსნა/ჩამოტვირთვა</a></span>
      <span class="k">ტიპი</span><span>${d.kind === "pdf" ? "PDF (გვერდები ორიგინალისაა)" : "Word (გვერდები საჩვენებელი რენდერისაა)"}</span>
      <span class="k">SHA-256</span><span>${esc(d.source_sha256.slice(0, 24))}… ${d.source_verified ? '<span class="badge ok">დადასტურდა</span>' : '<span class="badge warn">არ შემოწმდა</span>'}</span>
      <span class="k">ჩანაწერები</span><span>${chunks.length}</span>
    </div>
    <h3>გვერდები (${d.page_images.length})</h3>
    <div class="thumbs">${d.page_images.map((p) => `<figure data-doc-page="${esc(p)}"><img loading="lazy" src="${fileUrl(documentId, p)}" alt=""><figcaption>${pageNum(p)}</figcaption></figure>`).join("")}</div>
    <h3>ჩანაწერები</h3>
    <div>${chunks.map((c) => `<div class="card" style="margin-bottom:.4rem"><div class="cid">[${esc(c.chunk_id)}] · ${esc(c.location.label)}</div><div class="csnip">${esc(c.snippet)}</div><div class="cbtns"><button data-excerpt="${esc(c.chunk_id)}">ამონარიდი</button></div></div>`).join("")}</div>`);
  $("#modalBody").querySelectorAll("[data-doc-page]").forEach((f) => { f.onclick = () => openPage(documentId, f.dataset.docPage, false); });
}

// ------------------------------------------------------------ wiring

document.addEventListener("click", (e) => {
  const t = e.target.closest("button, figure, span.cite");
  if (!t) return;
  if (t.dataset.tab) {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === t));
    ["docs", "index", "costs"].forEach((n) => $(`#tab-${n}`).classList.toggle("hidden", n !== t.dataset.tab));
  } else if (t.dataset.excerpt) openExcerpt(t.dataset.excerpt);
  else if (t.dataset.page) openPageForChunk(t.dataset.page);
  else if (t.dataset.openDoc) openDocument(t.dataset.openDoc);
  else if (t.dataset.cite) {
    const card = t.closest(".msg")?.querySelector(`[data-card="${CSS.escape(t.dataset.cite)}"]`);
    if (card) { card.scrollIntoView({ behavior: "smooth", block: "center" }); card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash"); }
    else openExcerpt(t.dataset.cite);
  } else if (t.dataset.example) { $("#question").value = t.dataset.example; ask(t.dataset.example); $("#question").value = ""; }
});

$("#docList").addEventListener("change", (e) => {
  const id = e.target.dataset.doc;
  if (!id) return;
  if (e.target.checked) state.selected.add(id); else state.selected.delete(id);
  updateScopeHint();
});
$("#docFilter").addEventListener("input", renderDocs);
$("#selAll").onclick = () => { state.docs.forEach((d) => state.selected.add(d.document_id)); renderDocs(); };
$("#selNone").onclick = () => { state.selected.clear(); renderDocs(); };
$("#btnEstimate").onclick = estimate;
$("#btnStart").onclick = startIndex;
$("#btnStop").onclick = () => api("/api/index/stop", { method: "POST", body: "{}" });
$("#btnCheck").onclick = checkOpenAI;
$("#btnReimport").onclick = reimport;
$("#btnNew").onclick = () => {
  state.conversationId = null;
  $("#messages").innerHTML = "";
};
$("#modalClose").onclick = closeModal;
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value;
  $("#question").value = "";
  ask(q);
});
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
});

$("#examples").innerHTML = EXAMPLES.map((q) => `<button data-example="${esc(q)}">${esc(q)}</button>`).join("");
loadDocuments().catch((e) => showBanner(`დოკუმენტების ჩატვირთვა ვერ მოხერხდა: ${esc(e.message)}`));
loadStatus();
