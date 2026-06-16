/* ============================================================
   PULMONET — front-end logic
   1. scroll reveals   2. render metrics   3. in-browser ONNX demo
   ============================================================ */
"use strict";

// Enable scroll-reveal styling only when JS runs (content stays visible if it doesn't).
document.documentElement.classList.add("js");

const M = window.LUNG_METRICS || null;

const CLASS_DISPLAY = (M && M.dataset && M.dataset.classes) ||
  ["Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous cell carcinoma"];
const CLASS_COLOR = ["#bd7b18", "#c2422a", "#0c6b5b", "#2f6db0"]; // adeno · large · normal · squamous
const NORMAL_IDX  = CLASS_DISPLAY.findIndex(c => /normal/i.test(c));
const short = s => s.replace(" cell carcinoma", " cell").replace("carcinoma", "").trim();

/* ---------- 1. scroll reveals ---------- */
const io = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("is-visible"); io.unobserve(e.target); } });
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach(el => io.observe(el));
// hero is above the fold — reveal immediately
requestAnimationFrame(() => document.querySelectorAll(".hero .reveal").forEach(el => el.classList.add("is-visible")));

/* ---------- 2. render metrics ---------- */
function pct(x) { return Math.round(x * 100); }

function countUp(el, target, suffix) {
  const dur = 1100, t0 = performance.now();
  function step(t) {
    const k = Math.min(1, (t - t0) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    el.textContent = Math.round(target * eased) + (suffix || "");
    if (k < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function metricValue(key) {
  if (!M) return null;
  if (key in M) return M[key];
  if (M.binary && key in M.binary) return M.binary[key];
  return null;
}

function renderMetrics() {
  if (!M) { console.warn("LUNG_METRICS not found — run the notebook to generate assets/data.js"); return; }

  // headline metric numbers (hero + results), shown as integer percentages
  document.querySelectorAll("[data-metric]").forEach(el => {
    const v = metricValue(el.dataset.metric);
    if (v != null) countUp(el, pct(v), "%");
  });

  // dataset stat cards
  document.querySelectorAll("[data-ds]").forEach(el => {
    const v = M.dataset[el.dataset.ds];
    if (v != null) countUp(el, v, "");
  });

  // hero stamp + model bar
  const stamp = document.getElementById("genstamp");
  if (stamp && M.generated) stamp.textContent = M.generated;

  const mb = document.getElementById("modelbar");
  if (mb) {
    const md = M.model;
    mb.innerHTML = [
      ["Architecture", md.arch],
      ["Parameters", (md.params / 1e6).toFixed(1) + "M"],
      ["Trainable", (md.trainable / 1e3).toFixed(1) + "K (head)"],
      ["Trained on", md.device.toUpperCase()],
      ["Epochs", md.epochs],
      ["Best val acc", pct(M.best_val_acc) + "%"],
    ].map(([k, v]) => `<span><b>${k}:</b> ${v}</span>`).join("");
  }

  // class chips
  const cc = document.getElementById("classchips");
  if (cc) {
    cc.innerHTML = CLASS_DISPLAY.map((c, i) => {
      const sup = M.per_class[c] ? M.per_class[c].support : "";
      return `<span class="chip"><i style="background:${CLASS_COLOR[i]}"></i>${c}${sup !== "" ? ` · ${sup} test` : ""}</span>`;
    }).join("");
  }

  renderConfusion();
  renderPerClass();
}

function renderConfusion() {
  const host = document.getElementById("confusion");
  if (!host || !M.confusion_matrix) return;
  const cm = M.confusion_matrix;
  const gmax = Math.max(...cm.flat());

  let html = `<div class="cm-row"><span class="cm-corner">true ╲ pred</span>` +
    CLASS_DISPLAY.map(c => `<span class="cm-head">${short(c)}</span>`).join("") + `</div>`;

  cm.forEach((row, i) => {
    const rowSum = row.reduce((a, b) => a + b, 0) || 1;
    html += `<div class="cm-row"><span class="cm-rowlabel">${short(CLASS_DISPLAY[i])}</span>` +
      row.map((v, j) => {
        const frac = v / rowSum;
        const diag = i === j;
        const bg = diag
          ? `rgba(12,107,91,${0.10 + 0.85 * frac})`
          : (v === 0 ? "var(--surface)" : `rgba(194,66,42,${0.10 + 0.55 * (v / gmax)})`);
        const fg = (diag && frac > 0.45) ? "#fff" : "var(--ink)";
        return `<span class="cm-cell" style="background:${bg};color:${fg}">${v}</span>`;
      }).join("") + `</div>`;
  });
  host.innerHTML = html;
}

function renderPerClass() {
  const host = document.getElementById("perclass");
  if (!host || !M.per_class) return;
  host.innerHTML = CLASS_DISPLAY.map((c, i) => {
    const p = M.per_class[c]; if (!p) return "";
    return `<div class="pc">
      <div class="pc__top">
        <span class="pc__name"><i style="background:${CLASS_COLOR[i]}"></i>${c}</span>
        <span class="pc__f1">F1 ${p.f1.toFixed(2)}</span>
      </div>
      <div class="pc__track"><div class="pc__fill" data-w="${pct(p.f1)}" style="background:${CLASS_COLOR[i]}"></div></div>
      <div class="pc__sub"><span>precision ${p.precision.toFixed(2)}</span><span>recall ${p.recall.toFixed(2)}</span><span>AUC ${p.auc.toFixed(2)}</span></div>
    </div>`;
  }).join("");
  // animate bar widths once visible
  const bars = host.querySelectorAll(".pc__fill");
  const obs = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { e.target.style.width = e.target.dataset.w + "%"; obs.unobserve(e.target); }
  }), { threshold: 0.4 });
  bars.forEach(b => obs.observe(b));
}

renderMetrics();

/* ---------- 3. in-browser ONNX demo ---------- */
const PRE = { size: 224, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };

const els = {
  dz: document.getElementById("dropzone"),
  file: document.getElementById("fileInput"),
  empty: document.getElementById("dzEmpty"),
  preview: document.getElementById("preview"),
  run: document.getElementById("runBtn"),
  status: document.getElementById("status"),
  samples: document.getElementById("samples"),
  verdict: document.getElementById("verdict"),
  placeholder: document.getElementById("verdictPlaceholder"),
  vName: document.getElementById("verdictName"),
  vConf: document.getElementById("verdictConf"),
  vRisk: document.getElementById("verdictRisk"),
  probs: document.getElementById("probs"),
};

let session = null;
let haveImage = false;

function setStatus(msg, cls) { els.status.textContent = msg; els.status.className = "status" + (cls ? " " + cls : ""); }

async function initModel() {
  if (typeof ort === "undefined") { setStatus("ONNX runtime failed to load (no internet for CDN).", "err"); return; }
  ort.env.wasm.numThreads = 1;            // single-thread: no COOP/COEP headers required
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
  if (location.protocol === "file:") {
    setStatus("Open via a local server to enable the live demo (see README → ./serve.sh).", "err");
    els.run.disabled = true;
    return;
  }
  try {
    setStatus("Loading model…");
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
    setStatus("Model ready. Drop a CT scan to begin.", "ok");
    if (haveImage) els.run.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("Could not load model.onnx — run the notebook, then serve the folder over HTTP.", "err");
  }
}

function showImage(src) {
  els.preview.src = src;
  els.preview.hidden = false;
  els.empty.style.display = "none";
  haveImage = true;
  if (session) els.run.disabled = false;
}

/* upload + drag/drop */
els.dz.addEventListener("click", () => els.file.click());
els.dz.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.file.click(); } });
els.file.addEventListener("change", e => { const f = e.target.files[0]; if (f) showImage(URL.createObjectURL(f)); });
["dragenter", "dragover"].forEach(ev => els.dz.addEventListener(ev, e => { e.preventDefault(); els.dz.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => els.dz.addEventListener(ev, e => { e.preventDefault(); els.dz.classList.remove("drag"); }));
els.dz.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f && f.type.startsWith("image/")) showImage(URL.createObjectURL(f)); });

/* sample thumbnails (copied from the test set into assets/samples/) */
const SAMPLES = [
  { src: "assets/samples/adenocarcinoma.png", label: "Adeno" },
  { src: "assets/samples/large.cell.carcinoma.png", label: "Large cell" },
  { src: "assets/samples/normal.png", label: "Normal" },
  { src: "assets/samples/squamous.cell.carcinoma.png", label: "Squamous" },
];
if (els.samples) {
  els.samples.innerHTML = SAMPLES.map(s => `<img src="${s.src}" alt="${s.label} sample" title="${s.label}" />`).join("");
  els.samples.querySelectorAll("img").forEach(img => {
    img.addEventListener("error", () => img.remove());
    img.addEventListener("click", () => showImage(img.src));
  });
}

/* preprocessing identical to training: resize 224², /255, ImageNet normalise, NCHW */
function preprocess(img) {
  const s = PRE.size;
  const cv = document.createElement("canvas"); cv.width = s; cv.height = s;
  const ctx = cv.getContext("2d");
  ctx.drawImage(img, 0, 0, s, s);
  const { data } = ctx.getImageData(0, 0, s, s);
  const out = new Float32Array(3 * s * s);
  for (let i = 0; i < s * s; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * s * s + i] = (data[i * 4 + c] / 255 - PRE.mean[c]) / PRE.std[c];
    }
  }
  return new ort.Tensor("float32", out, [1, 3, s, s]);
}

function softmax(arr) {
  const m = Math.max(...arr);
  const ex = arr.map(v => Math.exp(v - m));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map(v => v / sum);
}

async function analyse() {
  if (!session || !haveImage) return;
  els.run.disabled = true;
  els.dz.classList.add("scanning");
  setStatus("Analysing…");
  try {
    await new Promise(r => (els.preview.complete ? r() : (els.preview.onload = r)));
    const input = preprocess(els.preview);
    const res = await session.run({ input });
    const logits = Array.from(res[Object.keys(res)[0]].data);
    const probs = softmax(logits);
    renderVerdict(probs);
    setStatus("Done. Educational result — not a diagnosis.", "ok");
  } catch (e) {
    console.error(e); setStatus("Inference failed: " + e.message, "err");
  } finally {
    els.dz.classList.remove("scanning");
    els.run.disabled = false;
  }
}
els.run.addEventListener("click", analyse);

function renderVerdict(probs) {
  const top = probs.indexOf(Math.max(...probs));
  const cancerProb = probs.reduce((a, p, i) => a + (i === NORMAL_IDX ? 0 : p), 0);
  const isCancer = top !== NORMAL_IDX;

  els.placeholder.hidden = true;
  els.verdict.hidden = false;
  els.vName.textContent = CLASS_DISPLAY[top];
  els.vName.style.color = isCancer ? "var(--danger)" : "var(--accent-ink)";
  els.vConf.textContent = `${pct(probs[top])}% confidence`;
  els.vRisk.textContent = isCancer ? `⚠ cancer signal ${pct(cancerProb)}%` : `✓ no cancer signal`;
  els.vRisk.className = "verdict__risk " + (isCancer ? "risk-cancer" : "risk-normal");

  const order = probs.map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]);
  els.probs.innerHTML = order.map(([i, p]) => `
    <div class="prob">
      <div class="prob__top"><span>${CLASS_DISPLAY[i]}</span><span class="prob__pct">${(p * 100).toFixed(1)}%</span></div>
      <div class="prob__track"><div class="prob__fill" style="background:${CLASS_COLOR[i]}"></div></div>
    </div>`).join("");
  requestAnimationFrame(() => {
    els.probs.querySelectorAll(".prob__fill").forEach((el, k) => { el.style.width = (order[k][1] * 100) + "%"; });
  });
}

initModel();
