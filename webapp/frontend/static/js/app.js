// Web demo N3-08C: nối giao diện với API FastAPI (webapp/backend/main.py)
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const pct = v => v == null || isNaN(v) ? "–" : Math.round(100 * v) + "%";
const CAM_VI = {CAM_FRONT_LEFT: "Trước trái", CAM_FRONT: "Trước", CAM_FRONT_RIGHT: "Trước phải",
  CAM_BACK_LEFT: "Sau trái", CAM_BACK: "Sau", CAM_BACK_RIGHT: "Sau phải"};
const SURROUND = Object.keys(CAM_VI);
const LEVEL_VI = {light: "nhẹ", moderate: "vừa", heavy: "to", dense: "dày", storm: "giông bão"};
const WEATHER_VI = {none: "Trời trong", rain: "Mưa", fog: "Sương mù", storm: "Giông / bão"};
let META = null, TASK = null, GEN_DATA = null, RESULTS = null, MODE = "preset";
const TYPE_VI = {car: "Ô tô (thật)", truck: "Xe tải (thật)", bus: "Xe buýt (thật)", moto: "Xe máy 1 người (thật)",
  bicycle: "Xe đạp (thật)", pedestrian: "Người đi bộ (thật)", moto_multi: "Xe máy chở 2-3 người", moto_cargo: "Xe máy chở hàng cồng kềnh",
  ba_gac: "Xe ba gác", moto_tow: "Xe máy kéo thùng", vendor_cart: "Xe hàng rong", dog: "Chó", cow: "Trâu / bò",
  ox_cart: "Xe bò (bò + xe gỗ)", car_rc: "Ô tô (ray-cast)"};
const PLACE_VI = {traffic: "cùng chiều quanh xe", ahead: "phía trước", swarm: "bầy quanh xe (15 m)", curb: "sát lề phía trước",
  roadside: "lấn từ lề xuống đường", crossing: "băng ngang trước xe", group_crossing: "cả nhóm băng ngang",
  crowd: "đi lại quanh xe", wander: "đứng / đi chậm trên đường"};
const CUSTOM_PRESETS = {
  "Giờ cao điểm VN": [["moto", 10, "swarm"], ["moto_multi", 3, "swarm"], ["moto_cargo", 2, "ahead"], ["ba_gac", 1, "curb"], ["car", 3, "traffic"]],
  "Chợ ven đường": [["vendor_cart", 2, "roadside"], ["pedestrian", 8, "crowd"], ["moto", 4, "traffic"], ["dog", 1, "crossing"]],
  "Tan trường": [["pedestrian", 6, "group_crossing"], ["bicycle", 4, "traffic"], ["moto", 8, "swarm"]],
  "Đường quê": [["ox_cart", 1, "curb"], ["cow", 2, "wander"], ["moto_tow", 1, "ahead"], ["dog", 1, "crossing"]]};

function customRow(type = "moto", count = 5, placement = "traffic") {
  const row = document.createElement("div");
  row.className = "grid grid-cols-[1fr_3.2rem_1.6rem] gap-1 items-center bg-dark-800 rounded-lg p-1.5";
  row.innerHTML = `<select class="sel c-type">${Object.keys(META.custom_types).map(t => `<option value="${t}" ${t === type ? "selected" : ""}>${TYPE_VI[t] || t}</option>`).join("")}</select>
    <input type="number" min="1" max="${META.max_custom_objects}" value="${count}" class="sel c-count !px-1 text-center">
    <button class="text-slate-400 hover:text-red-400" title="Xoá"><i class="fa-solid fa-xmark"></i></button>
    <select class="sel c-place col-span-3">${META.placements.map(p => `<option value="${p}" ${p === placement ? "selected" : ""}>bố trí: ${PLACE_VI[p] || p}</option>`).join("")}</select>`;
  row.querySelector("button").onclick = () => { row.remove(); customTotal(); };
  row.querySelector(".c-count").oninput = customTotal;
  $("#custom-rows").appendChild(row); customTotal();
}
function customTotal() { $("#custom-total").textContent = customGroups().reduce((a, g) => a + g.count, 0); }
function customGroups() {
  return [...$("#custom-rows").children].map(r => ({type: r.querySelector(".c-type").value,
    count: Math.max(0, +r.querySelector(".c-count").value || 0), placement: r.querySelector(".c-place").value})).filter(g => g.count > 0);
}
function loadCustomPreset(name) { $("#custom-rows").innerHTML = ""; CUSTOM_PRESETS[name].forEach(([t, n, p]) => customRow(t, n, p)); }
let CAM_MODE = "photorealistic_box";
const viewers = {};

async function api(url, opt) {
  const r = await fetch(url, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}
function seg(id, cb) {
  $$(`#${id} button`).forEach(b => b.onclick = () => { $$(`#${id} button`).forEach(x => x.classList.toggle("on", x === b)); cb(b.dataset.v); });
}
function lightbox(src) { const lb = $("#lightbox"); lb.querySelector("img").src = src; lb.classList.remove("hidden"); }
$("#lightbox").onclick = () => $("#lightbox").classList.add("hidden");
document.addEventListener("keydown", e => { if (e.key === "Escape") $("#lightbox").classList.add("hidden"); });

function renderCams(el, cams) {  // cams: [{channel, objects, image}]
  el.innerHTML = cams.map(c => `<div class="cam${c.objects ? " hit" : ""}" data-src="${c.image}">
    <img src="${c.image}" alt="${c.channel}"><span class="lab">${CAM_VI[c.channel] || c.channel}</span>
    ${c.objects ? `<span class="badge">${c.objects} object chèn</span>` : ""}</div>`).join("");
  el.querySelectorAll(".cam").forEach(d => d.onclick = () => lightbox(d.dataset.src));
}
function centerOf(boxes) {
  const ins = boxes.filter(b => b.inserted);
  if (!ins.length) return [0, 0];
  const c = ins.map(b => b.translation || b.c);
  return [c.reduce((a, p) => a + p[0], 0) / c.length / 2, c.reduce((a, p) => a + p[1], 0) / c.length / 2];
}

// ---------- tabs ----------
$$(".tab").forEach(t => t.onclick = () => {
  $$(".tab").forEach(x => x.classList.toggle("on", x === t));
  ["gen", "cases", "results"].forEach(v => $(`#view-${v}`).classList.toggle("hidden", v !== t.dataset.view));
  if (t.dataset.view === "cases") initCases();
  if (t.dataset.view === "results") initResults();
  Object.values(viewers).forEach(v => v.resize());
});

// ---------- sinh kịch bản ----------
async function initGen() {
  viewers.gen = new PointCloudViewer3D("viewer-gen");
  seg("seg-color", v => viewers.gen.setColorMode(v));
  seg("seg-disp", v => viewers.gen.setDisplay(v));
  seg("seg-view", v => viewers.gen.setView(v, GEN_DATA ? centerOf(GEN_DATA.boxes) : [0, 0]));
  $("#inp-psize").oninput = e => viewers.gen.setPointSize(+e.target.value);
  seg("seg-cam-mode", async v => {
    CAM_MODE = v;
    if (TASK) {
      $("#loading").classList.remove("hidden");
      $("#loading-text").textContent = "Đang kết xuất 6 camera quang học (" + v + ")…";
      try {
        const res = await api(`/api/cameras/${TASK}?mode=${CAM_MODE}`);
        renderCams($("#cams-gen"), res.cameras);
      } catch (e) {
        alertBox(e.message);
      } finally {
        $("#loading").classList.add("hidden");
      }
    }
  });
  $("#btn-cam-toggle").onclick = () => { const g = $("#cams-gen"); g.classList.toggle("hidden"); $("#btn-cam-toggle").textContent = g.classList.contains("hidden") ? "Mở rộng" : "Thu gọn"; viewers.gen.resize(); };
  $("#btn-rand").onclick = () => $("#inp-seed").value = Math.floor(Math.random() * 10000);
  META = await api("/api/meta");
  // cảnh nền: mini_val trước (scene model chưa thấy khi train)
  const byScene = {};
  META.samples.forEach(s => (byScene[s.scene] = byScene[s.scene] || []).push(s));
  const scenes = Object.keys(byScene).sort((a, b) => (byScene[a][0].split === "mini_val" ? -1 : 1) - (byScene[b][0].split === "mini_val" ? -1 : 1));
  $("#sel-sample").innerHTML = scenes.map(sc => `<optgroup label="${sc} (${byScene[sc][0].split})">` +
    byScene[sc].map(s => `<option value="${s.token}" ${sc === "scene-0103" && s.index === 10 ? "selected" : ""}>${sc} · khung ${s.index}</option>`).join("") + "</optgroup>").join("");
  const showDesc = () => { const s = META.samples.find(x => x.token === $("#sel-sample").value); $("#sample-desc").textContent = s ? s.description : ""; };
  $("#sel-sample").onchange = showDesc; showDesc();
  const groups = {};
  META.scenarios.forEach(s => (groups[s.group] = groups[s.group] || []).push(s));
  $("#sel-scenario").innerHTML = Object.entries(groups).map(([g, list]) => `<optgroup label="${g}">` +
    list.map(s => `<option value="${s.name}" ${s.name === "moto_swarm" ? "selected" : ""}>${s.name} — ${s.description}</option>`).join("") + "</optgroup>").join("");
  const showSc = () => { const s = META.scenarios.find(x => x.name === $("#sel-scenario").value); $("#scenario-desc").textContent = s ? `${s.description} · ${s.count[0]}-${s.count[1]} object` : ""; };
  $("#sel-scenario").onchange = showSc; showSc();
  $("#sel-weather").innerHTML = Object.keys(META.weather).map(w => `<option value="${w}">${WEATHER_VI[w] || w}</option>`).join("");
  const showLv = () => {
    const w = $("#sel-weather").value, lv = META.weather[w];
    $("#sel-level").innerHTML = lv.length ? lv.map(l => `<option value="${l}" ${l === lv[lv.length - 1] ? "selected" : ""}>${LEVEL_VI[l] || l}</option>`).join("") : "<option value=''>–</option>";
    $("#sel-level").disabled = !lv.length; $("#puddle-box").classList.toggle("hidden", w !== "storm");
  };
  $("#sel-weather").onchange = showLv; showLv();
  $("#inp-puddles").oninput = e => $("#puddle-val").textContent = e.target.value;
  $("#det-note").textContent = META.detector ? "Detector sẵn sàng (GPU). Bật 'Sinh đủ 10 sweep' trước khi sinh."
    : "Máy này không chạy được detector trực tiếp (" + (META.detector_error || "không có GPU / OpenPCDet") + "). Kết quả detector đã đo: tab 'Trường hợp đã đo' và 'Kết quả đo'. Chạy trong WSL env pcdet để dùng nút này.";
  seg("seg-mode", v => { MODE = v; $("#preset-box").classList.toggle("hidden", v !== "preset"); $("#custom-box").classList.toggle("hidden", v !== "custom"); });
  $("#custom-presets").innerHTML = Object.keys(CUSTOM_PRESETS).map(n => `<button class="btn bg-dark-700 hover:bg-cyan-800 !py-1 !text-[11px]" data-p="${n}">${n}</button>`).join("");
  $$("#custom-presets button").forEach(b => b.onclick = () => loadCustomPreset(b.dataset.p));
  $("#btn-add-row").onclick = () => customRow();
  loadCustomPreset("Giờ cao điểm VN");
  $("#btn-run").onclick = runGen;
  $("#btn-detect").onclick = runDetect;

  // Xử lý upload point cloud đa định dạng
  const inpUpload = $("#inp-upload-file");
  if (inpUpload) {
    $("#btn-trigger-upload").onclick = () => inpUpload.click();
    inpUpload.onchange = async () => {
      if (!inpUpload.files.length) return;
      const file = inpUpload.files[0];
      $("#upload-status").textContent = `Đang tải lên ${file.name}…`;
      $("#loading").classList.remove("hidden");
      $("#loading-text").textContent = `Đang xử lý ${file.name} (chuyển đổi đa định dạng)…`;
      const fd = new FormData();
      fd.append("file", file);
      try {
        const resp = await fetch("/api/upload", { method: "POST", body: fd });
        if (!resp.ok) throw new Error(await resp.text());
        const r = await resp.json();
        TASK = r.task_id;
        GEN_DATA = r.preview;
        viewers.gen.setData({augmented: r.preview.augmented, original: r.preview.original, obj: r.preview.obj, boxes: [], preds: [], stride: 4});
        $("#stats").innerHTML = `<div class="flex justify-between"><b class="text-cyan-300">Tùy chỉnh: ${file.name}</b><span class="text-emerald-400 font-mono">${r.stats.elapsed_ms} ms</span></div>
          <div class="flex justify-between"><span class="text-slate-400">Số điểm LiDAR</span><span class="font-mono">${r.stats.points_after.toLocaleString()}</span></div>`;
        $("#stats").classList.remove("hidden");
        ["pcd", "bin", "ply", "xyz", "zip"].forEach(k => {
          const el = $(`#dl-${k}`);
          if (el && r.downloads[k]) el.href = r.downloads[k];
        });
        $("#dl").classList.remove("hidden");
        $("#upload-status").textContent = `✓ Đã nạp ${r.stats.points_after.toLocaleString()} điểm từ ${file.name}`;
      } catch (err) {
        alertBox("Lỗi upload: " + err.message);
        $("#upload-status").textContent = "❌ Tải lên thất bại.";
      } finally {
        $("#loading").classList.add("hidden");
      }
    };
  }
}

async function runGen() {
  $("#loading").classList.remove("hidden"); $("#loading-text").textContent = "Đang sinh kịch bản + chèn theo vật lý LiDAR…";
  try {
    const body = {sample_token: $("#sel-sample").value, scenario: MODE === "custom" ? "custom" : $("#sel-scenario").value, seed: +$("#inp-seed").value,
      weather: $("#sel-weather").value, level: $("#sel-level").value || null, puddles: +$("#inp-puddles").value, sweeps: $("#chk-sweeps").checked,
      custom: MODE === "custom" ? customGroups() : null};
    const r = await api("/api/generate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    TASK = r.task_id; GEN_DATA = r.preview;
    viewers.gen.setData({augmented: r.preview.augmented, original: r.preview.original, obj: r.preview.obj, boxes: r.preview.boxes, preds: [], stride: 4});
    viewers.gen.setView("iso", centerOf(r.preview.boxes));
    const s = r.stats, rej = Object.entries(s.reject_stats || {}).map(([k, v]) => `${k} ${v}`).join(", ") || "–";
    $("#stats").innerHTML = `<div class="flex justify-between"><b class="text-cyan-300">${s.scenario}</b><span class="text-emerald-400 font-mono">${s.elapsed_ms} ms</span></div>
      <div class="flex justify-between"><span class="text-slate-400">Object chèn</span><b>${s.inserted}/${s.requested}</b></div>
      <div class="flex justify-between"><span class="text-slate-400">Object "khó" (&lt; 5 điểm)</span><b>${s.difficult}</b></div>
      <div class="flex justify-between"><span class="text-slate-400">Điểm LiDAR</span><span class="font-mono">${s.points_before.toLocaleString()} → ${s.points_after.toLocaleString()}</span></div>
      <div class="flex justify-between"><span class="text-slate-400">Thời tiết</span><span>${WEATHER_VI[s.weather]} ${s.level ? LEVEL_VI[s.level] || s.level : ""}</span></div>
      ${(s.group_stats || []).length > 1 || MODE === "custom" ? `<div class="pt-1 border-t border-dark-600">${(s.group_stats || []).map(g =>
        `<div class="flex justify-between text-[11px]"><span class="text-slate-400">${TYPE_VI[g.what] || g.what}</span>
         <span class="${g.placed < g.requested ? "text-amber-400" : "text-emerald-400"}">${g.placed}/${g.requested}</span></div>`).join("")}</div>` : ""}
      <div class="text-[11px] text-slate-500">Vị trí bị loại: ${rej}</div><div id="det-summary"></div>`;
    $("#stats").classList.remove("hidden");
    ["pcd", "bin", "ply", "xyz", "labels", "zip"].forEach(k => {
      const el = $(`#dl-${k}`);
      if (el && r.downloads[k]) el.href = r.downloads[k];
    });
    $("#dl").classList.remove("hidden");
    $("#btn-detect").disabled = !(META.detector && body.sweeps);
    $("#loading-text").textContent = "Đang kết xuất 6 camera quang học…";
    renderCams($("#cams-gen"), (await api(`/api/cameras/${TASK}?mode=${CAM_MODE}`)).cameras);
  } catch (e) { alertBox(e.message); }
  $("#loading").classList.add("hidden");
}

async function runDetect() {
  $("#loading").classList.remove("hidden"); $("#loading-text").textContent = "Detector đang chạy (10 sweep, GPU)…";
  try {
    const r = await api(`/api/detect/${TASK}`, {method: "POST"});
    if (!r.available) { alertBox(r.reason); return; }
    let k = 0;
    GEN_DATA.boxes.forEach(b => { if (b.inserted) { const st = r.statuses[k] || r.statuses[String(k)]; if (st) b.status = st; k++; } });
    viewers.gen.setData({augmented: GEN_DATA.augmented, original: GEN_DATA.original, obj: GEN_DATA.obj, boxes: GEN_DATA.boxes, preds: r.preds, stride: 4});
    const sm = r.summary;
    $("#det-summary").innerHTML = `<div class="mt-1 pt-1 border-t border-dark-600"><b>Detector:</b> đúng <b class="text-emerald-400">${sm.detected}</b>/${sm.total} · nhầm lớp <b class="text-amber-400">${sm.confused}</b> · bỏ sót <b class="text-red-400">${sm.missed}</b></div>`;
    renderCams($("#cams-gen"), (await api(`/api/cameras/${TASK}?mode=${CAM_MODE}`)).cameras);
  } catch (e) { alertBox(e.message); }
  finally { $("#loading").classList.add("hidden"); }
}
function alertBox(msg) {  // thông báo trong trang (không dùng alert() chặn trình duyệt)
  $("#stats").classList.remove("hidden");
  const d = document.createElement("div");
  d.className = "text-red-400 text-[11px] mb-1";
  d.textContent = "⚠ " + msg;
  $("#stats").prepend(d);
}

// ---------- trường hợp đã đo ----------
let SEQS = {}, SEQ = null;  // chuỗi khung có video (tools/sequence_export.py)
const seqCache = new Map();
async function seqFrame(name, k) {
  const key = `${name}/${k}`;
  if (!seqCache.has(key)) seqCache.set(key, api(`/sequences/${name}/${k}.json`));
  return seqCache.get(key);
}
function stopPlayer() { if (SEQ && SEQ.timer) clearTimeout(SEQ.timer); if (SEQ) SEQ.timer = null; $("#pl-play").innerHTML = '<i class="fa-solid fa-play"></i>'; }
async function showSeq(name) {
  stopPlayer();
  $$("#case-list button").forEach(b => b.classList.toggle("border-cyan-400", b.dataset.seq === name));
  SEQ = {name, k: 0, timer: null, first: true};
  const s = SEQS[name];
  $("#player").classList.remove("hidden");
  $("#pl-slider").max = s.frames.length - 1; $("#pl-slider").value = 0;
  $("#pl-video").onclick = () => { stopPlayer(); const vb = $("#videobox"); vb.querySelector("video").src = `/videos/${s.video}`; vb.classList.remove("hidden"); vb.querySelector("video").play(); };
  await seqGo(0);
}
async function seqGo(k) {
  if (!SEQ) return;
  const s = SEQS[SEQ.name], n = s.frames.length;
  k = (k + n) % n; SEQ.k = k; $("#pl-slider").value = k;
  const d = await seqFrame(SEQ.name, k);
  if (!SEQ || SEQ.k !== k) return;
  const boxes = d.gt.map(b => ({c: b.c, s: b.s, yaw: b.yaw, status: b.status, inserted: b.inserted}));
  viewers.cases.setData({augmented: d.points, obj: d.obj, boxes, preds: d.pred, stride: 3});
  if (SEQ.first) { viewers.cases.setView("iso", [0, 4]); SEQ.first = false; }
  const f = s.frames[k], tt = s.totals;
  $("#pl-label").textContent = `khung ${k + 1}/${n} · t=${f.t.toFixed(1)}s`;
  $("#case-info").innerHTML = `<b class="text-cyan-300">${s.label}</b> · ${s.scene}${s.weather ? " · " + (WEATHER_VI[s.weather] || s.weather) : ""} ·
    <span class="text-white">khung này: đúng <b class="text-emerald-400">${f.ok}</b>/${f.n}, nhầm lớp <b class="text-amber-400">${f.confused}</b>, sót <b class="text-red-400">${f.missed}</b></span>
    <span class="text-slate-400"> · cả chuỗi: đúng ${tt.ok}/${tt.objects} (${pct(tt.ok / Math.max(tt.objects, 1))}), nhầm ${tt.confused}, sót ${tt.missed} · detector chạy thật trên từng khung</span>`;
  renderCams($("#cams-case"), SURROUND.map(c => ({channel: c, objects: (f.cams || {})[c] || 0, image: `/sequences/${SEQ.name}/${k}/${c}.jpg`})));
  for (let j = 1; j <= 2; j++) {  // tải trước 2 khung sau cho phát mượt
    const kn = (k + j) % n; seqFrame(SEQ.name, kn);
    SURROUND.forEach(c => { new Image().src = `/sequences/${SEQ.name}/${kn}/${c}.jpg`; });
  }
}
function playLoop() {
  if (!SEQ) return;
  seqGo(SEQ.k + 1).then(() => { if (SEQ && SEQ.timer !== null) SEQ.timer = setTimeout(playLoop, 1000 / +$("#pl-fps").value); });
}
$("#pl-play").onclick = () => {
  if (!SEQ) return;
  if (SEQ.timer) { stopPlayer(); return; }
  $("#pl-play").innerHTML = '<i class="fa-solid fa-pause"></i>';
  SEQ.timer = setTimeout(playLoop, 0);
};
$("#pl-prev").onclick = () => { stopPlayer(); seqGo(SEQ.k - 1); };
$("#pl-next").onclick = () => { stopPlayer(); seqGo(SEQ.k + 1); };
$("#pl-slider").oninput = e => { stopPlayer(); seqGo(+e.target.value); };
$("#videobox-close").onclick = () => { const vb = $("#videobox"); vb.querySelector("video").pause(); vb.classList.add("hidden"); };
document.addEventListener("keydown", e => { if (e.key === "Escape") $("#videobox-close").click(); });

async function initCases() {
  if (viewers.cases) return;
  viewers.cases = new PointCloudViewer3D("viewer-case");
  RESULTS = RESULTS || await api("/api/results");
  SEQS = await api("/api/sequences");
  const groups = {t6: "Kịch bản giao thông (xe cắt mẫu thật)", vn: "Vật thể Việt Nam (ray-cast)", weather: "Thời tiết (cùng 1 khung bầy xe)"};
  const seqHtml = Object.keys(SEQS).length ? `<div class="text-[11px] uppercase tracking-wide text-slate-500 mt-1 mb-1">Chuỗi khung có video (phát từng khung)</div>` +
    Object.entries(SEQS).map(([k, s]) => `<button data-seq="${k}" class="block w-full text-left px-2 py-1.5 rounded-lg border border-dark-600 bg-dark-800 hover:border-cyan-600">
      <i class="fa-solid fa-film text-indigo-400 mr-1"></i>${s.label}<span class="block text-[11px] text-slate-400">${s.frames.length} khung · đúng ${pct(s.totals.ok / Math.max(s.totals.objects, 1))} (${s.totals.ok}/${s.totals.objects})</span></button>`).join("") : "";
  $("#case-list").innerHTML = seqHtml + Object.entries(groups).map(([g, title]) =>
    `<div class="text-[11px] uppercase tracking-wide text-slate-500 mt-3 mb-1">${title}</div>` +
    Object.entries(RESULTS.cases).filter(([k, v]) => v.group === g).map(([k, v]) =>
      `<button data-k="${k}" class="block w-full text-left px-2 py-1.5 rounded-lg border border-dark-600 bg-dark-800 hover:border-cyan-600">${v.label}
       <span class="block text-[11px] text-slate-400">${v.recall_all != null ? "cả kịch bản: " + pct(v.recall_all) + " (n=" + v.n_all + ")" : v.frame}</span></button>`).join("")).join("");
  $$("#case-list button").forEach(b => b.onclick = () => b.dataset.seq ? showSeq(b.dataset.seq) : showCase(b.dataset.k));
  if (location.hash === "#sequence" && Object.keys(SEQS).length) showSeq(Object.keys(SEQS)[0]);
  else showCase(Object.keys(RESULTS.cases)[0]);
}
async function showCase(k) {
  stopPlayer(); SEQ = null; $("#player").classList.add("hidden");
  $$("#case-list button").forEach(b => b.classList.toggle("border-cyan-400", b.dataset.k === k));
  const v = RESULTS.cases[k], d = await api(`/api/case/${k}`);
  const boxes = d.gt.map(b => ({c: b.c, s: b.s, yaw: b.yaw, status: b.status, inserted: b.inserted}));
  viewers.cases.setData({augmented: d.points, obj: d.obj, boxes, preds: d.pred, stride: 3});
  viewers.cases.setView("iso", centerOf(boxes));
  $("#case-info").innerHTML = `<b class="text-cyan-300">${v.label}</b> · ${v.recall_all != null ? "cả kịch bản phát hiện <b>" + pct(v.recall_all) + "</b> (n=" + v.n_all + ") · " : ""}khung minh hoạ: ${v.frame}
    <span class="text-slate-500"> · detector CenterPoint-PointPillar pretrained, đo sẵn trên mini_val</span>`;
  renderCams($("#cams-case"), SURROUND.map(c => ({channel: c, objects: (v.cams || {})[c] || 0, image: `/cases/camtiles/${k}/${c}.jpg`})));
}

// ---------- kết quả đo ----------
async function initResults() {
  if ($("#kpis").dataset.done) return;
  RESULTS = RESULTS || await api("/api/results");
  const S = RESULTS.summary, m = S.t6_motorcycle || [], g = k => m.find(r => r.group === k) || {};
  const W = {}; ((RESULTS.weather || {}).rows || []).forEach(r => (W[r.cls] = W[r.cls] || {})[r.group] = r.recall);
  const WS = {}; ((RESULTS.weather_swarm || {}).rows || []).forEach(r => (WS[r.cls] = WS[r.cls] || {})[r.group] = r.recall);
  const diff = S.t6_swarm_minus_control, cargo = (S.vn_moto_vs_rc_control || {}).cargo_moto;
  const kpis = [[pct(g("real").recall), "xe máy thật được phát hiện (nuScenes)"],
    [pct(g("control_2w").recall), "xe máy chèn, vị trí bình thường (đối chứng)"],
    [pct(g("moto_swarm").recall), "xe máy trong bầy 10-20 xe"],
    [diff ? Math.round(100 * diff.diff) + " điểm" : "–", "bầy so với đối chứng (CI " + (diff ? diff.ci95.map(x => Math.round(100 * x)).join("..") : "") + ")"],
    [cargo ? Math.round(100 * cargo.diff_vs_rc_control.diff) + " điểm" : "–", "xe máy chở hàng cồng kềnh"],
    [pct((W.pedestrian || {}).clear) + " → " + pct((W.pedestrian || {}).storm), "người đi bộ: trời trong → giông bão"]];
  $("#kpis").innerHTML = kpis.map(([v, t]) => `<div class="bg-dark-800 border border-dark-600 rounded-xl p-4"><div class="text-2xl font-black text-cyan-300">${v}</div><div class="text-xs text-slate-400 mt-1">${t}</div></div>`).join("");
  $("#kpis").dataset.done = 1;
  const card = (title, head, rows) => `<div class="bg-dark-800 border border-dark-600 rounded-xl p-4"><h3 class="font-bold mb-2 text-sm">${title}</h3><table><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</table></div>`;
  const ci = r => r.ci95 ? r.ci95.map(x => Math.round(100 * x)).join("–") + "%" : "";
  const conds = ["clear", "rain_heavy", "fog_dense", "storm"], cv = {clear: "Trời trong", rain_heavy: "Mưa to", fog_dense: "Sương dày", storm: "Giông/bão"};
  $("#tables").innerHTML =
    card("Xe máy — detector pretrained (CI 95%)", ["Nhóm", "Recall", "CI 95%", "n"], m.map(r => [r.label, pct(r.recall), ci(r), r.n_objects])) +
    card("Vật thể Việt Nam (ray-cast)", ["Vật thể", "Recall", "CI 95%", "n"], (S.vn_objects || []).map(r => [r.label, pct(r.recall), ci(r), r.n_objects])) +
    card("Thời tiết (41 khung mini_val)", ["Lớp", ...conds.map(c => cv[c])], [["Ô tô", "car", W], ["Người đi bộ", "pedestrian", W], ["Xe máy (thật)", "motorcycle", W], ["Xe máy trong bầy", "motorcycle", WS]].map(([l, c, src]) => [l, ...conds.map(k => pct((src[c] || {})[k]))])) +
    card("Người đi bộ: 1 / nhóm / đám đông", ["Số người", "Recall", "CI 95%", "n"], Object.entries((S.pedestrians || {}).recall || {}).map(([k, r]) => [k, pct(r.recall), ci(r), r.n_objects]));
  $("#charts").innerHTML = RESULTS.charts.map(c => `<img src="/report/${c}" class="rounded-lg bg-white cursor-zoom-in" onclick="lightbox(this.src)">`).join("");
  $("#videos").innerHTML = RESULTS.videos.map(v => `<div class="bg-dark-800 rounded-xl p-2"><video src="/videos/${v}" controls preload="metadata" class="w-full rounded"></video><div class="text-xs text-slate-400 mt-1">${v}</div></div>`).join("");
}

initGen().then(() => {  // mở thẳng: /#auto (sinh ngay), /#cases, /#results
  const h = location.hash.slice(1);
  if (h === "auto") runGen();
  else if (h === "custom") { $('#seg-mode button[data-v="custom"]').click(); runGen(); }
  else if (h === "cases" || h === "results" || h === "sequence") $(`.tab[data-view="${h === "sequence" ? "cases" : h}"]`).click();
}).catch(e => alertBox("Không tải được dữ liệu: " + e.message));
