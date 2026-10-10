// Drives the real dashboard (frontend/index.html) in jsdom against a LIVE server.
// Usage: BASE=http://127.0.0.1:8000 SEED_PASSWORD=... node scripts/ui_test.mjs   (needs `npm i jsdom`)
import { JSDOM, VirtualConsole } from "jsdom";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.env.BASE || "http://127.0.0.1:8000";
const PW = process.env.SEED_PASSWORD;
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = fs.readFileSync(path.join(ROOT, "frontend", "index.html"), "utf8");
let failures = 0, passed = 0;
const errors = [];
process.on("unhandledRejection", (e) => errors.push("unhandledRejection: " + (e && e.message)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function check(name, cond, extra = "") { if (cond) { passed++; console.log("  ok  ", name); } else { failures++; console.log("  FAIL", name, extra); } }

// Minimal Leaflet stub: exercises the dashboard's map data flow (styles, popups, live markers) without a browser engine.
const chain = () => { const o = { addTo() { return o; }, bindTooltip() { return o; }, bindPopup() { return o; }, setLatLng() { return o; }, addLatLng() { return o; } }; return o; };
const leafletStub = (calls) => ({
  map: () => ({ remove() {}, fitBounds() {}, setView() { return this; } }),
  tileLayer: () => chain(), polygon: () => chain(),
  polyline: () => { calls.polylines++; return chain(); },
  circleMarker: () => { calls.markers++; return chain(); },
  geoJSON(fc, opts) {
    for (const f of fc.features) {
      const layer = { bindPopup: (h) => calls.popups.push(h), bindTooltip() {} };
      opts.style?.(f);
      if (f.geometry.type === "Point") opts.pointToLayer?.(f, [f.geometry.coordinates[1], f.geometry.coordinates[0]]);
      opts.onEachFeature?.(f, layer);
      calls.kinds.add(f.properties.kind);
    }
    return { addTo() { return this; }, getBounds: () => ({ pad: () => ({}) }) };
  },
});

async function openApp(role) {
  const vc = new VirtualConsole();
  vc.on("jsdomError", (e) => { if (!/Not implemented/.test(e.message)) errors.push("jsdomError: " + e.message); });
  vc.on("error", (...a) => errors.push("console.error: " + a.join(" ")));
  const calls = { popups: [], kinds: new Set(), markers: 0, polylines: 0 };
  const dom = new JSDOM(html, {
    url: BASE + "/", runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) {
      w.fetch = (u, o) => fetch(new URL(u, BASE), o);
      w.URL.createObjectURL = () => "blob:stub";
      w.L = leafletStub(calls);
      w.alert = (m) => errors.push("alert: " + m);
      w.addEventListener("error", (e) => errors.push("window.onerror: " + e.message));
    },
  });
  const { window } = dom, $ = (s) => window.document.querySelector(s), $$ = (s) => [...window.document.querySelectorAll(s)];
  const waitFor = async (fn, what, ms = 20000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) { try { const v = fn(); if (v) return v; } catch {} await sleep(80); }
    throw new Error("timeout waiting for " + what + "\nVIEW: " + ($("#view")?.textContent || "").slice(0, 300));
  };
  const click = (el) => el.dispatchEvent(new window.MouseEvent("click", { bubbles: true, clientX: el._x, clientY: el._y }));
  $("#u").value = role; $("#p").value = PW; click($("#go"));
  await waitFor(() => !$("#app").classList.contains("hide"), role + " login");
  const tab = async (name, ready) => {
    click($(`#nav button[data-t="${name}"]`));
    await waitFor(() => $("#ptitle").textContent && $("#view").textContent.includes(ready), name + " view");
    check(`tab ${name} renders without error card`, !$("#view .card.err"), $("#view .card.err")?.textContent);
  };
  return { window, $, $$, waitFor, click, tab, calls, dom };
}
async function api(role, p, opt = {}) {
  const lr = await (await fetch(BASE + "/api/v1/auth/login", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ username: role, password: PW }) })).json();
  return fetch(BASE + "/api/v1" + p, { ...opt, headers: { ...(opt.headers || {}), Authorization: "Bearer " + lr.access_token } });
}

console.log("== ADMIN session");
const A = await openApp("admin");
check("sidebar shows user + role", A.$("#who").textContent.includes("admin") && A.$("#role").textContent.includes("admin"));
await A.waitFor(() => A.$("#live").classList.contains("on"), "live pill turns green (WebSocket connected)", 8000);
check("live pill is green: WebSocket connected", A.$("#live").classList.contains("on"));
for (const [t, r] of [["overview", "Tracked objects"], ["live", "Objects in frame"], ["missions", "New mission"], ["incidents", "Incident Center"], ["analytics", "Object counts"], ["assistant", "Investigation Assistant"], ["reports", "Generate report"], ["admin", "Audit log"]]) await A.tab(t, r);
check("KPI cards + live panel + event feed present on Command Center", await (async () => { await A.tab("overview", "Tracked objects"); return A.$$(".kpi").length === 4 && A.$(".livebox") && A.$("#feed"); })());

console.log("== create mission through the UI");
await A.tab("missions", "New mission");
A.click(A.$("#mk"));
await A.waitFor(() => A.$("#mdetail")?.textContent.includes("MISSION-"), "mission created");
const mid = +A.$("#msel").value;
check("mission appears in header selector and mission list", mid > 0 && A.$$(".mcard").length >= 1);

console.log("== upload (API) then draw a zone on the frame through the UI");
const fd = new FormData();
fd.append("mission_id", String(mid));
fd.append("file", new Blob([fs.readFileSync(path.join(ROOT, "datasets", "demo_aerial.mp4"))], { type: "video/mp4" }), "demo_aerial.mp4");
check("upload accepted", (await api("operator", "/videos/upload", { method: "POST", body: fd })).status === 201);
await A.tab("missions", "New mission");
await A.waitFor(() => A.$("[data-p]"), "Process button");
check("stepper + Process + 'Process live' + 'Draw zone' controls", A.$(".stepper") && A.$("[data-pl]") && A.$("[data-z]"));
A.click(A.$("[data-z]"));
await A.waitFor(() => A.$("#zsvg") && A.$("#zdraw").classList.contains("on"), "zone modal");
A.$("#zdn").value = "Drawn UI Zone";
for (const [x, y] of [[560, 250], [860, 250], [860, 90], [560, 90]]) { const s = A.$("#zsvg"); s._x = x; s._y = y; A.click(s); }
check("polygon + 4 handles drawn in SVG overlay", A.$("#zsvg polygon") && A.$$("#zsvg circle").length === 4 && A.$("#zdm").textContent.includes("4 points"));
A.click(A.$("#zs"));
await A.waitFor(() => !A.$("#zdraw").classList.contains("on") && A.$("#view").textContent.includes("Drawn UI Zone"), "zone saved, modal closed, zone listed");
check("zone saved and listed", true);
const zones = await (await api("admin", "/zones")).json();
check("zone stored as GeoJSON polygon", zones.some((z) => z.name === "Drawn UI Zone" && z.geojson.type === "Polygon"));
for (const z of zones.filter((z) => z.name !== "Drawn UI Zone")) await api("admin", "/zones/" + z.id, { method: "DELETE" });

console.log("== process via UI");
await A.waitFor(() => A.$("[data-p]"), "Process button again");
A.click(A.$("[data-p]"));
await A.waitFor(() => A.$("#mdetail video") && /FPS/.test(A.$("#mdetail").textContent), "processing done + annotated video", 60000);
check("annotated video + processing stats shown", !!A.$("#mdetail video") && /FPS/.test(A.$("#mdetail").textContent));
check("stepper reached Done", A.$$(".step.on").length === 3);
check("live toast and event feed received WebSocket events", A.$$(".toast").length > 0);

console.log("== REAL-TIME: 'Process live' streams frames to the dashboard");
await A.waitFor(() => A.$("[data-pl]"), "Process live button");
A.click(A.$("[data-pl]"));
await A.waitFor(() => A.$("#ptitle").textContent === "Live Ops", "navigated to Live Ops");
await A.waitFor(() => A.$(".livebox.has"), "first live frame", 15000);
check("live annotated image appears", !!A.$(".livebox.has img.liveimg") && A.$(".liveimg").src.startsWith("data:image/jpeg;base64,"));
const srcs = new Set(), frames = new Set();
for (let k = 0; k < 14; k++) { srcs.add(A.$(".liveimg").src.slice(-40)); frames.add(A.$(".ln").textContent); await sleep(500); }
check("image keeps changing in real time (>=4 distinct frames in 7 s)", srcs.size >= 4 && frames.size >= 4, `distinct=${srcs.size}/${frames.size}`);
check("HUD shows time, GPS and object count", /t=\d/.test(A.$(".lt").textContent) && /GPS 24\./.test(A.$(".lg").textContent) && /\d+ objects/.test(A.$(".lo").textContent));
check("objects table lists tracked cars/people with lat/lon", A.$$("#objs tr").length >= 2 && /car|person/.test(A.$("#objs").textContent));
check("progress bar advances", parseFloat(A.$(".lpbar").style.width) > 5);
check("geo-map markers + trails created from live telemetry", A.calls.markers >= 3 && A.calls.polylines >= 3, `markers=${A.calls.markers} lines=${A.calls.polylines}`);
await A.waitFor(() => /Processing finished/.test(A.$("#feed").textContent), "live run finished event", 40000);
check("feed shows incident events + 'Processing finished'", /EVT-\d+/.test(A.$("#feed").textContent) && /Processing finished/.test(A.$("#feed").textContent));
await A.waitFor(() => A.$("#icnt") && !A.$("#icnt").classList.contains("hide"), "nav incident badge", 8000);
check("sidebar shows pending HIGH/CRITICAL incident count badge", +A.$("#icnt").textContent >= 1);

console.log("== incident center + human review");
await A.tab("incidents", "Incident Center");
await A.waitFor(() => A.$(".icard"), "incident cards");
check("CRITICAL incident card shown first", A.$(".icard.CRITICAL") !== null && A.$(".icard").classList.contains("CRITICAL"));
A.click(A.$$("#sevchips .chip").find((c) => c.dataset.s === "HIGH"));
await A.waitFor(() => A.$$("#ilist .icard").every((c) => c.classList.contains("HIGH")) && A.$$("#ilist .icard").length >= 1, "severity filter");
check("severity filter chips work", true);
A.click(A.$$("#sevchips .chip").find((c) => c.dataset.s === ""));
await A.waitFor(() => A.$$("#ilist .icard").length >= 2, "filter cleared");
A.click(A.$("#ilist .icard"));
await A.waitFor(() => A.$("#drawer").classList.contains("on") && A.$("#idet").textContent.includes("Why this score"), "incident drawer");
const det = A.$("#idet").textContent;
check("drawer: risk gauge, reasons, evidence timeline, recommendation", A.$("#idet .gauge") && /intrusion/i.test(det) && /Drawn UI Zone/.test(det) && /human investigation/i.test(det) && A.$$("#idet .tl div").length >= 2);
check("drawer states 'no autonomous physical action'", /no autonomous physical action/.test(det));
A.click(A.$('[data-d="dismissed"]'));                      // no comment -> must be rejected by the server
await A.waitFor(() => /comment is required/i.test(A.$("#rmsg")?.textContent || ""), "dismiss requires comment");
check("dismiss without comment is rejected", true);
A.$("#rc").value = "Confirmed from UI test";
A.click(A.$('[data-d="confirmed"]'));
await A.waitFor(() => /confirmed/i.test(A.$("#idet").textContent) && /Confirmed from UI test/.test(A.$("#idet").textContent), "decision saved in drawer", 10000);
check("approval stored and shown in decisions", true);
A.click(A.$("#dclose")); await sleep(300);
check("drawer closes", !A.$("#drawer").classList.contains("on"));
await A.waitFor(() => A.$$(".st.confirmed").length >= 1, "list shows confirmed");
check("list refreshed with confirmed status", true);

console.log("== map view data flow");
await A.tab("map", "Mission map");
await A.waitFor(() => A.calls.kinds.size >= 3, "geojson layers", 5000).catch(() => {});
check("map received zone + trajectory + incident features", ["zone", "trajectory", "incident"].every((k) => A.calls.kinds.has(k)), [...A.calls.kinds].join());
check("incident popup + side list built", A.calls.popups.some((p) => /EVT-/.test(p)) && A.$$("#mlist .icard").length >= 1);

console.log("== assistant");
await A.tab("assistant", "Investigation Assistant");
A.$("#q").value = "Show all high-risk events today."; A.click(A.$("#ask"));
await A.waitFor(() => /CRITICAL/.test(A.$("#chat").textContent), "assistant reply");
check("reply grounded in DB with tool trace", /tools: search_events/.test(A.$("#chat").textContent));
A.$("#q").value = "<img src=x onerror=alert(1)> why vehicle #99?"; A.click(A.$("#ask"));
await A.waitFor(() => A.$$("#chat .msg").length >= 5 && !A.$("#chat .typing"), "second reply");
check("assistant output is text-only (no HTML injection)", A.$("#chat img") === null);

console.log("== reports, analytics, admin");
await A.tab("reports", "Generate report");
A.click(A.$("#gen"));
await A.waitFor(() => A.$("#view table a[data-r]"), "report listed");
check("report row with PDF/JSON/CSV links", A.$("a[data-f='pdf']") && A.$("a[data-f='csv']"));
await A.tab("analytics", "Object counts");
check("analytics: KPIs, donuts and detection stats", A.$$(".kpi").length === 4 && A.$$(".donut").length >= 2 && /stored detections/.test(A.$("#view").textContent));
check("analytics hides informational OBJECT_ENTRY/EXIT noise", !/OBJECT_ENTRY/.test(A.$("#view").textContent));
await A.tab("admin", "Audit log");
const at = A.$("#view").textContent;
check("audit shows review + zone-from-video + login", /incident_reviewed/.test(at) && /zone_created_from_video/.test(at) && /login/.test(at));
check("model registry listed", /bytelike-tracker/.test(at));
const th = A.window.document.documentElement.dataset.theme; A.click(A.$("#thm"));
check("theme toggle switches dark/light", A.window.document.documentElement.dataset.theme !== th);

console.log("== VIEWER session (RBAC reflected in UI)");
const V = await openApp("viewer");
check("viewer has no Admin tab", !V.$('#nav button[data-t="admin"]'));
await V.tab("incidents", "Incident Center");
await V.waitFor(() => V.$(".icard"), "viewer incident cards");
V.click(V.$(".icard"));
await V.waitFor(() => V.$("#idet")?.textContent.includes("Why this score"), "viewer drawer");
check("viewer cannot see Approve/Dismiss", !V.$("[data-d]"));
await V.tab("missions", "Missions");
check("viewer has no upload / process / draw / create controls", !V.$("#vu") && !V.$("[data-p]") && !V.$("[data-pl]") && !V.$("[data-z]") && !V.$("#mk"));
await V.tab("assistant", "available to operators");
check("viewer assistant blocked", /available to operators/.test(V.$("#view").textContent));

console.log("== bad login");
const B = new JSDOM(html, { url: BASE + "/", runScripts: "dangerously", pretendToBeVisual: true, beforeParse(w) { w.fetch = (u, o) => fetch(new URL(u, BASE), o); w.L = {}; } });
B.window.document.querySelector("#u").value = "admin"; B.window.document.querySelector("#p").value = "wrong-password";
B.window.document.querySelector("#go").dispatchEvent(new B.window.MouseEvent("click", { bubbles: true }));
await sleep(1200);
check("wrong password shows error, stays on login", /Invalid credentials/.test(B.window.document.querySelector("#lerr").textContent) && B.window.document.querySelector("#app").classList.contains("hide"));

check("no JS errors / uncaught exceptions during whole run", errors.length === 0, "\n" + errors.join("\n"));
console.log(`\n${passed} passed, ${failures} failed`);
process.exit(failures ? 1 : 0);
