from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from wildwatch.theme import CSS_VARS

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>WILDWATCH</title>
  <style>
    :root {
      /* Injected from wildwatch.theme — Pocket LLM / Pixel Live Translate tokens */
      __CSS_VARS__
    }
    * { box-sizing: border-box; }
    html, body { margin:0; height:100%; background:var(--bg); color:var(--txt);
      font:15px/1.5 -apple-system, Roboto, Segoe UI, sans-serif; }
    header { display:flex; align-items:center; justify-content:space-between; padding:12px 16px;
             border-bottom:1px solid var(--line); background:var(--panel); }
    header h1 { font-size:15px; font-weight:800; margin:0; letter-spacing:.5px;
                background:var(--grad); -webkit-background-clip:text; background-clip:text;
                -webkit-text-fill-color:transparent; }
    header .meta { color:var(--mut); letter-spacing:.06em; font-size:12px; }
    .layout { display:grid; grid-template-columns: 1fr 320px; gap:12px; padding:12px; }
    @media (max-width: 980px) { .layout { grid-template-columns: 1fr; } }
    .feed { background:var(--bot); border:1px solid var(--line); border-radius:12px;
            overflow:hidden; position:relative; min-height:240px; }
    .feed img { width:100%; height:auto; display:block; background:#0B0A1A; }
    aside { display:flex; flex-direction:column; gap:12px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }
    .card h2 { margin:0 0 8px; font-size:12px; color:var(--accent); font-weight:600; letter-spacing:.08em; }
    .stat { display:flex; justify-content:space-between; padding:3px 0; color:var(--txt); }
    .stat span:first-child { color:var(--mut); }
    .banner { display:none; margin:12px 12px 0; padding:10px 14px; background:var(--panel2);
              border:1px solid var(--danger); border-radius:12px;
              color:var(--danger); letter-spacing:.2em; font-weight:700; text-align:center; }
    .banner.on { display:block; }
    ol { list-style:none; margin:0; padding:0; max-height:340px; overflow:auto; }
    ol li { padding:8px 0; border-bottom:1px solid var(--line); font-size:12px; }
    ol li .lab { color:var(--purple); font-weight:600; letter-spacing:.04em; }
    ol li .when { color:var(--mut); }
    a { color:var(--accent); }
    ::-webkit-scrollbar { width:10px; }
    ::-webkit-scrollbar-thumb { background:var(--line); border-radius:5px; }
    ::-webkit-scrollbar-thumb:hover { background:var(--purple); }
    .hint { color:var(--mut); font-size:12px; margin:8px 0 0; }
    .row { display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-top:8px; }
    input[type=text] { background:var(--panel2); color:var(--txt); border:1px solid var(--line);
      border-radius:9px; padding:7px 9px; font:inherit; min-width:0; flex:1; }
    button { cursor:pointer; border:none; color:var(--txt); background:var(--panel2);
      border:1px solid var(--line); border-radius:10px; padding:7px 11px; font:inherit; }
    button.primary { background:var(--grad); color:var(--send-on); border:none; font-weight:700; }
    button.danger { color:var(--danger); }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th { text-align:left; color:var(--mut); font-weight:600; padding:3px 4px; }
    td { padding:4px; border-top:1px solid var(--line); }
    td.high { color:var(--danger); font-weight:700; }
    td.medium { color:var(--purple); }
    td.low { color:var(--ok); }
    .zone-item { display:flex; justify-content:space-between; align-items:center;
      padding:4px 0; border-bottom:1px solid var(--line); font-size:13px; }
  </style>
</head>
<body>
  <header>
    <h1>WILDWATCH  /  AERIAL AWARENESS</h1>
    <div class="meta" id="clock">--</div>
  </header>
  <div class="banner" id="banner"></div>
  <div class="layout">
    <div>
      <div class="feed"><img id="video" src="/stream.mjpg" alt="annotated live feed"/></div>
      <p class="hint">Click the video to drop geofence vertices (frame-space, not GPS). Close the zone when you have 3+ points.</p>
    </div>
    <aside>
      <div class="card">
        <h2>TELEMETRY</h2>
        <div class="stat"><span>FPS</span><span id="fps">--</span></div>
        <div class="stat"><span>FRAME</span><span id="frame">--</span></div>
        <div class="stat"><span>SOURCE</span><span id="source">--</span></div>
        <div class="stat"><span>DETECTOR</span><span id="detector">--</span></div>
        <div class="stat"><span>DEVICE</span><span id="device">--</span></div>
        <div class="stat"><span>DROPS</span><span id="drops">0</span></div>
      </div>
      <div class="card">
        <h2>THREATS</h2>
        <div id="threats">NO CONTACTS</div>
      </div>
      <div class="card">
        <h2>GEOFENCES</h2>
        <div id="zonelist"></div>
        <div class="row">
          <input id="zonename" type="text" placeholder="zone name"/>
          <button class="primary" id="zoneclose">Close zone</button>
          <button id="zoneundo">Undo</button>
          <button id="zonecancel">Cancel</button>
        </div>
        <div class="hint" id="drafthint">Draft: 0 points</div>
      </div>
      <div class="card">
        <h2>EVENT LOG</h2>
        <ol id="events"><li class="when">Waiting for alerts…</li></ol>
      </div>
    </aside>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    function tickClock() { $("clock").textContent = new Date().toLocaleString(); }
    setInterval(tickClock, 1000); tickClock();

    $("video").addEventListener("click", async (ev) => {
      const r = ev.target.getBoundingClientRect();
      const x = (ev.clientX - r.left) / r.width;
      const y = (ev.clientY - r.top) / r.height;
      await fetch("/api/zones/draft", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({x, y})
      });
    });
    $("zoneclose").addEventListener("click", async () => {
      const r = await fetch("/api/zones/draft/close", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({name: $("zonename").value || ""})
      });
      if (!r.ok) { alert(await r.text()); return; }
      $("zonename").value = "";
    });
    $("zoneundo").addEventListener("click", () => fetch("/api/zones/draft/pop", {method:"POST"}));
    $("zonecancel").addEventListener("click", () => fetch("/api/zones/draft/cancel", {method:"POST"}));

    async function refresh() {
      try {
        const st = await (await fetch("/api/stats")).json();
        $("fps").textContent = (st.fps ?? 0).toFixed(1);
        $("frame").textContent = st.frame ?? "--";
        $("source").textContent = st.source ?? "--";
        $("detector").textContent = (st.detector || "") + (st.classifier && st.classifier !== "none" ? " + " + st.classifier : "");
        $("device").textContent = st.device ?? "--";
        $("drops").textContent = st.drops ?? 0;
        const banner = $("banner");
        if (st.banner) { banner.textContent = "ALERT  " + st.banner; banner.classList.add("on"); }
        else { banner.classList.remove("on"); }
        const threats = st.threats || [];
        $("threats").innerHTML = threats.length
          ? `<table><tr><th>ID</th><th>Class</th><th>Lvl</th><th>Threat</th><th>Zone</th><th>Status</th></tr>` +
            threats.map(t => {
              const lvl = (t.threat_level || "low");
              return `<tr>
                <td>${t.track_id ?? "-"}</td>
                <td>${(t.label||"").toUpperCase()}</td>
                <td class="${lvl}">${lvl.toUpperCase()}</td>
                <td>${(t.threat??0).toFixed(2)}</td>
                <td>${(t.zones && t.zones[0]) || "-"}</td>
                <td>${t.approaching ? "APPROACHING" : (t.status || "")}</td>
              </tr>`;
            }).join("") + `</table>`
          : "NO CONTACTS";
        const draft = st.draft || [];
        $("drafthint").textContent = "Draft: " + draft.length + " point" + (draft.length===1?"":"s");
        const zones = st.zones || [];
        $("zonelist").innerHTML = zones.length
          ? zones.map(z => `<div class="zone-item"><span>${z.name}</span>
              <button class="danger" data-zone="${z.name}">Delete</button></div>`).join("")
          : "<div class='hint'>No zones yet.</div>";
        $("zonelist").querySelectorAll("button[data-zone]").forEach(btn => {
          btn.onclick = () => fetch("/api/zones/" + encodeURIComponent(btn.dataset.zone), {method:"DELETE"});
        });
      } catch (e) {}
      try {
        const ev = await (await fetch("/api/events")).json();
        const items = ev.events || [];
        $("events").innerHTML = items.length
          ? items.slice(0, 25).map(e => `<li><div class="lab">${(e.event_type||"alert").toUpperCase()}  ${(e.label||"").toUpperCase()}  ${e.track_id!=="" && e.track_id!=null ? "ID "+e.track_id : ""}</div>
              <div class="when">${e.timestamp || ""}  ${e.zone || ""}  frame ${e.frame ?? ""}</div></li>`).join("")
          : "<li class='when'>Waiting for alerts…</li>";
      } catch (e) {}
    }
    setInterval(refresh, 700); refresh();
  </script>
</body>
</html>
"""


def create_app(pipeline) -> FastAPI:
    app = FastAPI(title="Wildwatch", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_HTML.replace("__CSS_VARS__", CSS_VARS))

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            while pipeline.running:
                with pipeline.lock:
                    jpeg = pipeline.latest_jpeg
                if jpeg:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                time.sleep(0.03)

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/stats")
    def stats():
        with pipeline.lock:
            payload = dict(pipeline.stats)
        return JSONResponse(payload)

    @app.get("/api/events")
    def events():
        return JSONResponse({"events": list(pipeline.event_log.recent)})

    @app.get("/api/zones")
    def get_zones():
        return JSONResponse({"items": pipeline.zones.as_list(), "draft": list(pipeline.zones.draft)})

    @app.post("/api/zones/draft")
    def draft_point(payload: dict = Body(...)):
        try:
            draft = pipeline.zones.add_draft(float(payload["x"]), float(payload["y"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        return {"draft": draft}

    @app.post("/api/zones/draft/pop")
    def draft_pop():
        pipeline.zones.pop_draft()
        return {"draft": list(pipeline.zones.draft)}

    @app.post("/api/zones/draft/cancel")
    def draft_cancel():
        pipeline.zones.cancel_draft()
        return {"draft": []}

    @app.post("/api/zones/draft/close")
    def draft_close(payload: dict = Body(default={})):
        try:
            zone = pipeline.zones.close_draft(name=(payload or {}).get("name") or "")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"zone": zone.as_dict(), "items": pipeline.zones.as_list()}

    @app.post("/api/zones")
    def add_zone(payload: dict = Body(...)):
        try:
            zone = pipeline.zones.add(str(payload.get("name") or ""), payload.get("polygon") or [])
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        return {"zone": zone.as_dict(), "items": pipeline.zones.as_list()}

    @app.delete("/api/zones/{name}")
    def delete_zone(name: str):
        if not pipeline.zones.remove(name):
            raise HTTPException(404, "zone not found")
        return {"ok": True, "items": pipeline.zones.as_list()}

    return app


def run_browser(pipeline, host: str, port: int) -> None:
    import uvicorn

    def loop():
        try:
            while pipeline.running:
                if pipeline.step() is None:
                    # File ended: keep the last frame on the dashboard.
                    time.sleep(0.25)
                    if not pipeline.capture.live:
                        time.sleep(0.5)
                        continue
        except Exception as exc:
            print(f"[wildwatch] pipeline thread error: {exc}")
            pipeline.running = False

    worker = threading.Thread(target=loop, daemon=True, name="wildwatch-pipeline")
    worker.start()
    # Wait briefly so the first JPEG exists before browsers connect.
    for _ in range(50):
        if pipeline.latest_jpeg is not None:
            break
        time.sleep(0.1)
    print(f"[wildwatch] dashboard at http://{host}:{port}/  (Ctrl+C to stop)")
    try:
        uvicorn.run(create_app(pipeline), host=host, port=port, log_level="warning")
    finally:
        pipeline.running = False
        worker.join(timeout=2)
        pipeline.close()
