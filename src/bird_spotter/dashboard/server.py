"""Live MJPEG streaming and responsive desktop/mobile dashboard for British Bird Spotter."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from bird_spotter.vision.detector import Detection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Bird Spotter — British Bird Watch & PTZ Tracker</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --panel-header: #21262d;
    --border: #30363d;
    --text: #f0f6fc;
    --muted: #8b949e;
    --green: #2ea043;
    --green-bright: #3fb950;
    --cyan: #388bfd;
    --orange: #d29922;
    --red: #f85149;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-tap-highlight-color: transparent;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 12px 20px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 50;
  }
  header h1 {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--green-bright);
  }
  .header-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }
  .badge {
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: #21262d;
    border: 1px solid var(--border);
    white-space: nowrap;
  }
  .badge-active {
    background: rgba(46, 160, 67, 0.2);
    color: var(--green-bright);
    border-color: var(--green);
  }
  .badge-warn {
    background: rgba(210, 153, 34, 0.2);
    color: #e3b341;
    border-color: var(--orange);
  }
  .badge-idle {
    background: rgba(139, 148, 158, 0.15);
    color: var(--muted);
  }
  .power-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .power-on {
    background: rgba(46, 160, 67, 0.15);
    color: #3fb950;
    border: 2px solid var(--green);
  }
  .power-off {
    background: rgba(248, 81, 73, 0.15);
    color: #f85149;
    border: 2px solid var(--red);
  }
  main {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 16px;
    padding: 16px 20px;
    max-width: 1720px;
    margin: 0 auto;
  }
  @media (max-width: 1024px) {
    main { grid-template-columns: 1fr; padding: 12px; }
    header { padding: 10px 12px; }
  }
  .left-pane, .right-pane {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .card-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: var(--panel-header);
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
    font-weight: 600;
  }
  .card-body { padding: 14px; }
  .video-container {
    width: 100%;
    aspect-ratio: 16 / 9;
    background: #090d13;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
  }
  #roiEditorOverlay {
    position: absolute;
    inset: 0;
    z-index: 20;
    display: none;
    touch-action: none;
    user-select: none;
  }
  #roiSelectionBox {
    position: absolute;
    border: 3px dashed #ffee00;
    outline: 2px solid #000;
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.70);
    cursor: move;
    box-sizing: border-box;
  }
  .roi-handle {
    width: 22px;
    height: 22px;
    background: #ffee00;
    border: 2px solid #000;
    border-radius: 50%;
    position: absolute;
    touch-action: none;
    box-sizing: border-box;
    box-shadow: 0 0 8px rgba(0, 0, 0, 0.8);
  }
  .roi-handle-nw { top: -11px; left: -11px; cursor: nwse-resize; }
  .roi-handle-ne { top: -11px; right: -11px; cursor: nesw-resize; }
  .roi-handle-sw { bottom: -11px; left: -11px; cursor: nesw-resize; }
  .roi-handle-se { bottom: -11px; right: -11px; cursor: nwse-resize; }
  .video-container img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .video-off {
    display: none;
    color: var(--muted);
    font-size: 0.95rem;
  }
  .override-banner {
    display: none;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 8px 14px;
    background: rgba(210, 153, 34, 0.18);
    border-bottom: 1px solid var(--orange);
    color: #e3b341;
    font-size: 0.82rem;
    font-weight: 600;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  @media (max-width: 640px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
  }
  .stat-box {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
  }
  .stat-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .stat-val {
    font-size: 0.98rem;
    font-weight: 600;
    margin-top: 3px;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  button {
    background: #21262d;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    touch-action: manipulation;
  }
  button:hover:not(:disabled) {
    background: #30363d;
    border-color: #8b949e;
  }
  button.btn-primary {
    background: var(--green);
    border-color: var(--green-bright);
    color: #fff;
  }
  button.btn-primary:hover { background: var(--green-bright); }
  button.btn-danger {
    background: rgba(248, 81, 73, 0.18);
    border-color: var(--red);
    color: #ff7b72;
  }
  button.btn-danger:hover { background: rgba(248, 81, 73, 0.32); }
  button.active {
    background: rgba(46, 160, 67, 0.25);
    border-color: var(--green-bright);
    color: var(--green-bright);
  }
  .dpad {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    max-width: 300px;
    margin: 8px auto;
  }
  .dpad button {
    padding: 12px 10px;
    font-size: 0.9rem;
  }
  .zoom-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    justify-content: center;
  }
  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
    background: #11161d;
  }
  .tab-btn {
    flex: 1;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    background: transparent;
    padding: 10px 8px;
    font-size: 0.82rem;
    color: var(--muted);
  }
  .tab-btn.active {
    color: var(--green-bright);
    border-bottom-color: var(--green-bright);
    background: rgba(46, 160, 67, 0.08);
  }
  .tab-panel { display: none; padding: 14px; }
  .tab-panel.active { display: block; }
  .dex-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 10px;
  }
  .dex-card {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .dex-card img {
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    background: #05070a;
    cursor: pointer;
  }
  .dex-info {
    padding: 9px 11px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .dex-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 700;
    font-size: 0.92rem;
  }
  .dex-sci {
    font-size: 0.74rem;
    color: var(--muted);
    font-style: italic;
  }
  .dex-meta {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 3px;
  }
  .species-count-badge {
    background: rgba(46, 160, 67, 0.2);
    border: 1px solid var(--green);
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--green-bright);
  }
  .snapshot-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
  }
  .snap-card {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 0.15s;
  }
  .snap-card:hover { border-color: var(--green-bright); }
  .snap-card img {
    width: 100%;
    aspect-ratio: 16 / 10;
    object-fit: cover;
    display: block;
    background: #000;
  }
  .snap-card-caption {
    padding: 6px 8px;
    font-size: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .chart-bars {
    display: flex;
    align-items: flex-end;
    gap: 4px;
    height: 120px;
    padding-top: 14px;
    border-bottom: 1px solid var(--border);
  }
  .chart-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    justify-content: flex-end;
  }
  .chart-bar {
    width: 100%;
    background: var(--green);
    border-radius: 3px 3px 0 0;
    min-height: 2px;
    transition: height 0.2s;
  }
  .chart-lbl {
    font-size: 0.62rem;
    color: var(--muted);
    margin-top: 4px;
  }
  .status-toast {
    font-size: 0.8rem;
    color: var(--green-bright);
    min-height: 18px;
    margin-top: 6px;
    text-align: center;
  }
  /* Modal for Snapshot Inspection, Relabeling & Deletion */
  .modal-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.8);
    z-index: 200;
    align-items: center;
    justify-content: center;
    padding: 16px;
  }
  .modal-backdrop.open { display: flex; }
  .modal-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    max-width: 780px;
    width: 100%;
    max-height: 92vh;
    overflow-y: auto;
  }
  .modal-box img {
    width: 100%;
    max-height: 55vh;
    object-fit: contain;
    background: #000;
    display: block;
  }
  select, input[type="range"] {
    background: #0d1117;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 0.84rem;
  }
</style>
</head>
<body>
<header>
  <h1>🐦 British Bird Spotter</h1>
  <div class="header-actions">
    <div id="fsmBadge" class="badge badge-active">WATCHING (HOME 1.0x)</div>
    <div id="targetBadge" class="badge badge-idle">TARGET: NONE</div>
    <button id="btnSolar" class="badge badge-active" style="cursor:pointer;" onclick="sendCtrl({action:'toggle_solar_schedule'})" title="Automatic Sunset/Dawn Scheduler">
      🌅 Auto-Sleep: ON
    </button>
    <button id="pwrBtn" class="power-btn power-on" onclick="sendCtrl({action:'toggle_system_power'})">
      ⏻ CAM ACTIVE
    </button>
    <button class="btn-danger" style="padding:5px 10px; font-size:0.76rem; border-radius:999px;" onclick="if(confirm('Safely shut down Raspberry Pi so you can unplug power?')) { showToast('Shutting down Raspberry Pi...'); sendCtrl({action:'shutdown_pi'}); }">
      🛑 Shutdown Pi
    </button>
  </div>
</header>

<main>
  <!-- LEFT PANE: VIDEO STREAM, TELEMETRY & ANALYTICS -->
  <div class="left-pane">
    <div class="card">
      <div class="card-header">
        <span>📹 Live OBSBOT Tiny SE Feed</span>
        <div style="display:flex; flex-wrap:wrap; gap:6px;">
          <button class="btn-primary" onclick="takeSnapshot()">📸 Manual Snapshot</button>
          <button id="btnEditRoi" onclick="toggleRoiEditor()" title="Configure Zone of Interest on feeder">🎯 Select Zone</button>
          <label id="lblLockAspect" style="display:none; align-items:center; gap:5px; font-size:0.75rem; color:var(--text); cursor:pointer; background:#21262d; padding:4px 8px; border-radius:6px; border:1px solid var(--border);"><input type="checkbox" id="chkLockAspect"> 🔒 16:9</label>
          <button id="btnApplyRoi" class="btn-primary" style="display:none;" onclick="applyRoiSelection()" title="Save Zone of Interest">✅ Set Zone</button>
          <button id="btnClearRoi" class="btn-danger" style="display:none;" onclick="clearRoiSelection()" title="Disable Zone of Interest">❌ Unset Zone</button>
          <button id="btnPreviewBbox" class="active" onclick="sendCtrl({action:'toggle_preview_bbox'})" title="Toggle bounding boxes on live camera preview">📦 Preview Box: ON</button>
          <button id="btnCaptureBbox" class="active" onclick="sendCtrl({action:'toggle_capture_bbox'})" title="Toggle bounding boxes on saved bird snapshots">🖼️ Capture Box: ON</button>
          <button id="btnHudMode" class="active" onclick="toggleStreamHud()">HUD: ON</button>
          <button id="btnToggleCam" class="active" onclick="toggleCamPreview()">Stream: ON</button>
        </div>
      </div>
      <div id="overrideBanner" class="override-banner">
        <span id="overrideText">Manual Override Active — Auto-Framing paused</span>
        <button class="btn-primary" style="padding:4px 10px; font-size:0.76rem;" onclick="sendCtrl({action:'resume_auto_tracking'})">
          ▶ Resume Auto-Framing Now
        </button>
      </div>
      <div class="video-container">
        <img id="streamImg" src="/stream.mjpg" alt="Bird Spotter Live Stream">
        <div id="streamOffText" class="video-off">Camera stream paused</div>
        <div id="roiEditorOverlay">
          <div id="roiSelectionBox">
            <div class="roi-handle roi-handle-nw" data-handle="nw"></div>
            <div class="roi-handle roi-handle-ne" data-handle="ne"></div>
            <div class="roi-handle roi-handle-sw" data-handle="sw"></div>
            <div class="roi-handle roi-handle-se" data-handle="se"></div>
            <div style="position:absolute; bottom:6px; left:8px; background:rgba(13,17,23,0.85); color:#ffee00; font-size:0.72rem; font-weight:700; padding:2px 6px; border-radius:4px; pointer-events:none;">
              🎯 ZONE OF INTEREST
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <span>📊 Live Telemetry</span>
        <span id="classifierTag" style="font-size:0.75rem; color:var(--muted);">Classifier: --</span>
      </div>
      <div class="card-body">
        <div class="stat-grid">
          <div class="stat-box">
            <div class="stat-label">Camera FPS</div>
            <div id="fpsVal" class="stat-val">--</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Hailo-8 Latency</div>
            <div id="inferVal" class="stat-val">-- ms</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Optical/Dig Zoom</div>
            <div id="zoomVal" class="stat-val">1.0x</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Birds in View</div>
            <div id="birdCountVal" class="stat-val">0</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Locked Target</div>
            <div id="primaryTargetVal" class="stat-val">None</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Frame Fill (25-40%)</div>
            <div id="targetAreaVal" class="stat-val">0.0%</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Pan / Tilt</div>
            <div id="panTiltVal" class="stat-val">0 / 0</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Species / Total</div>
            <div id="totalSightingsVal" class="stat-val">0 / 0</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT PANE: PTZ GIMBAL CONTROLS & LIFE LIST / GALLERY TABS -->
  <div class="right-pane">
    <div class="card">
      <div class="card-header">
        <span>🎯 OBSBOT PTZ Gimbal & Home Preset</span>
        <button id="btnUkFilter" class="active" style="padding:4px 9px; font-size:0.75rem;" onclick="sendCtrl({action:'toggle_uk_filter'})">
          🇬🇧 UK Filter: ON
        </button>
      </div>
      <div class="card-body">
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px;">
          <button id="btnAutoTrack" class="active" style="flex:1;" onclick="sendCtrl({action:'toggle_tracking'})">
            🤖 Auto-Framing: ON
          </button>
          <button onclick="sendCtrl({action:'go_home'})" title="Return to saved feeder/home position">🏠 Go Home</button>
          <button onclick="saveHomePreset()" title="Save current pan/tilt as the default detection view">📌 Set Home</button>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.74rem; color:var(--muted); font-weight:600; text-transform:uppercase;">
          <span>Manual Jog D-Pad (Pauses Auto-Framing 60s)</span>
          <select id="nudgeStepSel" style="padding:2px 6px; font-size:0.74rem;">
            <option value="18000">Fine Step</option>
            <option value="36000" selected>Medium Step</option>
            <option value="72000">Large Step</option>
          </select>
        </div>
        <div class="dpad">
          <div></div>
          <button onclick="nudgePTZ(0, 1)">▲ Up</button>
          <div></div>
          <button onclick="nudgePTZ(1, 0)">◀ Left</button>
          <button onclick="sendCtrl({action:'center'})">🎯 0,0</button>
          <button onclick="nudgePTZ(-1, 0)">Right ▶</button>
          <div></div>
          <button onclick="nudgePTZ(0, -1)">▼ Down</button>
          <div></div>
        </div>

        <div style="margin-top:10px; font-size:0.74rem; color:var(--muted); font-weight:600; text-transform:uppercase; display:flex; justify-content:space-between;">
          <span>Hardware Field of View (OBSBOT ISP)</span>
          <span id="fovLbl">65° Narrow</span>
        </div>
        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:6px; margin-top:6px;">
          <button id="btnFov86" onclick="sendCtrl({action:'set_fov', fov:86})">86° Wide</button>
          <button id="btnFov78" onclick="sendCtrl({action:'set_fov', fov:78})">78° Medium</button>
          <button id="btnFov65" class="active" onclick="sendCtrl({action:'set_fov', fov:65})">65° Narrow</button>
        </div>

        <div style="margin-top:10px; font-size:0.74rem; color:var(--muted); font-weight:600; text-transform:uppercase; display:flex; justify-content:space-between;">
          <span>Hardware Zoom Control (1.0x – 4.0x)</span>
          <span id="zoomSliderLbl">1.0x</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
          <button onclick="sendCtrl({action:'zoom_out', step:0.25})">🔍 -</button>
          <input id="zoomSlider" type="range" min="1.0" max="4.0" step="0.25" value="1.0" style="flex:1;" onchange="sendCtrl({action:'set_zoom', zoom:parseFloat(this.value)})">
          <button onclick="sendCtrl({action:'zoom_in', step:0.25})">🔍 +</button>
        </div>
        <div class="zoom-bar">
          <button onclick="sendCtrl({action:'set_zoom', zoom:1.0})">1.0x</button>
          <button onclick="sendCtrl({action:'set_zoom', zoom:1.5})">1.5x</button>
          <button onclick="sendCtrl({action:'set_zoom', zoom:2.0})">2.0x</button>
          <button onclick="sendCtrl({action:'set_zoom', zoom:3.0})">3.0x</button>
          <button onclick="sendCtrl({action:'set_zoom', zoom:4.0})">4.0x</button>
        </div>

        <div style="margin-top:10px; font-size:0.74rem; color:var(--muted); font-weight:600; text-transform:uppercase; display:flex; justify-content:space-between; align-items:center;">
          <span>Motorized Lens Focus (Window / Feeder)</span>
          <span id="focusLbl">Auto</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
          <button id="btnAutofocus" class="active" style="padding:5px 10px; font-size:0.76rem;" onclick="sendCtrl({action:'toggle_autofocus'})">AF: ON</button>
          <input id="focusSlider" type="range" min="0" max="100" step="1" value="50" style="flex:1;" onchange="sendCtrl({action:'set_focus', focus:parseInt(this.value, 10)})">
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;">
          <button id="btnInvPan" onclick="sendCtrl({action:'toggle_invert_pan'})">Invert Pan: ON</button>
          <button id="btnInvTilt" onclick="sendCtrl({action:'toggle_invert_tilt'})">Invert Tilt: ON</button>
        </div>
        <div id="statusToast" class="status-toast"></div>
      </div>
    </div>

    <!-- TABS: BIRD DEX / CAPTURES & REVIEW / ANALYTICS -->
    <div class="card">
      <div class="tabs">
        <button id="tabBtnDex" class="tab-btn active" onclick="switchTab('dex')">🦉 Bird Dex (<span id="dexCountLbl">0</span>)</button>
        <button id="tabBtnSnaps" class="tab-btn" onclick="switchTab('snaps')">🖼️ Captures (<span id="snapCountLbl">0</span>)</button>
        <button id="tabBtnCharts" class="tab-btn" onclick="switchTab('charts')">📈 Analytics</button>
      </div>

      <!-- TAB 1: LIFE LIST / BIRD DEX -->
      <div id="tabPanelDex" class="tab-panel active">
        <div id="dexGrid" class="dex-grid">
          <div style="grid-column: 1 / -1; color:var(--muted); text-align:center; padding:20px; font-size:0.86rem;">
            No bird species recorded yet — watching feeder...
          </div>
        </div>
      </div>

      <!-- TAB 2: RECENT CAPTURES & MANUAL REVIEW -->
      <div id="tabPanelSnaps" class="tab-panel">
        <div style="font-size:0.76rem; color:var(--muted); margin-bottom:10px;">
          Click any photograph to view full resolution, inspect species confidence, re-label, or delete false positives.
        </div>
        <div id="snapshotGallery" class="snapshot-gallery"></div>
      </div>

      <!-- TAB 3: TIME-OF-DAY & DAILY ANALYTICS -->
      <div id="tabPanelCharts" class="tab-panel">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-size:0.8rem; font-weight:600;">Sightings by Hour of Day (00:00 – 23:00)</span>
          <button onclick="if(confirm('Clear all sighting statistics?')) sendCtrl({action:'clear_stats'})" style="padding:3px 8px; font-size:0.72rem;">Reset DB</button>
        </div>
        <div id="hourlyChart" class="chart-bars"></div>

        <div style="font-size:0.8rem; font-weight:600; margin-top:18px; margin-bottom:6px;">
          14-Day Activity Trend
        </div>
        <div id="dailyChart" class="chart-bars"></div>
      </div>
    </div>
  </div>
</main>

<!-- SNAPSHOT REVIEW / RELABEL / DELETE MODAL -->
<div id="reviewModal" class="modal-backdrop" onclick="if(event.target===this) closeModal()">
  <div class="modal-box">
    <div class="card-header">
      <span id="modalTitle">Sighting Inspection</span>
      <button onclick="closeModal()" style="padding:4px 10px;">✕ Close</button>
    </div>
    <img id="modalImg" src="" alt="Bird Snapshot">
    <div class="card-body" style="display:flex; flex-direction:column; gap:12px;">
      <div id="modalMeta" style="font-size:0.84rem; color:var(--muted);"></div>
      <div id="modalTopPreds" style="font-size:0.82rem;"></div>
      <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center; border-top:1px solid var(--border); padding-top:12px;">
        <label style="font-size:0.82rem; font-weight:600;">Re-label Species:</label>
        <select id="relabelSelect" style="flex:1; min-width:180px;"></select>
        <button class="btn-primary" onclick="submitRelabel()">✔ Save Label</button>
        <button class="btn-danger" onclick="submitDelete()">🗑️ Delete Photo</button>
      </div>
    </div>
  </div>
</div>

<script>
let camPreviewOn = true;
let streamHudOn = true;
let ukCatalog = [];
let currentSightings = [];
let selectedSightingId = null;

let isEditingRoi = false;
let currentRoiState = null;
let roiDraft = { x: 0.25, y: 0.25, w: 0.50, h: 0.50 };

function toggleRoiEditor() {
  if (isEditingRoi) {
    cancelRoiEditor();
  } else {
    startRoiEditor();
  }
}

function startRoiEditor() {
  isEditingRoi = true;
  if (currentRoiState && currentRoiState.enabled) {
    roiDraft.x = currentRoiState.x;
    roiDraft.y = currentRoiState.y;
    roiDraft.w = currentRoiState.w;
    roiDraft.h = currentRoiState.h;
  } else {
    roiDraft.x = 0.25;
    roiDraft.y = 0.25;
    roiDraft.w = 0.50;
    roiDraft.h = 0.50;
  }
  document.getElementById('roiEditorOverlay').style.display = 'block';
  document.getElementById('btnApplyRoi').style.display = 'inline-block';
  const lblLock = document.getElementById('lblLockAspect');
  if (lblLock) lblLock.style.display = 'inline-flex';
  const editBtn = document.getElementById('btnEditRoi');
  editBtn.textContent = '❌ Cancel Editing';
  editBtn.classList.add('active');
  renderRoiBox();
  showToast('Drag box to move, drag corners to resize around feeder');
}

function cancelRoiEditor() {
  isEditingRoi = false;
  document.getElementById('roiEditorOverlay').style.display = 'none';
  document.getElementById('btnApplyRoi').style.display = 'none';
  const lblLock = document.getElementById('lblLockAspect');
  if (lblLock) lblLock.style.display = 'none';
  const editBtn = document.getElementById('btnEditRoi');
  editBtn.textContent = (currentRoiState && currentRoiState.enabled) ? '🎯 Zone: ON' : '🎯 Select Zone';
  if (!(currentRoiState && currentRoiState.enabled)) {
    editBtn.classList.remove('active');
  }
}

async function applyRoiSelection() {
  const lockAspect = document.getElementById('chkLockAspect') ? document.getElementById('chkLockAspect').checked : false;
  const payload = {
    action: 'set_roi',
    x: roiDraft.x,
    y: roiDraft.y,
    w: roiDraft.w,
    h: roiDraft.h,
    lock_aspect: lockAspect,
  };
  await sendCtrl(payload);
  isEditingRoi = false;
  document.getElementById('roiEditorOverlay').style.display = 'none';
  document.getElementById('btnApplyRoi').style.display = 'none';
  const lblLock = document.getElementById('lblLockAspect');
  if (lblLock) lblLock.style.display = 'none';
  showToast('Zone of Interest set and saved!');
}

async function clearRoiSelection() {
  await sendCtrl({ action: 'clear_roi' });
  isEditingRoi = false;
  document.getElementById('roiEditorOverlay').style.display = 'none';
  document.getElementById('btnApplyRoi').style.display = 'none';
  const lblLock = document.getElementById('lblLockAspect');
  if (lblLock) lblLock.style.display = 'none';
  showToast('Zone of Interest cleared');
}

function renderRoiBox() {
  const box = document.getElementById('roiSelectionBox');
  if (!box) return;
  box.style.left = (roiDraft.x * 100) + '%';
  box.style.top = (roiDraft.y * 100) + '%';
  box.style.width = (roiDraft.w * 100) + '%';
  box.style.height = (roiDraft.h * 100) + '%';
}

(function initRoiInteractions() {
  const overlay = document.getElementById('roiEditorOverlay');
  const box = document.getElementById('roiSelectionBox');
  if (!overlay || !box) return;

  let dragMode = null; // 'move', 'nw', 'ne', 'sw', 'se'
  let startX = 0;
  let startY = 0;
  let initX = 0;
  let initY = 0;
  let initW = 0;
  let initH = 0;

  function onPointerDown(e) {
    if (!isEditingRoi) return;
    const handle = e.target.closest('.roi-handle');
    if (handle) {
      dragMode = handle.dataset.handle;
    } else if (e.target.closest('#roiSelectionBox')) {
      dragMode = 'move';
    } else {
      return;
    }
    startX = e.clientX;
    startY = e.clientY;
    initX = roiDraft.x;
    initY = roiDraft.y;
    initW = roiDraft.w;
    initH = roiDraft.h;

    try {
      box.setPointerCapture(e.pointerId);
    } catch (err) {}
    e.preventDefault();
    e.stopPropagation();
  }

  function onPointerMove(e) {
    if (!dragMode) return;
    const rect = overlay.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const dx = (e.clientX - startX) / rect.width;
    const dy = (e.clientY - startY) / rect.height;
    const lockAspect = document.getElementById('chkLockAspect') ? document.getElementById('chkLockAspect').checked : false;

    if (dragMode === 'move') {
      const maxX = 1.0 - initW;
      const maxY = 1.0 - initH;
      roiDraft.x = Math.round(Math.max(0.0, Math.min(maxX, initX + dx)) * 10000) / 10000;
      roiDraft.y = Math.round(Math.max(0.0, Math.min(maxY, initY + dy)) * 10000) / 10000;
    } else if (lockAspect) {
      // 16:9 locked mode (w === h in normalized overlay)
      const initS = (initW + initH) * 0.5;
      if (dragMode === 'se') {
        const ds = (dx + dy) * 0.5;
        const maxS = Math.min(1.0 - initX, 1.0 - initY);
        const s = Math.round(Math.max(0.10, Math.min(maxS, initS + ds)) * 10000) / 10000;
        roiDraft.x = initX;
        roiDraft.y = initY;
        roiDraft.w = s;
        roiDraft.h = s;
      } else if (dragMode === 'nw') {
        const anchorX = initX + initS;
        const anchorY = initY + initS;
        const ds = (-dx + -dy) * 0.5;
        const maxS = Math.min(anchorX, anchorY);
        const s = Math.round(Math.max(0.10, Math.min(maxS, initS + ds)) * 10000) / 10000;
        roiDraft.w = s;
        roiDraft.h = s;
        roiDraft.x = Math.round((anchorX - s) * 10000) / 10000;
        roiDraft.y = Math.round((anchorY - s) * 10000) / 10000;
      } else if (dragMode === 'ne') {
        const anchorX = initX;
        const anchorY = initY + initS;
        const ds = (dx + -dy) * 0.5;
        const maxS = Math.min(1.0 - anchorX, anchorY);
        const s = Math.round(Math.max(0.10, Math.min(maxS, initS + ds)) * 10000) / 10000;
        roiDraft.w = s;
        roiDraft.h = s;
        roiDraft.x = anchorX;
        roiDraft.y = Math.round((anchorY - s) * 10000) / 10000;
      } else if (dragMode === 'sw') {
        const anchorX = initX + initS;
        const anchorY = initY;
        const ds = (-dx + dy) * 0.5;
        const maxS = Math.min(anchorX, 1.0 - anchorY);
        const s = Math.round(Math.max(0.10, Math.min(maxS, initS + ds)) * 10000) / 10000;
        roiDraft.w = s;
        roiDraft.h = s;
        roiDraft.x = Math.round((anchorX - s) * 10000) / 10000;
        roiDraft.y = anchorY;
      }
    } else {
      // Free aspect ratio resize
      if (dragMode === 'se') {
        const nw = Math.max(0.10, Math.min(1.0 - initX, initW + dx));
        const nh = Math.max(0.10, Math.min(1.0 - initY, initH + dy));
        roiDraft.x = initX;
        roiDraft.y = initY;
        roiDraft.w = Math.round(nw * 10000) / 10000;
        roiDraft.h = Math.round(nh * 10000) / 10000;
      } else if (dragMode === 'sw') {
        const nw = Math.max(0.10, Math.min(initX + initW, initW - dx));
        const nh = Math.max(0.10, Math.min(1.0 - initY, initH + dy));
        roiDraft.w = Math.round(nw * 10000) / 10000;
        roiDraft.x = Math.round((initX + initW - nw) * 10000) / 10000;
        roiDraft.y = initY;
        roiDraft.h = Math.round(nh * 10000) / 10000;
      } else if (dragMode === 'ne') {
        const nw = Math.max(0.10, Math.min(1.0 - initX, initW + dx));
        const nh = Math.max(0.10, Math.min(initY + initH, initH - dy));
        roiDraft.x = initX;
        roiDraft.w = Math.round(nw * 10000) / 10000;
        roiDraft.h = Math.round(nh * 10000) / 10000;
        roiDraft.y = Math.round((initY + initH - nh) * 10000) / 10000;
      } else if (dragMode === 'nw') {
        const nw = Math.max(0.10, Math.min(initX + initW, initW - dx));
        const nh = Math.max(0.10, Math.min(initY + initH, initH - dy));
        roiDraft.w = Math.round(nw * 10000) / 10000;
        roiDraft.h = Math.round(nh * 10000) / 10000;
        roiDraft.x = Math.round((initX + initW - nw) * 10000) / 10000;
        roiDraft.y = Math.round((initY + initH - nh) * 10000) / 10000;
      }
    }

    renderRoiBox();
  }

  function onPointerUp(e) {
    if (dragMode) {
      try {
        box.releasePointerCapture(e.pointerId);
      } catch (err) {}
      dragMode = null;
    }
  }

  box.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);
})();

async function loadCatalog() {
  try {
    const res = await fetch('/api/catalog');
    if (res.ok) {
      ukCatalog = await res.json();
      const sel = document.getElementById('relabelSelect');
      sel.innerHTML = ukCatalog.map(item =>
        `<option value="${item.species}">${item.species} (${item.scientific_name})</option>`
      ).join('');
    }
  } catch (e) {}
}
loadCatalog();

function switchTab(tab) {
  for (const t of ['dex', 'snaps', 'charts']) {
    document.getElementById('tabBtn' + t.charAt(0).toUpperCase() + t.slice(1)).classList.toggle('active', t === tab);
    document.getElementById('tabPanel' + t.charAt(0).toUpperCase() + t.slice(1)).classList.toggle('active', t === tab);
  }
}

function updateStreamSrc() {
  const img = document.getElementById('streamImg');
  if (!camPreviewOn) {
    img.src = '';
    return;
  }
  const cleanParam = streamHudOn ? '0' : '1';
  img.src = `/stream.mjpg?clean=${cleanParam}&t=${Date.now()}`;
}

function toggleCamPreview() {
  camPreviewOn = !camPreviewOn;
  const img = document.getElementById('streamImg');
  const offText = document.getElementById('streamOffText');
  const btn = document.getElementById('btnToggleCam');
  if (camPreviewOn) {
    updateStreamSrc();
    img.style.display = 'block';
    offText.style.display = 'none';
    btn.textContent = 'Stream: ON';
    btn.className = 'active';
  } else {
    img.src = '';
    img.style.display = 'none';
    offText.style.display = 'block';
    btn.textContent = 'Stream: OFF';
    btn.className = '';
  }
}

function toggleStreamHud() {
  streamHudOn = !streamHudOn;
  const btn = document.getElementById('btnHudMode');
  btn.textContent = 'HUD: ' + (streamHudOn ? 'ON' : 'CLEAN');
  btn.className = streamHudOn ? 'active' : '';
  if (camPreviewOn) updateStreamSrc();
}

function nudgePTZ(panSign, tiltSign) {
  const step = parseInt(document.getElementById('nudgeStepSel').value || '36000', 10);
  sendCtrl({action: 'nudge', d_pan: panSign * step, d_tilt: roundTilt(tiltSign * step)});
}

function roundTilt(val) {
  return Math.round(val * 0.8);
}

async function saveHomePreset() {
  await sendCtrl({action: 'save_home_preset'});
  const z = document.getElementById('zoomVal').textContent;
  showToast(`Saved current Pan, Tilt & Zoom (${z}) as Home Preset!`);
}

async function sendCtrl(payload) {
  try {
    const res = await fetch('/api/control', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const st = await res.json();
    updateUI(st);
  } catch (e) {
    console.error(e);
  }
}

async function takeSnapshot() {
  try {
    showToast('Capturing clean high-res snapshot...');
    const res = await fetch('/api/snapshot', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      showToast(`Saved: ${data.species || 'Bird'} (${data.filename})`);
    } else {
      showToast('Snapshot failed');
    }
  } catch (e) {
    showToast('Error taking snapshot');
  }
}

function showToast(msg) {
  const el = document.getElementById('statusToast');
  if (!el) return;
  el.textContent = msg;
  setTimeout(() => {
    if (el.textContent === msg) el.textContent = '';
  }, 4000);
}

function openModalById(id) {
  const item = currentSightings.find(s => s.id === id);
  if (!item) return;
  selectedSightingId = item.id;
  document.getElementById('modalTitle').textContent = `#${item.id} — ${item.species} (${(item.species_confidence * 100 || item.yolo_confidence * 100).toFixed(0)}%)`;
  document.getElementById('modalImg').src = '/snapshots/' + encodeURI(item.snapshot_filename);
  document.getElementById('modalMeta').innerHTML =
    `<strong>Timestamp:</strong> ${item.timestamp} &nbsp;|&nbsp; ` +
    `<strong>Scientific:</strong> <em>${item.scientific_name || 'N/A'}</em> &nbsp;|&nbsp; ` +
    `<strong>Zoom:</strong> ${(item.zoom_level || 1.0).toFixed(1)}x &nbsp;|&nbsp; ` +
    `<strong>Sharpness:</strong> ${(item.sharpness || 0).toFixed(0)} &nbsp;|&nbsp; ` +
    `<strong>Status:</strong> ${item.verified_status || 'auto'}`;

  const preds = item.top_predictions || [];
  if (preds.length > 0) {
    document.getElementById('modalTopPreds').innerHTML =
      '<strong>Top Classifier Predictions:</strong> ' +
      preds.map(p => `${p.species} (${(p.confidence * 100).toFixed(1)}%)`).join(' · ');
  } else {
    document.getElementById('modalTopPreds').textContent = '';
  }

  const sel = document.getElementById('relabelSelect');
  if (sel) sel.value = item.species;
  document.getElementById('reviewModal').classList.add('open');
}

function closeModal() {
  document.getElementById('reviewModal').classList.remove('open');
  selectedSightingId = null;
}

async function submitRelabel() {
  if (!selectedSightingId) return;
  const newSp = document.getElementById('relabelSelect').value;
  const res = await fetch('/api/sighting/relabel', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: selectedSightingId, species: newSp})
  });
  if (res.ok) {
    showToast('Re-labeled sighting to ' + newSp);
    closeModal();
  }
}

async function submitDelete() {
  if (!selectedSightingId) return;
  if (!confirm('Delete this sighting and remove its snapshot from disk?')) return;
  const res = await fetch('/api/sighting/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: selectedSightingId})
  });
  if (res.ok) {
    showToast('Deleted sighting #' + selectedSightingId);
    closeModal();
  }
}

function renderBars(containerId, values, labels) {
  const el = document.getElementById(containerId);
  if (!el || !Array.isArray(values)) return;
  const maxVal = Math.max(1, ...values);
  el.innerHTML = values.map((v, i) => {
    const pct = Math.round((v / maxVal) * 100);
    return `<div class="chart-col" title="${labels[i]}: ${v} sightings">
      <div class="chart-bar" style="height:${ Math.max(3, pct) }%; opacity:${ v > 0 ? 1 : 0.2 }"></div>
      <div class="chart-lbl">${labels[i]}</div>
    </div>`;
  }).join('');
}

function updateUI(st) {
  if (!st) return;
  document.getElementById('fpsVal').textContent = (st.fps || 0).toFixed(1);
  document.getElementById('inferVal').textContent = (st.infer_ms || 0).toFixed(1) + ' ms';
  const zm = st.zoom || 1.0;
  document.getElementById('zoomVal').textContent = zm.toFixed(1) + 'x';
  document.getElementById('zoomSliderLbl').textContent = zm.toFixed(1) + 'x';
  const slider = document.getElementById('zoomSlider');
  if (document.activeElement !== slider) slider.value = zm;

  document.getElementById('birdCountVal').textContent = st.detections_count || 0;
  document.getElementById('primaryTargetVal').textContent = st.primary_target || 'None';
  document.getElementById('targetAreaVal').textContent = ((st.primary_area || 0) * 100).toFixed(1) + '%';
  document.getElementById('panTiltVal').textContent = `${st.pan || 0} / ${st.tilt || 0}`;
  document.getElementById('classifierTag').textContent =
    `Classifier: ${st.classifier_backend || 'Hailo/ONNX'} (${st.uk_filter_enabled ? 'UK Garden Mask' : 'All 707 EU'})`;

  const fsmBadge = document.getElementById('fsmBadge');
  fsmBadge.textContent = st.fsm_label || (st.tracking_enabled ? 'AUTO-FRAMING: ON' : 'MANUAL');
  fsmBadge.className = 'badge ' + (st.fsm_state === 'MANUAL_OVERRIDE' ? 'badge-warn' : 'badge-active');

  const banner = document.getElementById('overrideBanner');
  if (st.fsm_state === 'MANUAL_OVERRIDE' && st.manual_override_remaining > 0) {
    banner.style.display = 'flex';
    document.getElementById('overrideText').textContent =
      `Manual PTZ Override Active — Auto-framing resumes in ${Math.ceil(st.manual_override_remaining)}s`;
  } else {
    banner.style.display = 'none';
  }

  const targetBadge = document.getElementById('targetBadge');
  if (st.primary_target) {
    targetBadge.textContent = 'TARGET: ' + st.primary_target;
    targetBadge.className = 'badge badge-active';
  } else {
    targetBadge.textContent = 'TARGET: NONE';
    targetBadge.className = 'badge badge-idle';
  }

  const btnSolar = document.getElementById('btnSolar');
  if (btnSolar && st.solar) {
    const s = st.solar;
    if (!s.enabled) {
      btnSolar.className = 'badge badge-idle';
      btnSolar.textContent = '🌅 Auto-Sleep: OFF';
      btnSolar.title = 'Solar Scheduler is disabled. Click to enable automatic sunset sleep & dawn wake.';
    } else if (s.is_night) {
      btnSolar.className = 'badge badge-warn';
      btnSolar.textContent = `🌙 NIGHT SLEEP (Wakes ${s.wake_time || '--:--'})`;
      btnSolar.title = `Night Sleep Active (${s.location || 'Local'})\nDawn: ${s.dawn} | Sunrise: ${s.sunrise}\nSunset: ${s.sunset}\nWake (-30m): ${s.wake_time}\nSleep (+30m): ${s.sleep_time}\nSource: ${s.source}`;
    } else {
      btnSolar.className = 'badge badge-active';
      btnSolar.textContent = `🌅 Day Mode (Sleeps ${s.sleep_time || '--:--'})`;
      btnSolar.title = `Day Active (${s.location || 'Local'})\nDawn: ${s.dawn} | Sunrise: ${s.sunrise}\nSunset: ${s.sunset}\nWake (-30m): ${s.wake_time}\nSleep (+30m): ${s.sleep_time}\nSource: ${s.source}`;
    }
  }

  const pwrBtn = document.getElementById('pwrBtn');
  const isOn = st.system_power_on !== false;
  pwrBtn.className = 'power-btn ' + (isOn ? 'power-on' : 'power-off');
  pwrBtn.textContent = isOn ? '⏻ CAM ACTIVE' : '⏻ CAM STANDBY';

  const btnAutoTrack = document.getElementById('btnAutoTrack');
  btnAutoTrack.textContent = '🤖 Auto-Framing: ' + (st.tracking_enabled ? 'ON' : 'LOCKED OFF');
  btnAutoTrack.className = st.tracking_enabled ? 'active' : '';

  const btnUk = document.getElementById('btnUkFilter');
  btnUk.textContent = '🇬🇧 UK Filter: ' + (st.uk_filter_enabled !== false ? 'ON' : 'ALL 707');
  btnUk.className = st.uk_filter_enabled !== false ? 'active' : '';

  const btnPrevBox = document.getElementById('btnPreviewBbox');
  if (btnPrevBox) {
    const prevOn = st.show_preview_bbox !== false;
    btnPrevBox.textContent = '📦 Preview Box: ' + (prevOn ? 'ON' : 'OFF');
    btnPrevBox.className = prevOn ? 'active' : '';
  }

  const btnCapBox = document.getElementById('btnCaptureBbox');
  if (btnCapBox) {
    const capOn = st.show_capture_bbox !== false;
    btnCapBox.textContent = '🖼️ Capture Box: ' + (capOn ? 'ON' : 'OFF');
    btnCapBox.className = capOn ? 'active' : '';
  }

  const fov = st.fov || 65;
  const fovLbl = document.getElementById('fovLbl');
  if (fovLbl) {
    fovLbl.textContent = fov === 86 ? '86° Wide' : (fov === 78 ? '78° Medium' : '65° Narrow');
  }
  [86, 78, 65].forEach(deg => {
    const b = document.getElementById('btnFov' + deg);
    if (b) b.className = (fov === deg) ? 'active' : '';
  });

  const afOn = st.autofocus !== false;
  const focusVal = st.focus !== undefined ? st.focus : 50;
  const btnAf = document.getElementById('btnAutofocus');
  if (btnAf) {
    btnAf.textContent = 'AF: ' + (afOn ? 'ON' : 'MANUAL');
    btnAf.className = afOn ? 'active' : '';
  }
  const focusSlider = document.getElementById('focusSlider');
  if (focusSlider && document.activeElement !== focusSlider) {
    focusSlider.value = focusVal;
  }
  const focusLbl = document.getElementById('focusLbl');
  if (focusLbl) {
    focusLbl.textContent = afOn ? `Auto (${focusVal})` : `Manual: ${focusVal}/100`;
  }

  document.getElementById('btnInvPan').textContent = 'Invert Pan: ' + (st.invert_pan ? 'ON' : 'OFF');
  document.getElementById('btnInvTilt').textContent = 'Invert Tilt: ' + (st.invert_tilt ? 'ON' : 'OFF');

  // Zone of Interest (ROI) button state
  if (st.roi) {
    currentRoiState = st.roi;
    const btnClearRoi = document.getElementById('btnClearRoi');
    const btnEditRoi = document.getElementById('btnEditRoi');
    if (!isEditingRoi) {
      if (st.roi.enabled) {
        btnEditRoi.textContent = '🎯 Zone: ON';
        btnEditRoi.classList.add('active');
        btnClearRoi.style.display = 'inline-block';
      } else {
        btnEditRoi.textContent = '🎯 Select Zone';
        btnEditRoi.classList.remove('active');
        btnClearRoi.style.display = 'none';
      }
    } else {
      btnClearRoi.style.display = st.roi.enabled ? 'inline-block' : 'none';
    }
  }

  // Life List / Bird Dex
  const dex = st.species_dex || [];
  document.getElementById('dexCountLbl').textContent = dex.length;
  let totalSightings = 0;
  dex.forEach(d => { totalSightings += (d.total_sightings || 0); });
  document.getElementById('totalSightingsVal').textContent = `${dex.length} sp / ${totalSightings}`;

  const dexGrid = document.getElementById('dexGrid');
  if (dex.length === 0) {
    dexGrid.innerHTML = '<div style="grid-column: 1 / -1; color:var(--muted); text-align:center; padding:20px; font-size:0.86rem;">No bird species recorded yet — watching feeder...</div>';
  } else {
    dexGrid.innerHTML = dex.map(d => {
      const imgPath = d.best_thumb_filename || d.best_snapshot_filename;
      const imgHtml = imgPath
        ? `<img src="/snapshots/${encodeURI(imgPath)}" alt="${d.species}" onclick="if(${d.best_sighting_id || 'null'}) openModalById(${d.best_sighting_id})">`
        : `<div style="aspect-ratio:4/3; background:#090d13; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:0.8rem;">No Photo</div>`;
      return `<div class="dex-card">
        ${imgHtml}
        <div class="dex-info">
          <div class="dex-title">
            <span>${d.species}</span>
            <span class="species-count-badge">${d.total_sightings}</span>
          </div>
          <div class="dex-sci">${d.scientific_name || ''}</div>
          <div class="dex-meta">Peak Conf: ${((d.best_confidence || 0) * 100).toFixed(0)}% · Last: ${(d.last_seen || '').split(' ').pop()}</div>
        </div>
      </div>`;
    }).join('');
  }

  // Recent Captures Gallery
  if (Array.isArray(st.recent_snapshots)) {
    currentSightings = st.recent_snapshots;
    document.getElementById('snapCountLbl').textContent = currentSightings.length;
    const gal = document.getElementById('snapshotGallery');
    if (currentSightings.length === 0) {
      gal.innerHTML = '<div style="grid-column: 1 / -1; color:var(--muted); font-size:0.85rem; text-align:center; padding:16px;">No snapshots captured yet</div>';
    } else {
      gal.innerHTML = currentSightings.map(s => {
        const thumb = s.thumb_filename || s.snapshot_filename || s.filename;
        const confPct = (((s.species_confidence || s.yolo_confidence || 0)) * 100).toFixed(0);
        return `<div class="snap-card" onclick="openModalById(${s.id})">
          <img src="/snapshots/${encodeURI(thumb)}" alt="${s.species}">
          <div class="snap-card-caption">
            <strong>${s.species || 'Bird'}</strong>
            <span style="color:var(--green-bright);">${confPct}%</span>
          </div>
        </div>`;
      }).join('');
    }
  }

  // Charts
  if (Array.isArray(st.hourly_distribution)) {
    const hLabels = st.hourly_distribution.map((_, i) => (i % 3 === 0 ? String(i).padStart(2, '0') : ''));
    renderBars('hourlyChart', st.hourly_distribution, hLabels);
  }
  if (Array.isArray(st.daily_trends)) {
    renderBars(
      'dailyChart',
      st.daily_trends.map(d => d.total),
      st.daily_trends.map((d, i) => (i % 2 === 0 ? d.label : ''))
    );
  }
}

setInterval(async () => {
  try {
    const res = await fetch('/api/state');
    if (res.ok) updateUI(await res.json());
  } catch (e) {}
}, 450);
</script>
</body>
</html>
"""


def draw_capture_bbox(
    frame: np.ndarray,
    det: Detection | None,
    species_name: str = "",
    confidence: float = 0.0,
) -> np.ndarray:
    """Draw a clean bounding box and species label onto a saved bird snapshot."""
    if frame is None or det is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    x1, y1 = int(det.xmin * w), int(det.ymin * h)
    x2, y2 = int(det.xmax * w), int(det.ymax * h)
    color = (67, 185, 46)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

    label_text = species_name or det.display_name
    if species_name and confidence > 0.0:
        label_text = f"{species_name} ({confidence:.0%})"

    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
    cv2.putText(
        out,
        label_text,
        (x1 + 5, max(16, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (15, 20, 15),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    primary: Detection | None,
    fps: float,
    infer_ms: float,
    zoom: float = 1.0,
    deadzone_norm: float = 0.08,
    fsm_label: str = "",
    show_bbox: bool = True,
    roi: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Draw bounding boxes, aim crosshairs, deadzone, Zone of Interest, and telemetry overlay on camera frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Dim outside Zone of Interest (ROI) if configured
    if roi is not None:
        rx1 = max(0, min(w, int(round(roi[0] * w))))
        ry1 = max(0, min(h, int(round(roi[1] * h))))
        rx2 = max(0, min(w, int(round(roi[2] * w))))
        ry2 = max(0, min(h, int(round(roi[3] * h))))
        if rx2 > rx1 and ry2 > ry1:
            # Dim the outside of the Zone of Interest by 70%
            roi_patch = out[ry1:ry2, rx1:rx2].copy()
            out = (out.astype(np.float32) * 0.30).astype(np.uint8)
            out[ry1:ry2, rx1:rx2] = roi_patch

            # Dual-stroke high-contrast border (outer black shadow 8px, inner Neon Yellow 3px)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 0, 0), 8)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)

            # Neon Cyan corner brackets (BGR: 255, 255, 0, thickness 6)
            cyan_bgr = (255, 255, 0)
            bracket_len = max(18, int(min(rx2 - rx1, ry2 - ry1) * 0.15))
            # NW
            cv2.line(out, (rx1, ry1), (rx1 + bracket_len, ry1), cyan_bgr, 6)
            cv2.line(out, (rx1, ry1), (rx1, ry1 + bracket_len), cyan_bgr, 6)
            # NE
            cv2.line(out, (rx2, ry1), (rx2 - bracket_len, ry1), cyan_bgr, 6)
            cv2.line(out, (rx2, ry1), (rx2, ry1 + bracket_len), cyan_bgr, 6)
            # SW
            cv2.line(out, (rx1, ry2), (rx1 + bracket_len, ry2), cyan_bgr, 6)
            cv2.line(out, (rx1, ry2), (rx1, ry2 - bracket_len), cyan_bgr, 6)
            # SE
            cv2.line(out, (rx2, ry2), (rx2 - bracket_len, ry2), cyan_bgr, 6)
            cv2.line(out, (rx2, ry2), (rx2, ry2 - bracket_len), cyan_bgr, 6)

            # High-contrast Neon Yellow badge with dark black text
            label_text = "ZONE OF INTEREST (ACTIVE)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            font_thickness = 2
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            badge_y2 = min(h - 4, ry1 + th + 12)
            badge_y1 = max(4, badge_y2 - th - 12)
            badge_x1 = max(4, rx1 + 6)
            badge_x2 = min(w - 4, badge_x1 + tw + 14)
            cv2.rectangle(out, (badge_x1 - 1, badge_y1 - 1), (badge_x2 + 1, badge_y2 + 1), (0, 0, 0), -1)
            cv2.rectangle(out, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 255, 255), -1)
            cv2.putText(
                out,
                label_text,
                (badge_x1 + 7, badge_y2 - 6),
                font,
                font_scale,
                (10, 10, 10),
                font_thickness,
                cv2.LINE_AA,
            )

    if show_bbox:
        for det in detections:
            x1, y1 = int(det.xmin * w), int(det.ymin * h)
            x2, y2 = int(det.xmax * w), int(det.ymax * h)
            is_primary = primary is not None and det.track_id == primary.track_id

            color = (67, 185, 46) if is_primary else (250, 150, 50)
            thickness = 3 if is_primary else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            label_text = f"#{det.track_id} {det.display_name}"
            if is_primary:
                label_text = f"[LOCKED] {label_text}"
                ax, ay = int(det.aim_x * w), int(det.aim_y * h)
                cv2.drawMarker(out, (ax, ay), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
                cv2.circle(out, (ax, ay), 5, (0, 255, 255), -1)
                cv2.line(out, (w // 2, h // 2), (ax, ay), color, 2)

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                out,
                label_text,
                (x1 + 4, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (15, 20, 15),
                2,
                cv2.LINE_AA,
            )

    hud = f"FPS: {fps:.1f} | Hailo: {infer_ms:.1f}ms | Zoom: {zoom:.1f}x | {fsm_label or f'Birds: {len(detections)}'}"
    (hud_w, _), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    cv2.rectangle(out, (10, 10), (min(w - 10, hud_w + 30), 44), (20, 26, 20), -1)
    cv2.putText(
        out,
        hud,
        (20, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (67, 215, 60),
        2,
        cv2.LINE_AA,
    )
    return out


@dataclass
class DashboardServer:
    """Thread-safe MJPEG streaming and REST API server for Bird Spotter."""

    host: str = "0.0.0.0"
    port: int = 8080
    on_control: Callable[[dict[str, Any]], None] | None = None
    on_snapshot_requested: Callable[[], dict[str, Any] | str | None] | None = None
    on_relabel_sighting: Callable[[int, str], bool] | None = None
    on_delete_sighting: Callable[[int], bool] | None = None
    get_catalog_cb: Callable[[], list[dict[str, str]]] | None = None

    _raw_frame: np.ndarray | None = field(default=None, init=False, repr=False)
    _jpeg_bytes: bytes = field(default=b"", init=False, repr=False)
    _clean_jpeg_bytes: bytes = field(default=b"", init=False, repr=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _cond: threading.Condition = field(
        default_factory=threading.Condition, init=False, repr=False
    )
    _httpd: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        (SNAPSHOTS_DIR / "thumbs").mkdir(parents=True, exist_ok=True)

    def update_frame_and_state(
        self,
        raw_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        state: dict[str, Any],
    ) -> None:
        ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return
        with self._cond:
            self._raw_frame = raw_bgr
            self._jpeg_bytes = buf.tobytes()
            self._state = state
            self._cond.notify_all()

    def get_raw_frame(self) -> np.ndarray | None:
        with self._cond:
            return self._raw_frame.copy() if self._raw_frame is not None else None

    def get_state(self) -> dict[str, Any]:
        with self._cond:
            return dict(self._state)

    def capture_snapshot(self) -> str | None:
        """Save a clean high quality JPEG snapshot of the current raw frame."""
        frame = self.get_raw_frame()
        if frame is None:
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bird_{ts}_manual.jpg"
        filepath = SNAPSHOTS_DIR / filename

        cv2.imwrite(str(filepath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return filename

    def start_background(self) -> None:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path

                if path in ("/", "/index.html"):
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/state":
                    self._send_json(dashboard.get_state())
                elif path == "/api/catalog":
                    cat = dashboard.get_catalog_cb() if dashboard.get_catalog_cb else []
                    self._send_json(cat)
                elif path.startswith("/snapshots/"):
                    rel_str = unquote(path[len("/snapshots/"):]).lstrip("/")
                    snap_path = (SNAPSHOTS_DIR / rel_str).resolve()
                    if (
                        snap_path.is_file()
                        and SNAPSHOTS_DIR.resolve() in snap_path.parents
                    ):
                        img_bytes = snap_path.read_bytes()
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(img_bytes)))
                        self.end_headers()
                        self.wfile.write(img_bytes)
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
                elif path == "/stream.mjpg":
                    qs = parse_qs(parsed.query)
                    want_clean = qs.get("clean", ["0"])[0] == "1"
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        last_sent = b""
                        while True:
                            with dashboard._cond:
                                dashboard._cond.wait(timeout=1.0)
                                if want_clean and dashboard._raw_frame is not None:
                                    ok, cbuf = cv2.imencode(
                                        ".jpg",
                                        dashboard._raw_frame,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                                    )
                                    jpg = cbuf.tobytes() if ok else dashboard._jpeg_bytes
                                else:
                                    jpg = dashboard._jpeg_bytes
                            if not jpg or jpg is last_sent:
                                continue
                            last_sent = jpg
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    payload = {}

                if self.path == "/api/control":
                    if dashboard.on_control is not None:
                        dashboard.on_control(payload)
                    self._send_json(dashboard.get_state())
                elif self.path == "/api/snapshot":
                    res = (
                        dashboard.on_snapshot_requested()
                        if dashboard.on_snapshot_requested is not None
                        else dashboard.capture_snapshot()
                    )
                    if isinstance(res, dict):
                        self._send_json(res)
                    else:
                        self._send_json({"success": res is not None, "filename": res})
                elif self.path == "/api/sighting/relabel":
                    sid = int(payload.get("id", 0))
                    new_sp = str(payload.get("species", "")).strip()
                    ok = (
                        dashboard.on_relabel_sighting(sid, new_sp)
                        if (dashboard.on_relabel_sighting and sid and new_sp)
                        else False
                    )
                    self._send_json({"success": ok})
                elif self.path == "/api/sighting/delete":
                    sid = int(payload.get("id", 0))
                    ok = (
                        dashboard.on_delete_sighting(sid)
                        if (dashboard.on_delete_sighting and sid)
                        else False
                    )
                    self._send_json({"success": ok})
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
