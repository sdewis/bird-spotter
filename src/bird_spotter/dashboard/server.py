"""Web Dashboard and MJPEG Streaming Server for Bird Spotter.

Provides a responsive web interface at http://<pi-ip>:8080 with:
- Live MJPEG video feed with Hailo-8 bounding box overlays, Zone of Interest (ROI), and HUD telemetry
- Camera preview pop-out and full-screen controls
- Night mode handling with "Night mode \nCamera Offline" display over black background and greyed-out camera controls
- Bird Dex life-list with per-species deletion trash cans
- Recent captures gallery with multi-card selection and batch classification override
- Deep-zoom capture modal supporting magnifying glass cursor, click zoom, scroll wheel zoom, and mobile pinch
- PTZ gimbal jog D-pad, FOV, motorized focus, optical/digital zoom slider, and Home preset save
- Zone of Interest interactive click-and-drag selector on the video canvas
- Hourly and 14-day sighting analytics charts
- Collapsible real-time execution log console
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from bird_spotter.logger import spotter_log
from bird_spotter.vision.detector import Detection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"

POPOUT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>British Bird Spotter — Live Pop-out Feed</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' fill='none'%3E%3Cdefs%3E%3ClinearGradient id='btBlue' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%2358a6ff'/%3E%3Cstop offset='100%25' stop-color='%231f6feb'/%3E%3C/linearGradient%3E%3ClinearGradient id='btYellow' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23ffd33d'/%3E%3Cstop offset='100%25' stop-color='%23d29922'/%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='32' cy='32' r='28' stroke='%2358a6ff' stroke-width='2'/%3E%3Cpath d='M12 52 Q32 50 52 53' stroke='%2330363d' stroke-width='3' stroke-linecap='round'/%3E%3Cpath d='M28 47 L27 51 M31 47 L31 51 M36 48 L36 52 M39 48 L40 52' stroke='%238b949e' stroke-width='1.8' stroke-linecap='round'/%3E%3Cpath d='M16 43 L8 53 L13 55 L21 46 Z' fill='url(%23btBlue)'/%3E%3Cpath d='M28 29 C38 29 45 33 44 44 C43 50 34 51 27 46 C24 42 24 35 28 29 Z' fill='url(%23btYellow)'/%3E%3Cpath d='M27 30 C25 39 19 46 13 44 C12 39 18 30 27 30 Z' fill='url(%23btBlue)'/%3E%3Cpath d='M18 38 L25 35' stroke='%23f0f6fc' stroke-width='2' stroke-linecap='round'/%3E%3Cpath d='M25 19 C35 21 41 20 44 17 C47 22 45 28 39 31 C32 31 25 27 25 19 Z' fill='%23f0f6fc'/%3E%3Cpath d='M28 21 Q37 23 44 20 L43 23 Q36 24 28 22 Z' fill='%23161b22'/%3E%3Cpath d='M37 29 C35 32 31 32 29 30 Z' fill='%23161b22'/%3E%3Cpath d='M25 17 C31 11 39 12 44 17 C40 20 33 20 25 17 Z' fill='url(%23btBlue)'/%3E%3Cpolygon points='44,19 52,22 44,25' fill='%2330363d'/%3Ccircle cx='36' cy='21.5' r='1.8' fill='%23161b22'/%3Ccircle cx='36.5' cy='21' r='0.6' fill='%23ffffff'/%3E%3C/svg%3E">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #000000;
    color: #f0f6fc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  .spotter-icon { display: inline-block; vertical-align: middle; overflow: visible; }
  .spotter-brand-icon { width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .popout-bar {
    background: #161b22;
    padding: 8px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #30363d;
    z-index: 10;
  }
  .popout-title {
    font-size: 0.88rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .popout-actions {
    display: flex;
    gap: 8px;
  }
  button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    cursor: pointer;
  }
  button:hover { background: #30363d; color: #fff; }
  .video-stage {
    flex: 1;
    position: relative;
    background: #000000;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
  }
  .night-msg {
    display: none;
    position: absolute;
    inset: 0;
    background: #000000;
    color: #f0f6fc;
    font-size: 1.6rem;
    font-weight: 600;
    align-items: center;
    justify-content: center;
    text-align: center;
    white-space: pre-line;
    line-height: 1.4;
    z-index: 20;
  }
</style>
</head>
<body>
  <div class="popout-bar">
    <div class="popout-title">
      <span class="spotter-brand-icon"><svg class="spotter-icon bluetit-icon" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="popBtBlue" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#58a6ff"/><stop offset="100%" stop-color="#1f6feb"/></linearGradient><linearGradient id="popBtYellow" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#ffd33d"/><stop offset="100%" stop-color="#d29922"/></linearGradient></defs><circle class="bt-aura" cx="32" cy="32" r="28" stroke="#58a6ff" stroke-width="2"/><path d="M12 52 Q32 50 52 53" stroke="#30363d" stroke-width="3" stroke-linecap="round"/><path d="M28 47 L27 51 M31 47 L31 51 M36 48 L36 52 M39 48 L40 52" stroke="#8b949e" stroke-width="1.8" stroke-linecap="round"/><path d="M16 43 L8 53 L13 55 L21 46 Z" fill="url(#popBtBlue)"/><path d="M28 29 C38 29 45 33 44 44 C43 50 34 51 27 46 C24 42 24 35 28 29 Z" fill="url(#popBtYellow)"/><path d="M27 30 C25 39 19 46 13 44 C12 39 18 30 27 30 Z" fill="url(#popBtBlue)"/><path d="M18 38 L25 35" stroke="#f0f6fc" stroke-width="2" stroke-linecap="round"/><path d="M25 19 C35 21 41 20 44 17 C47 22 45 28 39 31 C32 31 25 27 25 19 Z" fill="#f0f6fc"/><path d="M28 21 Q37 23 44 20 L43 23 Q36 24 28 22 Z" fill="#161b22"/><path d="M37 29 C35 32 31 32 29 30 Z" fill="#161b22"/><path d="M25 17 C31 11 39 12 44 17 C40 20 33 20 25 17 Z" fill="url(#popBtBlue)"/><polygon points="44,19 52,22 44,25" fill="#30363d"/><circle cx="36" cy="21.5" r="1.8" fill="#161b22"/><circle cx="36.5" cy="21" r="0.6" fill="#ffffff"/></svg></span>
      <span>British Bird Spotter</span>
      <span id="popoutBadge" style="background:#238636; color:#fff; font-size:0.7rem; font-weight:700; padding:2px 6px; border-radius:4px;">LIVE</span>
    </div>
    <div class="popout-actions">
      <button onclick="toggleCleanStream()" id="btnClean">Clean: OFF</button>
      <button onclick="toggleFullscreen()">⛶ Fullscreen</button>
      <button onclick="window.close()">✕ Close</button>
    </div>
  </div>
  <div class="video-stage" id="stage">
    <img id="popoutImg" src="/stream.mjpg" alt="Live Feed">
    <div id="popoutNight" class="night-msg">Night mode &#10;Camera Offline</div>
  </div>
<script>
  let clean = false;
  function toggleCleanStream() {
    clean = !clean;
    document.getElementById('btnClean').textContent = 'Clean: ' + (clean ? 'ON' : 'OFF');
    document.getElementById('popoutImg').src = clean ? '/stream.mjpg?clean=1' : '/stream.mjpg';
  }
  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(()=>{});
    } else {
      document.exitFullscreen().catch(()=>{});
    }
  }
  setInterval(async () => {
    try {
      const res = await fetch('/api/state');
      if (res.ok) {
        const st = await res.json();
        const nightEl = document.getElementById('popoutNight');
        const imgEl = document.getElementById('popoutImg');
        const isNight = !!st.is_night_mode;
        if (isNight) {
          nightEl.style.display = 'flex';
          imgEl.style.display = 'none';
        } else {
          nightEl.style.display = 'none';
          imgEl.style.display = 'block';
        }
      }
    } catch(e){}
  }, 900);
</script>
</body>
</html>
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>British Bird Spotter — Live Dashboard</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' fill='none'%3E%3Cdefs%3E%3ClinearGradient id='btBlue' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%2358a6ff'/%3E%3Cstop offset='100%25' stop-color='%231f6feb'/%3E%3C/linearGradient%3E%3ClinearGradient id='btYellow' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' stop-color='%23ffd33d'/%3E%3Cstop offset='100%25' stop-color='%23d29922'/%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='32' cy='32' r='28' stroke='%2358a6ff' stroke-width='2'/%3E%3Cpath d='M12 52 Q32 50 52 53' stroke='%2330363d' stroke-width='3' stroke-linecap='round'/%3E%3Cpath d='M28 47 L27 51 M31 47 L31 51 M36 48 L36 52 M39 48 L40 52' stroke='%238b949e' stroke-width='1.8' stroke-linecap='round'/%3E%3Cpath d='M16 43 L8 53 L13 55 L21 46 Z' fill='url(%23btBlue)'/%3E%3Cpath d='M28 29 C38 29 45 33 44 44 C43 50 34 51 27 46 C24 42 24 35 28 29 Z' fill='url(%23btYellow)'/%3E%3Cpath d='M27 30 C25 39 19 46 13 44 C12 39 18 30 27 30 Z' fill='url(%23btBlue)'/%3E%3Cpath d='M18 38 L25 35' stroke='%23f0f6fc' stroke-width='2' stroke-linecap='round'/%3E%3Cpath d='M25 19 C35 21 41 20 44 17 C47 22 45 28 39 31 C32 31 25 27 25 19 Z' fill='%23f0f6fc'/%3E%3Cpath d='M28 21 Q37 23 44 20 L43 23 Q36 24 28 22 Z' fill='%23161b22'/%3E%3Cpath d='M37 29 C35 32 31 32 29 30 Z' fill='%23161b22'/%3E%3Cpath d='M25 17 C31 11 39 12 44 17 C40 20 33 20 25 17 Z' fill='url(%23btBlue)'/%3E%3Cpolygon points='44,19 52,22 44,25' fill='%2330363d'/%3Ccircle cx='36' cy='21.5' r='1.8' fill='%23161b22'/%3Ccircle cx='36.5' cy='21' r='0.6' fill='%23ffffff'/%3E%3C/svg%3E">
<style>
  :root {
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-bright: #f0f6fc;
    --muted: #8b949e;
    --green-bright: #3fb950;
    --accent: #238636;
    --accent-hover: #2ea043;
    --danger: #da3633;
    --warn: #d29922;
    --cyan: #58a6ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.45;
    padding-bottom: 24px;
  }
  header {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 10px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 50;
  }
  header h1 {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-bright);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .header-actions {
    display: flex;
    gap: 10px;
    align-items: center;
  }
  .badge {
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }
  .badge-active { background: rgba(46, 160, 67, 0.2); color: var(--green-bright); border: 1px solid rgba(46, 160, 67, 0.4); }
  .badge-idle { background: rgba(139, 148, 158, 0.15); color: var(--muted); border: 1px solid var(--border); }
  .badge-warn { background: rgba(210, 153, 34, 0.2); color: var(--warn); border: 1px solid rgba(210, 153, 34, 0.4); }
  .badge-night { background: rgba(88, 166, 255, 0.2); color: var(--cyan); border: 1px solid rgba(88, 166, 255, 0.4); }

  .power-btn {
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    cursor: pointer;
    border: 1px solid var(--border);
    transition: all 0.15s ease;
  }
  .power-on { background: #238636; color: #fff; border-color: #2ea043; }
  .power-off { background: #21262d; color: var(--muted); }

  main {
    max-width: 1440px;
    margin: 14px auto 0 auto;
    padding: 0 16px;
    display: grid;
    grid-template-columns: 1.45fr 1fr;
    gap: 16px;
  }
  @media (max-width: 1080px) {
    main { grid-template-columns: 1fr; }
  }

  .left-pane, .right-pane {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .card-header {
    background: #1c2128;
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text-bright);
  }
  .card-body {
    padding: 14px;
  }

  /* VIDEO PREVIEW AREA */
  .video-container {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: #000000;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .video-container img#streamImg {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .video-off {
    display: none;
    position: absolute;
    inset: 0;
    background: #000000;
    color: var(--muted);
    font-size: 1.1rem;
    font-weight: 600;
    align-items: center;
    justify-content: center;
    z-index: 10;
  }

  /* NIGHT MODE OVERLAY: Requirement 1 */
  .night-mode-overlay {
    position: absolute;
    inset: 0;
    background: #000000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 25;
    color: #f0f6fc;
    text-align: center;
    padding: 20px;
  }
  .night-mode-text {
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.45;
    letter-spacing: 0.02em;
    white-space: pre-line;
    color: #f0f6fc;
  }

  .override-banner {
    display: none;
    background: #382404;
    border-bottom: 1px solid #7a4706;
    color: #e3b341;
    padding: 6px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    justify-content: space-between;
    align-items: center;
  }

  /* ROI EDITOR OVERLAY */
  #roiEditorOverlay {
    position: absolute;
    inset: 0;
    display: none;
    cursor: crosshair;
    z-index: 15;
    user-select: none;
  }
  #roiSelectionBox {
    position: absolute;
    border: 2px dashed #ffff00;
    background: rgba(255, 255, 0, 0.12);
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.65);
    display: none;
  }
  .roi-handle {
    position: absolute;
    width: 12px;
    height: 12px;
    background: #ffff00;
    border: 1px solid #000;
    border-radius: 2px;
  }
  .roi-handle-nw { top: -6px; left: -6px; cursor: nwse-resize; }
  .roi-handle-ne { top: -6px; right: -6px; cursor: nesw-resize; }
  .roi-handle-sw { bottom: -6px; left: -6px; cursor: nesw-resize; }
  .roi-handle-se { bottom: -6px; right: -6px; cursor: nwse-resize; }

  /* BUTTONS & CONTROLS */
  button {
    background: #21262d;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s, color 0.15s;
    user-select: none;
  }
  button:hover {
    background: #30363d;
    color: var(--text-bright);
    border-color: #8b949e;
  }
  button.active {
    background: rgba(56, 139, 253, 0.2);
    color: #58a6ff;
    border-color: #388bfd;
  }
  .btn-primary {
    background: var(--accent);
    color: #fff;
    border-color: #2ea043;
  }
  .btn-primary:hover {
    background: var(--accent-hover);
    border-color: #3fb950;
  }
  .btn-danger {
    background: rgba(248, 81, 73, 0.15);
    color: #f85149;
    border-color: rgba(248, 81, 73, 0.35);
  }
  .btn-danger:hover {
    background: #da3633;
    color: #fff;
    border-color: #f85149;
  }
  .btn-subtle {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 3px 8px;
    font-size: 0.76rem;
  }
  .btn-subtle:hover {
    background: #21262d;
    color: var(--text-bright);
  }

  /* GREYED OUT CAMERA CONTROLS IN NIGHT MODE: Requirement 1 */
  .controls-disabled {
    opacity: 0.35 !important;
    pointer-events: none !important;
    filter: grayscale(0.9) !important;
    user-select: none !important;
  }
  button:disabled, select:disabled, input:disabled {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
  }

  /* STATS TELEMETRY */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  @media (max-width: 720px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
  }
  .stat-box {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
  }
  .stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .stat-val {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-bright);
    margin-top: 3px;
  }

  /* PTZ DPAD */
  .dpad {
    display: grid;
    grid-template-columns: repeat(3, 44px);
    grid-template-rows: repeat(3, 38px);
    gap: 6px;
    justify-content: center;
    margin: 10px auto;
  }
  .dpad button {
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
  }

  .zoom-bar {
    display: flex;
    gap: 6px;
    margin-top: 8px;
  }
  .zoom-bar button { flex: 1; }

  /* TABS */
  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
    background: #1c2128;
  }
  .tab-btn {
    flex: 1;
    border-radius: 0;
    border: none;
    border-bottom: 2px solid transparent;
    background: transparent;
    padding: 9px 12px;
    font-size: 0.84rem;
    color: var(--muted);
  }
  .tab-btn:hover { background: #21262d; color: var(--text-bright); }
  .tab-btn.active {
    color: var(--green-bright);
    border-bottom-color: var(--green-bright);
    background: #161b22;
  }
  .tab-panel {
    display: none;
    padding: 12px;
  }
  .tab-panel.active { display: block; }

  /* BIRD DEX CARDS & TRASH CAN: Requirement 3 */
  .dex-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 10px;
    max-height: 480px;
    overflow-y: auto;
  }
  .dex-card {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    position: relative;
    transition: transform 0.15s, border-color 0.15s;
  }
  .dex-card:hover {
    border-color: #58a6ff;
    transform: translateY(-2px);
  }
  .dex-card img {
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    background: #000;
    cursor: pointer;
  }
  .dex-info {
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
  }
  .dex-title {
    font-size: 0.84rem;
    font-weight: 700;
    color: var(--text-bright);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 4px;
  }
  .dex-sci {
    font-size: 0.72rem;
    color: var(--muted);
    font-style: italic;
  }
  .dex-meta {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 4px;
  }
  .species-count-badge {
    background: rgba(46, 160, 67, 0.2);
    color: var(--green-bright);
    border: 1px solid rgba(46, 160, 67, 0.4);
    font-size: 0.68rem;
    padding: 1px 6px;
    border-radius: 999px;
    font-weight: 700;
  }
  .dex-trash-btn {
    background: rgba(248, 81, 73, 0.15);
    border: 1px solid rgba(248, 81, 73, 0.35);
    color: #f85149;
    border-radius: 4px;
    padding: 2px 5px;
    font-size: 0.76rem;
    cursor: pointer;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .dex-trash-btn:hover {
    background: #da3633;
    color: #fff;
    border-color: #f85149;
  }

  /* CAPTURES GALLERY & BATCH RELABEL TOOLBAR: Requirement 4 */
  .batch-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    background: #11151c;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 10px;
  }
  .snapshot-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 8px;
    max-height: 480px;
    overflow-y: auto;
  }
  .snap-card {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    position: relative;
    transition: transform 0.15s, border-color 0.15s;
  }
  .snap-card:hover {
    border-color: var(--cyan);
    transform: translateY(-2px);
  }
  .snap-card.selected {
    border-color: #58a6ff;
    box-shadow: 0 0 0 2px #58a6ff;
  }
  .snap-card img {
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    display: block;
    background: #000;
  }
  .snap-select-cb {
    position: absolute;
    top: 6px;
    left: 6px;
    width: 18px;
    height: 18px;
    accent-color: #238636;
    cursor: pointer;
    z-index: 5;
  }
  .snap-card-caption {
    padding: 4px 6px;
    font-size: 0.72rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #161b22;
  }

  /* ANALYTICS CHARTS */
  .chart-bars {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 100px;
    padding: 8px 0;
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
    background: var(--green-bright);
    border-radius: 2px 2px 0 0;
    min-height: 2px;
  }
  .chart-lbl {
    font-size: 0.6rem;
    color: var(--muted);
    margin-top: 4px;
    height: 12px;
  }

  /* DETAILED CAPTURE POPUP MODAL & DEEP ZOOM: Requirement 5 */
  .modal-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.85);
    z-index: 100;
    align-items: center;
    justify-content: center;
    padding: 16px;
  }
  .modal-backdrop.open { display: flex; }
  .modal-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    max-width: 900px;
    width: 100%;
    max-height: 94vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .modal-img-frame {
    position: relative;
    width: 100%;
    height: 54vh;
    min-height: 280px;
    background: #000000;
    overflow: hidden; /* strictly within image frame area */
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: zoom-in; /* magnifying glass */
    user-select: none;
    touch-action: none;
  }
  .modal-img-frame.zoomed-in {
    cursor: zoom-out; /* magnifying glass when zoomed */
  }
  .modal-zoom-img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
    transform-origin: center center;
    transition: transform 0.08s ease-out;
    pointer-events: auto;
    cursor: inherit;
  }
  .zoom-indicator-pill {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(13, 17, 23, 0.8);
    color: #58a6ff;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    pointer-events: none;
    z-index: 10;
  }

  .status-toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #161b22;
    border: 1px solid var(--green-bright);
    color: var(--text-bright);
    padding: 8px 16px;
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    font-size: 0.82rem;
    display: none;
    z-index: 200;
  }

  /* LIVE LOGGING PANE */
  .log-terminal {
    background: #080b0f;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 0.74rem;
    height: 220px;
    overflow-y: auto;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    line-height: 1.4;
  }
  .log-line {
    display: flex;
    gap: 8px;
    word-break: break-all;
    white-space: pre-wrap;
  }
  .log-ts { color: var(--muted); flex-shrink: 0; }
  .log-lvl { font-weight: 700; flex-shrink: 0; width: 44px; text-align: center; border-radius: 3px; padding: 0 2px; }
  .log-lvl-INFO { color: var(--green-bright); background: rgba(46, 160, 67, 0.15); }
  .log-lvl-DEBUG { color: #58a6ff; background: rgba(56, 139, 253, 0.15); }
  .log-lvl-WARN { color: #e3b341; background: rgba(210, 153, 34, 0.15); }
  .log-lvl-ERROR { color: #f85149; background: rgba(248, 81, 73, 0.15); }
  .log-cat { color: #d2a8ff; font-weight: 600; flex-shrink: 0; min-width: 75px; }
  .log-msg { color: var(--text); }

  /* Style 3: Blue Tit Vector Icon & Reactive Badge */
  .spotter-icon {
    display: inline-block;
    vertical-align: middle;
    overflow: visible;
  }
  .spotter-brand-icon {
    width: 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .badge-icon-slot {
    width: 17px;
    height: 17px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .spotter-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.25s ease;
  }
  .bluetit-icon .bt-aura {
    opacity: 0;
    transform-origin: 32px 32px;
    transition: opacity 0.3s ease;
  }
  .badge-detected {
    background: rgba(46, 160, 67, 0.25) !important;
    border-color: var(--green-bright) !important;
    color: var(--green-bright) !important;
    box-shadow: 0 0 10px rgba(63, 185, 80, 0.35);
  }
  .badge-detected .bluetit-icon .bt-aura {
    opacity: 0.8;
    animation: btAuraPulse 1.8s ease-in-out infinite;
  }
  .badge-detected .bluetit-icon {
    animation: btCuteHop 2s ease-in-out infinite;
    transform-origin: 32px 50px;
  }
  @keyframes btAuraPulse {
    0%, 100% { opacity: 0.15; transform: scale(0.95); }
    50% { opacity: 0.65; transform: scale(1.08); }
  }
  @keyframes btCuteHop {
    0%, 80%, 100% { transform: translateY(0); }
    88% { transform: translateY(-3px) rotate(-1deg); }
    94% { transform: translateY(0) rotate(1deg); }
  }
</style>
</head>
<body>
<header>
  <h1>
    <span class="spotter-brand-icon"><svg class="spotter-icon bluetit-icon" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="btBlue" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#58a6ff"/><stop offset="100%" stop-color="#1f6feb"/></linearGradient><linearGradient id="btYellow" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#ffd33d"/><stop offset="100%" stop-color="#d29922"/></linearGradient></defs><circle class="bt-aura" cx="32" cy="32" r="28" stroke="#58a6ff" stroke-width="2"/><path d="M12 52 Q32 50 52 53" stroke="#30363d" stroke-width="3" stroke-linecap="round"/><path d="M28 47 L27 51 M31 47 L31 51 M36 48 L36 52 M39 48 L40 52" stroke="#8b949e" stroke-width="1.8" stroke-linecap="round"/><path d="M16 43 L8 53 L13 55 L21 46 Z" fill="url(#btBlue)"/><path d="M28 29 C38 29 45 33 44 44 C43 50 34 51 27 46 C24 42 24 35 28 29 Z" fill="url(#btYellow)"/><path d="M27 30 C25 39 19 46 13 44 C12 39 18 30 27 30 Z" fill="url(#btBlue)"/><path d="M18 38 L25 35" stroke="#f0f6fc" stroke-width="2" stroke-linecap="round"/><path d="M25 19 C35 21 41 20 44 17 C47 22 45 28 39 31 C32 31 25 27 25 19 Z" fill="#f0f6fc"/><path d="M28 21 Q37 23 44 20 L43 23 Q36 24 28 22 Z" fill="#161b22"/><path d="M37 29 C35 32 31 32 29 30 Z" fill="#161b22"/><path d="M25 17 C31 11 39 12 44 17 C40 20 33 20 25 17 Z" fill="url(#btBlue)"/><polygon points="44,19 52,22 44,25" fill="#30363d"/><circle cx="36" cy="21.5" r="1.8" fill="#161b22"/><circle cx="36.5" cy="21" r="0.6" fill="#ffffff"/></svg></span>
    <span>British Bird Spotter</span>
  </h1>
  <div class="header-actions">
    <button id="btnToggleLogs" class="badge badge-active" style="cursor:pointer;" onclick="toggleLogsPane()" title="Toggle Live Logging Pane">
      📜 Live Logs
    </button>
    <div id="fsmBadge" class="badge badge-active">WATCHING (HOME 1.0x)</div>
    <div id="targetBadge" class="badge badge-idle spotter-badge" title="Live Bird Detection Target">
      <span class="badge-icon-slot"><svg class="spotter-icon bluetit-icon" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle class="bt-aura" cx="32" cy="32" r="28" stroke="#58a6ff" stroke-width="2"/><path d="M12 52 Q32 50 52 53" stroke="#30363d" stroke-width="3" stroke-linecap="round"/><path d="M28 47 L27 51 M31 47 L31 51 M36 48 L36 52 M39 48 L40 52" stroke="#8b949e" stroke-width="1.8" stroke-linecap="round"/><path d="M16 43 L8 53 L13 55 L21 46 Z" fill="url(#btBlue)"/><path d="M28 29 C38 29 45 33 44 44 C43 50 34 51 27 46 C24 42 24 35 28 29 Z" fill="url(#btYellow)"/><path d="M27 30 C25 39 19 46 13 44 C12 39 18 30 27 30 Z" fill="url(#btBlue)"/><path d="M18 38 L25 35" stroke="#f0f6fc" stroke-width="2" stroke-linecap="round"/><path d="M25 19 C35 21 41 20 44 17 C47 22 45 28 39 31 C32 31 25 27 25 19 Z" fill="#f0f6fc"/><path d="M28 21 Q37 23 44 20 L43 23 Q36 24 28 22 Z" fill="#161b22"/><path d="M37 29 C35 32 31 32 29 30 Z" fill="#161b22"/><path d="M25 17 C31 11 39 12 44 17 C40 20 33 20 25 17 Z" fill="url(#btBlue)"/><polygon points="44,19 52,22 44,25" fill="#30363d"/><circle cx="36" cy="21.5" r="1.8" fill="#161b22"/><circle cx="36.5" cy="21" r="0.6" fill="#ffffff"/></svg></span>
      <span id="targetBadgeText">TARGET: NONE</span>
    </div>
    <button id="btnSolar" class="badge badge-active" style="cursor:pointer;" onclick="sendCtrl({action:'toggle_solar_schedule'})" title="Automatic Sunset/Dawn Scheduler">
      🌅 Auto-Sleep: ON
    </button>
    <button id="pwrBtn" class="power-btn power-on" onclick="togglePowerOrDayNight()">
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
        <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
          <!-- Requirement 2: Pop-out and Full Screen buttons -->
          <button id="btnPopout" class="cam-ctrl" onclick="openPopout()" title="Pop out camera stream in a separate window">↗ Pop-out</button>
          <button id="btnFullscreen" class="cam-ctrl" onclick="toggleFullscreen()" title="Toggle full screen preview">⛶ Fullscreen</button>
          <button id="btnManualSnap" class="btn-primary cam-ctrl" onclick="takeSnapshot()">📸 Manual Snapshot</button>
          <button id="btnEditRoi" class="cam-ctrl" onclick="toggleRoiEditor()" title="Configure Zone of Interest on feeder">🎯 Select Zone</button>
          <label id="lblLockAspect" class="cam-ctrl" style="display:none; align-items:center; gap:5px; font-size:0.75rem; color:var(--text); cursor:pointer; background:#21262d; padding:4px 8px; border-radius:6px; border:1px solid var(--border);"><input type="checkbox" id="chkLockAspect"> 🔒 16:9</label>
          <button id="btnApplyRoi" class="btn-primary cam-ctrl" style="display:none;" onclick="applyRoiSelection()" title="Save Zone of Interest">✅ Set Zone</button>
          <button id="btnClearRoi" class="btn-danger cam-ctrl" style="display:none;" onclick="clearRoiSelection()" title="Disable Zone of Interest">❌ Unset Zone</button>
          <button id="btnPreviewBbox" class="active cam-ctrl" onclick="sendCtrl({action:'toggle_preview_bbox'})" title="Toggle bounding boxes on live camera preview">📦 Preview Box: ON</button>
          <button id="btnCaptureBbox" class="active cam-ctrl" onclick="sendCtrl({action:'toggle_capture_bbox'})" title="Toggle bounding boxes on saved bird snapshots">🖼️ Capture Box: ON</button>
          <button id="btnHudMode" class="active cam-ctrl" onclick="toggleStreamHud()">HUD: ON</button>
          <button id="btnToggleCam" class="active cam-ctrl" onclick="toggleCamPreview()">Stream: ON</button>
        </div>
      </div>
      <div id="overrideBanner" class="override-banner">
        <span id="overrideText">Manual Override Active — Auto-Framing paused</span>
        <button class="btn-primary" style="padding:4px 10px; font-size:0.76rem;" onclick="sendCtrl({action:'resume_auto_tracking'})">
          ▶ Resume Auto-Framing Now
        </button>
      </div>
      <div class="video-container" id="videoContainer">
        <img id="streamImg" src="/stream.mjpg" alt="Bird Spotter Live Stream">
        <div id="streamOffText" class="video-off">Camera stream paused</div>

        <!-- Requirement 1: Night mode message over black background -->
        <div id="nightOverlay" class="night-mode-overlay" style="display:none;">
          <div class="night-mode-text">Night mode &#10;Camera Offline</div>
          <button class="btn-primary" style="margin-top:16px; padding:7px 16px; font-size:0.88rem;" onclick="sendCtrl({action:'activate_day_mode'})">
            ☀️ Activate Day Mode
          </button>
        </div>

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
            <div class="stat-label">Frame Fill (10-35%)</div>
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

    <!-- LIVE LOGGING PANE (TOGGLEABLE) -->
    <div id="liveLogsCard" class="card" style="display:none;">
      <div class="card-header">
        <span>📜 Live Execution Logs</span>
        <div style="display:flex; align-items:center; gap:8px;">
          <label style="font-size:0.75rem; color:var(--muted); font-weight:600;">Level:</label>
          <select id="logLevelSelect" onchange="changeLogLevel(this.value)" style="padding:2px 8px; font-size:0.75rem;">
            <option value="INFO" selected>INFO</option>
            <option value="DEBUG">DEBUG</option>
          </select>
          <button id="btnAutoScroll" class="active" onclick="toggleLogAutoScroll()" style="padding:2px 8px; font-size:0.75rem;">Scroll: ON</button>
          <button onclick="clearLogPane()" style="padding:2px 8px; font-size:0.75rem;">Clear</button>
          <button onclick="toggleLogsPane()" style="padding:2px 8px; font-size:0.75rem;">✕ Hide</button>
        </div>
      </div>
      <div class="card-body" style="padding:10px;">
        <div id="logTerminal" class="log-terminal">
          <div style="color:var(--muted); font-size:0.72rem;">Connecting to live logs...</div>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT PANE: PTZ GIMBAL CONTROLS & LIFE LIST / GALLERY TABS -->
  <div class="right-pane">
    <div id="ptzCard" class="card">
      <div class="card-header">
        <span>🎯 OBSBOT PTZ Gimbal & Home Preset</span>
        <button id="btnUkFilter" class="active cam-ctrl" style="padding:4px 9px; font-size:0.75rem;" onclick="sendCtrl({action:'toggle_uk_filter'})">
          🇬🇧 UK Filter: ON
        </button>
      </div>
      <div class="card-body">
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px;">
          <button id="btnAutoTrack" class="active cam-ctrl" style="flex:1;" onclick="sendCtrl({action:'toggle_tracking'})">
            🤖 Auto-Framing: ON
          </button>
          <button class="cam-ctrl" onclick="sendCtrl({action:'go_home'})" title="Return to saved feeder/home position">🏠 Go Home</button>
          <button class="cam-ctrl" onclick="saveHomePreset()" title="Save current pan/tilt as the default detection view">📌 Set Home</button>
        </div>

        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.74rem; color:var(--muted); font-weight:600; text-transform:uppercase;">
          <span>Manual Jog D-Pad (Pauses Auto-Framing 60s)</span>
          <select id="nudgeStepSel" class="cam-ctrl" style="padding:2px 6px; font-size:0.74rem;">
            <option value="18000">Fine Step</option>
            <option value="36000" selected>Medium Step</option>
            <option value="72000">Large Step</option>
          </select>
        </div>
        <div class="dpad">
          <div></div>
          <button class="cam-ctrl" onclick="nudgePTZ(0, 1)">▲ Up</button>
          <div></div>
          <button class="cam-ctrl" onclick="nudgePTZ(1, 0)">◀ Left</button>
          <button class="cam-ctrl" onclick="sendCtrl({action:'center'})">🎯 0,0</button>
          <button class="cam-ctrl" onclick="nudgePTZ(-1, 0)">Right ▶</button>
          <div></div>
          <button class="cam-ctrl" onclick="nudgePTZ(0, -1)">▼ Down</button>
          <div></div>
        </div>

        <div style="margin-top:10px; font-size:0.74rem; color:var(--muted); font-weight:600; text-transform:uppercase; display:flex; justify-content:space-between;">
          <span>Hardware Field of View (OBSBOT ISP)</span>
          <span id="fovLbl">65° Narrow</span>
        </div>
        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:6px; margin-top:6px;">
          <button id="btnFov86" class="cam-ctrl" onclick="sendCtrl({action:'set_fov', fov:86})">86° Wide</button>
          <button id="btnFov78" class="cam-ctrl" onclick="sendCtrl({action:'set_fov', fov:78})">78° Medium</button>
          <button id="btnFov65" class="active cam-ctrl" onclick="sendCtrl({action:'set_fov', fov:65})">65° Narrow</button>
        </div>

        <div style="margin-top:10px; font-size:0.74rem; color:var(--muted); font-weight:600; text-transform:uppercase; display:flex; justify-content:space-between;">
          <span>Hardware Zoom Control (1.0x – 4.0x)</span>
          <span id="zoomSliderLbl">1.0x</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
          <button class="cam-ctrl" onclick="sendCtrl({action:'zoom_out', step:0.25})">🔍 -</button>
          <input id="zoomSlider" class="cam-ctrl" type="range" min="1.0" max="4.0" step="0.25" value="1.0" style="flex:1;" onchange="sendCtrl({action:'set_zoom', zoom:parseFloat(this.value)})">
          <button class="cam-ctrl" onclick="sendCtrl({action:'zoom_in', step:0.25})">🔍 +</button>
        </div>
        <div class="zoom-bar">
          <button class="cam-ctrl" onclick="sendCtrl({action:'set_zoom', zoom:1.0})">1.0x</button>
          <button class="cam-ctrl" onclick="sendCtrl({action:'set_zoom', zoom:1.5})">1.5x</button>
          <button class="cam-ctrl" onclick="sendCtrl({action:'set_zoom', zoom:2.0})">2.0x</button>
          <button class="cam-ctrl" onclick="sendCtrl({action:'set_zoom', zoom:3.0})">3.0x</button>
          <button class="cam-ctrl" onclick="sendCtrl({action:'set_zoom', zoom:4.0})">4.0x</button>
        </div>

        <div style="margin-top:10px; font-size:0.74rem; color:var(--muted); font-weight:600; text-transform:uppercase; display:flex; justify-content:space-between; align-items:center;">
          <span>Motorized Lens Focus (Window / Feeder)</span>
          <span id="focusLbl">Auto</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
          <button id="btnAutofocus" class="active cam-ctrl" style="padding:5px 10px; font-size:0.76rem;" onclick="sendCtrl({action:'toggle_autofocus'})">AF: ON</button>
          <input id="focusSlider" class="cam-ctrl" type="range" min="0" max="100" step="1" value="50" style="flex:1;" onchange="sendCtrl({action:'set_focus', focus:parseInt(this.value, 10)})">
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;">
          <button id="btnInvPan" class="cam-ctrl" onclick="sendCtrl({action:'toggle_invert_pan'})">Invert Pan: ON</button>
          <button id="btnInvTilt" class="cam-ctrl" onclick="sendCtrl({action:'toggle_invert_tilt'})">Invert Tilt: ON</button>
        </div>
      </div>
    </div>

    <!-- TABS: BIRD DEX / CAPTURES & REVIEW / ANALYTICS (Requirement 1: always available) -->
    <div class="card">
      <div class="tabs">
        <button id="tabBtnDex" class="tab-btn active" onclick="switchTab('dex')">🦉 Bird Dex (<span id="dexCountLbl">0</span>)</button>
        <button id="tabBtnSnaps" class="tab-btn" onclick="switchTab('snaps')">🖼️ Captures (<span id="snapCountLbl">0</span>)</button>
        <button id="tabBtnCharts" class="tab-btn" onclick="switchTab('charts')">📈 Analytics</button>
      </div>

      <!-- TAB 1: LIFE LIST / BIRD DEX (Requirement 3: trash can icon on each card) -->
      <div id="tabPanelDex" class="tab-panel active">
        <div id="dexGrid" class="dex-grid">
          <div style="grid-column: 1 / -1; color:var(--muted); text-align:center; padding:20px; font-size:0.86rem;">
            No bird species recorded yet — watching feeder...
          </div>
        </div>
      </div>

      <!-- TAB 2: RECENT CAPTURES & BATCH OVERRIDE (Requirement 4: multi-select & batch relabel) -->
      <div id="tabPanelSnaps" class="tab-panel">
        <div id="batchRelabelBar" class="batch-bar">
          <div style="display:flex; align-items:center; gap:8px;">
            <label style="display:flex; align-items:center; gap:4px; font-size:0.78rem; cursor:pointer; font-weight:600;">
              <input type="checkbox" id="selectAllCapturesCb" onchange="toggleSelectAllCaptures(this.checked)">
              <span>Select All</span>
            </label>
            <span id="selectedCapturesCount" style="font-size:0.78rem; color:var(--muted); font-weight:700;">0 selected</span>
            <button id="btnClearSelection" class="btn-subtle" onclick="clearCapturesSelection()" style="display:none;">Clear</button>
          </div>
          <div style="display:flex; align-items:center; gap:6px;">
            <select id="batchSpeciesSelect" style="padding:4px 8px; font-size:0.78rem; max-width:180px;">
              <option value="">-- Choose Species --</option>
            </select>
            <button id="btnBatchRelabel" class="btn-primary" onclick="batchRelabelSelected()" style="padding:4px 10px; font-size:0.78rem;">
              Override Species
            </button>
          </div>
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

<div id="statusToast" class="status-toast"></div>

<!-- DETAILED CAPTURE POPUP MODAL & DEEP ZOOM (Requirement 5) -->
<div id="reviewModal" class="modal-backdrop" onclick="if(event.target===this) closeModal()">
  <div class="modal-box">
    <div class="card-header">
      <span id="modalTitle">Sighting Inspection</span>
      <div style="display:flex; align-items:center; gap:8px;">
        <span id="modalZoomVal" style="font-size:0.75rem; color:var(--muted); font-weight:700;">1.0x</span>
        <button class="btn-subtle" onclick="resetModalZoom(event)">Reset Zoom</button>
        <button onclick="closeModal()" style="padding:4px 10px;">✕ Close</button>
      </div>
    </div>
    <div id="modalImgFrame" class="modal-img-frame" title="Click or scroll to zoom in/out">
      <div id="modalZoomPill" class="zoom-indicator-pill">1.0x</div>
      <img id="modalImg" src="" alt="Bird Snapshot" class="modal-zoom-img">
    </div>
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
let selectedCaptures = new Set();
let isNightModeActive = false;

let isEditingRoi = false;
let currentRoiState = null;
let roiDraft = { x: 0.25, y: 0.25, w: 0.50, h: 0.50 };

/* LOGS STATE */
let logsPaneOpen = false;
let logAutoScroll = true;
let currentLogLevel = "INFO";
let lastLogSeq = 0;
let logPollTimer = null;

/* Requirement 2: Pop-out and Fullscreen helpers */
function openPopout() {
  window.open('/popout', 'BirdSpotterPopout', 'width=960,height=580,menubar=no,toolbar=no,location=no,status=no,resizable=yes');
}

function toggleFullscreen() {
  const vContainer = document.getElementById('videoContainer');
  if (!vContainer) return;
  if (!document.fullscreenElement) {
    if (vContainer.requestFullscreen) {
      vContainer.requestFullscreen().catch(()=>{});
    } else if (vContainer.webkitRequestFullscreen) {
      vContainer.webkitRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen().catch(()=>{});
    }
  }
}

function initLogsState() {
  const savedOpen = localStorage.getItem('spotter_logs_open');
  if (savedOpen === 'true') {
    setLogsPaneVisible(true);
  }
  const savedLevel = localStorage.getItem('spotter_logs_level');
  if (savedLevel && (savedLevel === 'INFO' || savedLevel === 'DEBUG')) {
    currentLogLevel = savedLevel;
    const sel = document.getElementById('logLevelSelect');
    if (sel) sel.value = savedLevel;
  }
}

function toggleLogsPane() {
  setLogsPaneVisible(!logsPaneOpen);
}

function setLogsPaneVisible(visible) {
  logsPaneOpen = visible;
  localStorage.setItem('spotter_logs_open', visible ? 'true' : 'false');
  const card = document.getElementById('liveLogsCard');
  const btn = document.getElementById('btnToggleLogs');
  if (card) card.style.display = visible ? 'block' : 'none';
  if (btn) btn.classList.toggle('active', visible);

  if (visible) {
    pollLogs();
    if (!logPollTimer) {
      logPollTimer = setInterval(pollLogs, 600);
    }
  } else {
    if (logPollTimer) {
      clearInterval(logPollTimer);
      logPollTimer = null;
    }
  }
}

function toggleLogAutoScroll() {
  logAutoScroll = !logAutoScroll;
  const btn = document.getElementById('btnAutoScroll');
  if (btn) {
    btn.textContent = 'Scroll: ' + (logAutoScroll ? 'ON' : 'OFF');
    btn.classList.toggle('active', logAutoScroll);
  }
}

async function changeLogLevel(newLevel) {
  currentLogLevel = newLevel;
  localStorage.setItem('spotter_logs_level', newLevel);
  await sendCtrl({ action: 'set_log_level', level: newLevel });
  clearLogPane();
  pollLogs();
}

function clearLogPane() {
  const term = document.getElementById('logTerminal');
  if (term) term.innerHTML = '<div style="color:var(--muted); font-size:0.72rem;">Logs cleared</div>';
  lastLogSeq = 0;
}

async function pollLogs() {
  if (!logsPaneOpen) return;
  try {
    const res = await fetch(`/api/logs?since=${lastLogSeq}&level=${currentLogLevel}`);
    if (!res.ok) return;
    const data = await res.json();
    const entries = data.entries || [];
    if (entries.length === 0) return;

    const term = document.getElementById('logTerminal');
    if (!term) return;

    if (lastLogSeq === 0) {
      term.innerHTML = '';
    }

    const frag = document.createDocumentFragment();
    for (const e of entries) {
      lastLogSeq = Math.max(lastLogSeq, e.seq);
      const row = document.createElement('div');
      row.className = 'log-line';
      row.innerHTML =
        `<span class="log-ts">${escapeHtml(e.ts)}</span>` +
        `<span class="log-lvl log-lvl-${escapeHtml(e.level)}">${escapeHtml(e.level)}</span>` +
        `<span class="log-cat">[${escapeHtml(e.category)}]</span>` +
        `<span class="log-msg">${escapeHtml(e.msg)}</span>`;
      frag.appendChild(row);
    }
    term.appendChild(frag);

    while (term.childNodes.length > 350) {
      term.removeChild(term.firstChild);
    }

    if (logAutoScroll) {
      term.scrollTop = term.scrollHeight;
    }
  } catch (err) {}
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeJs(str) {
  if (!str) return '';
  return String(str).replace(/'/g, "\\'").replace(/"/g, '\\"');
}

function toggleRoiEditor() {
  if (isEditingRoi) {
    cancelRoiEditor();
  } else {
    startRoiEditor();
  }
}

function startRoiEditor() {
  isEditingRoi = true;
  document.getElementById('btnEditRoi').textContent = 'Cancel Zone';
  document.getElementById('btnEditRoi').classList.add('active');
  document.getElementById('btnApplyRoi').style.display = 'inline-block';
  document.getElementById('btnClearRoi').style.display = 'inline-block';
  document.getElementById('lblLockAspect').style.display = 'inline-flex';
  const ov = document.getElementById('roiEditorOverlay');
  ov.style.display = 'block';

  if (currentRoiState && currentRoiState.enabled) {
    roiDraft = {
      x: currentRoiState.norm_x,
      y: currentRoiState.norm_y,
      w: currentRoiState.norm_w,
      h: currentRoiState.norm_h,
    };
  } else {
    roiDraft = { x: 0.20, y: 0.20, w: 0.60, h: 0.60 };
  }
  renderRoiSelectionBox();
}

function cancelRoiEditor() {
  isEditingRoi = false;
  document.getElementById('btnEditRoi').textContent = (currentRoiState && currentRoiState.enabled) ? '🎯 Zone: ON' : '🎯 Select Zone';
  document.getElementById('btnEditRoi').classList.toggle('active', !!(currentRoiState && currentRoiState.enabled));
  document.getElementById('btnApplyRoi').style.display = 'none';
  document.getElementById('btnClearRoi').style.display = (currentRoiState && currentRoiState.enabled) ? 'inline-block' : 'none';
  document.getElementById('lblLockAspect').style.display = 'none';
  const ov = document.getElementById('roiEditorOverlay');
  ov.style.display = 'none';
  document.getElementById('roiSelectionBox').style.display = 'none';
}

function renderRoiSelectionBox() {
  const box = document.getElementById('roiSelectionBox');
  box.style.left = (roiDraft.x * 100).toFixed(2) + '%';
  box.style.top = (roiDraft.y * 100).toFixed(2) + '%';
  box.style.width = (roiDraft.w * 100).toFixed(2) + '%';
  box.style.height = (roiDraft.h * 100).toFixed(2) + '%';
  box.style.display = 'block';
}

function applyRoiSelection() {
  const lock = document.getElementById('chkLockAspect').checked;
  sendCtrl({
    action: 'set_roi',
    x: roiDraft.x,
    y: roiDraft.y,
    w: roiDraft.w,
    h: roiDraft.h,
    lock_aspect: lock
  });
  cancelRoiEditor();
  showToast('Zone of Interest saved');
}

function clearRoiSelection() {
  sendCtrl({ action: 'clear_roi' });
  cancelRoiEditor();
  showToast('Zone of Interest cleared');
}

/* ROI drag setup */
(function setupRoiInteractions() {
  const ov = document.getElementById('roiEditorOverlay');
  const box = document.getElementById('roiSelectionBox');
  let activeHandle = null;
  let isDraggingBox = false;
  let startX = 0, startY = 0;
  let startDraft = { ...roiDraft };

  ov.addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('roi-handle')) {
      activeHandle = e.target.dataset.handle;
      startX = e.clientX;
      startY = e.clientY;
      startDraft = { ...roiDraft };
      e.stopPropagation();
      return;
    }
    if (e.target === box || box.contains(e.target)) {
      isDraggingBox = true;
      startX = e.clientX;
      startY = e.clientY;
      startDraft = { ...roiDraft };
      e.stopPropagation();
      return;
    }
    const rect = ov.getBoundingClientRect();
    const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    roiDraft = { x: nx, y: ny, w: 0.05, h: 0.05 };
    activeHandle = 'se';
    startX = e.clientX;
    startY = e.clientY;
    startDraft = { ...roiDraft };
    renderRoiSelectionBox();
  });

  window.addEventListener('mousemove', (e) => {
    if (!isEditingRoi) return;
    const rect = ov.getBoundingClientRect();
    if (isDraggingBox) {
      const dx = (e.clientX - startX) / rect.width;
      const dy = (e.clientY - startY) / rect.height;
      roiDraft.x = Math.max(0, Math.min(1 - startDraft.w, startDraft.x + dx));
      roiDraft.y = Math.max(0, Math.min(1 - startDraft.h, startDraft.y + dy));
      renderRoiSelectionBox();
      return;
    }
    if (activeHandle) {
      const dx = (e.clientX - startX) / rect.width;
      const dy = (e.clientY - startY) / rect.height;
      let x1 = startDraft.x;
      let y1 = startDraft.y;
      let x2 = startDraft.x + startDraft.w;
      let y2 = startDraft.y + startDraft.h;

      if (activeHandle === 'nw') { x1 += dx; y1 += dy; }
      else if (activeHandle === 'ne') { x2 += dx; y1 += dy; }
      else if (activeHandle === 'sw') { x1 += dx; y2 += dy; }
      else if (activeHandle === 'se') { x2 += dx; y2 += dy; }

      const nx1 = Math.max(0, Math.min(x1, x2));
      const nx2 = Math.min(1, Math.max(x1, x2));
      const ny1 = Math.max(0, Math.min(y1, y2));
      const ny2 = Math.min(1, Math.max(y1, y2));
      roiDraft = {
        x: nx1,
        y: ny1,
        w: Math.max(0.04, nx2 - nx1),
        h: Math.max(0.04, ny2 - ny1)
      };
      renderRoiSelectionBox();
    }
  });

  window.addEventListener('mouseup', () => {
    activeHandle = null;
    isDraggingBox = false;
  });
})();

async function sendCtrl(payload) {
  try {
    const res = await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      const st = await res.json();
      updateUI(st);
    }
  } catch (err) {
    console.error('sendCtrl failed', err);
  }
}

function togglePowerOrDayNight() {
  if (isNightModeActive) {
    sendCtrl({ action: 'activate_day_mode' });
  } else {
    sendCtrl({ action: 'toggle_system_power' });
  }
}

function nudgePTZ(dp, dt) {
  const step = parseInt(document.getElementById('nudgeStepSel').value, 10) || 36000;
  sendCtrl({ action: 'nudge', d_pan: dp * step, d_tilt: dt * step });
}

function saveHomePreset() {
  sendCtrl({ action: 'save_home_preset' });
  showToast('Current PTZ position saved as Home Preset');
}

async function takeSnapshot() {
  try {
    const res = await fetch('/api/snapshot', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      showToast(`Snapshot captured: ${data.species || data.filename}`);
    } else {
      showToast('Snapshot failed (camera sleeping or busy)');
    }
  } catch (err) {
    showToast('Snapshot error');
  }
}

function toggleCamPreview() {
  camPreviewOn = !camPreviewOn;
  const img = document.getElementById('streamImg');
  const txt = document.getElementById('streamOffText');
  const btn = document.getElementById('btnToggleCam');
  if (camPreviewOn) {
    img.src = streamHudOn ? '/stream.mjpg' : '/stream.mjpg?clean=1';
    img.style.display = 'block';
    txt.style.display = 'none';
    btn.textContent = 'Stream: ON';
    btn.classList.add('active');
  } else {
    img.src = '';
    img.style.display = 'none';
    txt.style.display = 'flex';
    btn.textContent = 'Stream: OFF';
    btn.classList.remove('active');
  }
}

function toggleStreamHud() {
  streamHudOn = !streamHudOn;
  const btn = document.getElementById('btnHudMode');
  btn.textContent = 'HUD: ' + (streamHudOn ? 'ON' : 'OFF');
  btn.classList.toggle('active', streamHudOn);
  if (camPreviewOn) {
    document.getElementById('streamImg').src = streamHudOn ? '/stream.mjpg' : '/stream.mjpg?clean=1';
  }
}

function switchTab(tab) {
  document.getElementById('tabBtnDex').classList.toggle('active', tab === 'dex');
  document.getElementById('tabBtnSnaps').classList.toggle('active', tab === 'snaps');
  document.getElementById('tabBtnCharts').classList.toggle('active', tab === 'charts');

  document.getElementById('tabPanelDex').classList.toggle('active', tab === 'dex');
  document.getElementById('tabPanelSnaps').classList.toggle('active', tab === 'snaps');
  document.getElementById('tabPanelCharts').classList.toggle('active', tab === 'charts');
}

function showToast(msg) {
  const el = document.getElementById('statusToast');
  if (!el) return;
  el.textContent = msg;
  el.style.display = 'block';
  setTimeout(() => {
    if (el.textContent === msg) el.style.display = 'none';
  }, 4000);
}

/* Requirement 3: Bird Dex deletion */
async function deleteDexSpecies(species) {
  if (!confirm(`Delete all records and photos for "${species}"?`)) return;
  try {
    const res = await fetch('/api/dex/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ species })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Deleted ${data.count} records for ${species}`);
      const sRes = await fetch('/api/state');
      if (sRes.ok) updateUI(await sRes.json());
    } else {
      showToast('No records were deleted');
    }
  } catch (e) {
    showToast('Network error deleting species');
  }
}

/* Requirement 4: Captures Multi-Select & Batch Relabel */
function toggleCaptureSelected(id, checked) {
  if (checked) {
    selectedCaptures.add(id);
  } else {
    selectedCaptures.delete(id);
  }
  updateCapturesSelectionUI();
}

function onSnapCardClick(event, id) {
  if (event.target && (event.target.classList.contains('snap-select-cb') || event.target.tagName === 'INPUT')) {
    return;
  }
  openModalById(id);
}

function toggleSelectAllCaptures(checked) {
  if (checked) {
    currentSightings.forEach(s => selectedCaptures.add(s.id));
  } else {
    selectedCaptures.clear();
  }
  updateCapturesSelectionUI();
}

function clearCapturesSelection() {
  selectedCaptures.clear();
  updateCapturesSelectionUI();
}

function updateCapturesSelectionUI() {
  const count = selectedCaptures.size;
  const countLbl = document.getElementById('selectedCapturesCount');
  if (countLbl) countLbl.textContent = `${count} selected`;

  const clearBtn = document.getElementById('btnClearSelection');
  if (clearBtn) clearBtn.style.display = count > 0 ? 'inline-block' : 'none';

  const selectAllCb = document.getElementById('selectAllCapturesCb');
  if (selectAllCb) {
    selectAllCb.checked = (currentSightings.length > 0 && count === currentSightings.length);
  }

  // Update classes and checkbox states in gallery
  document.querySelectorAll('.snap-card').forEach(card => {
    const cb = card.querySelector('.snap-select-cb');
    if (cb) {
      const id = parseInt(cb.dataset.id, 10);
      const isSel = selectedCaptures.has(id);
      cb.checked = isSel;
      card.classList.toggle('selected', isSel);
    }
  });
}

async function batchRelabelSelected() {
  if (selectedCaptures.size === 0) {
    showToast('Please select at least one capture to override.');
    return;
  }
  const sel = document.getElementById('batchSpeciesSelect');
  const newSp = sel ? sel.value.trim() : '';
  if (!newSp) {
    showToast('Please choose a replacement species from the dropdown.');
    return;
  }
  const count = selectedCaptures.size;
  // Exact requirement: with an "Are you sure?" confirmation
  const confirmed = confirm(`Are you sure? This will override classification to "${newSp}" for ${count} selected capture(s).`);
  if (!confirmed) return;

  try {
    const res = await fetch('/api/sightings/batch_relabel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: Array.from(selectedCaptures), species: newSp })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Successfully relabeled ${data.count} captures to "${newSp}"`);
      selectedCaptures.clear();
      updateCapturesSelectionUI();
      const sRes = await fetch('/api/state');
      if (sRes.ok) updateUI(await sRes.json());
    } else {
      showToast('Batch override failed');
    }
  } catch (e) {
    showToast('Network error during batch override');
  }
}

/* Requirement 5: Detailed Capture Popup & Deep-Zoom */
let modalZoom = 1.0;
let modalPanX = 0;
let modalPanY = 0;
let isPanningModal = false;
let panStartX = 0, panStartY = 0;
let pinchStartDist = null;
let pinchStartZoom = 1.0;

function resetModalZoom(e) {
  if (e) e.stopPropagation();
  modalZoom = 1.0;
  modalPanX = 0;
  modalPanY = 0;
  applyModalZoomTransform();
}

function applyModalZoomTransform() {
  const img = document.getElementById('modalImg');
  const frame = document.getElementById('modalImgFrame');
  const pill = document.getElementById('modalZoomPill');
  const valLbl = document.getElementById('modalZoomVal');

  if (img) {
    img.style.transform = `scale(${modalZoom}) translate(${modalPanX / modalZoom}px, ${modalPanY / modalZoom}px)`;
  }
  if (frame) {
    frame.classList.toggle('zoomed-in', modalZoom > 1.05);
  }
  const text = `${modalZoom.toFixed(1)}x`;
  if (pill) pill.textContent = text;
  if (valLbl) valLbl.textContent = text;
}

function clampModalPan() {
  const frame = document.getElementById('modalImgFrame');
  if (!frame) return;
  const maxPanX = (frame.clientWidth * (modalZoom - 1.0)) / 2;
  const maxPanY = (frame.clientHeight * (modalZoom - 1.0)) / 2;
  modalPanX = Math.max(-maxPanX, Math.min(maxPanX, modalPanX));
  modalPanY = Math.max(-maxPanY, Math.min(maxPanY, modalPanY));
}

(function setupModalDeepZoom() {
  const frame = document.getElementById('modalImgFrame');
  if (!frame) return;

  // 1) Click to zoom in / out
  frame.addEventListener('click', (e) => {
    if (isPanningModal) return;
    if (modalZoom <= 1.05) {
      modalZoom = 2.5;
      modalPanX = 0;
      modalPanY = 0;
    } else {
      modalZoom = 1.0;
      modalPanX = 0;
      modalPanY = 0;
    }
    applyModalZoomTransform();
  });

  // 2) Mouse scroll wheel zoom in / out
  frame.addEventListener('wheel', (e) => {
    e.preventDefault();
    const prevZoom = modalZoom;
    const delta = e.deltaY < 0 ? 0.35 : -0.35;
    modalZoom = Math.max(1.0, Math.min(6.0, modalZoom + delta));
    if (modalZoom <= 1.0) {
      modalPanX = 0;
      modalPanY = 0;
    } else {
      clampModalPan();
    }
    applyModalZoomTransform();
  }, { passive: false });

  // 3) Mouse drag when zoomed in
  frame.addEventListener('mousedown', (e) => {
    if (modalZoom > 1.05 && e.button === 0) {
      isPanningModal = true;
      panStartX = e.clientX - modalPanX;
      panStartY = e.clientY - modalPanY;
      e.preventDefault();
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (isPanningModal) {
      modalPanX = e.clientX - panStartX;
      modalPanY = e.clientY - panStartY;
      clampModalPan();
      applyModalZoomTransform();
    }
  });

  window.addEventListener('mouseup', () => {
    setTimeout(() => { isPanningModal = false; }, 20);
  });

  // 4) Mobile pinch action & touch pan
  frame.addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) {
      pinchStartDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      pinchStartZoom = modalZoom;
    } else if (e.touches.length === 1 && modalZoom > 1.05) {
      isPanningModal = true;
      panStartX = e.touches[0].clientX - modalPanX;
      panStartY = e.touches[0].clientY - modalPanY;
    }
  }, { passive: false });

  frame.addEventListener('touchmove', (e) => {
    if (e.touches.length === 2 && pinchStartDist) {
      e.preventDefault();
      const currentDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      const ratio = currentDist / pinchStartDist;
      modalZoom = Math.max(1.0, Math.min(6.0, pinchStartZoom * ratio));
      if (modalZoom <= 1.0) {
        modalPanX = 0;
        modalPanY = 0;
      }
      applyModalZoomTransform();
    } else if (e.touches.length === 1 && isPanningModal) {
      e.preventDefault();
      modalPanX = e.touches[0].clientX - panStartX;
      modalPanY = e.touches[0].clientY - panStartY;
      clampModalPan();
      applyModalZoomTransform();
    }
  }, { passive: false });

  frame.addEventListener('touchend', (e) => {
    if (e.touches.length < 2) pinchStartDist = null;
    if (e.touches.length === 0) isPanningModal = false;
  });
})();

function openModalById(id) {
  const item = currentSightings.find(s => s.id === id);
  if (!item) return;
  selectedSightingId = item.id;
  resetModalZoom();

  document.getElementById('modalTitle').textContent = `#${item.id} — ${item.species} (${((item.species_confidence || item.yolo_confidence || 0) * 100).toFixed(0)}%)`;
  const snapSrc = item.snapshot_filename || item.thumb_filename || '';
  const mImg = document.getElementById('modalImg');
  mImg.src = snapSrc ? ('/snapshots/' + encodeURI(snapSrc)) : '';
  mImg.onerror = function() {
    if (item.thumb_filename && mImg.src.indexOf(encodeURI(item.thumb_filename)) === -1) {
      mImg.src = '/snapshots/' + encodeURI(item.thumb_filename);
    }
  };
  document.getElementById('modalMeta').innerHTML =
    `<strong>Timestamp:</strong> ${item.timestamp || 'N/A'} &nbsp;|&nbsp; ` +
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
  resetModalZoom();
  selectedSightingId = null;
}

async function submitRelabel() {
  if (!selectedSightingId) return;
  const newSp = document.getElementById('relabelSelect').value;
  const res = await fetch('/api/sighting/relabel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: selectedSightingId, species: newSp })
  });
  if (res.ok) {
    showToast('Re-labeled sighting to ' + newSp);
    closeModal();
    const sRes = await fetch('/api/state');
    if (sRes.ok) updateUI(await sRes.json());
  }
}

async function submitDelete() {
  if (!selectedSightingId) return;
  if (!confirm('Delete this sighting and remove its snapshot from disk?')) return;
  const res = await fetch('/api/sighting/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: selectedSightingId })
  });
  if (res.ok) {
    showToast('Deleted sighting #' + selectedSightingId);
    closeModal();
    const sRes = await fetch('/api/state');
    if (sRes.ok) updateUI(await sRes.json());
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

function setCameraControlsDisabled(disabled) {
  const ptzCard = document.getElementById('ptzCard');
  if (ptzCard) {
    ptzCard.classList.toggle('controls-disabled', disabled);
  }
  document.querySelectorAll('.cam-ctrl').forEach(el => {
    el.disabled = disabled;
  });
}

function updateUI(st) {
  if (!st) return;

  // Requirement 1: Night mode check
  const isNight = !!st.is_night_mode;
  isNightModeActive = isNight;
  const nightOverlay = document.getElementById('nightOverlay');
  const streamImg = document.getElementById('streamImg');
  const btnPopout = document.getElementById('btnPopout');
  const btnFullscreen = document.getElementById('btnFullscreen');
  const streamOffText = document.getElementById('streamOffText');

  if (isNight) {
    if (nightOverlay) nightOverlay.style.display = 'flex';
    if (streamImg) streamImg.style.display = 'none';
    if (streamOffText) streamOffText.style.display = 'none';
    if (btnPopout) btnPopout.style.display = 'none';
    if (btnFullscreen) btnFullscreen.style.display = 'none';
    setCameraControlsDisabled(true);
  } else {
    if (nightOverlay) nightOverlay.style.display = 'none';
    if (camPreviewOn && streamImg) streamImg.style.display = 'block';
    if (btnPopout) btnPopout.style.display = 'inline-block';
    if (btnFullscreen) btnFullscreen.style.display = 'inline-block';
    setCameraControlsDisabled(!st.system_power_on);
  }

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
  if (isNight) {
    fsmBadge.textContent = 'NIGHT MODE (CAMERA OFFLINE)';
    fsmBadge.className = 'badge badge-night';
  } else {
    fsmBadge.textContent = st.fsm_label || (st.tracking_enabled ? 'AUTO-FRAMING: ON' : 'MANUAL');
    fsmBadge.className = 'badge ' + (st.fsm_state === 'MANUAL_OVERRIDE' ? 'badge-warn' : 'badge-active');
  }

  const targetBadge = document.getElementById('targetBadge');
  const targetBadgeText = document.getElementById('targetBadgeText');
  const birdCount = st.detections_count || 0;
  const target = st.primary_target && st.primary_target !== 'None' ? st.primary_target : null;

  if (targetBadge) {
    if (isNight) {
      targetBadge.className = 'badge badge-idle spotter-badge';
      if (targetBadgeText) targetBadgeText.textContent = 'CAMERA OFFLINE';
    } else if (target || birdCount > 0) {
      targetBadge.className = 'badge badge-active spotter-badge badge-detected';
      const label = target ? `TARGET: ${target.toUpperCase()}` : `BIRDS IN VIEW: ${birdCount}`;
      if (targetBadgeText) targetBadgeText.textContent = label;
    } else {
      targetBadge.className = 'badge badge-idle spotter-badge';
      if (targetBadgeText) targetBadgeText.textContent = 'TARGET: NONE';
    }
  }

  const pwrBtn = document.getElementById('pwrBtn');
  if (isNight) {
    pwrBtn.textContent = '🌙 NIGHT MODE';
    pwrBtn.className = 'power-btn power-off';
  } else if (st.system_power_on) {
    pwrBtn.textContent = '⏻ CAM ACTIVE';
    pwrBtn.className = 'power-btn power-on';
  } else {
    pwrBtn.textContent = '⏻ CAM STANDBY';
    pwrBtn.className = 'power-btn power-off';
  }

  const btnSolar = document.getElementById('btnSolar');
  if (st.solar) {
    const s = st.solar;
    btnSolar.textContent = s.enabled ? `🌅 ${s.state === 'night_sleep' ? 'Night Sleep' : 'Day Active'}` : '🌅 Auto-Sleep: OFF';
    btnSolar.classList.toggle('badge-active', s.enabled);
    btnSolar.classList.toggle('badge-idle', !s.enabled);
  }

  const btnAutoTrack = document.getElementById('btnAutoTrack');
  btnAutoTrack.textContent = st.tracking_enabled ? '🤖 Auto-Framing: ON' : '🤖 Auto-Framing: OFF';
  btnAutoTrack.classList.toggle('active', !!st.tracking_enabled);

  const btnUk = document.getElementById('btnUkFilter');
  btnUk.textContent = st.uk_filter_enabled ? '🇬🇧 UK Filter: ON' : '🌍 UK Filter: OFF';
  btnUk.classList.toggle('active', !!st.uk_filter_enabled);

  const btnPrev = document.getElementById('btnPreviewBbox');
  btnPrev.textContent = '📦 Preview Box: ' + (st.show_preview_bbox ? 'ON' : 'OFF');
  btnPrev.classList.toggle('active', !!st.show_preview_bbox);

  const btnCap = document.getElementById('btnCaptureBbox');
  btnCap.textContent = '🖼️ Capture Box: ' + (st.show_capture_bbox ? 'ON' : 'OFF');
  btnCap.classList.toggle('active', !!st.show_capture_bbox);

  const fov = st.fov || 65;
  document.getElementById('fovLbl').textContent = `${fov}° ${fov === 86 ? 'Wide' : fov === 78 ? 'Medium' : 'Narrow'}`;
  document.getElementById('btnFov86').classList.toggle('active', fov === 86);
  document.getElementById('btnFov78').classList.toggle('active', fov === 78);
  document.getElementById('btnFov65').classList.toggle('active', fov === 65);

  const btnAf = document.getElementById('btnAutofocus');
  btnAf.textContent = 'AF: ' + (st.autofocus ? 'ON' : 'OFF');
  btnAf.classList.toggle('active', !!st.autofocus);
  document.getElementById('focusLbl').textContent = st.autofocus ? 'Auto' : `Manual (${st.focus || 50})`;
  const fSlider = document.getElementById('focusSlider');
  if (document.activeElement !== fSlider && st.focus !== undefined) fSlider.value = st.focus;

  document.getElementById('btnInvPan').textContent = 'Invert Pan: ' + (st.invert_pan ? 'ON' : 'OFF');
  document.getElementById('btnInvPan').classList.toggle('active', !!st.invert_pan);
  document.getElementById('btnInvTilt').textContent = 'Invert Tilt: ' + (st.invert_tilt ? 'ON' : 'OFF');
  document.getElementById('btnInvTilt').classList.toggle('active', !!st.invert_tilt);

  const ovBanner = document.getElementById('overrideBanner');
  if (st.fsm_state === 'MANUAL_OVERRIDE' && (st.manual_override_remaining || 0) > 0) {
    ovBanner.style.display = 'flex';
    document.getElementById('overrideText').textContent = `Manual Override Active — Auto-Framing resumes in ${Math.round(st.manual_override_remaining)}s`;
  } else {
    ovBanner.style.display = 'none';
  }

  currentRoiState = st.roi;
  const btnEditRoi = document.getElementById('btnEditRoi');
  const btnClearRoi = document.getElementById('btnClearRoi');
  if (st.roi) {
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

  // Life List / Bird Dex (Requirement 3: trash can icon on each card)
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
        ? `<img src="/snapshots/${encodeURI(imgPath)}" alt="${escapeHtml(d.species)}" onerror="this.style.display='none'; if(this.nextElementSibling) this.nextElementSibling.style.display='flex';" onclick="if(${d.best_sighting_id || 'null'}) openModalById(${d.best_sighting_id})"><div style="display:none; aspect-ratio:4/3; background:#090d13; align-items:center; justify-content:center; color:var(--muted); font-size:0.8rem;">No Photo</div>`
        : `<div style="aspect-ratio:4/3; background:#090d13; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:0.8rem;">No Photo</div>`;
      return `<div class="dex-card">
        ${imgHtml}
        <div class="dex-info">
          <div class="dex-title">
            <span>${escapeHtml(d.species)}</span>
            <div style="display:flex; align-items:center; gap:6px;">
              <span class="species-count-badge">${d.total_sightings}</span>
              <button class="dex-trash-btn" onclick="event.stopPropagation(); deleteDexSpecies('${escapeJs(d.species)}')" title="Delete all records for this species">
                🗑️
              </button>
            </div>
          </div>
          <div class="dex-sci">${escapeHtml(d.scientific_name || '')}</div>
          <div class="dex-meta">Peak Conf: ${((d.best_confidence || 0) * 100).toFixed(0)}% · Last: ${(d.last_seen || '').split(' ').pop()}</div>
        </div>
      </div>`;
    }).join('');
  }

  // Recent Captures Gallery (Requirement 4: multi-select & batch override)
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
        const isSel = selectedCaptures.has(s.id);
        const imgHtml = thumb
          ? `<img src="/snapshots/${encodeURI(thumb)}" alt="${escapeHtml(s.species)}" onerror="this.style.display='none'; if(this.nextElementSibling) this.nextElementSibling.style.display='flex';"><div style="display:none; aspect-ratio:4/3; background:#090d13; align-items:center; justify-content:center; color:var(--muted); font-size:0.75rem;">No Photo</div>`
          : `<div style="aspect-ratio:4/3; background:#090d13; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:0.75rem;">No Photo</div>`;
        return `<div class="snap-card ${isSel ? 'selected' : ''}" onclick="onSnapCardClick(event, ${s.id})">
          <input type="checkbox" class="snap-select-cb" data-id="${s.id}" ${isSel ? 'checked' : ''} onclick="event.stopPropagation(); toggleCaptureSelected(${s.id}, this.checked)">
          ${imgHtml}
          <div class="snap-card-caption">
            <strong>${escapeHtml(s.species || 'Bird')}</strong>
            <span style="color:var(--green-bright);">${confPct}%</span>
          </div>
        </div>`;
      }).join('');
    }
    updateCapturesSelectionUI();
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

// Populate Species Dropdowns from /api/catalog
fetch('/api/catalog')
  .then(r => r.json())
  .then(list => {
    ukCatalog = list;
    const relabelSel = document.getElementById('relabelSelect');
    const batchSel = document.getElementById('batchSpeciesSelect');
    if (relabelSel) {
      relabelSel.innerHTML = list.map(item => {
        const cName = item.common_name || item.species || item.name || '';
        const sName = item.scientific_name ? ` (${item.scientific_name})` : '';
        return `<option value="${escapeHtml(cName)}">${escapeHtml(cName)}${escapeHtml(sName)}</option>`;
      }).join('');
    }
    if (batchSel) {
      batchSel.innerHTML = '<option value="">-- Choose New Species --</option>' +
        list.map(item => {
          const cName = item.common_name || item.species || item.name || '';
          return `<option value="${escapeHtml(cName)}">${escapeHtml(cName)}</option>`;
        }).join('');
    }
  })
  .catch(() => {});

setInterval(async () => {
  try {
    const res = await fetch('/api/state');
    if (res.ok) updateUI(await res.json());
  } catch (e) {}
}, 450);

initLogsState();
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
    """Draw a clean, elegant detection box and label badge on saved snapshot photo."""
    if det is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    x1, y1 = int(round(det.xmin * w)), int(round(det.ymin * h))
    x2, y2 = int(round(det.xmax * w)), int(round(det.ymax * h))

    color = (67, 185, 46)  # Green
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

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
            roi_patch = out[ry1:ry2, rx1:rx2].copy()
            out = (out.astype(np.float32) * 0.30).astype(np.uint8)
            out[ry1:ry2, rx1:rx2] = roi_patch

            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 0, 0), 8)
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)

            cyan_bgr = (255, 255, 0)
            bracket_len = max(18, int(min(rx2 - rx1, ry2 - ry1) * 0.15))
            cv2.line(out, (rx1, ry1), (rx1 + bracket_len, ry1), cyan_bgr, 6)
            cv2.line(out, (rx1, ry1), (rx1, ry1 + bracket_len), cyan_bgr, 6)
            cv2.line(out, (rx2, ry1), (rx2 - bracket_len, ry1), cyan_bgr, 6)
            cv2.line(out, (rx2, ry1), (rx2, ry1 + bracket_len), cyan_bgr, 6)
            cv2.line(out, (rx1, ry2), (rx1 + bracket_len, ry2), cyan_bgr, 6)
            cv2.line(out, (rx1, ry2), (rx1, ry2 - bracket_len), cyan_bgr, 6)
            cv2.line(out, (rx2, ry2), (rx2 - bracket_len, ry2), cyan_bgr, 6)
            cv2.line(out, (rx2, ry2), (rx2 - bracket_len, ry2), cyan_bgr, 6)

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
    on_batch_relabel: Callable[[list[int], str], int] | None = None
    on_delete_sighting: Callable[[int], bool] | None = None
    on_delete_species: Callable[[str], int] | None = None
    get_catalog_cb: Callable[[], list[dict[str, str]]] | None = None
    snapshots_dir: Path = field(default_factory=lambda: SNAPSHOTS_DIR)

    _raw_frame: np.ndarray | None = field(default=None, init=False, repr=False)
    _jpeg_bytes: bytes = field(default=b"", init=False, repr=False)
    _clean_jpeg_bytes: bytes = field(default=b"", init=False, repr=False)
    _offline_jpeg_bytes: bytes = field(default=b"", init=False, repr=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _cond: threading.Condition = field(
        default_factory=threading.Condition, init=False, repr=False
    )
    _httpd: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        (self.snapshots_dir / "thumbs").mkdir(parents=True, exist_ok=True)
        self._offline_jpeg_bytes = self._create_offline_frame()

    def _create_offline_frame(self) -> bytes:
        """Create a clean 960x540 black frame with centered 'Night mode \\nCamera Offline'."""
        img = np.zeros((540, 960, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Night mode", (370, 240), font, 1.25, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(img, "Camera Offline", (330, 310), font, 1.25, (160, 160, 160), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else b""

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

    def update_state_only(self, state: dict[str, Any]) -> None:
        """Update telemetry/state when camera is sleeping in night mode without stalling threads."""
        with self._cond:
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
        filepath = self.snapshots_dir / filename

        cv2.imwrite(str(filepath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return filename

    def _find_snapshot_file(self, rel_str: str) -> Path | None:
        """Search for snapshot file across configured and standard snapshot directories."""
        candidates = [
            self.snapshots_dir,
            PROJECT_ROOT / "snapshots",
            PROJECT_ROOT / "data" / "snapshots",
        ]
        # Also check relative to current working directory
        cwd_snap = Path.cwd() / "snapshots"
        if cwd_snap not in candidates:
            candidates.append(cwd_snap)

        for c_dir in candidates:
            resolved_c = c_dir.resolve()
            p = (resolved_c / rel_str).resolve()
            if p.is_file() and resolved_c in p.parents:
                return p

        # Fallback: if thumb requested and missing, try non-thumb, or vice-versa
        if rel_str.startswith("thumbs/"):
            bare = rel_str[len("thumbs/"):]
            if bare.startswith("thumb_"):
                bare_snap = bare.replace("thumb_", "bird_", 1)
                for c_dir in candidates:
                    resolved_c = c_dir.resolve()
                    p = (resolved_c / bare_snap).resolve()
                    if p.is_file() and resolved_c in p.parents:
                        return p
        elif rel_str.startswith("bird_"):
            thumb_name = "thumbs/thumb_" + rel_str[len("bird_"):]
            for c_dir in candidates:
                resolved_c = c_dir.resolve()
                p = (resolved_c / thumb_name).resolve()
                if p.is_file() and resolved_c in p.parents:
                    return p
        return None

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

            def do_HEAD(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                if path.startswith("/snapshots/"):
                    rel_str = unquote(path[len("/snapshots/"):]).lstrip("/")
                    found = dashboard._find_snapshot_file(rel_str)
                    if found is not None:
                        size = found.stat().st_size
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(size))
                        self.send_header("Cache-Control", "public, max-age=86400")
                        self.end_headers()
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

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
                elif path == "/popout":
                    body = POPOUT_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/state":
                    self._send_json(dashboard.get_state())
                elif path == "/api/logs":
                    qs = parse_qs(parsed.query)
                    since_seq = int(qs.get("since", ["0"])[0])
                    lvl = qs.get("level", [None])[0]
                    entries = spotter_log.get_entries(min_level=lvl, since_seq=since_seq, limit=300)
                    self._send_json({"level": spotter_log.level, "entries": entries})
                elif path == "/api/catalog":
                    cat = dashboard.get_catalog_cb() if dashboard.get_catalog_cb else []
                    self._send_json(cat)
                elif path.startswith("/snapshots/"):
                    rel_str = unquote(path[len("/snapshots/"):]).lstrip("/")
                    found = dashboard._find_snapshot_file(rel_str)
                    if found is not None:
                        img_bytes = found.read_bytes()
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(img_bytes)))
                        self.send_header("Cache-Control", "public, max-age=86400")
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
                                dashboard._cond.wait(timeout=0.6)
                                is_night = bool(dashboard._state.get("is_night_mode"))
                                if is_night:
                                    jpg = dashboard._offline_jpeg_bytes
                                elif want_clean and dashboard._raw_frame is not None:
                                    ok, cbuf = cv2.imencode(
                                        ".jpg",
                                        dashboard._raw_frame,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                                    )
                                    jpg = cbuf.tobytes() if ok else dashboard._jpeg_bytes
                                else:
                                    jpg = dashboard._jpeg_bytes or dashboard._offline_jpeg_bytes
                            if not jpg or (jpg is last_sent and not is_night):
                                continue
                            last_sent = jpg
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                            if is_night:
                                time.sleep(0.5)
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
                elif self.path == "/api/sightings/batch_relabel":
                    raw_ids = payload.get("ids", [])
                    sids = [int(x) for x in raw_ids if str(x).isdigit() or isinstance(x, int)]
                    new_sp = str(payload.get("species", "")).strip()
                    count = (
                        dashboard.on_batch_relabel(sids, new_sp)
                        if (dashboard.on_batch_relabel and sids and new_sp)
                        else 0
                    )
                    self._send_json({"success": count > 0, "count": count})
                elif self.path == "/api/sighting/delete":
                    sid = int(payload.get("id", 0))
                    ok = (
                        dashboard.on_delete_sighting(sid)
                        if (dashboard.on_delete_sighting and sid)
                        else False
                    )
                    self._send_json({"success": ok})
                elif self.path == "/api/dex/delete":
                    sp = str(payload.get("species", "")).strip()
                    count = (
                        dashboard.on_delete_species(sp)
                        if (dashboard.on_delete_species and sp)
                        else 0
                    )
                    self._send_json({"success": count > 0, "count": count})
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
