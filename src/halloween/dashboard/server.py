"""Live MJPEG streaming and interactive control dashboard for the Halloween Skull."""

from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Callable

import cv2
import numpy as np

from halloween.personality.engine import Mood
from halloween.vision.detector import Detection

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Halloween Animatronic Skull — v0.2 Dashboard Advanced</title>
<style>
  :root {
    --bg: #0c0c10;
    --panel: #16161e;
    --border: #2a2a38;
    --text: #e6e6f0;
    --muted: #8e8ea6;
    --orange: #ff6b1a;
    --sweet: #2ee59d;
    --menacing: #ff2e55;
    --chaotic: #c84bff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, sans-serif;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
  }
  header h1 {
    margin: 0;
    font-size: 1.15rem;
    letter-spacing: 0.5px;
    color: var(--orange);
  }
  .mood-badge {
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.88rem;
    background: #252533;
    border: 1px solid var(--border);
  }
  .mood-SWEET { background: rgba(46, 229, 157, 0.18); color: var(--sweet); border-color: var(--sweet); }
  .mood-MENACING { background: rgba(255, 46, 85, 0.2); color: var(--menacing); border-color: var(--menacing); }
  .mood-CHAOTIC { background: rgba(200, 75, 255, 0.2); color: var(--chaotic); border-color: var(--chaotic); }
  .tv-power-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 0.88rem;
    letter-spacing: 0.6px;
    cursor: pointer;
    transition: all 0.2s ease;
    background: radial-gradient(circle at 30% 30%, #2b2b3c, #151520);
  }
  .tv-power-btn .led {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }
  .tv-power-on {
    color: #eafff5;
    border: 2px solid var(--sweet);
    box-shadow: 0 0 14px rgba(46, 229, 157, 0.55), inset 0 0 8px rgba(46, 229, 157, 0.25);
  }
  .tv-power-on .led {
    background: var(--sweet);
    box-shadow: 0 0 8px var(--sweet);
  }
  .tv-power-off {
    color: #ffd6de;
    border: 2px solid var(--menacing);
    box-shadow: 0 0 14px rgba(255, 46, 85, 0.65), inset 0 0 8px rgba(255, 46, 85, 0.3);
  }
  .tv-power-off .led {
    background: var(--menacing);
    box-shadow: 0 0 8px var(--menacing);
  }
  main {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 16px;
    padding: 16px;
    max-width: 1640px;
    margin: 0 auto;
    align-items: start;
  }
  @media (max-width: 1024px) {
    main { grid-template-columns: 1fr; }
  }
  .left-pane, .right-pane {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .video-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .video-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    background: #12121a;
    border-bottom: 1px solid var(--border);
  }
  .video-header .title {
    font-size: 0.86rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: var(--muted);
  }
  .video-frame-wrap {
    width: 100%;
    aspect-ratio: 16 / 9;
    background: #2f2f37;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  .video-frame-wrap img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .video-off-box {
    width: 100%;
    height: 100%;
    background: #2f2f37;
    display: none;
  }
  .speech-banner {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.92rem;
  }
  .speech-banner .tag {
    font-weight: 700;
    color: var(--orange);
    white-space: nowrap;
  }
  .speech-banner .line {
    color: var(--text);
    font-style: italic;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
  }
  .card h2 {
    margin: 0 0 10px 0;
    font-size: 0.86rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--muted);
  }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  @media (max-width: 680px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
  }
  .stat {
    background: #101017;
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid #222230;
  }
  .stat .label { font-size: 0.71rem; color: var(--muted); }
  .stat .val { font-size: 0.94rem; font-weight: 600; margin-top: 2px; }
  .tab-bar {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
    background: #111118;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid var(--border);
  }
  .tab-btn {
    background: transparent;
    border: 1px solid transparent;
    color: var(--muted);
    padding: 9px 6px;
    border-radius: 7px;
    font-size: 0.81rem;
    font-weight: 700;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
  }
  .tab-btn:hover {
    color: var(--text);
    background: #1b1b26;
  }
  .tab-btn.active {
    background: var(--orange);
    color: #000;
    border-color: var(--orange);
  }
  .tab-panel {
    display: none;
  }
  .tab-panel.active {
    display: block;
  }
  .dpad {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 8px;
  }
  .sound-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
  }
  button {
    background: #232332;
    color: var(--text);
    border: 1px solid #38384f;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.83rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover:not(:disabled) { background: #303046; }
  button:disabled {
    opacity: 0.38;
    cursor: not-allowed;
  }
  button.active {
    background: var(--orange);
    color: #000;
    border-color: var(--orange);
  }
  button.calib-btn:not(:disabled) {
    border-color: var(--sweet);
    color: var(--sweet);
  }
  button.danger-btn {
    border-color: var(--menacing);
    color: #ff7a93;
  }
  button.sweet-btn {
    border-color: var(--sweet);
    color: var(--sweet);
  }
  .toggle-row {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .tts-box {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 10px;
  }
  .tts-row {
    display: flex;
    gap: 6px;
  }
  input[type="text"], select, textarea {
    background: #101017;
    color: var(--text);
    border: 1px solid #323246;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.85rem;
    width: 100%;
    font-family: inherit;
  }
  textarea {
    resize: vertical;
    min-height: 70px;
  }
  .slider-group {
    background: #111119;
    border: 1px solid #252536;
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .slider-row {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .slider-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.77rem;
    color: var(--muted);
    font-weight: 600;
  }
  .slider-head .sval {
    color: var(--orange);
    font-weight: 700;
  }
  input[type="range"] {
    width: 100%;
    accent-color: var(--orange);
    cursor: pointer;
  }
  .obj-group {
    margin-bottom: 8px;
  }
  .obj-group-title {
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 4px;
    font-weight: 600;
  }
  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .chip {
    padding: 5px 9px;
    font-size: 0.76rem;
    border-radius: 999px;
    background: #191924;
    border: 1px solid #323246;
    color: var(--muted);
  }
  .chip.active {
    background: rgba(255, 107, 26, 0.22);
    color: #ffb085;
    border-color: var(--orange);
  }
</style>
</head>
<body>
<header>
  <h1>💀 HALLOWEEN — v0.2 Dashboard Advanced</h1>
  <div style="display:flex; gap:12px; align-items:center;">
    <div id="pi2Badge" class="mood-badge">NODE: CHECKING...</div>
    <div id="moodBadge" class="mood-badge">MOOD: IDLE</div>
    <button id="powerBtn" class="tv-power-btn tv-power-on" onclick="sendCtrl({action:'toggle_system_power'})" title="System Power / Standby (Camera + Skull Servos)">
      <span class="led"></span>
      <span id="powerBtnLabel">⏻ ON</span>
    </button>
  </div>
</header>

<main>
  <!-- ================= LEFT PANE: VIDEO PREVIEW, TELEMETRY & OBJECTS ================= -->
  <div class="left-pane">
    <div class="video-card">
      <div class="video-header">
        <span class="title">📹 Camera Vision Preview</span>
        <button id="btnCamPreview" class="active" onclick="toggleCamPreview()" style="padding:5px 12px; font-size:0.78rem;">Cam Preview: ON</button>
      </div>
      <div class="video-frame-wrap">
        <img id="videoStreamImg" src="/stream.mjpg" alt="Live MJPEG Stream">
        <div id="videoOffBox" class="video-off-box"></div>
      </div>
    </div>

    <div class="speech-banner">
      <span class="tag">🔊 Last Skull Speech:</span>
      <span id="lastSpeechVal" class="line">(Waiting for visitor...)</span>
    </div>

    <div class="card">
      <h2>Live Telemetry</h2>
      <div class="stat-grid">
        <div class="stat"><div class="label">Camera FPS</div><div id="fpsVal" class="val">--</div></div>
        <div class="stat"><div class="label">Hailo-8 + Face</div><div id="inferVal" class="val">-- ms</div></div>
        <div class="stat"><div class="label">Locked Target</div><div id="targetVal" class="val">None</div></div>
        <div class="stat"><div class="label">Target Area</div><div id="areaVal" class="val">0%</div></div>
        <div class="stat"><div class="label">UVC Pan / Tilt</div><div id="uvcVal" class="val">0 / 0</div></div>
        <div class="stat"><div class="label">Skull Servo (°)</div><div id="servoVal" class="val">90° / 90°</div></div>
        <div class="stat"><div class="label">Node Heartbeat</div><div id="pi2PingVal" class="val">--</div></div>
        <div class="stat"><div class="label">Node Uptime</div><div id="pi2UptimeVal" class="val">--</div></div>
      </div>
    </div>

    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <h2 style="margin:0;">Detection Objects & Room Filter</h2>
        <div style="display:flex; gap:6px;">
          <button id="btnAllPanel" class="active" onclick="sendCtrl({action:'class_preset', preset:'toggle_all_panel'})" style="padding:4px 8px; font-size:0.75rem;">Toggle All: ON</button>
          <button onclick="sendCtrl({action:'class_preset', preset:'people_pets'})" style="padding:4px 8px; font-size:0.75rem;">People + Pets</button>
          <button onclick="sendCtrl({action:'class_preset', preset:'faces_only'})" style="padding:4px 8px; font-size:0.75rem;">Faces Only</button>
        </div>
      </div>

      <div style="margin-bottom:10px; padding:8px; background:#12121b; border:1px solid #28283a; border-radius:6px;">
        <div style="font-size:0.72rem; color:var(--muted); margin-bottom:4px; font-weight:600;">🎯 Detection Sensitivity (Room Filter)</div>
        <select id="sensSel" onchange="sendCtrl({action:'set_sensitivity', mode:this.value})" style="padding:5px 8px; font-size:0.78rem;">
          <option value="strict">Strict Room (70% Conf / 6-Frame Lock)</option>
          <option value="room" selected>Furnished Room (62% Conf / 5-Frame Lock)</option>
          <option value="high">High Sensitivity (48% Conf / 3-Frame Lock)</option>
        </select>
      </div>

      <div class="obj-group">
        <div class="obj-group-title">1. Person & Face (Gaze locks onto Face / Eyes)</div>
        <div class="chip-row">
          <button class="chip" data-classes="person" onclick="toggleClasses(['person'])">👤 Person</button>
          <button class="chip" data-classes="upper body" onclick="toggleClasses(['upper body'])">🧥 Upper Body</button>
          <button class="chip" data-classes="frontal face" onclick="toggleClasses(['frontal face'])">😀 Frontal Face</button>
          <button class="chip" data-classes="side face" onclick="toggleClasses(['side face'])">🗣️ Side Face</button>
        </div>
      </div>

      <div class="obj-group">
        <div class="obj-group-title">2. Animals</div>
        <div class="chip-row">
          <button class="chip" data-classes="cat" onclick="toggleClasses(['cat'])">🐱 Cat</button>
          <button class="chip" data-classes="dog" onclick="toggleClasses(['dog'])">🐶 Dog</button>
          <button class="chip" data-classes="bird" onclick="toggleClasses(['bird'])">🐦 Bird</button>
          <button class="chip" data-classes="horse" onclick="toggleClasses(['horse'])">🐴 Horse</button>
        </div>
      </div>

      <div class="obj-group">
        <div class="obj-group-title">3. Bags, Rain & Handheld</div>
        <div class="chip-row">
          <button class="chip" data-classes="backpack" onclick="toggleClasses(['backpack'])">🎒 Backpack</button>
          <button class="chip" data-classes="umbrella" onclick="toggleClasses(['umbrella'])">☂️ Umbrella</button>
          <button class="chip" data-classes="cell phone" onclick="toggleClasses(['cell phone'])">📱 Cell Phone</button>
          <button class="chip" data-classes="book" onclick="toggleClasses(['book'])">📖 Book</button>
        </div>
      </div>

      <div class="obj-group">
        <div class="obj-group-title">4. Drinks & Cutlery</div>
        <div class="chip-row">
          <button class="chip" data-classes="bottle" onclick="toggleClasses(['bottle'])">🍾 Bottle</button>
          <button class="chip" data-classes="wine glass" onclick="toggleClasses(['wine glass'])">🍷 Wine Glass</button>
          <button class="chip" data-classes="cup" onclick="toggleClasses(['cup'])">☕ Cup</button>
          <button class="chip" data-classes="knife,fork,spoon" onclick="toggleClasses(['knife','fork','spoon'])">🍴 Cutlery {Knife, Fork, Spoon}</button>
        </div>
      </div>

      <div class="obj-group" style="margin-bottom:0;">
        <div class="obj-group-title">5. Seating, Decor, TV & Vehicles</div>
        <div class="chip-row">
          <button class="chip" data-classes="chair,couch" onclick="toggleClasses(['chair','couch'])">🛋️ Seating {Chair, Couch}</button>
          <button class="chip" data-classes="potted plant" onclick="toggleClasses(['potted plant'])">🪴 Potted Plant</button>
          <button class="chip" data-classes="tv" onclick="toggleClasses(['tv'])">📺 TV</button>
          <button class="chip" data-classes="car" onclick="toggleClasses(['car'])">🚗 Car</button>
          <button class="chip" data-classes="motorcycle" onclick="toggleClasses(['motorcycle'])">🏍️ Motorcycle</button>
          <button class="chip" data-classes="truck" onclick="toggleClasses(['truck'])">🚚 Truck</button>
        </div>
      </div>
    </div>
  </div>

  <!-- ================= RIGHT PANE: TABBED CONFIGURATION ================= -->
  <div class="right-pane">
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('voice', this)">🔊 Voice</button>
      <button class="tab-btn" onclick="switchTab('tracking', this)">🎯 Tracking</button>
      <button class="tab-btn" onclick="switchTab('calibration', this)">📐 Calibration</button>
      <button class="tab-btn" onclick="switchTab('node', this)">🖧 Node</button>
      <button class="tab-btn" onclick="switchTab('dialogue', this)">💬 Dialogue</button>
    </div>

    <!-- TAB 1: VOICE -->
    <div id="tab-voice" class="tab-panel active">
      <div class="card">
        <h2>Voice & Soundboard (Piper en_GB-aru-medium 01(3))</h2>
        <div class="toggle-row">
          <button id="btnAutoSpeak" class="active" onclick="sendCtrl({action:'toggle_auto_speak'})">Auto-Speak on Target Lock: ON</button>
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:4px;">
            <span style="font-size:0.78rem; color:var(--muted); white-space:nowrap;">⏱️ Auto-Speech Cooldown:</span>
            <select id="cooldownSel" onchange="sendCtrl({action:'set_speech_cooldown', sec:parseFloat(this.value)})" style="width:auto; flex:1;">
              <option value="15">Every 15s (Frequent)</option>
              <option value="30" selected>Every 30s (Balanced)</option>
              <option value="60">Every 60s (Calm)</option>
              <option value="120">Every 120s (Rare)</option>
            </select>
          </div>
        </div>

        <div class="sound-grid" style="margin-top:12px;">
          <button onclick="sendCtrl({action:'preset_audio', preset:'sweet'})">😇 Sweet Host</button>
          <button onclick="sendCtrl({action:'preset_audio', preset:'pet'})">🐾 Sweet Pet</button>
          <button onclick="sendCtrl({action:'preset_audio', preset:'menacing'})">😈 Menacing</button>
          <button onclick="sendCtrl({action:'preset_audio', preset:'chaotic'})">🌀 Chaotic Switch</button>
          <button onclick="sendCtrl({action:'preset_audio', preset:'laugh'})">💀 Evil Laugh</button>
          <button onclick="sendCtrl({action:'preset_audio', preset:'chatter'})">🦷 Jaw Chatter</button>
        </div>

        <div style="margin-top:8px;">
          <button class="danger-btn" style="width:100%;" onclick="sendCtrl({action:'stop_audio'})">⏹ Stop Active Speech / Close Jaw</button>
        </div>

        <div class="tts-box" style="margin-top:14px; border-top:1px solid var(--border); padding-top:12px;">
          <div style="font-size:0.75rem; color:var(--muted); font-weight:600;">Quick Custom Speech Line</div>
          <input id="customText" type="text" placeholder="Type custom line for skull to speak..." onkeydown="if(event.key==='Enter') speakCustom()">
          <div class="tts-row">
            <select id="customMood">
              <option value="MENACING">Voice: MENACING (Crypt Demon)</option>
              <option value="SWEET">Voice: SWEET (Polite Host)</option>
              <option value="CHAOTIC">Voice: CHAOTIC (Erratic Switch)</option>
            </select>
            <button class="active" onclick="speakCustom()">Speak</button>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: TRACKING (ONLY AUTO-TRACK TOGGLE, INVERT CONTROLS & DPADS) -->
    <div id="tab-tracking" class="tab-panel">
      <div class="card">
        <h2>Tracking & Manual D-Pads</h2>
        <div class="toggle-row">
          <button id="btnTrack" class="active" onclick="sendCtrl({action:'toggle_tracking'})">Auto-Track: ON</button>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <button id="btnInvPan" onclick="sendCtrl({action:'toggle_invert_pan'})">Invert Pan: ON</button>
            <button id="btnInvTilt" onclick="sendCtrl({action:'toggle_invert_tilt'})">Invert Tilt: ON</button>
          </div>
        </div>

        <div style="margin-top:14px; font-size:0.75rem; color:var(--muted); font-weight:600;">1. Camera PTZ D-Pad (Auto-Pauses Tracking)</div>
        <div class="dpad">
          <div></div>
          <button onclick="sendCtrl({action:'nudge', d_pan:0, d_tilt:28800})">▲ Cam Up</button>
          <div></div>
          <button onclick="sendCtrl({action:'nudge', d_pan:36000, d_tilt:0})">◀ Cam Left</button>
          <button onclick="sendCtrl({action:'center'})">🎯 Cam Centre</button>
          <button onclick="sendCtrl({action:'nudge', d_pan:-36000, d_tilt:0})">Cam Right ▶</button>
          <div></div>
          <button onclick="sendCtrl({action:'nudge', d_pan:0, d_tilt:-28800})">▼ Cam Down</button>
          <div></div>
        </div>

        <div style="margin-top:14px; display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:0.75rem; color:var(--muted); font-weight:600;">2. Skull Servo D-Pad (Auto-Pauses Tracking)</span>
          <select id="servoStepSel" style="width:auto; padding:3px 6px; font-size:0.75rem;">
            <option value="3">Step: ±3° (Fine)</option>
            <option value="8" selected>Step: ±8° (Med)</option>
            <option value="16">Step: ±16° (Fast)</option>
          </select>
        </div>
        <div class="dpad">
          <div></div>
          <button onclick="nudgeServo(0, 1, 'servoStepSel')">▲ Skull Up</button>
          <div></div>
          <button onclick="nudgeServo(1, 0, 'servoStepSel')">◀ Skull Left</button>
          <button onclick="sendCtrl({action:'center_servo'})">🎯 Skull Centre</button>
          <button onclick="nudgeServo(-1, 0, 'servoStepSel')">Skull Right ▶</button>
          <div></div>
          <button onclick="nudgeServo(0, -1, 'servoStepSel')">▼ Skull Down</button>
          <div></div>
        </div>
      </div>
    </div>

    <!-- TAB 3: CALIBRATION (DUPLICATED DPADS + CALIBRATION CONTROLS) -->
    <div id="tab-calibration" class="tab-panel">
      <div class="card">
        <h2>Coupled Camera & Skull Calibration</h2>
        <div class="toggle-row">
          <button id="btnTrackCalib" class="active" onclick="sendCtrl({action:'toggle_tracking'})">Auto-Track: ON (Click to Pause for Calibration)</button>
        </div>

        <div style="margin-top:12px; font-size:0.75rem; color:var(--muted); font-weight:600;">1. Camera PTZ D-Pad (Position Camera)</div>
        <div class="dpad">
          <div></div>
          <button onclick="sendCtrl({action:'nudge', d_pan:0, d_tilt:28800})">▲ Cam Up</button>
          <div></div>
          <button onclick="sendCtrl({action:'nudge', d_pan:36000, d_tilt:0})">◀ Cam Left</button>
          <button onclick="sendCtrl({action:'center'})">🎯 Cam Centre</button>
          <button onclick="sendCtrl({action:'nudge', d_pan:-36000, d_tilt:0})">Cam Right ▶</button>
          <div></div>
          <button onclick="sendCtrl({action:'nudge', d_pan:0, d_tilt:-28800})">▼ Cam Down</button>
          <div></div>
        </div>

        <div style="margin-top:12px; display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:0.75rem; color:var(--muted); font-weight:600;">2. Skull Servo D-Pad (Position Skull)</span>
          <select id="servoStepSelCalib" style="width:auto; padding:3px 6px; font-size:0.75rem;">
            <option value="3">Step: ±3° (Fine)</option>
            <option value="8" selected>Step: ±8° (Med)</option>
            <option value="16">Step: ±16° (Fast)</option>
          </select>
        </div>
        <div class="dpad">
          <div></div>
          <button onclick="nudgeServo(0, 1, 'servoStepSelCalib')">▲ Skull Up</button>
          <div></div>
          <button onclick="nudgeServo(1, 0, 'servoStepSelCalib')">◀ Skull Left</button>
          <button onclick="sendCtrl({action:'center_servo'})">🎯 Skull Centre</button>
          <button onclick="nudgeServo(-1, 0, 'servoStepSelCalib')">Skull Right ▶</button>
          <div></div>
          <button onclick="nudgeServo(0, -1, 'servoStepSelCalib')">▼ Skull Down</button>
          <div></div>
        </div>

        <div style="margin-top:14px; border-top:1px solid var(--border); padding-top:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:0.75rem; color:var(--muted); font-weight:600;">3. Save Limits & Skewed Centres</span>
            <span id="calibStatusTag" style="font-size:0.72rem; color:var(--orange);">Pause Track to Calibrate</span>
          </div>
          <div class="sound-grid">
            <button id="btnCalibLeft" class="calib-btn" disabled onclick="sendCtrl({action:'calibrate', direction:'left'})">📐 Calib Left</button>
            <button id="btnCalibRight" class="calib-btn" disabled onclick="sendCtrl({action:'calibrate', direction:'right'})">📐 Calib Right</button>
            <button id="btnCalibUp" class="calib-btn" disabled onclick="sendCtrl({action:'calibrate', direction:'up'})">📐 Calib Up</button>
            <button id="btnCalibDown" class="calib-btn" disabled onclick="sendCtrl({action:'calibrate', direction:'down'})">📐 Calib Down</button>
            <button id="btnCalibCenterCam" class="calib-btn" disabled onclick="sendCtrl({action:'calibrate', direction:'center_cam'})">📐 Calib Cam Centre</button>
            <button id="btnCalibCenterServo" class="calib-btn" disabled onclick="sendCtrl({action:'calibrate', direction:'center_servo'})">📐 Calib Skull Centre</button>
          </div>
          <div style="margin-top:8px;">
            <button id="btnCalibCenterBoth" class="calib-btn" style="width:100%;" disabled onclick="sendCtrl({action:'calibrate', direction:'center'})">📐 Calib Centre (Both Cam & Skull)</button>
          </div>
          <div id="calibSummary" style="margin-top:8px; font-size:0.74rem; color:var(--muted); line-height:1.45;"></div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;">
            <button onclick="sendCtrl({action:'center_all'})">🎯 Go To Centre (Both)</button>
            <button class="danger-btn" onclick="sendCtrl({action:'calib_reset'})">🗑️ Calib Reset</button>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: NODE (PICON ZERO & DEPLOYMENT CONTROLS) -->
    <div id="tab-node" class="tab-panel">
      <div class="card">
        <h2>Picon Zero Servo Node (<span id="pi2HostVal">192.168.100.125</span>)</h2>
        <div class="sound-grid">
          <button class="active" onclick="sendCtrl({action:'pi2_init'})">🔄 Startup / Reset Node</button>
          <button onclick="sendCtrl({action:'pi2_standby'})">💤 Standby Servos</button>
          <button onclick="sendCtrl({action:'test_servo', channel:'pan'})">↔ Test Pan (Ch 0)</button>
          <button onclick="sendCtrl({action:'test_servo', channel:'tilt'})">↕ Test Tilt (Ch 1)</button>
          <button onclick="sendCtrl({action:'test_servo', channel:'jaw'})">🦷 Test Jaw (Ch 2)</button>
          <button onclick="sendCtrl({action:'test_servo', channel:'center'})">🎯 Centre All</button>
        </div>

        <div class="tts-box" style="margin-top:14px; border-top:1px solid var(--border); padding-top:12px;">
          <div style="font-size:0.75rem; color:var(--muted); font-weight:600;">Deploy Packaged Servo & Audio Node to Raspberry Pi</div>
          <div class="tts-row">
            <input id="deployHost" type="text" placeholder="Host / IP (e.g. 192.168.100.125)">
          </div>
          <div class="tts-row">
            <input id="deployUser" type="text" placeholder="SSH User (sean)" value="sean">
            <input id="deployPass" type="text" placeholder="Password" value=" ">
            <button class="active" onclick="deployNode()">🚀 Deploy</button>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: DIALOGUE (PHRASE LIBRARY, ADD/EDIT/REMOVE, PREVIEW & EFFECT SLIDERS) -->
    <div id="tab-dialogue" class="tab-panel">
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <h2 style="margin:0;">Dialogue Library & Phrase DSP Editor</h2>
          <div style="display:flex; gap:6px;">
            <button class="sweet-btn" onclick="startNewPhrase()" style="padding:5px 9px; font-size:0.76rem;">➕ New Phrase</button>
            <button class="danger-btn" onclick="deleteSelectedPhrase()" style="padding:5px 9px; font-size:0.76rem;">🗑️ Delete</button>
          </div>
        </div>

        <div style="font-size:0.73rem; color:var(--muted); margin-bottom:4px; font-weight:600;">Select Built-In or Saved Phrase (Title + Mood Category)</div>
        <select id="phraseSelect" onchange="onSelectPhrase(this.value)"></select>

        <div style="display:grid; grid-template-columns:1.2fr 0.8fr; gap:8px; margin-top:10px;">
          <div>
            <div style="font-size:0.72rem; color:var(--muted); margin-bottom:3px;">Phrase Title</div>
            <input id="dlgTitle" type="text" placeholder="Phrase title...">
          </div>
          <div>
            <div style="font-size:0.72rem; color:var(--muted); margin-bottom:3px;">Mood Category</div>
            <select id="dlgMood">
              <option value="SWEET">SWEET (Host)</option>
              <option value="SWEET_PET">SWEET_PET (Animal)</option>
              <option value="MENACING">MENACING (Demon)</option>
              <option value="CHAOTIC">CHAOTIC (Switch)</option>
            </select>
          </div>
        </div>

        <div style="margin-top:8px;">
          <div style="font-size:0.72rem; color:var(--muted); margin-bottom:3px;">Spoken Line Text</div>
          <textarea id="dlgText" placeholder="Enter what the skull should say..."></textarea>
        </div>

        <div class="slider-group">
          <div style="font-size:0.76rem; font-weight:700; color:var(--text);">🎛️ Individual Phrase Voice Effects</div>

          <div class="slider-row">
            <div class="slider-head">
              <span>Pitch / Speed Scale (<small>lower = deeper</small>)</span>
              <span id="valPitch" class="sval">1.00x</span>
            </div>
            <input id="slPitch" type="range" min="0.60" max="1.40" step="0.02" value="1.00" oninput="updateSliderLabels()">
          </div>

          <div class="slider-row">
            <div class="slider-head">
              <span>Cavern Echo Intensity (Wet Mix)</span>
              <span id="valEchoWet" class="sval">48%</span>
            </div>
            <input id="slEchoWet" type="range" min="0.00" max="1.00" step="0.02" value="0.48" oninput="updateSliderLabels()">
          </div>

          <div class="slider-row">
            <div class="slider-head">
              <span>Echo Delay (Cavern Size)</span>
              <span id="valEchoDelay" class="sval">75 ms</span>
            </div>
            <input id="slEchoDelay" type="range" min="15" max="180" step="5" value="75" oninput="updateSliderLabels()">
          </div>

          <div class="slider-row">
            <div class="slider-head">
              <span>Output Gain / Jaw Drive</span>
              <span id="valGain" class="sval">95%</span>
            </div>
            <input id="slGain" type="range" min="0.30" max="1.50" step="0.02" value="0.95" oninput="updateSliderLabels()">
          </div>

          <div class="slider-row">
            <div class="slider-head">
              <span>Piper Voice (en_GB-aru-medium Speaker)</span>
            </div>
            <select id="dlgSpeakerId" style="padding:5px 8px; font-size:0.8rem;">
              <option value="3">Speaker 01 (3) — Default British Host</option>
              <option value="0">Speaker 03 (0)</option>
              <option value="1">Speaker 06 (1)</option>
              <option value="2">Speaker 10 (2)</option>
              <option value="4">Speaker 09 (4)</option>
              <option value="5">Speaker 08 (5)</option>
              <option value="6">Speaker 11 (6)</option>
              <option value="7">Speaker 05 (7)</option>
              <option value="8">Speaker 12 (8)</option>
              <option value="9">Speaker 02 (9)</option>
              <option value="10">Speaker 07 (10)</option>
              <option value="11">Speaker 04 (11)</option>
            </select>
          </div>
        </div>

        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;">
          <button id="btnPreviewPlay" class="active" onclick="previewCurrentPhrase()">▶ Play Preview</button>
          <button id="btnPreviewStop" class="danger-btn" onclick="sendCtrl({action:'stop_audio'})">⏹ Stop Preview</button>
        </div>
        <div style="margin-top:8px;">
          <button class="sweet-btn" style="width:100%;" onclick="saveCurrentPhrase()">💾 Save Phrase & Voice Effects</button>
        </div>
        <div id="dlgSaveStatus" style="margin-top:6px; font-size:0.74rem; color:var(--sweet); text-align:center; min-height:16px;"></div>
      </div>
    </div>
  </div>
</main>

<script>
let camPreviewOn = true;
let phrasesCache = [];
let selectedPhraseId = null;
let isEditingNew = false;

function switchTab(tabName, btnEl) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const target = document.getElementById('tab-' + tabName);
  if (target) target.classList.add('active');
  if (btnEl) btnEl.classList.add('active');
}

function toggleCamPreview() {
  camPreviewOn = !camPreviewOn;
  const img = document.getElementById('videoStreamImg');
  const offBox = document.getElementById('videoOffBox');
  const btn = document.getElementById('btnCamPreview');
  if (camPreviewOn) {
    img.src = '/stream.mjpg?t=' + Date.now();
    img.style.display = 'block';
    offBox.style.display = 'none';
    btn.textContent = 'Cam Preview: ON';
    btn.className = 'active';
  } else {
    img.src = '';
    img.style.display = 'none';
    offBox.style.display = 'block';
    btn.textContent = 'Cam Preview: OFF';
    btn.className = '';
  }
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
  } catch (e) { console.error(e); }
}

function toggleClasses(clsList) {
  sendCtrl({action: 'toggle_classes', classes: clsList});
}

function nudgeServo(panDir, tiltDir, selId) {
  const sel = document.getElementById(selId || 'servoStepSel');
  const step = parseInt((sel && sel.value) || '8', 10);
  sendCtrl({action: 'nudge_servo', d_pan_deg: panDir * step, d_tilt_deg: tiltDir * step});
}

function speakCustom() {
  const input = document.getElementById('customText');
  const mood = document.getElementById('customMood').value;
  if (!input.value.trim()) return;
  sendCtrl({action: 'speak_custom', text: input.value.trim(), mood: mood});
  input.value = '';
}

function deployNode() {
  const host = document.getElementById('deployHost').value.trim();
  const user = document.getElementById('deployUser').value.trim() || 'sean';
  const password = document.getElementById('deployPass').value;
  if (!host) return;
  sendCtrl({action: 'deploy_node', host: host, user: user, password: password});
}

function updateSliderLabels() {
  const pitch = parseFloat(document.getElementById('slPitch').value || '1.0');
  const wet = parseFloat(document.getElementById('slEchoWet').value || '0.48');
  const dly = parseInt(document.getElementById('slEchoDelay').value || '75', 10);
  const gain = parseFloat(document.getElementById('slGain').value || '0.95');
  document.getElementById('valPitch').textContent = pitch.toFixed(2) + 'x';
  document.getElementById('valEchoWet').textContent = Math.round(wet * 100) + '%';
  document.getElementById('valEchoDelay').textContent = dly + ' ms';
  document.getElementById('valGain').textContent = Math.round(gain * 100) + '%';
}

function populatePhraseDropdown(phrases) {
  const sel = document.getElementById('phraseSelect');
  if (!sel) return;
  const sig = JSON.stringify(phrases.map(p => `${p.id}:${p.mood}:${p.title}`));
  if (sel.getAttribute('data-sig') === sig) return;
  sel.setAttribute('data-sig', sig);

  sel.innerHTML = '';
  phrases.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = `[${p.mood}] ${p.title}`;
    sel.appendChild(opt);
  });

  if (!isEditingNew) {
    const valid = phrases.find(p => p.id === selectedPhraseId) || phrases[0];
    if (valid) {
      selectedPhraseId = valid.id;
      sel.value = valid.id;
      loadPhraseIntoEditor(valid);
    }
  }
}

function loadPhraseIntoEditor(p) {
  if (!p) return;
  document.getElementById('dlgTitle').value = p.title || '';
  document.getElementById('dlgMood').value = p.mood || 'MENACING';
  document.getElementById('dlgText').value = p.text || '';
  document.getElementById('slPitch').value = p.pitch_scale ?? 1.0;
  document.getElementById('slEchoWet').value = p.echo_wet ?? 0.35;
  document.getElementById('slEchoDelay').value = p.echo_delay_ms ?? 65;
  document.getElementById('slGain').value = p.gain ?? 0.95;
  document.getElementById('dlgSpeakerId').value = String(p.speaker_id ?? 3);
  updateSliderLabels();
}

function onSelectPhrase(pid) {
  isEditingNew = false;
  selectedPhraseId = pid;
  const found = phrasesCache.find(p => p.id === pid);
  if (found) loadPhraseIntoEditor(found);
}

function startNewPhrase() {
  isEditingNew = true;
  selectedPhraseId = 'custom_' + Date.now();
  document.getElementById('dlgTitle').value = 'New Haunted Line';
  document.getElementById('dlgMood').value = 'MENACING';
  document.getElementById('dlgText').value = '';
  document.getElementById('slPitch').value = 0.82;
  document.getElementById('slEchoWet').value = 0.48;
  document.getElementById('slEchoDelay').value = 75;
  document.getElementById('slGain').value = 0.95;
  document.getElementById('dlgSpeakerId').value = '3';
  updateSliderLabels();
  showDlgStatus('Creating new phrase — edit and click Save');
}

function collectEditorPhrase() {
  return {
    id: selectedPhraseId || ('custom_' + Date.now()),
    title: document.getElementById('dlgTitle').value.trim() || 'Untitled Phrase',
    mood: document.getElementById('dlgMood').value,
    text: document.getElementById('dlgText').value.trim(),
    pitch_scale: parseFloat(document.getElementById('slPitch').value || '1.0'),
    echo_wet: parseFloat(document.getElementById('slEchoWet').value || '0.35'),
    echo_delay_ms: parseInt(document.getElementById('slEchoDelay').value || '65', 10),
    gain: parseFloat(document.getElementById('slGain').value || '0.95'),
    speaker_id: parseInt(document.getElementById('dlgSpeakerId').value || '3', 10)
  };
}

function previewCurrentPhrase() {
  const phrase = collectEditorPhrase();
  if (!phrase.text) {
    showDlgStatus('Enter spoken text first!');
    return;
  }
  sendCtrl({action: 'preview_phrase', phrase: phrase});
  showDlgStatus('▶ Playing preview on skull...');
}

function saveCurrentPhrase() {
  const phrase = collectEditorPhrase();
  if (!phrase.text) {
    showDlgStatus('Cannot save an empty phrase!');
    return;
  }
  isEditingNew = false;
  selectedPhraseId = phrase.id;
  document.getElementById('phraseSelect').removeAttribute('data-sig');
  sendCtrl({action: 'save_phrase', phrase: phrase});
  showDlgStatus('✔ Saved phrase & voice effects!');
}

function deleteSelectedPhrase() {
  if (!selectedPhraseId) return;
  const oldId = selectedPhraseId;
  selectedPhraseId = null;
  isEditingNew = false;
  document.getElementById('phraseSelect').removeAttribute('data-sig');
  sendCtrl({action: 'delete_phrase', id: oldId});
  showDlgStatus('🗑️ Phrase deleted.');
}

function showDlgStatus(msg) {
  const el = document.getElementById('dlgSaveStatus');
  if (!el) return;
  el.textContent = msg;
  setTimeout(() => {
    if (el.textContent === msg) el.textContent = '';
  }, 3200);
}

function updateUI(st) {
  document.getElementById('fpsVal').textContent = st.fps.toFixed(1);
  document.getElementById('inferVal').textContent = st.infer_ms.toFixed(1) + ' ms';
  document.getElementById('targetVal').textContent = st.primary_target || 'None';
  document.getElementById('areaVal').textContent = (st.primary_area * 100).toFixed(1) + '%';
  document.getElementById('uvcVal').textContent = `${st.pan} / ${st.tilt}`;
  document.getElementById('servoVal').textContent = `${st.servo_pan}° / ${st.servo_tilt}°`;

  if (st.pi2_host) {
    document.getElementById('pi2HostVal').textContent = st.pi2_host;
  }
  const pi2Badge = document.getElementById('pi2Badge');
  if (st.pi2_online) {
    pi2Badge.textContent = `NODE: ${st.pi2_state} (${st.pi2_ping_ms}ms)`;
    pi2Badge.className = 'mood-badge ' + (st.pi2_state === 'READY' ? 'mood-SWEET' : 'mood-CHAOTIC');
    document.getElementById('pi2PingVal').textContent = `${st.pi2_state} (${st.pi2_ping_ms} ms)`;
    document.getElementById('pi2UptimeVal').textContent = `${st.pi2_uptime_sec}s (${st.pi2_throttled || '0x0'})`;
  } else {
    pi2Badge.textContent = 'NODE: OFFLINE';
    pi2Badge.className = 'mood-badge mood-MENACING';
    document.getElementById('pi2PingVal').textContent = 'OFFLINE';
    document.getElementById('pi2UptimeVal').textContent = '--';
  }

  if (st.last_spoken_text) {
    document.getElementById('lastSpeechVal').textContent = `[${st.last_spoken_mood}] "${st.last_spoken_text}"`;
  }

  const badge = document.getElementById('moodBadge');
  badge.textContent = 'MOOD: ' + (st.system_power_on === false ? 'STANDBY' : st.mood);
  badge.className = 'mood-badge mood-' + (st.system_power_on === false ? 'IDLE' : st.mood);

  const pwrBtn = document.getElementById('powerBtn');
  const pwrLbl = document.getElementById('powerBtnLabel');
  if (pwrBtn && pwrLbl) {
    const isOn = st.system_power_on !== false;
    pwrBtn.className = 'tv-power-btn ' + (isOn ? 'tv-power-on' : 'tv-power-off');
    pwrLbl.textContent = isOn ? '⏻ ON' : '⏻ STANDBY';
  }

  const btnAutoSpeak = document.getElementById('btnAutoSpeak');
  btnAutoSpeak.textContent = 'Auto-Speak on Target Lock: ' + (st.auto_speak ? 'ON' : 'MUTED');
  btnAutoSpeak.className = st.auto_speak ? 'active' : '';

  const btnTrack = document.getElementById('btnTrack');
  btnTrack.textContent = 'Auto-Track: ' + (st.tracking_enabled ? 'ON' : 'PAUSED (Manual Mode)');
  btnTrack.className = st.tracking_enabled ? 'active' : '';

  const btnTrackCalib = document.getElementById('btnTrackCalib');
  if (btnTrackCalib) {
    btnTrackCalib.textContent = st.tracking_enabled
      ? 'Auto-Track: ON (Click to Pause for Calibration)'
      : 'Auto-Track: PAUSED (Ready to Calibrate)';
    btnTrackCalib.className = st.tracking_enabled ? 'active' : '';
  }

  const calibDisabled = !!st.tracking_enabled;
  [
    'btnCalibLeft',
    'btnCalibRight',
    'btnCalibUp',
    'btnCalibDown',
    'btnCalibCenterCam',
    'btnCalibCenterServo',
    'btnCalibCenterBoth'
  ].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = calibDisabled;
  });
  const calibTag = document.getElementById('calibStatusTag');
  if (calibTag) {
    calibTag.textContent = calibDisabled ? 'Pause Track to Calibrate' : 'MANUAL MODE: Ready to Calibrate';
    calibTag.style.color = calibDisabled ? 'var(--muted)' : 'var(--sweet)';
  }
  if (st.calib) {
    const c = st.calib;
    const sumEl = document.getElementById('calibSummary');
    if (sumEl) {
      const cPan = c.cam_pan_center ?? 0;
      const cTilt = c.cam_tilt_center ?? 0;
      const sPan = c.servo_pan_center ?? 90;
      const sTilt = c.servo_tilt_center ?? 90;
      sumEl.innerHTML =
        `<b>Centre (${c.calibrated ? 'Custom' : 'Default'}):</b> Cam(${cPan}, ${cTilt}) → Skull(${sPan}°, ${sTilt}°)<br>` +
        `<b>Limits:</b> L[${c.cam_pan_left}→${c.servo_pan_left}°] R[${c.cam_pan_right}→${c.servo_pan_right}°] U[${c.cam_tilt_up}→${c.servo_tilt_up}°] D[${c.cam_tilt_down}→${c.servo_tilt_down}°]`;
    }
  }

  document.getElementById('btnInvPan').textContent = 'Invert Pan: ' + (st.invert_pan ? 'ON' : 'OFF');
  document.getElementById('btnInvTilt').textContent = 'Invert Tilt: ' + (st.invert_tilt ? 'ON' : 'OFF');

  if (Array.isArray(st.phrases)) {
    phrasesCache = st.phrases;
    populatePhraseDropdown(st.phrases);
  }

  if (Array.isArray(st.enabled_classes)) {
    const enabledSet = new Set(st.enabled_classes);
    const chips = document.querySelectorAll('.chip');
    let allActive = true;
    chips.forEach(btn => {
      const clsAttr = btn.getAttribute('data-classes') || '';
      const clsArr = clsAttr.split(',').map(s => s.trim()).filter(Boolean);
      const isOn = clsArr.length > 0 && clsArr.every(c => enabledSet.has(c));
      if (!isOn) allActive = false;
      btn.className = isOn ? 'chip active' : 'chip';
    });
    const btnAllPanel = document.getElementById('btnAllPanel');
    if (btnAllPanel) {
      btnAllPanel.textContent = allActive ? 'Toggle All: ON' : 'Toggle All: OFF';
      btnAllPanel.className = allActive ? 'active' : '';
    }
  }
}

setInterval(async () => {
  try {
    const res = await fetch('/api/state');
    if (res.ok) updateUI(await res.json());
  } catch (e) {}
}, 400);
</script>
</body>
</html>
"""


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    primary: Detection | None,
    mood: Mood,
    fps: float,
    infer_ms: float,
    deadzone_norm: float = 0.08,
) -> np.ndarray:
    """Draw bounding boxes, face/eye-level target crosshair, deadzone, and HUD onto BGR frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    dz_x1 = int((0.5 - deadzone_norm) * w)
    dz_y1 = int((0.5 - deadzone_norm) * h)
    dz_x2 = int((0.5 + deadzone_norm) * w)
    dz_y2 = int((0.5 + deadzone_norm) * h)
    cv2.rectangle(out, (dz_x1, dz_y1), (dz_x2, dz_y2), (90, 90, 110), 1)
    cv2.drawMarker(out, (w // 2, h // 2), (120, 120, 140), cv2.MARKER_CROSS, 14, 1)

    mood_colors = {
        Mood.IDLE: (180, 180, 180),
        Mood.SWEET: (157, 229, 46),     # Greenish-teal (BGR)
        Mood.MENACING: (85, 46, 255),   # Crimson red (BGR)
        Mood.CHAOTIC: (255, 75, 200),   # Purple/Magenta (BGR)
    }
    mood_color = mood_colors.get(mood, (200, 200, 200))

    for det in detections:
        x1, y1 = int(det.xmin * w), int(det.ymin * h)
        x2, y2 = int(det.xmax * w), int(det.ymax * h)
        is_primary = primary is not None and det.track_id == primary.track_id

        if det.label in ("frontal face", "side face"):
            base_color = (0, 255, 255)  # Bright yellow/gold for faces
        elif det.label == "upper body":
            base_color = (255, 200, 0)  # Cyan-blue for upper body
        else:
            base_color = (255, 140, 0)

        color = mood_color if is_primary else base_color
        thickness = 3 if is_primary else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        tag = f"#{det.track_id} {det.label} {det.confidence:.2f}"
        if is_primary:
            tag = f"[LOCKED] {tag}"
            # Draw gaze target marker at (aim_x, aim_y) — looks at face/eye level instead of mid-torso!
            ax, ay = int(det.aim_x * w), int(det.aim_y * h)
            cv2.drawMarker(out, (ax, ay), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
            cv2.circle(out, (ax, ay), 6, color, -1)
            cv2.line(out, (w // 2, h // 2), (ax, ay), color, 2)

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            out,
            tag,
            (x1 + 4, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (10, 10, 15),
            2,
            cv2.LINE_AA,
        )

    hud = f"FPS: {fps:.1f} | Hailo+Face: {infer_ms:.1f}ms | Mood: {mood.value}"
    cv2.rectangle(out, (10, 10), (490, 44), (15, 15, 22), -1)
    cv2.putText(
        out,
        hud,
        (20, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        mood_color,
        2,
        cv2.LINE_AA,
    )
    return out


@dataclass
class DashboardServer:
    """Thread-safe MJPEG and JSON telemetry server."""

    host: str = "0.0.0.0"
    port: int = 8080
    on_control: Callable[[dict[str, Any]], None] | None = None

    _jpeg_bytes: bytes = field(default=b"", init=False, repr=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _cond: threading.Condition = field(
        default_factory=threading.Condition, init=False, repr=False
    )
    _httpd: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)

    def update_frame_and_state(
        self, annotated_bgr: np.ndarray, state: dict[str, Any]
    ) -> None:
        ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._cond:
            self._jpeg_bytes = buf.tobytes()
            self._state = state
            self._cond.notify_all()

    def get_state(self) -> dict[str, Any]:
        with self._cond:
            return dict(self._state)

    def start_background(self) -> None:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/api/state":
                    body = json.dumps(dashboard.get_state()).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/stream.mjpg"):
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
                if self.path == "/api/control":
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    payload = json.loads(raw.decode("utf-8"))
                    if dashboard.on_control is not None:
                        dashboard.on_control(payload)
                    body = json.dumps(dashboard.get_state()).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

        ThreadingHTTPServer.allow_reuse_address = True
        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
