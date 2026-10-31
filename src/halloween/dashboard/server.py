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
<title>Halloween Animatronic Skull — Live Vision, PTZ & Bipolar Voice</title>
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
    font-size: 1.2rem;
    letter-spacing: 0.5px;
    color: var(--orange);
  }
  .mood-badge {
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
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
    grid-template-columns: 1fr 390px;
    gap: 16px;
    padding: 16px;
    max-width: 1560px;
    margin: 0 auto;
  }
  @media (max-width: 980px) {
    main { grid-template-columns: 1fr; }
  }
  .left-col {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .video-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 400px;
  }
  .video-card img {
    width: 100%;
    height: auto;
    display: block;
  }
  .speech-banner {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.95rem;
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
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
  }
  .card h2 {
    margin: 0 0 10px 0;
    font-size: 0.88rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--muted);
  }
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat {
    background: #101017;
    padding: 8px 10px;
    border-radius: 6px;
    border: 1px solid #222230;
  }
  .stat .label { font-size: 0.72rem; color: var(--muted); }
  .stat .val { font-size: 0.98rem; font-weight: 600; margin-top: 2px; }
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
  input[type="text"], select {
    background: #101017;
    color: var(--text);
    border: 1px solid #323246;
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.85rem;
    width: 100%;
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
  <h1>💀 HALLOWEEN — Hailo-8 + OBSBOT + Bipolar Skull</h1>
  <div style="display:flex; gap:12px; align-items:center;">
    <div id="pi2Badge" class="mood-badge">PI2: CHECKING...</div>
    <div id="moodBadge" class="mood-badge">MOOD: IDLE</div>
    <button id="powerBtn" class="tv-power-btn tv-power-on" onclick="sendCtrl({action:'toggle_system_power'})" title="System Power / Standby (Camera + Skull Servos)">
      <span class="led"></span>
      <span id="powerBtnLabel">⏻ ON</span>
    </button>
  </div>
</header>
<main>
  <div class="left-col">
    <div class="video-card">
      <img src="/stream.mjpg" alt="Live MJPEG Stream">
    </div>
    <div class="speech-banner">
      <span class="tag">🔊 Last Skull Speech:</span>
      <span id="lastSpeechVal" class="line">(Waiting for visitor...)</span>
    </div>

    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <h2 style="margin:0;">Detection Objects & Face/Eye Targeting</h2>
        <div style="display:flex; gap:6px;">
          <button id="btnAllPanel" class="active" onclick="sendCtrl({action:'class_preset', preset:'toggle_all_panel'})" style="padding:4px 8px; font-size:0.75rem;">Toggle All: ON</button>
          <button onclick="sendCtrl({action:'class_preset', preset:'people_pets'})" style="padding:4px 8px; font-size:0.75rem;">People + Pets</button>
          <button onclick="sendCtrl({action:'class_preset', preset:'faces_only'})" style="padding:4px 8px; font-size:0.75rem;">Faces Only</button>
        </div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:10px; padding:8px; background:#12121b; border:1px solid #28283a; border-radius:6px;">
        <div>
          <div style="font-size:0.72rem; color:var(--muted); margin-bottom:4px; font-weight:600;">🎯 Detection Sensitivity (Room Filter)</div>
          <select id="sensSel" onchange="sendCtrl({action:'set_sensitivity', mode:this.value})" style="padding:5px 8px; font-size:0.78rem;">
            <option value="strict">Strict Room (70% Conf / 6-Frame Lock)</option>
            <option value="room" selected>Furnished Room (62% Conf / 5-Frame Lock)</option>
            <option value="high">High Sensitivity (48% Conf / 3-Frame Lock)</option>
          </select>
        </div>
        <div>
          <div style="font-size:0.72rem; color:var(--muted); margin-bottom:4px; font-weight:600;">⏱️ Auto-Speech Cooldown</div>
          <select id="cooldownSel" onchange="sendCtrl({action:'set_speech_cooldown', sec:parseFloat(this.value)})" style="padding:5px 8px; font-size:0.78rem;">
            <option value="15">Every 15s (Frequent)</option>
            <option value="30" selected>Every 30s (Balanced)</option>
            <option value="60">Every 60s (Calm)</option>
            <option value="120">Every 120s (Rare)</option>
          </select>
        </div>
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

  <div class="sidebar">
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
      <h2>Bipolar Voice & Jaw Sync (3.5mm)</h2>
      <div class="toggle-row">
        <button id="btnAutoSpeak" class="active" onclick="sendCtrl({action:'toggle_auto_speak'})">Auto-Speak on Target Lock: ON</button>
      </div>
      <div class="sound-grid">
        <button onclick="sendCtrl({action:'preset_audio', preset:'sweet'})">😇 Sweet Host</button>
        <button onclick="sendCtrl({action:'preset_audio', preset:'pet'})">🐾 Sweet Pet</button>
        <button onclick="sendCtrl({action:'preset_audio', preset:'menacing'})">😈 Menacing</button>
        <button onclick="sendCtrl({action:'preset_audio', preset:'chaotic'})">🌀 Chaotic Dual</button>
        <button onclick="sendCtrl({action:'preset_audio', preset:'laugh'})">💀 Evil Laugh</button>
        <button onclick="sendCtrl({action:'preset_audio', preset:'chatter'})">🦷 Jaw Chatter</button>
      </div>
      <div class="tts-box">
        <input id="customText" type="text" placeholder="Type custom line for skull to speak..." onkeydown="if(event.key==='Enter') speakCustom()">
        <div class="tts-row">
          <select id="customMood">
            <option value="MENACING">Voice: MENACING (Crypt Demon)</option>
            <option value="SWEET">Voice: SWEET (Polite Host)</option>
            <option value="CHAOTIC">Voice: CHAOTIC (Dual Bipolar)</option>
          </select>
          <button class="active" onclick="speakCustom()">Speak</button>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Tracking, Dual D-Pads & Calibration</h2>
      <div class="toggle-row">
        <button id="btnTrack" onclick="sendCtrl({action:'toggle_tracking'})">Auto-Track: ON</button>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
          <button id="btnInvPan" onclick="sendCtrl({action:'toggle_invert_pan'})">Invert Pan: ON</button>
          <button id="btnInvTilt" onclick="sendCtrl({action:'toggle_invert_tilt'})">Invert Tilt: ON</button>
        </div>
      </div>

      <div style="margin-top:12px; font-size:0.75rem; color:var(--muted); font-weight:600;">1. Camera PTZ D-Pad (Auto-Pauses Tracking)</div>
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
        <span style="font-size:0.75rem; color:var(--muted); font-weight:600;">2. Skull Servo D-Pad (Auto-Pauses Tracking)</span>
        <select id="servoStepSel" style="width:auto; padding:3px 6px; font-size:0.75rem;">
          <option value="3">Step: ±3° (Fine)</option>
          <option value="8" selected>Step: ±8° (Med)</option>
          <option value="16">Step: ±16° (Fast)</option>
        </select>
      </div>
      <div class="dpad">
        <div></div>
        <button onclick="nudgeServo(0, 1)">▲ Skull Up</button>
        <div></div>
        <button onclick="nudgeServo(1, 0)">◀ Skull Left</button>
        <button onclick="sendCtrl({action:'center_servo'})">🎯 Skull Centre</button>
        <button onclick="nudgeServo(-1, 0)">Skull Right ▶</button>
        <div></div>
        <button onclick="nudgeServo(0, -1)">▼ Skull Down</button>
        <div></div>
      </div>

      <div style="margin-top:12px; border-top:1px solid #2a2a38; padding-top:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-size:0.75rem; color:var(--muted); font-weight:600;">3. Coupled Cam + Skull Limit & Centre Calibration</span>
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
        <div id="calibSummary" style="margin-top:6px; font-size:0.72rem; color:var(--muted); line-height:1.4;"></div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px;">
          <button onclick="sendCtrl({action:'center_all'})">🎯 Go To Centre (Both)</button>
          <button onclick="sendCtrl({action:'calib_reset'})">🗑️ Calib Reset</button>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Picon Zero Servo Node (<span id="pi2HostVal">192.168.100.125</span>)</h2>
      <div class="sound-grid">
        <button class="active" onclick="sendCtrl({action:'pi2_init'})">🔄 Startup / Reset Node</button>
        <button onclick="sendCtrl({action:'pi2_standby'})">💤 Standby Servos</button>
        <button onclick="sendCtrl({action:'test_servo', channel:'pan'})">↔ Test Pan (Ch 0)</button>
        <button onclick="sendCtrl({action:'test_servo', channel:'tilt'})">↕ Test Tilt (Ch 1)</button>
        <button onclick="sendCtrl({action:'test_servo', channel:'jaw'})">🦷 Test Jaw (Ch 2)</button>
        <button onclick="sendCtrl({action:'test_servo', channel:'center'})">🎯 Center 90°</button>
      </div>
      <div class="tts-box" style="margin-top:12px; border-top:1px solid #2a2a38; padding-top:10px;">
        <div style="font-size:0.75rem; color:var(--muted);">Deploy Packaged Bundle to New Pi:</div>
        <div class="tts-row">
          <input id="deployHost" type="text" placeholder="Host / IP (e.g. pi4 or 192.168.100.x)">
        </div>
        <div class="tts-row">
          <input id="deployUser" type="text" placeholder="SSH User (sean)" value="sean">
          <input id="deployPass" type="text" placeholder="Password" value=" ">
          <button onclick="deployNode()">🚀 Deploy</button>
        </div>
      </div>
    </div>
  </div>
</main>
<script>
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

function nudgeServo(panDir, tiltDir) {
  const step = parseInt(document.getElementById('servoStepSel').value || '8', 10);
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
  btnTrack.textContent = 'Auto-Track: ' + (st.tracking_enabled ? 'ON' : 'PAUSED (Manual/Calib Mode)');
  btnTrack.className = st.tracking_enabled ? 'active' : '';

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
