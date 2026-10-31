"""Low-latency socket client & heartbeat monitor on RPi5 for the RPi2 Picon Zero daemon."""

import base64
from dataclasses import dataclass, field
import json
import socket
import threading
import time
from typing import Any


@dataclass
class PiconZeroClient:
    """Manages servo/audio streaming, heartbeat monitoring, and safe initialization of RPi 2."""

    host: str = "192.168.100.121"
    port: int = 5555
    enabled: bool = True
    heartbeat_interval_sec: float = 1.5
    min_send_interval_sec: float = 0.03

    # Live telemetry fields exposed to the web dashboard
    pi2_online: bool = False
    pi2_state: str = "OFFLINE"
    pi2_ping_ms: float = 0.0
    pi2_uptime_sec: float = 0.0
    pi2_throttled: str = "unknown"
    pi2_revision: list[int] = field(default_factory=lambda: [0, 0])
    auto_init_enabled: bool = True
    init_pan_center: int = 90
    init_tilt_center: int = 90

    _sock: socket.socket | None = field(default=None, init=False, repr=False)
    _last_sent: tuple[int, int, int | None, int | None] | None = field(
        default=None, init=False, repr=False
    )
    _last_send_ts: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _monitor_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start_monitor(self) -> None:
        """Start background heartbeat & power-outage auto-recovery thread."""
        if not self.enabled or (self._monitor_thread and self._monitor_thread.is_alive()):
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._close_socket_unlocked()

    def _close_socket_unlocked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._last_sent = None

    def _request_cmd(self, payload: dict[str, Any], timeout: float = 2.0) -> dict[str, Any] | None:
        """Send a command that expects a JSON line reply (ping, init, standby)."""
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout) as s:
                s.settimeout(timeout)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                buf = b""
                while b"\n" not in buf:
                    chunk = s.recv(1024)
                    if not chunk:
                        break
                    buf += chunk
                if not buf:
                    return None
                return json.loads(buf.decode("utf-8").splitlines()[0])
        except (OSError, json.JSONDecodeError):
            return None

    def send_startup_init(self, pan_center: int | None = None, tilt_center: int | None = None) -> bool:
        """Send explicit startup/reset command to RPi 4 to initialize Picon Zero servos at center."""
        if pan_center is not None:
            self.init_pan_center = pan_center
        if tilt_center is not None:
            self.init_tilt_center = tilt_center
        self.auto_init_enabled = True
        t0 = time.perf_counter()
        resp = self._request_cmd(
            {
                "cmd": "init",
                "pan_center": self.init_pan_center,
                "tilt_center": self.init_tilt_center,
            },
            timeout=3.2,
        )
        if resp and resp.get("status") == "ok":
            self.pi2_online = True
            self.pi2_state = str(resp.get("state", "READY"))
            self.pi2_revision = list(resp.get("rev", [0, 0]))
            self.pi2_uptime_sec = float(resp.get("uptime", 0.0))
            self.pi2_throttled = str(resp.get("throttled", "unknown"))
            self.pi2_ping_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            with self._lock:
                self._close_socket_unlocked()
            print(
                f"[PiconZeroClient] Sent INIT to RPi 2 ({self.host}) -> "
                f"State: {self.pi2_state}, Rev: {self.pi2_revision}, "
                f"Uptime: {self.pi2_uptime_sec}s, Throttled: {self.pi2_throttled}"
            )
            return True
        return False

    def send_standby(self) -> bool:
        """Place RPi 2 Picon Zero into de-energized STANDBY until next manual init."""
        self.auto_init_enabled = False
        resp = self._request_cmd({"cmd": "standby"}, timeout=1.0)
        if resp and resp.get("status") == "ok":
            self.pi2_state = "STANDBY"
            return True
        return False

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            resp = self._request_cmd({"cmd": "ping"}, timeout=0.8)
            if resp is None:
                if self.pi2_online:
                    print(f"[PiconZeroClient] WARNING: Lost heartbeat from RPi 2 ({self.host})!")
                self.pi2_online = False
                self.pi2_state = "OFFLINE"
                self.pi2_ping_ms = 0.0
                with self._lock:
                    self._close_socket_unlocked()
            else:
                rtt = round((time.perf_counter() - t0) * 1000.0, 1)
                self.pi2_online = True
                self.pi2_ping_ms = rtt
                self.pi2_state = str(resp.get("state", "UNKNOWN"))
                self.pi2_revision = list(resp.get("rev", [0, 0]))
                self.pi2_uptime_sec = float(resp.get("uptime", 0.0))
                self.pi2_throttled = str(resp.get("throttled", "unknown"))

                # Auto-initialize whenever system power is ON and node is in STANDBY
                if self.pi2_state == "STANDBY" and self.auto_init_enabled:
                    print(
                        f"[PiconZeroClient] Servo Node in STANDBY (Uptime: {self.pi2_uptime_sec}s) "
                        f"-> sending INIT command (Centre: {self.init_pan_center}°/{self.init_tilt_center}°)..."
                    )
                    self.send_startup_init()

            self._stop_event.wait(self.heartbeat_interval_sec)

    def _send_stream_json(self, payload: dict[str, Any]) -> bool:
        if not self.enabled or not self.pi2_online or self.pi2_state != "READY":
            return False
        with self._lock:
            if self._sock is None:
                try:
                    sock = socket.create_connection((self.host, self.port), timeout=0.15)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._sock = sock
                except OSError:
                    self._sock = None
                    return False
            try:
                assert self._sock is not None
                self._sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                return True
            except OSError:
                self._close_socket_unlocked()
                return False

    def send_angles(
        self,
        pan: int = 90,
        tilt: int = 90,
        jaw: int | None = None,
        eyes: int | None = None,
        force: bool = False,
    ) -> bool:
        """Send servo target angles (0-180 degrees) to the RPi2 daemon, rate-limited."""
        now = time.monotonic()
        if not force and jaw is None and (now - self._last_send_ts) < self.min_send_interval_sec:
            return False

        if not force and self._last_sent is not None and jaw is None:
            prev_pan, prev_tilt, _, _ = self._last_sent
            if pan == prev_pan and tilt == prev_tilt:
                return True

        state_tuple = (pan, tilt, jaw, eyes)
        ok = self._send_stream_json({"pan": pan, "tilt": tilt, "jaw": jaw, "eyes": eyes})
        if ok:
            self._last_sent = state_tuple
            self._last_send_ts = now
        return ok

    def play_wav_with_jaw(
        self,
        wav_bytes: bytes,
        alsa_device: str = "plughw:1,0",
        jaw_closed: int = 85,
        jaw_open: int = 135,
    ) -> bool:
        """Stream synthesized WAV bytes to RPi4 for 3.5mm audio output + jaw servo sync."""
        if not self.enabled or not self.pi2_online or self.pi2_state != "READY":
            return False
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        payload = {
            "play_wav_b64": b64,
            "alsa_device": alsa_device,
            "jaw_closed": jaw_closed,
            "jaw_open": jaw_open,
        }
        try:
            with socket.create_connection((self.host, self.port), timeout=2.5) as s:
                s.settimeout(2.5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            return True
        except OSError:
            return False

    def stop_audio(self, alsa_device: str = "plughw:1,0", jaw_closed: int = 85) -> bool:
        """Immediately interrupt any playing WAV on RPi4 and return jaw to closed angle."""
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"\x00\x00" * 220)
        return self.play_wav_with_jaw(
            wav_bytes=buf.getvalue(),
            alsa_device=alsa_device,
            jaw_closed=jaw_closed,
            jaw_open=jaw_closed,
        )
