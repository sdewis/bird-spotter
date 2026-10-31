"""Standalone Python 3 daemon script to run on the Raspberry Pi 2 (192.168.100.121) with the Picon Zero HAT.

Anti-Brownout & Power-Resilience Architecture:
1. Boots in safe STANDBY mode (all outputs de-energized, 0 mA servo load).
2. Waits for explicit `{"cmd": "init"}` command from RPi 5 before enabling servo controller.
3. Single-Axis Time-Multiplexed Slew Controller:
   - Never moves two servos simultaneously (eliminates overlapping inrush current spikes).
   - Limits angular velocity to gentle steps (max 4 deg per 60ms tick) to avoid peak stall current.
   - Auto-Detaches PWM (`MODE_OFF = 0`) 350ms after a servo reaches its target angle so servos draw 0 holding current when stationary.
4. Reports RPi 2 `vcgencmd get_throttled` under-voltage telemetry in every `ping` heartbeat.
"""

import argparse
import base64
import io
import json
import math
import socketserver
import struct
import subprocess
import threading
import time
import wave

try:
    import smbus  # type: ignore
except ImportError:
    smbus = None

import ctypes
import fcntl
import os


class NativeLinuxSMBus:
    """Zero-dependency pure-Python Linux /dev/i2c-X SMBus driver using fcntl.ioctl.

    Works on any 32-bit or 64-bit Raspberry Pi kernel with `i2c-dev` loaded, even if
    `smbus.cpython-*.so` / `libi2c.so.0` are not installed or Python versions differ.
    """

    I2C_SLAVE = 0x0703
    I2C_SMBUS = 0x0720
    I2C_SMBUS_READ = 1
    I2C_SMBUS_WRITE = 0
    I2C_SMBUS_BYTE_DATA = 2
    I2C_SMBUS_WORD_DATA = 3

    class _SmbusData(ctypes.Union):
        _fields_ = [
            ("byte", ctypes.c_uint8),
            ("word", ctypes.c_uint16),
            ("block", ctypes.c_uint8 * 34),
        ]

    class _SmbusIoctlData(ctypes.Structure):
        pass

    _SmbusIoctlData._fields_ = [
        ("read_write", ctypes.c_uint8),
        ("command", ctypes.c_uint8),
        ("size", ctypes.c_uint32),
        ("data", ctypes.POINTER(_SmbusData)),
    ]

    def __init__(self, bus_id: int = 1) -> None:
        self.fd = os.open(f"/dev/i2c-{bus_id}", os.O_RDWR)
        self._last_addr: int | None = None

    def _select_addr(self, addr: int) -> None:
        if self._last_addr != addr:
            fcntl.ioctl(self.fd, self.I2C_SLAVE, addr)
            self._last_addr = addr

    def write_byte_data(self, addr: int, reg: int, val: int) -> None:
        self._select_addr(addr)
        data = self._SmbusData(byte=val & 0xFF)
        msg = self._SmbusIoctlData(
            read_write=self.I2C_SMBUS_WRITE,
            command=reg & 0xFF,
            size=self.I2C_SMBUS_BYTE_DATA,
            data=ctypes.pointer(data),
        )
        fcntl.ioctl(self.fd, self.I2C_SMBUS, msg)

    def read_word_data(self, addr: int, reg: int) -> int:
        self._select_addr(addr)
        data = self._SmbusData()
        msg = self._SmbusIoctlData(
            read_write=self.I2C_SMBUS_READ,
            command=reg & 0xFF,
            size=self.I2C_SMBUS_WORD_DATA,
            data=ctypes.pointer(data),
        )
        fcntl.ioctl(self.fd, self.I2C_SMBUS, msg)
        return int(data.word)


def open_i2c_bus(bus_id: int = 1):
    """Open SMBus using C-extension smbus.so if available, falling back to NativeLinuxSMBus."""
    if smbus is not None:
        try:
            return smbus.SMBus(bus_id)
        except Exception:
            pass
    try:
        return NativeLinuxSMBus(bus_id)
    except Exception:
        return None


def get_system_uptime() -> float:
    try:
        with open("/proc/uptime", "r", encoding="utf-8") as f:
            return float(f.read().split()[0])
    except Exception:
        return 0.0


def get_throttled_status() -> str:
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=0.4)
        # e.g. "throttled=0x50000"
        return out.strip().split("=")[-1]
    except Exception:
        return "unknown"


class PiconZeroBoard:
    """Thread-safe native Python 3 driver for the 4tronix Picon Zero I2C HAT (0x22) with Low-Current Slew & Auto-Detach."""

    ADDR = 0x22
    REG_OUTCFG0 = 2
    REG_OUTPUT0 = 8
    REG_RESET = 20
    MODE_OFF = 0
    MODE_SERVO = 2
    RETRIES = 5

    # 100 Hz kinematic acceleration/deceleration loop for Pan (Ch 0) & Tilt (Ch 1)
    TICK_SEC = 0.01
    MAX_SPEED_DEG_S = 92.0
    DECEL_CURVE_DEG_S2 = 145.0
    JAW_MAX_STEP_DEG = 22

    def __init__(self, bus_id: int = 1) -> None:
        self.bus = open_i2c_bus(bus_id)
        self._lock = threading.Lock()
        self.initialized = False
        self.last_revision: list[int] = [0, 0]
        self.last_activity_ts: float = time.monotonic()

        # Per-channel state: integer angle written to hardware, floating-point kinematic state, and target angle
        self._curr_angle: dict[int, int] = {0: 90, 1: 90, 2: 85}
        self._pos_deg: dict[int, float] = {0: 90.0, 1: 90.0, 2: 85.0}
        self._vel_deg_s: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
        self._target_angle: dict[int, int] = {0: 90, 1: 90, 2: 85}
        self._attached: dict[int, bool] = {0: False, 1: False, 2: False}
        self._last_move_ts: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}

        self.standby()
        threading.Thread(target=self._slew_worker, daemon=True).start()

    def _write_byte(self, reg: int, val: int) -> None:
        if self.bus is None:
            return
        with self._lock:
            for _ in range(self.RETRIES):
                try:
                    self.bus.write_byte_data(self.ADDR, reg, val)
                    return
                except OSError:
                    time.sleep(0.002)

    def get_revision(self) -> list[int]:
        if self.bus is None:
            return [0, 0]
        with self._lock:
            for _ in range(self.RETRIES):
                try:
                    rval = self.bus.read_word_data(self.ADDR, 0)
                    self.last_revision = [rval // 256, rval % 256]
                    return self.last_revision
                except OSError:
                    time.sleep(0.002)
        return [-1, -1]

    def _attach_and_write(self, ch: int, angle: int) -> None:
        clamped = max(0, min(180, int(angle)))
        if not self._attached.get(ch, False):
            self._write_byte(self.REG_OUTCFG0 + ch, self.MODE_SERVO)
            self._attached[ch] = True
            time.sleep(0.005)
        self._write_byte(self.REG_OUTPUT0 + ch, clamped)
        self._curr_angle[ch] = clamped
        self._last_move_ts[ch] = time.monotonic()

    def _detach_channel(self, ch: int) -> None:
        if self._attached.get(ch, False):
            self._write_byte(self.REG_OUTCFG0 + ch, self.MODE_OFF)
            self._write_byte(self.REG_OUTPUT0 + ch, 0)
            self._attached[ch] = False

    def init_servos(self, pan_center: int = 90, tilt_center: int = 90, jaw_closed: int = 85) -> list[int]:
        """Initialize board and keep servos continuously energized so the skull never flops forward."""
        self.last_activity_ts = time.monotonic()
        self._write_byte(self.REG_RESET, 0)
        time.sleep(0.08)
        rev = self.get_revision()

        for ch, center_deg in ((0, pan_center), (1, tilt_center), (2, jaw_closed)):
            clamped = max(0, min(180, int(center_deg)))
            self._curr_angle[ch] = clamped
            self._pos_deg[ch] = float(clamped)
            self._vel_deg_s[ch] = 0.0
            self._target_angle[ch] = clamped
            self._attach_and_write(ch, clamped)
            time.sleep(0.20)

        subprocess.run(
            ["amixer", "-c", "Headphones", "sset", "PCM", "95%", "unmute"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.initialized = True
        print(f"[pi2_daemon] INIT complete -> State: READY (Rev: {rev}, Holding at {pan_center}/{tilt_center})")
        return rev

    def set_servo(self, channel: int, angle: int) -> bool:
        """Set target angle for the single-axis slew controller."""
        self.last_activity_ts = time.monotonic()
        if not self.initialized or channel not in (0, 1, 2):
            return False
        clamped = max(0, min(180, int(angle)))
        if abs(clamped - self._target_angle.get(channel, 90)) >= 1:
            self._target_angle[channel] = clamped
        return True

    def _step_kinematic_axis(self, ch: int, dt: float) -> None:
        """Advance Pan/Tilt axis along a smooth acceleration/deceleration velocity profile."""
        pos = self._pos_deg[ch]
        vel = self._vel_deg_s[ch]
        targ = float(self._target_angle[ch])
        err = targ - pos
        dist = abs(err)

        if dist < 0.15 and abs(vel) < 5.0:
            self._pos_deg[ch] = targ
            self._vel_deg_s[ch] = 0.0
            targ_int = int(round(targ))
            if targ_int != self._curr_angle[ch]:
                self._attach_and_write(ch, targ_int)
            return

        direction = 1.0 if err >= 0.0 else -1.0
        # Small delta (<= 6°): slow gentle speed; Large delta (> 6°): kinematic ramp up/down curve
        if dist <= 6.0:
            target_speed = 4.5 + 3.0 * dist
        else:
            target_speed = min(
                self.MAX_SPEED_DEG_S,
                math.sqrt((22.5 ** 2) + 2.0 * self.DECEL_CURVE_DEG_S2 * (dist - 6.0)),
            )

        desired_vel = direction * target_speed
        dv = desired_vel - vel
        same_dir = (vel * direction) >= -1e-6
        speeding_up = same_dir and (abs(desired_vel) > abs(vel))

        if speeding_up:
            # Smooth acceleration ramp scaled by distance (soft for small deltas, strong ramp-up for large deltas)
            accel = 55.0 + min(195.0, dist * 5.0)
        else:
            # Smooth deceleration to follow the braking curve into target without overshoot
            accel = 260.0

        max_dv = accel * dt
        if abs(dv) > max_dv:
            vel += math.copysign(max_dv, dv)
        else:
            vel = desired_vel

        step = vel * dt
        if abs(step) >= dist:
            pos = targ
            vel = 0.0
        else:
            pos += step

        self._pos_deg[ch] = pos
        self._vel_deg_s[ch] = vel

        new_int = max(0, min(180, int(round(pos))))
        if new_int != self._curr_angle[ch]:
            self._attach_and_write(ch, new_int)

    def _slew_worker(self) -> None:
        """100 Hz background loop applying acceleration/deceleration curves to skull Pan/Tilt servos."""
        prev_t = time.monotonic()
        while True:
            time.sleep(self.TICK_SEC)
            now = time.monotonic()
            dt = max(0.002, min(0.05, now - prev_t))
            prev_t = now

            if not self.initialized:
                continue

            # Smooth kinematic acceleration/deceleration for Pan (Ch 0) and Tilt (Ch 1)
            self._step_kinematic_axis(0, dt)
            self._step_kinematic_axis(1, dt)

            # Snappy syllable tracking for Jaw (Ch 2)
            curr_jaw = self._curr_angle[2]
            targ_jaw = self._target_angle[2]
            diff_jaw = targ_jaw - curr_jaw
            if diff_jaw != 0:
                step_jaw = max(-self.JAW_MAX_STEP_DEG, min(self.JAW_MAX_STEP_DEG, diff_jaw))
                self._attach_and_write(2, curr_jaw + step_jaw)

    def standby(self) -> None:
        """De-energize all outputs into safe STANDBY mode."""
        self.initialized = False
        for ch in (0, 1, 2):
            self._attached[ch] = False
            self._vel_deg_s[ch] = 0.0
        self._write_byte(self.REG_RESET, 0)
        time.sleep(0.01)


BOARD = PiconZeroBoard()


class AudioJawPlayer:
    """Plays WAV audio through ALSA on RPi2 while synchronizing Picon Zero Servo Channel 2 (Jaw)."""

    def __init__(self, board: PiconZeroBoard, jaw_channel: int = 2) -> None:
        self.board = board
        self.jaw_channel = jaw_channel
        self._play_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._aplay_proc: subprocess.Popen | None = None

    def stop_current(self) -> None:
        self._stop_event.set()
        if self._aplay_proc is not None and self._aplay_proc.poll() is None:
            try:
                self._aplay_proc.terminate()
                self._aplay_proc.wait(timeout=0.3)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=0.5)

    def play_wav_with_jaw(
        self,
        wav_bytes: bytes,
        alsa_device: str = "plughw:CARD=Headphones,DEV=0",
        jaw_closed: int = 85,
        jaw_open: int = 135,
    ) -> None:
        self.stop_current()
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run_playback,
            args=(wav_bytes, alsa_device, jaw_closed, jaw_open),
            daemon=True,
        )
        self._worker.start()

    def _extract_envelope(self, wav_bytes: bytes, step_sec: float = 0.06) -> tuple[list[float], float]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            sr = w.getframerate()
            n_frames = w.getnframes()
            raw = w.readframes(n_frames)

        chunk_samples = max(1, int(sr * step_sec))
        total_samples = len(raw) // 2
        rms_vals: list[float] = []

        for start in range(0, total_samples, chunk_samples):
            end = min(total_samples, start + chunk_samples)
            count = end - start
            if count <= 0:
                break
            shorts = struct.unpack_from(f"<{count}h", raw, start * 2)
            mean_sq = sum(s * s for s in shorts) / float(count)
            rms_vals.append(math.sqrt(mean_sq))

        max_rms = max(rms_vals) if rms_vals else 1.0
        if max_rms < 200.0:
            max_rms = 200.0

        norm_env = [min(1.0, (val / (max_rms * 0.75))) for val in rms_vals]
        return norm_env, step_sec

    def _run_playback(
        self,
        wav_bytes: bytes,
        alsa_device: str,
        jaw_closed: int,
        jaw_open: int,
    ) -> None:
        with self._play_lock:
            wav_path = "/tmp/skull_voice.wav"
            try:
                with open(wav_path, "wb") as f:
                    f.write(wav_bytes)
                envelope, step_sec = self._extract_envelope(wav_bytes, step_sec=0.06)
            except Exception as exc:
                print(f"[pi2_daemon] WAV parse error: {exc}")
                return

            try:
                self._aplay_proc = subprocess.Popen(
                    ["aplay", "-q", "-D", alsa_device, wav_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                print(f"[pi2_daemon] aplay error: {exc}")
                return

            t0 = time.monotonic()
            for idx, amp in enumerate(envelope):
                if self._stop_event.is_set():
                    break
                if self._aplay_proc.poll() is not None and idx > 2:
                    break

                gated = 0.0 if amp < 0.16 else amp
                angle = int(round(jaw_closed + gated * (jaw_open - jaw_closed)))
                self.board.set_servo(self.jaw_channel, angle)

                target_t = t0 + (idx + 1) * step_sec
                sleep_dt = target_t - time.monotonic()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

            self.board.set_servo(self.jaw_channel, jaw_closed)
            if self._aplay_proc is not None:
                try:
                    self._aplay_proc.wait(timeout=1.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass


JAW_PLAYER = AudioJawPlayer(BOARD, jaw_channel=2)


class ServoTCPHandler(socketserver.StreamRequestHandler):
    """Handles newline-delimited JSON init, heartbeat ping, servo, and audio commands."""

    def _send_reply(self, payload: dict) -> None:
        try:
            self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
            self.wfile.flush()
        except OSError:
            pass

    def handle(self) -> None:
        for line in self.rfile:
            try:
                raw = line.decode("utf-8").strip()
                if not raw:
                    continue
                cmd = json.loads(raw)
                BOARD.last_activity_ts = time.monotonic()

                action = cmd.get("cmd")
                if action in ("init", "reset"):
                    p_center = int(cmd.get("pan_center", 90))
                    t_center = int(cmd.get("tilt_center", 90))
                    j_closed = int(cmd.get("jaw_closed", 85))
                    rev = BOARD.init_servos(
                        pan_center=p_center, tilt_center=t_center, jaw_closed=j_closed
                    )
                    self._send_reply(
                        {
                            "status": "ok",
                            "state": "READY",
                            "rev": rev,
                            "uptime": round(get_system_uptime(), 1),
                            "throttled": get_throttled_status(),
                        }
                    )
                    continue

                if action == "standby":
                    JAW_PLAYER.stop_current()
                    BOARD.standby()
                    self._send_reply(
                        {
                            "status": "ok",
                            "state": "STANDBY",
                            "rev": BOARD.last_revision,
                            "uptime": round(get_system_uptime(), 1),
                            "throttled": get_throttled_status(),
                        }
                    )
                    continue

                if action == "ping":
                    rev = BOARD.get_revision()
                    state_str = "READY" if (BOARD.initialized and rev[0] > 0) else "STANDBY"
                    self._send_reply(
                        {
                            "status": "ok",
                            "state": state_str,
                            "rev": rev,
                            "uptime": round(get_system_uptime(), 1),
                            "throttled": get_throttled_status(),
                        }
                    )
                    continue

                if not BOARD.initialized:
                    self._send_reply({"status": "standby_required", "state": "STANDBY"})
                    continue

                if "pan" in cmd and cmd["pan"] is not None:
                    BOARD.set_servo(0, int(cmd["pan"]))
                if "tilt" in cmd and cmd["tilt"] is not None:
                    BOARD.set_servo(1, int(cmd["tilt"]))
                if cmd.get("jaw") is not None:
                    BOARD.set_servo(2, int(cmd["jaw"]))

                if "play_wav_b64" in cmd:
                    wav_bytes = base64.b64decode(cmd["play_wav_b64"])
                    JAW_PLAYER.play_wav_with_jaw(
                        wav_bytes=wav_bytes,
                        alsa_device=cmd.get("alsa_device", "plughw:CARD=Headphones,DEV=0"),
                        jaw_closed=int(cmd.get("jaw_closed", 85)),
                        jaw_open=int(cmd.get("jaw_open", 135)),
                    )
            except Exception as exc:
                print(f"[pi2_daemon] Error processing command: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RPi2/RPi4 Picon Zero Servo & Audio Daemon")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    rev = BOARD.get_revision()
    BOARD.standby()
    subprocess.run(
        ["amixer", "-c", "Headphones", "sset", "PCM", "100%", "unmute"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    print(
        f"[pi2_daemon] Booted in STANDBY mode (Picon Zero Rev: {rev}). "
        f"Waiting for RPi 5 power-on/init command on {args.host}:{args.port}..."
    )

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((args.host, args.port), ServoTCPHandler) as srv:
        try:
            srv.serve_forever()
        finally:
            JAW_PLAYER.stop_current()
            BOARD.standby()


if __name__ == "__main__":
    main()
