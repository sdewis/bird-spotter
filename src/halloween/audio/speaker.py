"""Bipolar voice synthesizer, DSP effects engine, and RPi2 jaw-sync dispatcher."""

from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np

from halloween.personality.engine import Mood
from halloween.servos.client import PiconZeroClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPER_BIN = PROJECT_ROOT / "models" / "piper" / "piper"
PIPER_MODEL = PROJECT_ROOT / "models" / "piper" / "en_GB-aru-medium.onnx"
# Speaker "01" in en_GB-aru-medium maps to speaker_id 3 ("01(3)")
DEFAULT_PIPER_SPEAKER_ID = 3

SWEET_HUMAN_LINES: tuple[str, ...] = (
    "Oh hello there, lovely! Welcome! Please, take as much candy as your heart desires!",
    "Good evening, dear friend! What a delightful Halloween costume you have on!",
    "Why hello! Come right on up, don't be shy! I promise I don't bite... anymore!",
    "Oh how wonderful, a visitor! It gets so lonely sitting on this porch all night!",
    "Well aren't you looking fabulous tonight! Happy Halloween, darling!",
)

SWEET_PET_LINES: tuple[str, ...] = (
    "Aww, look at that precious little creature! Who is a good furry friend? You are!",
    "Oh my goodness, an animal visitor! Hello sweet little buddy, come say hi to the skull!",
    "Look at those adorable ears! You get extra treats tonight, little friend!",
)

MENACING_LOOMING_LINES: tuple[str, ...] = (
    "Step back, mortal! You are standing far too close to my crypt!",
    "Do not crowd me, fleshbag! One step closer and my jaw snaps shut!",
    "I can smell your fear from here... Take one piece of candy and run!",
    "Why are you leaning in so close? Do you want to join the skeleton crew tonight?",
    "Back away from the skull! My eyes are locked onto your soul!",
)

MENACING_GENERAL_LINES: tuple[str, ...] = (
    "I see you lurking in the shadows, human... Nothing escapes my gaze!",
    "Another greedy mortal coming for our treats... Proceed at your own peril!",
    "Keep walking, fleshling... before the spirits of Halloween claim your shoes!",
    "You call that a scary costume? I am literally a bare skull and I look better!",
)

CHAOTIC_CROWD_LINES: tuple[str, ...] = (
    "Wait, there are too many of them! Stay back! No, come in, have a treat! Shut up, they are surrounding us!",
    "Oh how wonderful, guests! Tell them to leave! Be polite! No, bite their ankles right now!",
    "Stop moving around! My eyes cannot track all of you at once! Who is in charge here?!",
    "We love company! We hate crowds! Take the candy! Drop the candy! Make up our mind!",
)


class _PersistentPiper:
    """Keeps the Piper aarch64 neural TTS process warm in RAM for low-latency synthesis."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._tmpdir = tempfile.mkdtemp(prefix="halloween_piper_")
        self._seq = 0

    def is_available(self) -> bool:
        return PIPER_BIN.is_file() and PIPER_MODEL.is_file()

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._proc = subprocess.Popen(
            [
                str(PIPER_BIN),
                "--model",
                str(PIPER_MODEL),
                "--speaker",
                str(DEFAULT_PIPER_SPEAKER_ID),
                "--json-input",
                "--output_dir",
                self._tmpdir,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        return self._proc

    def warmup_async(self) -> None:
        """Warm up the ONNX model in a background thread at startup."""
        if not self.is_available():
            return

        def _warm() -> None:
            try:
                self.synthesize_samples("Ready.", speaker_id=DEFAULT_PIPER_SPEAKER_ID)
                print(
                    f"[PiperTTS] Warmed up {PIPER_MODEL.name} (speaker 01 / id={DEFAULT_PIPER_SPEAKER_ID})"
                )
            except Exception as exc:
                print(f"[PiperTTS] Warmup warning: {exc}")

        threading.Thread(target=_warm, daemon=True).start()

    def synthesize_samples(
        self, text: str, speaker_id: int = DEFAULT_PIPER_SPEAKER_ID
    ) -> tuple[np.ndarray, int]:
        """Synthesize text using Piper en_GB-aru-medium and return (float32 mono samples, sr)."""
        with self._lock:
            proc = self._ensure_proc()
            self._seq += 1
            out_wav = os.path.join(self._tmpdir, f"utt_{self._seq}.wav")
            payload = (
                json.dumps(
                    {
                        "text": text,
                        "speaker_id": int(speaker_id),
                        "output_file": out_wav,
                    }
                )
                + "\n"
            )
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(payload)
            proc.stdin.flush()
            _ = proc.stdout.readline()

            with wave.open(out_wav, "rb") as w:
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())
            try:
                os.remove(out_wav)
            except OSError:
                pass

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        # Normalize peak amplitude so jaw sync and 3.5mm speaker get strong, consistent drive
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        if peak > 100.0:
            samples = samples * (28500.0 / peak)
        return samples, sr


_PIPER = _PersistentPiper()


def _espeak_raw_samples(text: str, voice: str, pitch: int, speed: int) -> tuple[np.ndarray, int]:
    """Fallback espeak-ng synthesizer returning (float32 mono samples, sample_rate)."""
    res = subprocess.run(
        [
            "espeak-ng",
            "-v",
            voice,
            "-p",
            str(pitch),
            "-s",
            str(speed),
            "--stdout",
            text,
        ],
        capture_output=True,
        check=True,
    )
    with wave.open(io.BytesIO(res.stdout), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return samples, sr


def _tts_samples(
    text: str,
    speaker_id: int = DEFAULT_PIPER_SPEAKER_ID,
    fallback_voice: str = "en-gb+f3",
    fallback_pitch: int = 80,
    fallback_speed: int = 160,
) -> tuple[np.ndarray, int]:
    """Synthesize with Piper en_GB-aru-medium (default speaker 01(3)), falling back to espeak-ng."""
    if _PIPER.is_available():
        try:
            return _PIPER.synthesize_samples(text, speaker_id=speaker_id)
        except Exception as exc:
            print(f"[PiperTTS] Fallback to espeak-ng due to: {exc}")
    return _espeak_raw_samples(
        text, voice=fallback_voice, pitch=fallback_pitch, speed=fallback_speed
    )


def _samples_to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    """Clip float32 samples to int16 and pack into a standard mono WAV byte buffer."""
    clipped = np.clip(samples, -32767.0, 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(clipped.tobytes())
    return buf.getvalue()


def _split_chaotic_chunks(text: str) -> list[tuple[str, bool]]:
    """Split text into clauses (or 3-5 word groups if a single long sentence) for chaotic voice switching."""
    raw_clauses = [c.strip() for c in re.findall(r"[^!?;.]+[!?;.]*", text) if c.strip()]
    if not raw_clauses:
        return [(text.strip(), True)]

    chunks: list[tuple[str, bool]] = []
    single_clause = len(raw_clauses) == 1
    for clause in raw_clauses:
        words = clause.split()
        max_direct = 3 if single_clause else 6
        if len(words) <= max_direct:
            chunks.append((clause, True))
        else:
            idx = 0
            while idx < len(words):
                step = random.randint(3, 5)
                rem = len(words) - idx
                if rem <= 5:
                    step = rem
                sub = " ".join(words[idx : idx + step])
                idx += step
                chunks.append((sub, idx >= len(words)))
    return chunks


def _trim_silence(samples: np.ndarray, threshold: float = 220.0) -> np.ndarray:
    """Trim dead leading and trailing silence around an utterance."""
    active = np.flatnonzero(np.abs(samples) > threshold)
    if len(active) == 0:
        return samples[:0]
    start = max(0, int(active[0]) - 140)
    end = min(len(samples), int(active[-1]) + 240)
    return samples[start:end]


def _synthesize_chaotic_chunk(chunk_text: str, persona: str) -> tuple[np.ndarray, int]:
    """Synthesize a single chunk using Piper en_GB-aru-medium 01(3) & alternating personas."""
    if persona == "sweet_high":
        # Natural Piper en_GB-aru-medium speaker 01(3) with bright presence
        raw, sr = _tts_samples(chunk_text, speaker_id=DEFAULT_PIPER_SPEAKER_ID)
        trimmed = _trim_silence(raw)
        d = int(sr * 0.020)
        out = np.zeros(len(trimmed) + d, dtype=np.float32)
        out[: len(trimmed)] += trimmed
        out[d : d + len(trimmed)] += 0.22 * trimmed
        return out * 0.96, sr

    if persona == "demon_deep":
        # Piper en_GB-aru-medium 01(3) pitch-shifted into deep demonic register + crypt echo (+25%)
        raw, sr = _tts_samples(chunk_text, speaker_id=DEFAULT_PIPER_SPEAKER_ID)
        trimmed = _trim_silence(raw)
        if len(trimmed) > 0:
            idx = np.arange(0, len(trimmed), 0.80)
            pitched = np.interp(idx, np.arange(len(trimmed)), trimmed).astype(np.float32)
        else:
            pitched = trimmed
        d1, d2 = int(sr * 0.070), int(sr * 0.140)
        out = np.zeros(len(pitched) + d2, dtype=np.float32)
        out[: len(pitched)] += pitched
        out[d1 : d1 + len(pitched)] += 0.45 * pitched
        out[d2 : d2 + len(pitched)] += 0.22 * pitched
        return out * 0.94, sr

    if persona == "manic_goblin":
        # Alternating aru speaker (id=1 / "06") pitched up slightly for manic contrast
        raw, sr = _tts_samples(chunk_text, speaker_id=1)
        trimmed = _trim_silence(raw)
        if len(trimmed) > 0:
            idx = np.arange(0, len(trimmed), 1.12)
            pitched = np.interp(idx, np.arange(len(trimmed)), trimmed).astype(np.float32)
        else:
            pitched = trimmed
        return pitched * 0.95, sr

    # "crypt_growl": Alternating aru speaker (id=5 / "08") pitched down with dark echo (+25%)
    raw, sr = _tts_samples(chunk_text, speaker_id=5)
    trimmed = _trim_silence(raw)
    if len(trimmed) > 0:
        idx = np.arange(0, len(trimmed), 0.86)
        pitched = np.interp(idx, np.arange(len(trimmed)), trimmed).astype(np.float32)
    else:
        pitched = trimmed
    d1, d2 = int(sr * 0.056), int(sr * 0.112)
    out = np.zeros(len(pitched) + d2, dtype=np.float32)
    out[: len(pitched)] += pitched
    out[d1 : d1 + len(pitched)] += 0.38 * pitched
    out[d2 : d2 + len(pitched)] += 0.18 * pitched
    return out * 0.94, sr


def synthesize_bipolar_wav(text: str, mood: Mood) -> bytes:
    """Synthesize speech with Piper en_GB-aru-medium 01(3) and mood-specific NumPy DSP."""
    if mood == Mood.SWEET:
        samples, sr = _tts_samples(text, speaker_id=DEFAULT_PIPER_SPEAKER_ID)
        # Subtle 20ms bright room presence on top of clean Piper 01(3) voice
        d = int(sr * 0.020)
        out = np.zeros(len(samples) + d, dtype=np.float32)
        out[: len(samples)] += samples
        out[d : d + len(samples)] += 0.18 * samples
        return _samples_to_wav_bytes(out * 0.96, sr)

    if mood == Mood.CHAOTIC:
        # Erratically switch voices from chunk to chunk (never playing two at once)
        chunks = _split_chaotic_chunks(text)
        personas = ("sweet_high", "demon_deep", "manic_goblin", "crypt_growl")
        last_persona: str | None = None
        segments: list[np.ndarray] = []
        sr = 22050

        for chunk_text, is_clause_end in chunks:
            choices = [p for p in personas if p != last_persona]
            persona = random.choice(choices)
            last_persona = persona

            seg_samples, sr = _synthesize_chaotic_chunk(chunk_text, persona)
            if len(seg_samples) > 0:
                segments.append(seg_samples)
                pause_sec = 0.075 if is_clause_end else 0.020
                segments.append(np.zeros(int(sr * pause_sec), dtype=np.float32))

        if not segments:
            return _samples_to_wav_bytes(np.zeros(1000, dtype=np.float32), sr)
        out = np.concatenate(segments)
        return _samples_to_wav_bytes(out, sr)

    # Default / MENACING: Piper en_GB-aru-medium 01(3) pitch-shifted + crypt cavern echo (+25%)
    samples, sr = _tts_samples(text, speaker_id=DEFAULT_PIPER_SPEAKER_ID)
    factor = 0.82
    idx = np.arange(0, len(samples), factor)
    pitched = np.interp(idx, np.arange(len(samples)), samples)

    d1, d2, d3 = int(sr * 0.075), int(sr * 0.150), int(sr * 0.225)
    out = np.zeros(len(pitched) + d3, dtype=np.float32)
    out[: len(pitched)] += pitched
    out[d1 : d1 + len(pitched)] += 0.48 * pitched
    out[d2 : d2 + len(pitched)] += 0.25 * pitched
    out[d3 : d3 + len(pitched)] += 0.12 * pitched
    return _samples_to_wav_bytes(out * 0.93, sr)


def synthesize_bone_chatter_wav(duration_sec: float = 1.6, sr: int = 22050) -> bytes:
    """Generate a rhythmic 9 Hz bone-clacking rattle WAV that drives rapid jaw chatter."""
    total_samples = int(sr * duration_sec)
    t = np.linspace(0.0, duration_sec, total_samples, endpoint=False)
    # 9 Hz square-ish burst envelope so the jaw opens/closes 9 times per second
    burst_env = (np.sin(2.0 * np.pi * 9.0 * t) > 0.15).astype(np.float32)
    # Mix percussive clicks + hollow skull resonance (180 Hz + 420 Hz + noise)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, size=total_samples).astype(np.float32)
    tone = (
        0.5 * np.sin(2.0 * np.pi * 185.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 410.0 * t)
        + 0.4 * noise
    ).astype(np.float32)
    samples = burst_env * tone * 26000.0
    return _samples_to_wav_bytes(samples, sr)


@dataclass
class BipolarSpeaker:
    """Manages automatic and manual Bipolar speech reactions and RPi2 jaw sync."""

    servo_client: PiconZeroClient
    alsa_device: str = "plughw:1,0"
    enabled: bool = True
    auto_speak: bool = True
    cooldown_sec: float = 30.0
    jaw_closed_deg: int = 85
    jaw_open_deg: int = 135

    last_spoken_text: str = ""
    last_spoken_mood: str = "IDLE"
    _last_speak_ts: float = field(default=0.0, init=False, repr=False)
    _last_auto_mood: Mood = field(default=Mood.IDLE, init=False, repr=False)
    _idle_since_ts: float = field(default=0.0, init=False, repr=False)
    _lock_since_ts: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        _PIPER.warmup_async()

    def speak_async(self, text: str, mood: Mood) -> None:
        """Synthesize text in background and send WAV to RPi2 for playback + jaw sync."""
        if not self.enabled or not text.strip():
            return
        with self._lock:
            self._last_speak_ts = time.monotonic()
            self.last_spoken_text = text.strip()
            self.last_spoken_mood = mood.value

        def _worker() -> None:
            try:
                wav_bytes = synthesize_bipolar_wav(text, mood)
                self.servo_client.play_wav_with_jaw(
                    wav_bytes=wav_bytes,
                    alsa_device=self.alsa_device,
                    jaw_closed=self.jaw_closed_deg,
                    jaw_open=self.jaw_open_deg,
                )
            except Exception as exc:
                print(f"[BipolarSpeaker] Speech synthesis error: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    def trigger_preset(self, preset: str) -> None:
        """Trigger a manual soundboard preset from the web dashboard."""
        if preset == "sweet":
            self.speak_async(random.choice(SWEET_HUMAN_LINES), Mood.SWEET)
        elif preset == "pet":
            self.speak_async(random.choice(SWEET_PET_LINES), Mood.SWEET)
        elif preset == "menacing":
            self.speak_async(random.choice(MENACING_LOOMING_LINES), Mood.MENACING)
        elif preset == "chaotic":
            self.speak_async(random.choice(CHAOTIC_CROWD_LINES), Mood.CHAOTIC)
        elif preset == "laugh":
            self.speak_async(
                "Mwa ha ha ha ha haaa! Welcome to your doom, mortals! Ha ha ha ha!",
                Mood.MENACING,
            )
        elif preset == "chatter":
            with self._lock:
                self._last_speak_ts = time.monotonic()
                self.last_spoken_text = "*Bone Jaw Chatter*"
                self.last_spoken_mood = "CHAOTIC"

            def _chatter_worker() -> None:
                wav_bytes = synthesize_bone_chatter_wav(1.6)
                self.servo_client.play_wav_with_jaw(
                    wav_bytes=wav_bytes,
                    alsa_device=self.alsa_device,
                    jaw_closed=self.jaw_closed_deg,
                    jaw_open=self.jaw_open_deg,
                )

            threading.Thread(target=_chatter_worker, daemon=True).start()

    def evaluate_auto_reaction(
        self,
        mood: Mood,
        target_label: str | None,
        area_ratio: float,
        looming_threshold: float = 0.22,
    ) -> None:
        """Automatically speak on sustained target lock or bipolar mood shift, respecting cooldown."""
        now = time.monotonic()
        if not self.enabled or not self.auto_speak or mood == Mood.IDLE or target_label is None:
            self._lock_since_ts = 0.0
            if mood == Mood.IDLE:
                if self._idle_since_ts == 0.0:
                    self._idle_since_ts = now
                elif (now - self._idle_since_ts) >= 8.0:
                    # Only reset last mood to IDLE after 8s of continuous absence (prevents flicker spam)
                    self._last_auto_mood = Mood.IDLE
            return

        self._idle_since_ts = 0.0
        if self._lock_since_ts == 0.0:
            self._lock_since_ts = now

        # Require sustained lock for >= 0.8s before speaking
        if (now - self._lock_since_ts) < 0.8:
            return

        mood_changed = mood != self._last_auto_mood
        elapsed = now - self._last_speak_ts

        # Enforce at least 16s between auto-speech even on mood shifts, and full cooldown_sec (30s) otherwise
        min_wait = (
            max(16.0, self.cooldown_sec * 0.55)
            if (mood_changed and mood in (Mood.MENACING, Mood.CHAOTIC))
            else self.cooldown_sec
        )
        if elapsed < min_wait:
            return

        self._last_auto_mood = mood
        if mood == Mood.CHAOTIC:
            line = random.choice(CHAOTIC_CROWD_LINES)
        elif mood == Mood.MENACING:
            if area_ratio >= looming_threshold:
                line = random.choice(MENACING_LOOMING_LINES)
            else:
                line = random.choice(MENACING_GENERAL_LINES)
        else:
            if target_label in ("cat", "dog", "bird"):
                line = random.choice(SWEET_PET_LINES)
            else:
                line = random.choice(SWEET_HUMAN_LINES)

        self.speak_async(line, mood)
