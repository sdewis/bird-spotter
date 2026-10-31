"""Bipolar voice synthesizer, DSP effects engine, and RPi2 jaw-sync dispatcher."""

from dataclasses import dataclass, field
import io
import random
import subprocess
import threading
import time
import wave

import numpy as np

from halloween.personality.engine import Mood
from halloween.servos.client import PiconZeroClient

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


def _espeak_raw_samples(text: str, voice: str, pitch: int, speed: int) -> tuple[np.ndarray, int]:
    """Run espeak-ng and return (float32 mono samples, sample_rate)."""
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


def synthesize_bipolar_wav(text: str, mood: Mood) -> bytes:
    """Synthesize speech with mood-specific NumPy DSP effects in ~20ms."""
    if mood == Mood.SWEET:
        samples, sr = _espeak_raw_samples(text, voice="en-gb+f3", pitch=82, speed=162)
        # Gentle bright slapback shimmer (22ms)
        d = int(sr * 0.022)
        out = np.zeros(len(samples) + d, dtype=np.float32)
        out[: len(samples)] += samples
        out[d : d + len(samples)] += 0.25 * samples
        return _samples_to_wav_bytes(out * 0.95, sr)

    if mood == Mood.CHAOTIC:
        # Dual-personality synthesis: high sweet voice + deep demonic voice fighting!
        s_high, sr = _espeak_raw_samples(text, voice="en-gb+f4", pitch=88, speed=175)
        s_low, _ = _espeak_raw_samples(text, voice="en-us+m3", pitch=15, speed=175)
        idx = np.arange(0, len(s_low), 0.80)
        s_low_pitched = np.interp(idx, np.arange(len(s_low)), s_low)

        offset = int(sr * 0.035)
        total_len = max(len(s_high), len(s_low_pitched) + offset)
        out = np.zeros(total_len, dtype=np.float32)
        out[: len(s_high)] += 0.65 * s_high
        out[offset : offset + len(s_low_pitched)] += 0.75 * s_low_pitched
        return _samples_to_wav_bytes(out * 0.9, sr)

    # Default / MENACING: Deep pitch-shifted demonic voice + crypt cavern echo
    samples, sr = _espeak_raw_samples(text, voice="en-us+m3", pitch=14, speed=135)
    factor = 0.76
    idx = np.arange(0, len(samples), factor)
    pitched = np.interp(idx, np.arange(len(samples)), samples)

    d1, d2 = int(sr * 0.065), int(sr * 0.13)
    out = np.zeros(len(pitched) + d2, dtype=np.float32)
    out[: len(pitched)] += pitched
    out[d1 : d1 + len(pitched)] += 0.44 * pitched
    out[d2 : d2 + len(pitched)] += 0.24 * pitched
    return _samples_to_wav_bytes(out * 0.92, sr)


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
    cooldown_sec: float = 12.0
    jaw_closed_deg: int = 85
    jaw_open_deg: int = 135

    last_spoken_text: str = ""
    last_spoken_mood: str = "IDLE"
    _last_speak_ts: float = field(default=0.0, init=False, repr=False)
    _last_auto_mood: Mood = field(default=Mood.IDLE, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

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
        """Automatically speak on target lock or bipolar mood shift, respecting cooldown."""
        if not self.enabled or not self.auto_speak or mood == Mood.IDLE or target_label is None:
            if mood == Mood.IDLE:
                self._last_auto_mood = Mood.IDLE
            return

        now = time.monotonic()
        mood_changed = mood != self._last_auto_mood
        elapsed = now - self._last_speak_ts

        # Allow faster interrupt (5s) when snapping into MENACING or CHAOTIC mood!
        min_wait = 5.0 if (mood_changed and mood in (Mood.MENACING, Mood.CHAOTIC)) else self.cooldown_sec
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
