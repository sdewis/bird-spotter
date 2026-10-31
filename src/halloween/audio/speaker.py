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
from typing import Any
import wave

import numpy as np

from halloween.personality.engine import Mood
from halloween.servos.client import PiconZeroClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPER_BIN = PROJECT_ROOT / "models" / "piper" / "piper"
PIPER_MODEL = PROJECT_ROOT / "models" / "piper" / "en_GB-aru-medium.onnx"
DIALOGUE_PATH = PROJECT_ROOT / "config" / "dialogue.json"
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


_PAUSE_TOKEN_RE = re.compile(
    r"(\[\s*\d+(?:\.\d+)?\s*(?:ms|s)?\s*\]|\[\s*pause\s*\]|\.{2,}|,{2,}|-{2,})",
    re.IGNORECASE,
)


def _parse_pause_duration(token: str) -> float:
    """Convert a pause marker ('...', ',,,', '--', '[0.8s]', '[500ms]', '[pause]') into seconds."""
    t = token.strip().lower()
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        if inner == "pause":
            return 0.65
        if inner.endswith("ms"):
            return max(0.05, min(5.0, float(inner[:-2].strip()) / 1000.0))
        if inner.endswith("s"):
            inner = inner[:-1].strip()
        try:
            return max(0.05, min(5.0, float(inner)))
        except ValueError:
            return 0.5
    if t.startswith("."):
        # '...' = 0.60s, '....' = 0.85s, etc.
        return min(4.0, 0.35 + 0.25 * max(0, len(t) - 2))
    if t.startswith(","):
        # ',,' = 0.40s, ',,,' = 0.62s, etc.
        return min(4.0, 0.40 + 0.22 * max(0, len(t) - 2))
    if t.startswith("-"):
        return min(4.0, 0.35 + 0.20 * max(0, len(t) - 2))
    return 0.4


def _trim_silence(samples: np.ndarray, threshold: float = 220.0) -> np.ndarray:
    """Trim dead leading and trailing silence around an utterance."""
    active = np.flatnonzero(np.abs(samples) > threshold)
    if len(active) == 0:
        return samples[:0]
    start = max(0, int(active[0]) - 140)
    end = min(len(samples), int(active[-1]) + 240)
    return samples[start:end]


def _single_tts_samples(
    text: str,
    speaker_id: int = DEFAULT_PIPER_SPEAKER_ID,
    fallback_voice: str = "en-gb+f3",
    fallback_pitch: int = 80,
    fallback_speed: int = 160,
) -> tuple[np.ndarray, int]:
    if _PIPER.is_available():
        try:
            s, sr = _PIPER.synthesize_samples(text, speaker_id=speaker_id)
            return _trim_silence(s), sr
        except Exception as exc:
            print(f"[PiperTTS] Fallback to espeak-ng due to: {exc}")
    s, sr = _espeak_raw_samples(
        text, voice=fallback_voice, pitch=fallback_pitch, speed=fallback_speed
    )
    return _trim_silence(s), sr


def _tts_samples(
    text: str,
    speaker_id: int = DEFAULT_PIPER_SPEAKER_ID,
    fallback_voice: str = "en-gb+f3",
    fallback_pitch: int = 80,
    fallback_speed: int = 160,
) -> tuple[np.ndarray, int]:
    """Synthesize with Piper en_GB-aru-medium, honoring '...', ',,,', '--', and '[0.8s]' pauses."""
    parts = _PAUSE_TOKEN_RE.split(text)
    if len(parts) <= 1:
        return _single_tts_samples(
            text, speaker_id, fallback_voice, fallback_pitch, fallback_speed
        )

    out_chunks: list[np.ndarray] = []
    sr = 22050
    for part in parts:
        if not part or not part.strip():
            continue
        if _PAUSE_TOKEN_RE.fullmatch(part):
            dur = _parse_pause_duration(part)
            out_chunks.append(np.zeros(int(sr * dur), dtype=np.float32))
        else:
            trimmed, sr = _single_tts_samples(
                part.strip(), speaker_id, fallback_voice, fallback_pitch, fallback_speed
            )
            if len(trimmed) > 0:
                out_chunks.append(trimmed)

    if not out_chunks:
        return np.zeros(1000, dtype=np.float32), sr
    return np.concatenate(out_chunks), sr


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
    raw_clauses = [c.strip() for c in re.findall(r"[^!?;]+[!?;]*", text) if c.strip()]
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


def _synthesize_chaotic_chunk(chunk_text: str, persona: str) -> tuple[np.ndarray, int]:
    """Synthesize a single chunk using Piper en_GB-aru-medium 01(3) & alternating personas."""
    if persona == "sweet_high":
        # Natural Piper en_GB-aru-medium speaker 01(3) with bright presence
        trimmed, sr = _tts_samples(chunk_text, speaker_id=DEFAULT_PIPER_SPEAKER_ID)
        d = int(sr * 0.020)
        out = np.zeros(len(trimmed) + d, dtype=np.float32)
        out[: len(trimmed)] += trimmed
        out[d : d + len(trimmed)] += 0.22 * trimmed
        return out * 0.96, sr

    if persona == "demon_deep":
        # Piper en_GB-aru-medium 01(3) pitch-shifted into deep demonic register + crypt echo (+25%)
        trimmed, sr = _tts_samples(chunk_text, speaker_id=DEFAULT_PIPER_SPEAKER_ID)
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
        trimmed, sr = _tts_samples(chunk_text, speaker_id=1)
        if len(trimmed) > 0:
            idx = np.arange(0, len(trimmed), 1.12)
            pitched = np.interp(idx, np.arange(len(trimmed)), trimmed).astype(np.float32)
        else:
            pitched = trimmed
        return pitched * 0.95, sr

    # "crypt_growl": Alternating aru speaker (id=5 / "08") pitched down with dark echo (+25%)
    trimmed, sr = _tts_samples(chunk_text, speaker_id=5)
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


DEFAULT_DIALOGUE_PHRASES: list[dict[str, Any]] = [
    {
        "id": "sweet_1",
        "title": "Warm Porch Welcome",
        "mood": "SWEET",
        "text": SWEET_HUMAN_LINES[0],
        "pitch_scale": 1.00,
        "echo_wet": 0.18,
        "echo_delay_ms": 20,
        "gain": 0.96,
        "speaker_id": 3,
    },
    {
        "id": "sweet_2",
        "title": "Delightful Costume",
        "mood": "SWEET",
        "text": SWEET_HUMAN_LINES[1],
        "pitch_scale": 1.00,
        "echo_wet": 0.18,
        "echo_delay_ms": 20,
        "gain": 0.96,
        "speaker_id": 3,
    },
    {
        "id": "sweet_3",
        "title": "Promise I Don't Bite",
        "mood": "SWEET",
        "text": SWEET_HUMAN_LINES[2],
        "pitch_scale": 1.00,
        "echo_wet": 0.18,
        "echo_delay_ms": 20,
        "gain": 0.96,
        "speaker_id": 3,
    },
    {
        "id": "sweet_4",
        "title": "Lonely Porch Host",
        "mood": "SWEET",
        "text": SWEET_HUMAN_LINES[3],
        "pitch_scale": 1.00,
        "echo_wet": 0.18,
        "echo_delay_ms": 20,
        "gain": 0.96,
        "speaker_id": 3,
    },
    {
        "id": "sweet_5",
        "title": "Looking Fabulous",
        "mood": "SWEET",
        "text": SWEET_HUMAN_LINES[4],
        "pitch_scale": 1.00,
        "echo_wet": 0.18,
        "echo_delay_ms": 20,
        "gain": 0.96,
        "speaker_id": 3,
    },
    {
        "id": "pet_1",
        "title": "Precious Furry Friend",
        "mood": "SWEET_PET",
        "text": SWEET_PET_LINES[0],
        "pitch_scale": 1.03,
        "echo_wet": 0.18,
        "echo_delay_ms": 20,
        "gain": 0.96,
        "speaker_id": 3,
    },
    {
        "id": "pet_2",
        "title": "Animal Visitor Greeting",
        "mood": "SWEET_PET",
        "text": SWEET_PET_LINES[1],
        "pitch_scale": 1.03,
        "echo_wet": 0.18,
        "echo_delay_ms": 20,
        "gain": 0.96,
        "speaker_id": 3,
    },
    {
        "id": "pet_3",
        "title": "Adorable Ears",
        "mood": "SWEET_PET",
        "text": SWEET_PET_LINES[2],
        "pitch_scale": 1.03,
        "echo_wet": 0.18,
        "echo_delay_ms": 20,
        "gain": 0.96,
        "speaker_id": 3,
    },
    {
        "id": "menacing_1",
        "title": "Step Back From My Crypt",
        "mood": "MENACING",
        "text": MENACING_LOOMING_LINES[0],
        "pitch_scale": 0.82,
        "echo_wet": 0.48,
        "echo_delay_ms": 75,
        "gain": 0.93,
        "speaker_id": 3,
    },
    {
        "id": "menacing_2",
        "title": "Jaw Snaps Shut",
        "mood": "MENACING",
        "text": MENACING_LOOMING_LINES[1],
        "pitch_scale": 0.82,
        "echo_wet": 0.48,
        "echo_delay_ms": 75,
        "gain": 0.93,
        "speaker_id": 3,
    },
    {
        "id": "menacing_3",
        "title": "Smell Your Fear",
        "mood": "MENACING",
        "text": MENACING_LOOMING_LINES[2],
        "pitch_scale": 0.82,
        "echo_wet": 0.48,
        "echo_delay_ms": 75,
        "gain": 0.93,
        "speaker_id": 3,
    },
    {
        "id": "menacing_4",
        "title": "Join Skeleton Crew",
        "mood": "MENACING",
        "text": MENACING_LOOMING_LINES[3],
        "pitch_scale": 0.82,
        "echo_wet": 0.48,
        "echo_delay_ms": 75,
        "gain": 0.93,
        "speaker_id": 3,
    },
    {
        "id": "menacing_5",
        "title": "Locked Onto Your Soul",
        "mood": "MENACING",
        "text": MENACING_LOOMING_LINES[4],
        "pitch_scale": 0.82,
        "echo_wet": 0.48,
        "echo_delay_ms": 75,
        "gain": 0.93,
        "speaker_id": 3,
    },
    {
        "id": "menacing_6",
        "title": "Lurking in Shadows",
        "mood": "MENACING",
        "text": MENACING_GENERAL_LINES[0],
        "pitch_scale": 0.82,
        "echo_wet": 0.48,
        "echo_delay_ms": 75,
        "gain": 0.93,
        "speaker_id": 3,
    },
    {
        "id": "menacing_7",
        "title": "Greedy Mortal",
        "mood": "MENACING",
        "text": MENACING_GENERAL_LINES[1],
        "pitch_scale": 0.82,
        "echo_wet": 0.48,
        "echo_delay_ms": 75,
        "gain": 0.93,
        "speaker_id": 3,
    },
    {
        "id": "menacing_8",
        "title": "Claim Your Shoes",
        "mood": "MENACING",
        "text": MENACING_GENERAL_LINES[2],
        "pitch_scale": 0.82,
        "echo_wet": 0.48,
        "echo_delay_ms": 75,
        "gain": 0.93,
        "speaker_id": 3,
    },
    {
        "id": "menacing_9",
        "title": "Bare Skull Roast",
        "mood": "MENACING",
        "text": MENACING_GENERAL_LINES[3],
        "pitch_scale": 0.82,
        "echo_wet": 0.48,
        "echo_delay_ms": 75,
        "gain": 0.93,
        "speaker_id": 3,
    },
    {
        "id": "menacing_laugh",
        "title": "Evil Crypt Laugh",
        "mood": "MENACING",
        "text": "Mwa ha ha ha ha haaa! Welcome to your doom, mortals! Ha ha ha ha!",
        "pitch_scale": 0.80,
        "echo_wet": 0.55,
        "echo_delay_ms": 85,
        "gain": 0.95,
        "speaker_id": 3,
    },
    {
        "id": "chaotic_1",
        "title": "Surrounded Argument",
        "mood": "CHAOTIC",
        "text": CHAOTIC_CROWD_LINES[0],
        "pitch_scale": 1.00,
        "echo_wet": 0.45,
        "echo_delay_ms": 70,
        "gain": 0.95,
        "speaker_id": 3,
    },
    {
        "id": "chaotic_2",
        "title": "Bite Their Ankles",
        "mood": "CHAOTIC",
        "text": CHAOTIC_CROWD_LINES[1],
        "pitch_scale": 1.00,
        "echo_wet": 0.45,
        "echo_delay_ms": 70,
        "gain": 0.95,
        "speaker_id": 3,
    },
    {
        "id": "chaotic_3",
        "title": "Who Is In Charge",
        "mood": "CHAOTIC",
        "text": CHAOTIC_CROWD_LINES[2],
        "pitch_scale": 1.00,
        "echo_wet": 0.45,
        "echo_delay_ms": 70,
        "gain": 0.95,
        "speaker_id": 3,
    },
    {
        "id": "chaotic_4",
        "title": "Make Up Our Mind",
        "mood": "CHAOTIC",
        "text": CHAOTIC_CROWD_LINES[3],
        "pitch_scale": 1.00,
        "echo_wet": 0.45,
        "echo_delay_ms": 70,
        "gain": 0.95,
        "speaker_id": 3,
    },
]


def load_dialogue_phrases() -> list[dict[str, Any]]:
    """Load dialogue phrases and per-phrase DSP settings from config/dialogue.json."""
    if DIALOGUE_PATH.is_file():
        try:
            with open(DIALOGUE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data
        except (OSError, json.JSONDecodeError):
            pass
    phrases = [dict(p) for p in DEFAULT_DIALOGUE_PHRASES]
    save_dialogue_phrases(phrases)
    return phrases


def save_dialogue_phrases(phrases: list[dict[str, Any]]) -> None:
    """Persist dialogue phrases and per-phrase DSP settings to config/dialogue.json."""
    try:
        DIALOGUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DIALOGUE_PATH, "w", encoding="utf-8") as f:
            json.dump(phrases, f, indent=2)
    except OSError as exc:
        print(f"[BipolarSpeaker] Failed to save dialogue.json: {exc}")


def synthesize_phrase_wav(phrase: dict[str, Any]) -> bytes:
    """Synthesize a dialogue phrase applying its individual slider effects."""
    text = str(phrase.get("text", "")).strip()
    mood_str = str(phrase.get("mood", "MENACING")).upper()
    pitch_scale = max(0.55, min(1.45, float(phrase.get("pitch_scale", 1.0))))
    echo_wet = max(0.0, min(1.0, float(phrase.get("echo_wet", 0.35))))
    echo_delay_ms = max(10, min(220, int(phrase.get("echo_delay_ms", 65))))
    gain = max(0.2, min(1.6, float(phrase.get("gain", 0.95))))
    speaker_id = max(0, min(11, int(phrase.get("speaker_id", DEFAULT_PIPER_SPEAKER_ID))))

    if mood_str == "CHAOTIC":
        # Erratically switch voices across chunks, modulated by the phrase's custom sliders
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
                if abs(pitch_scale - 1.0) > 0.02:
                    idx = np.arange(0, len(seg_samples), pitch_scale)
                    seg_samples = np.interp(idx, np.arange(len(seg_samples)), seg_samples).astype(
                        np.float32
                    )
                if echo_wet > 0.02:
                    d1 = max(1, int(sr * (echo_delay_ms / 1000.0)))
                    padded = np.zeros(len(seg_samples) + d1, dtype=np.float32)
                    padded[: len(seg_samples)] += seg_samples
                    padded[d1 : d1 + len(seg_samples)] += (echo_wet * 0.5) * seg_samples
                    seg_samples = padded
                segments.append(seg_samples * gain)
                pause_sec = 0.075 if is_clause_end else 0.020
                segments.append(np.zeros(int(sr * pause_sec), dtype=np.float32))

        if not segments:
            return _samples_to_wav_bytes(np.zeros(1000, dtype=np.float32), sr)
        return _samples_to_wav_bytes(np.concatenate(segments), sr)

    samples, sr = _tts_samples(text, speaker_id=speaker_id)
    if abs(pitch_scale - 1.0) > 0.01 and len(samples) > 1:
        idx = np.arange(0, len(samples), pitch_scale)
        pitched = np.interp(idx, np.arange(len(samples)), samples).astype(np.float32)
    else:
        pitched = samples

    if echo_wet > 0.01:
        d1 = max(1, int(sr * (echo_delay_ms / 1000.0)))
        d2 = d1 * 2
        d3 = d1 * 3
        out = np.zeros(len(pitched) + d3, dtype=np.float32)
        out[: len(pitched)] += pitched
        out[d1 : d1 + len(pitched)] += echo_wet * pitched
        out[d2 : d2 + len(pitched)] += (echo_wet * 0.52) * pitched
        out[d3 : d3 + len(pitched)] += (echo_wet * 0.25) * pitched
    else:
        out = pitched

    return _samples_to_wav_bytes(out * gain, sr)


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
    is_playing: bool = False
    phrases: list[dict[str, Any]] = field(default_factory=load_dialogue_phrases)
    _last_speak_ts: float = field(default=0.0, init=False, repr=False)
    _last_auto_mood: Mood = field(default=Mood.IDLE, init=False, repr=False)
    _idle_since_ts: float = field(default=0.0, init=False, repr=False)
    _lock_since_ts: float = field(default=0.0, init=False, repr=False)
    _play_token: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        _PIPER.warmup_async()

    def stop_playback(self) -> None:
        """Immediately stop any active speech or preview on RPi4 and close jaw."""
        with self._lock:
            self._play_token += 1
            self.is_playing = False
        threading.Thread(
            target=lambda: self.servo_client.stop_audio(
                alsa_device=self.alsa_device, jaw_closed=self.jaw_closed_deg
            ),
            daemon=True,
        ).start()

    def upsert_phrase(self, phrase_data: dict[str, Any]) -> dict[str, Any]:
        """Add or update a phrase with its individual DSP slider settings and save to disk."""
        pid = str(phrase_data.get("id", "")).strip() or f"custom_{int(time.time() * 1000)}"
        entry = {
            "id": pid,
            "title": str(phrase_data.get("title", "Untitled Phrase")).strip() or "Untitled Phrase",
            "mood": str(phrase_data.get("mood", "MENACING")).upper(),
            "text": str(phrase_data.get("text", "")).strip(),
            "pitch_scale": round(float(phrase_data.get("pitch_scale", 1.0)), 2),
            "echo_wet": round(float(phrase_data.get("echo_wet", 0.35)), 2),
            "echo_delay_ms": int(phrase_data.get("echo_delay_ms", 65)),
            "gain": round(float(phrase_data.get("gain", 0.95)), 2),
            "speaker_id": int(phrase_data.get("speaker_id", DEFAULT_PIPER_SPEAKER_ID)),
        }
        with self._lock:
            replaced = False
            for i, p in enumerate(self.phrases):
                if p.get("id") == pid:
                    self.phrases[i] = entry
                    replaced = True
                    break
            if not replaced:
                self.phrases.append(entry)
            save_dialogue_phrases(self.phrases)
        return entry

    def delete_phrase(self, phrase_id: str) -> bool:
        """Delete a phrase by ID and persist to config/dialogue.json."""
        with self._lock:
            before = len(self.phrases)
            self.phrases = [p for p in self.phrases if p.get("id") != phrase_id]
            if len(self.phrases) != before:
                save_dialogue_phrases(self.phrases)
                return True
        return False

    def speak_phrase_async(self, phrase: dict[str, Any]) -> None:
        """Synthesize a phrase dict using its custom slider settings and stream to RPi4."""
        text = str(phrase.get("text", "")).strip()
        if not self.enabled or not text:
            return
        mood_str = str(phrase.get("mood", "MENACING")).upper()
        display_mood = "SWEET" if mood_str == "SWEET_PET" else mood_str
        with self._lock:
            self._play_token += 1
            my_token = self._play_token
            self._last_speak_ts = time.monotonic()
            self.last_spoken_text = text
            self.last_spoken_mood = display_mood
            self.is_playing = True

        def _worker() -> None:
            try:
                wav_bytes = synthesize_phrase_wav(phrase)
                with self._lock:
                    if self._play_token != my_token:
                        return
                self.servo_client.play_wav_with_jaw(
                    wav_bytes=wav_bytes,
                    alsa_device=self.alsa_device,
                    jaw_closed=self.jaw_closed_deg,
                    jaw_open=self.jaw_open_deg,
                )
            except Exception as exc:
                print(f"[BipolarSpeaker] Phrase synthesis error: {exc}")
            finally:
                with self._lock:
                    if self._play_token == my_token:
                        self.is_playing = False

        threading.Thread(target=_worker, daemon=True).start()

    def speak_async(self, text: str, mood: Mood) -> None:
        """Synthesize text in background and send WAV to RPi2 for playback + jaw sync."""
        if not self.enabled or not text.strip():
            return
        with self._lock:
            self._play_token += 1
            my_token = self._play_token
            self._last_speak_ts = time.monotonic()
            self.last_spoken_text = text.strip()
            self.last_spoken_mood = mood.value
            self.is_playing = True

        def _worker() -> None:
            try:
                wav_bytes = synthesize_bipolar_wav(text, mood)
                with self._lock:
                    if self._play_token != my_token:
                        return
                self.servo_client.play_wav_with_jaw(
                    wav_bytes=wav_bytes,
                    alsa_device=self.alsa_device,
                    jaw_closed=self.jaw_closed_deg,
                    jaw_open=self.jaw_open_deg,
                )
            except Exception as exc:
                print(f"[BipolarSpeaker] Speech synthesis error: {exc}")
            finally:
                with self._lock:
                    if self._play_token == my_token:
                        self.is_playing = False

        threading.Thread(target=_worker, daemon=True).start()

    def _pick_phrase_by_mood(self, category: str) -> dict[str, Any] | None:
        with self._lock:
            matches = [p for p in self.phrases if str(p.get("mood", "")).upper() == category.upper()]
            if not matches and category.upper() == "SWEET_PET":
                matches = [p for p in self.phrases if str(p.get("mood", "")).upper() == "SWEET"]
            return random.choice(matches) if matches else None

    def trigger_preset(self, preset: str) -> None:
        """Trigger a manual soundboard preset from the web dashboard."""
        if preset == "sweet":
            p = self._pick_phrase_by_mood("SWEET")
            if p:
                self.speak_phrase_async(p)
            else:
                self.speak_async(random.choice(SWEET_HUMAN_LINES), Mood.SWEET)
        elif preset == "pet":
            p = self._pick_phrase_by_mood("SWEET_PET")
            if p:
                self.speak_phrase_async(p)
            else:
                self.speak_async(random.choice(SWEET_PET_LINES), Mood.SWEET)
        elif preset == "menacing":
            p = self._pick_phrase_by_mood("MENACING")
            if p:
                self.speak_phrase_async(p)
            else:
                self.speak_async(random.choice(MENACING_LOOMING_LINES), Mood.MENACING)
        elif preset == "chaotic":
            p = self._pick_phrase_by_mood("CHAOTIC")
            if p:
                self.speak_phrase_async(p)
            else:
                self.speak_async(random.choice(CHAOTIC_CROWD_LINES), Mood.CHAOTIC)
        elif preset == "laugh":
            with self._lock:
                laugh_matches = [p for p in self.phrases if p.get("id") == "menacing_laugh"]
            if laugh_matches:
                self.speak_phrase_async(laugh_matches[0])
            else:
                self.speak_async(
                    "Mwa ha ha ha ha haaa! Welcome to your doom, mortals! Ha ha ha ha!",
                    Mood.MENACING,
                )
        elif preset == "chatter":
            with self._lock:
                self._play_token += 1
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
            cat = "CHAOTIC"
        elif mood == Mood.MENACING:
            cat = "MENACING"
        else:
            cat = "SWEET_PET" if target_label in ("cat", "dog", "bird") else "SWEET"

        chosen = self._pick_phrase_by_mood(cat)
        if chosen is not None:
            self.speak_phrase_async(chosen)
        else:
            self.speak_async(random.choice(SWEET_HUMAN_LINES), mood)
