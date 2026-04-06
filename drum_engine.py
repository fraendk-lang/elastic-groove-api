"""Drum synthesis engine v2 — Step-based rendering with velocity, probability, retrig."""

import numpy as np
import logging
from typing import Any

from utils import get, should_trig_fire

logger = logging.getLogger("elastic-groove.drums")

SR = 44100


# ── Drum Synths ──

def synth_kick(sr: int = SR) -> np.ndarray:
    t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
    freq = 150 * np.exp(-t * 12) + 30
    phase = np.cumsum(2 * np.pi * freq / sr)
    env = np.exp(-t * 8)
    return np.sin(phase) * env * 0.9


def synth_snare(sr: int = SR) -> np.ndarray:
    length = int(sr * 0.15)
    t = np.linspace(0, 0.15, length, endpoint=False)
    noise = np.random.randn(length) * 0.4
    noise_env = np.exp(-t * 30)
    body = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 40) * 0.5
    return (noise * noise_env + body) * 0.8


def synth_hihat_closed(sr: int = SR) -> np.ndarray:
    length = int(sr * 0.05)
    t = np.linspace(0, 0.05, length, endpoint=False)
    return np.random.randn(length) * np.exp(-t * 80) * 0.3


def synth_hihat_open(sr: int = SR) -> np.ndarray:
    length = int(sr * 0.25)
    t = np.linspace(0, 0.25, length, endpoint=False)
    return np.random.randn(length) * np.exp(-t * 12) * 0.3


def synth_clap(sr: int = SR) -> np.ndarray:
    length = int(sr * 0.12)
    t = np.linspace(0, 0.12, length, endpoint=False)
    result = np.zeros(length)
    for i in range(3):
        offset = int(i * sr * 0.01)
        burst_len = min(int(sr * 0.03), length - offset)
        burst = np.random.randn(burst_len) * np.exp(-np.linspace(0, 1, burst_len) * 15)
        result[offset:offset + burst_len] += burst * 0.4
    return result * np.exp(-t * 25) * 0.7


def synth_perc(sr: int = SR) -> np.ndarray:
    length = int(sr * 0.08)
    t = np.linspace(0, 0.08, length, endpoint=False)
    freq = 800 * np.exp(-t * 30) + 200
    phase = np.cumsum(2 * np.pi * freq / sr)
    return np.sin(phase) * np.exp(-t * 40) * 0.5


SYNTHS = [synth_kick, synth_snare, synth_hihat_closed, synth_hihat_open, synth_clap, synth_perc]


def render_drums(
    tracks: list[Any],
    bpm: float = 120,
    swing: float = 0,
    mixer_channels: list[dict] | None = None,
    length: int = 16,
    loops: int = 2,
    is_fill: bool = False,
) -> np.ndarray:
    """Render drum tracks to a stereo numpy array."""

    step_duration = 60 / bpm / 4
    total_steps = length * loops
    total_duration = total_steps * step_duration
    total_samples = int(total_duration * SR) + SR

    mix = np.zeros((total_samples, 2))
    sounds = [s() for s in SYNTHS]

    for loop_idx in range(loops):
        for step_idx in range(length):
            global_step = loop_idx * length + step_idx
            swing_offset = (swing / 100) * step_duration * 0.5 if step_idx % 2 == 1 else 0
            step_time = global_step * step_duration + swing_offset

            for voice_idx, track in enumerate(tracks):
                if voice_idx >= len(sounds):
                    break
                steps = get(track, "steps", [])
                if step_idx >= len(steps):
                    continue

                step = steps[step_idx]
                if not get(step, "on", False):
                    continue

                if np.random.random() * 100 >= get(step, "probability", 100):
                    continue

                if not should_trig_fire(get(step, "condition", "always"), loop_idx, is_fill):
                    continue

                vel = get(step, "velocity", 100)
                vel_scale = vel / 100

                plock = get(step, "plock", {})
                plock_vol = get(plock, "volume")
                plock_pitch = get(plock, "pitch")
                plock_decay = get(plock, "decay")

                # Per-track swing (Phase 2d)
                track_swing = get(track, "swing")
                # Recalculate swing offset if per-track swing is set
                if track_swing is not None and step_idx % 2 == 1:
                    swing_offset = (track_swing / 100) * step_duration * 0.5
                    step_time = global_step * step_duration + swing_offset

                # Mixer channel
                ch_vol = 0.8
                ch_pan = 0.0
                ch_mute = False
                if mixer_channels and voice_idx < len(mixer_channels):
                    ch = mixer_channels[voice_idx]
                    ch_vol = ch.get("volume", 0.8)
                    ch_pan = ch.get("pan", 0.0)
                    ch_mute = ch.get("mute", False)

                if ch_mute:
                    continue

                vol = (plock_vol if plock_vol is not None else ch_vol) * vel_scale
                sound = sounds[voice_idx].copy()

                # Pitch P-Lock: resample to shift pitch
                if plock_pitch and plock_pitch != 0:
                    ratio = 2 ** (plock_pitch / 12)
                    orig_len = len(sound)
                    new_len = max(1, int(orig_len / ratio))
                    x_old = np.linspace(0, orig_len - 1, orig_len)
                    x_new = np.linspace(0, orig_len - 1, new_len)
                    sound = np.interp(x_new, x_old, sound)

                # Decay P-Lock: truncate + fade out
                if plock_decay is not None:
                    # 0.0 = 10% of original, 1.0 = full length
                    decay_frac = max(0.1, plock_decay)
                    cut_len = max(1, int(len(sound) * decay_frac))
                    sound = sound[:cut_len]
                    # Apply short fade-out (last 10%)
                    fade_len = max(1, int(cut_len * 0.1))
                    sound[-fade_len:] *= np.linspace(1, 0, fade_len)

                sound = sound * vol

                micro = get(step, "microTiming", 0)
                micro_offset = (micro / 100) * step_duration

                sample_offset = max(0, int((step_time + micro_offset) * SR))

                retrig = get(step, "retrig", 0)
                if retrig > 0:
                    retrig_interval = step_duration / retrig
                    for r in range(retrig):
                        r_offset = sample_offset + int(r * retrig_interval * SR)
                        r_vol = vol * (1 - r * 0.1)
                        r_sound = sounds[voice_idx] * max(0.1, r_vol)
                        end = min(r_offset + len(r_sound), total_samples)
                        slen = end - r_offset
                        if slen > 0 and r_offset >= 0:
                            l_gain = min(1.0, (1 - ch_pan))
                            r_gain = min(1.0, (1 + ch_pan))
                            mix[r_offset:end, 0] += r_sound[:slen] * l_gain
                            mix[r_offset:end, 1] += r_sound[:slen] * r_gain
                else:
                    end = min(sample_offset + len(sound), total_samples)
                    slen = end - sample_offset
                    if slen > 0 and sample_offset >= 0:
                        l_gain = min(1.0, (1 - ch_pan))
                        r_gain = min(1.0, (1 + ch_pan))
                        mix[sample_offset:end, 0] += sound[:slen] * l_gain
                        mix[sample_offset:end, 1] += sound[:slen] * r_gain

    logger.info(f"Rendered drums: {total_duration:.1f}s, {len(tracks)} tracks")
    return mix
