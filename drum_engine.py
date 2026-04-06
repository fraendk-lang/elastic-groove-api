"""Drum synthesis engine — generates WAV from pattern data using numpy."""

import numpy as np
import soundfile as sf
import tempfile
import logging

logger = logging.getLogger("elastic-groove.drums")

SR = 44100  # Sample rate


def synth_kick(sr: int = SR) -> np.ndarray:
    """808-style kick: sine with pitch envelope."""
    t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
    freq = 150 * np.exp(-t * 12) + 30
    phase = np.cumsum(2 * np.pi * freq / sr)
    env = np.exp(-t * 8)
    return np.sin(phase) * env * 0.9


def synth_snare(sr: int = SR) -> np.ndarray:
    """Snare: noise burst + sine body."""
    length = int(sr * 0.15)
    t = np.linspace(0, 0.15, length, endpoint=False)
    # Noise
    noise = np.random.randn(length) * 0.4
    noise_env = np.exp(-t * 30)
    # Body
    body = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 40) * 0.5
    return (noise * noise_env + body) * 0.8


def synth_hihat_closed(sr: int = SR) -> np.ndarray:
    """Closed hi-hat: short filtered noise."""
    length = int(sr * 0.05)
    t = np.linspace(0, 0.05, length, endpoint=False)
    noise = np.random.randn(length)
    env = np.exp(-t * 80)
    return noise * env * 0.3


def synth_hihat_open(sr: int = SR) -> np.ndarray:
    """Open hi-hat: longer filtered noise."""
    length = int(sr * 0.25)
    t = np.linspace(0, 0.25, length, endpoint=False)
    noise = np.random.randn(length)
    env = np.exp(-t * 12)
    return noise * env * 0.3


def synth_clap(sr: int = SR) -> np.ndarray:
    """Clap: layered noise bursts."""
    length = int(sr * 0.12)
    t = np.linspace(0, 0.12, length, endpoint=False)
    result = np.zeros(length)
    for i in range(3):
        offset = int(i * sr * 0.01)
        burst_len = min(int(sr * 0.03), length - offset)
        burst = np.random.randn(burst_len) * np.exp(-np.linspace(0, 1, burst_len) * 15)
        result[offset:offset + burst_len] += burst * 0.4
    env = np.exp(-t * 25)
    return result * env * 0.7


def synth_perc(sr: int = SR) -> np.ndarray:
    """Percussion: metallic click."""
    length = int(sr * 0.08)
    t = np.linspace(0, 0.08, length, endpoint=False)
    freq = 800 * np.exp(-t * 30) + 200
    phase = np.cumsum(2 * np.pi * freq / sr)
    env = np.exp(-t * 40)
    return np.sin(phase) * env * 0.5


SYNTHS = [synth_kick, synth_snare, synth_hihat_closed, synth_hihat_open, synth_clap, synth_perc]


def render_pattern(
    pattern: list[list[bool]],
    bpm: float = 120,
    swing: float = 0,
    track_vols: list[float] | None = None,
    master_vol: float = 0.8,
    loops: int = 2,
) -> str:
    """Render a drum pattern to a WAV file. Returns the file path."""

    if track_vols is None:
        track_vols = [0.9, 0.8, 0.6, 0.6, 0.7, 0.5]

    step_duration = 60 / bpm / 4  # 16th note duration in seconds
    total_steps = 16 * loops
    total_duration = total_steps * step_duration
    total_samples = int(total_duration * SR) + SR  # extra second for tails

    mix = np.zeros(total_samples)

    # Pre-render all drum sounds
    sounds = [s() for s in SYNTHS]

    for loop in range(loops):
        for step in range(16):
            global_step = loop * 16 + step
            # Swing: offset odd steps
            swing_offset = (swing / 100) * step_duration * 0.5 if step % 2 == 1 else 0
            step_time = global_step * step_duration + swing_offset
            sample_offset = int(step_time * SR)

            for voice in range(min(len(pattern), len(sounds))):
                if voice < len(pattern) and step < len(pattern[voice]) and pattern[voice][step]:
                    sound = sounds[voice] * track_vols[voice] * master_vol
                    end = min(sample_offset + len(sound), total_samples)
                    length = end - sample_offset
                    if length > 0:
                        mix[sample_offset:end] += sound[:length]

    # Normalize to prevent clipping
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    # Write to temp file
    output_path = tempfile.mktemp(suffix=".wav")
    sf.write(output_path, mix, SR, subtype="PCM_24")
    logger.info(f"Rendered WAV: {total_duration:.1f}s, {total_steps} steps, {bpm} BPM")

    return output_path
