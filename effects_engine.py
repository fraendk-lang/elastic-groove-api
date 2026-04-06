"""Effects engine — Reverb, Delay, Chorus, Flanger. All vectorized NumPy processing."""

import numpy as np
import logging

logger = logging.getLogger("elastic-groove.effects")

SR = 44100


# ── Reverb (Schroeder algorithm) ──

# Preset delay times and decay factors per size
REVERB_PRESETS = {
    "small": {"comb_delays": [0.0297, 0.0371, 0.0411, 0.0437], "decay": 0.3, "predelay": 0.005},
    "medium": {"comb_delays": [0.0297, 0.0371, 0.0411, 0.0437], "decay": 0.7, "predelay": 0.01},
    "large": {"comb_delays": [0.0297, 0.0371, 0.0411, 0.0437], "decay": 1.2, "predelay": 0.02},
    "hall": {"comb_delays": [0.0437, 0.0531, 0.0611, 0.0673], "decay": 2.0, "predelay": 0.03},
}

ALLPASS_DELAYS = [0.0050, 0.0017]  # seconds


def _comb_filter(signal: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
    """Feedback comb filter."""
    output = np.zeros(len(signal))
    buf = np.zeros(delay_samples)
    buf_idx = 0
    for i in range(len(signal)):
        delayed = buf[buf_idx]
        output[i] = delayed
        buf[buf_idx] = signal[i] + delayed * feedback
        buf_idx = (buf_idx + 1) % delay_samples
    return output


def _allpass_filter(signal: np.ndarray, delay_samples: int, gain: float = 0.7) -> np.ndarray:
    """Allpass filter for diffusion."""
    output = np.zeros(len(signal))
    buf = np.zeros(delay_samples)
    buf_idx = 0
    for i in range(len(signal)):
        delayed = buf[buf_idx]
        output[i] = -gain * signal[i] + delayed
        buf[buf_idx] = signal[i] + gain * delayed
        buf_idx = (buf_idx + 1) % delay_samples
    return output


def apply_reverb(signal: np.ndarray, size: str = "medium", mix: float = 30, sr: int = SR) -> np.ndarray:
    """Apply Schroeder reverb to stereo signal.

    Args:
        signal: stereo array (N, 2)
        size: "small", "medium", "large", "hall"
        mix: dry/wet 0-100
    """
    preset = REVERB_PRESETS.get(size, REVERB_PRESETS["medium"])
    wet_gain = mix / 100
    dry_gain = 1.0 - wet_gain * 0.5  # Keep some dry signal

    # Process mono sum for reverb
    mono = (signal[:, 0] + signal[:, 1]) * 0.5

    # Pre-delay
    predelay_samples = int(preset["predelay"] * sr)
    if predelay_samples > 0:
        mono = np.concatenate([np.zeros(predelay_samples), mono])

    # Feedback for comb filters based on decay time
    # feedback ≈ 10^(-3 * delay / decay)
    decay_time = preset["decay"]

    # 4 parallel comb filters
    comb_outputs = []
    for delay in preset["comb_delays"]:
        delay_samples = max(1, int(delay * sr))
        feedback = 10 ** (-3 * delay / max(0.1, decay_time))
        feedback = min(0.98, feedback)
        comb_out = _comb_filter(mono, delay_samples, feedback)
        comb_outputs.append(comb_out)

    # Sum comb outputs
    max_len = max(len(c) for c in comb_outputs)
    wet = np.zeros(max_len)
    for c in comb_outputs:
        wet[:len(c)] += c
    wet *= 0.25  # Average

    # 2 series allpass filters
    for delay in ALLPASS_DELAYS:
        delay_samples = max(1, int(delay * sr))
        wet = _allpass_filter(wet, delay_samples, 0.7)

    # Trim to original length
    orig_len = signal.shape[0]
    wet = wet[:orig_len] if len(wet) >= orig_len else np.pad(wet, (0, orig_len - len(wet)))

    # Create stereo output with slight L/R decorrelation
    output = np.zeros_like(signal)
    output[:, 0] = signal[:, 0] * dry_gain + wet * wet_gain
    shift = int(0.001 * sr)  # 1ms offset for stereo width
    wet_r = np.roll(wet, shift)
    output[:, 1] = signal[:, 1] * dry_gain + wet_r * wet_gain

    return output


# ── Delay (BPM-synced) ──

DELAY_BEAT_FRACTIONS = {
    "1/4": 1.0,
    "1/8": 0.5,
    "1/16": 0.25,
    "dotted": 0.75,  # dotted 1/8
}


def apply_delay(
    signal: np.ndarray,
    bpm: float = 120,
    time: str = "1/8",
    feedback: float = 40,
    mix: float = 25,
    sr: int = SR,
) -> np.ndarray:
    """Apply BPM-synced stereo delay.

    Args:
        signal: stereo array (N, 2)
        bpm: tempo
        time: beat division
        feedback: 0-100
        mix: dry/wet 0-100
    """
    beat_fraction = DELAY_BEAT_FRACTIONS.get(time, 0.5)
    delay_seconds = (60 / bpm) * beat_fraction
    delay_samples = max(1, int(delay_seconds * sr))

    feedback_gain = feedback / 100
    wet_gain = mix / 100

    output = signal.copy()

    for ch in range(2):
        # Slight L/R offset for stereo width
        ch_delay = delay_samples + (ch * int(0.003 * sr))  # 3ms offset on R
        buf = np.zeros(ch_delay)
        buf_idx = 0
        wet = np.zeros(len(signal))

        for i in range(len(signal)):
            delayed = buf[buf_idx]
            wet[i] = delayed
            buf[buf_idx] = signal[i, ch] + delayed * feedback_gain
            buf_idx = (buf_idx + 1) % ch_delay

        output[:, ch] = signal[:, ch] + wet * wet_gain

    return output


# ── Chorus (modulated delay) ──

def apply_chorus(
    signal: np.ndarray,
    rate: float = 1.5,
    depth: float = 50,
    mix: float = 30,
    sr: int = SR,
) -> np.ndarray:
    """Apply chorus effect (modulated delay).

    Args:
        signal: stereo array (N, 2)
        rate: LFO speed in Hz
        depth: modulation depth 0-100
        mix: dry/wet 0-100
    """
    base_delay = 0.007  # 7ms base delay
    max_mod = 0.003 * (depth / 100)  # max modulation: 3ms at full depth
    wet_gain = mix / 100

    n = signal.shape[0]
    t = np.arange(n) / sr
    lfo = np.sin(2 * np.pi * rate * t)

    output = signal.copy()

    for ch in range(2):
        # Phase offset between L/R for stereo width
        phase_offset = ch * np.pi / 2
        ch_lfo = np.sin(2 * np.pi * rate * t + phase_offset)
        delay_mod = (base_delay + ch_lfo * max_mod) * sr

        # Variable delay via interpolation
        indices = np.arange(n) - delay_mod
        indices = np.clip(indices, 0, n - 1)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, n - 1)
        frac = indices - idx_floor

        wet = signal[idx_floor, ch] * (1 - frac) + signal[idx_ceil, ch] * frac
        output[:, ch] = signal[:, ch] * (1 - wet_gain * 0.5) + wet * wet_gain

    return output


# ── Flanger (short modulated delay + feedback) ──

def apply_flanger(
    signal: np.ndarray,
    rate: float = 0.5,
    depth: float = 60,
    feedback: float = 50,
    mix: float = 20,
    sr: int = SR,
) -> np.ndarray:
    """Apply flanger effect (short modulated delay with feedback).

    Args:
        signal: stereo array (N, 2)
        rate: LFO speed in Hz
        depth: modulation depth 0-100
        feedback: feedback 0-100
        mix: dry/wet 0-100
    """
    base_delay = 0.001  # 1ms base
    max_mod = 0.002 * (depth / 100)  # max 2ms modulation
    feedback_gain = feedback / 100 * 0.9  # cap at 0.9 to prevent blowup
    wet_gain = mix / 100

    n = signal.shape[0]
    t = np.arange(n) / sr

    output = np.zeros_like(signal)

    for ch in range(2):
        phase_offset = ch * np.pi / 3
        ch_lfo = np.sin(2 * np.pi * rate * t + phase_offset)
        delay_samples = ((base_delay + ch_lfo * max_mod) * sr).astype(int)
        delay_samples = np.clip(delay_samples, 1, int(0.005 * sr))

        # Process with feedback (requires sample-by-sample)
        buf = np.zeros(int(0.01 * sr))  # 10ms buffer max
        buf_len = len(buf)
        buf_idx = 0
        wet = np.zeros(n)

        for i in range(n):
            d = delay_samples[i]
            read_idx = (buf_idx - d) % buf_len
            delayed = buf[read_idx]
            wet[i] = delayed
            buf[buf_idx] = signal[i, ch] + delayed * feedback_gain
            buf_idx = (buf_idx + 1) % buf_len

        output[:, ch] = signal[:, ch] * (1 - wet_gain * 0.5) + wet * wet_gain

    return output
