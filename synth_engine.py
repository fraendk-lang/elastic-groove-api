"""Synth/Bass synthesis engine v3 — biquad filter, configurable ADSR, bass modulation, LFO."""

import numpy as np
from scipy.signal import sosfilt, butter, sosfiltfilt
import logging
from typing import Any

from utils import get, note_to_freq, should_trig_fire, CHORD_INTERVALS, quantize_note

logger = logging.getLogger("elastic-groove.synth")

SR = 44100


# ── Biquad Filter (Phase 1a) ──

def _design_biquad(filter_type: str, freq: float, res: float, sr: int = SR):
    """Design a biquad filter, return second-order sections.

    Args:
        filter_type: "lowpass", "highpass", or "bandpass"
        freq: cutoff frequency in Hz
        res: resonance 0-1, mapped to Q 0.707-15
    """
    # Clamp frequency to valid range
    nyq = sr / 2
    freq = max(20, min(freq, nyq * 0.95))

    # Map resonance 0-1 → Q 0.707-15
    q = 0.707 + res * 14.293

    btype = {"lowpass": "low", "highpass": "high", "bandpass": "band"}.get(filter_type, "low")

    try:
        sos = butter(2, freq / nyq, btype=btype, output="sos")
        # Apply resonance by adjusting the Q of the second-order section
        # For a 2-pole filter, we modify the feedback coefficients
        if q > 0.707:
            # Recalculate with adjusted bandwidth for resonance
            w0 = 2 * np.pi * freq / sr
            alpha = np.sin(w0) / (2 * q)
            cos_w0 = np.cos(w0)

            if btype == "low":
                b0 = (1 - cos_w0) / 2
                b1 = 1 - cos_w0
                b2 = (1 - cos_w0) / 2
            elif btype == "high":
                b0 = (1 + cos_w0) / 2
                b1 = -(1 + cos_w0)
                b2 = (1 + cos_w0) / 2
            else:  # band
                b0 = alpha
                b1 = 0
                b2 = -alpha

            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha

            sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])
    except Exception:
        # Fallback to simple butterworth
        sos = butter(2, max(0.001, min(0.999, freq / nyq)), btype=btype, output="sos")

    return sos


def apply_filter(signal: np.ndarray, filter_type: str, freq: float, res: float, sr: int = SR) -> np.ndarray:
    """Apply a resonant biquad filter to a signal."""
    if len(signal) < 4:
        return signal
    sos = _design_biquad(filter_type, freq, res, sr)
    return sosfilt(sos, signal).astype(np.float64)


# ── Oscillators ──

def _oscillator(freq: float, duration: float, wave: str = "sine", sr: int = SR) -> np.ndarray:
    """Generate oscillator waveform."""
    length = int(sr * duration)
    t = np.linspace(0, duration, length, endpoint=False)
    phase = 2 * np.pi * freq * t

    if wave == "sawtooth":
        return 2 * (freq * t % 1) - 1
    elif wave == "square":
        return np.sign(np.sin(phase))
    elif wave == "triangle":
        return 2 * np.abs(2 * (freq * t % 1) - 1) - 1
    else:  # sine
        return np.sin(phase)


# ── ADSR Envelope (Phase 1b) ──

def _adsr_envelope(
    length: int,
    attack: float = 0.01,
    decay: float = 0.08,
    sustain: float = 0.6,
    release: float = 0.05,
    sr: int = SR,
) -> np.ndarray:
    """Generate ADSR envelope."""
    a_samples = min(int(sr * attack), length)
    d_samples = min(int(sr * decay), length - a_samples)
    r_samples = min(int(sr * release), length)
    s_samples = max(0, length - a_samples - d_samples - r_samples)

    env = np.zeros(length)

    # Attack
    if a_samples > 0:
        env[:a_samples] = np.linspace(0, 1, a_samples)

    # Decay
    if d_samples > 0:
        env[a_samples:a_samples + d_samples] = np.linspace(1, sustain, d_samples)

    # Sustain
    if s_samples > 0:
        env[a_samples + d_samples:a_samples + d_samples + s_samples] = sustain

    # Release
    release_start = length - r_samples
    if r_samples > 0 and release_start >= 0:
        start_level = env[release_start] if release_start > 0 else sustain
        env[release_start:] = np.linspace(start_level, 0, r_samples)

    return env


# ── Filter Envelope (Phase 1c) ──

def _filter_envelope(length: int, amount: float, attack: float, decay: float, sr: int = SR) -> np.ndarray:
    """Generate filter envelope: attack to peak, then exponential decay."""
    t = np.arange(length) / sr
    a_samples = int(sr * attack)

    env = np.zeros(length)
    # Attack: linear ramp to amount
    if a_samples > 0 and a_samples < length:
        env[:a_samples] = np.linspace(0, amount, a_samples)
    # Decay: exponential decay back to 0
    decay_start = min(a_samples, length)
    if decay_start < length:
        decay_t = t[decay_start:] - t[decay_start]
        env[decay_start:] = amount * np.exp(-decay_t * (3 / max(0.01, decay)))

    return env


# ── LFO (Phase 1c) ──

def _lfo_signal(length: int, rate: float, depth: float, sr: int = SR) -> np.ndarray:
    """Generate sine LFO signal (0 to depth)."""
    t = np.arange(length) / sr
    return depth * np.sin(2 * np.pi * rate * t)


# ── Synth Note (Phase 1a/1b combined) ──

def synth_note(
    note: int,
    wave: str = "sawtooth",
    filter_freq: float = 2000,
    filter_res: float = 0.0,
    filter_type: str = "lowpass",
    duration: float = 0.3,
    attack: float = 0.01,
    decay: float = 0.08,
    sustain: float = 0.6,
    release: float = 0.05,
    filter_env: dict | None = None,
    lfo: dict | None = None,
    sr: int = SR,
) -> np.ndarray:
    """Subtractive synth: oscillator → filter (with envelope/LFO) → ADSR."""
    freq = note_to_freq(note)
    length = int(sr * duration)
    if length < 1:
        return np.zeros(1)

    # Oscillator
    osc = _oscillator(freq, duration, wave, sr)

    # LFO modulation on pitch (regenerate osc if pitch LFO active)
    if lfo and lfo.get("target") == "pitch":
        lfo_sig = _lfo_signal(length, lfo.get("rate", 2.0), lfo.get("depth", 0.5), sr)
        # Modulate frequency by up to 2 semitones
        t = np.arange(length) / sr
        freq_mod = freq * (2 ** (lfo_sig * 2 / 12))
        phase = np.cumsum(2 * np.pi * freq_mod / sr)
        if wave == "sawtooth":
            osc = 2 * (np.cumsum(freq_mod / sr) % 1) - 1
        elif wave == "square":
            osc = np.sign(np.sin(phase))
        elif wave == "triangle":
            osc = 2 * np.abs(2 * (np.cumsum(freq_mod / sr) % 1) - 1) - 1
        else:
            osc = np.sin(phase)

    # Filter cutoff modulation
    effective_freq = filter_freq

    if filter_env:
        f_env = _filter_envelope(
            length,
            filter_env.get("amount", 2000),
            filter_env.get("attack", 0.01),
            filter_env.get("decay", 0.3),
            sr,
        )
        # Apply filter envelope per-block (simplified: use peak for static filter)
        # For more accurate: process in small blocks. Here we use a compromise:
        # apply filter at the average envelope-modulated frequency
        avg_mod = np.mean(f_env)
        effective_freq = filter_freq + avg_mod

    if lfo and lfo.get("target") == "filter":
        lfo_sig = _lfo_signal(length, lfo.get("rate", 2.0), lfo.get("depth", 0.5), sr)
        # Modulate filter by depth * cutoff range
        effective_freq = effective_freq + lfo_sig.mean() * effective_freq * 0.5

    # Apply filter
    filtered = apply_filter(osc, filter_type, effective_freq, filter_res, sr)

    # ADSR envelope
    env = _adsr_envelope(length, attack, decay, sustain, release, sr)
    result = filtered * env * 0.4

    # LFO on volume (tremolo)
    if lfo and lfo.get("target") == "volume":
        lfo_sig = _lfo_signal(length, lfo.get("rate", 2.0), lfo.get("depth", 0.5), sr)
        result *= (1 - lfo_sig * 0.5)  # Modulate between 50%-100%

    return result


# ── Bass Note (Phase 1c: wave support, filter env, LFO) ──

def synth_bass_note(
    note: int,
    wave: str = "sine",
    filter_freq: float = 800,
    filter_res: float = 0.0,
    filter_type: str = "lowpass",
    duration: float = 0.4,
    attack: float = 0.005,
    decay: float = 0.15,
    sustain: float = 0.4,
    release: float = 0.05,
    filter_env: dict | None = None,
    lfo: dict | None = None,
    sr: int = SR,
) -> np.ndarray:
    """Bass synth: oscillator → filter → ADSR. Supports all waveforms + modulation."""
    freq = note_to_freq(note)
    length = int(sr * duration)
    if length < 1:
        return np.zeros(1)

    t = np.linspace(0, duration, length, endpoint=False)

    # Pitch drop for bass character
    freq_env = freq * (1 - 0.03 * t / duration)

    # LFO pitch modulation
    if lfo and lfo.get("target") == "pitch":
        lfo_sig = _lfo_signal(length, lfo.get("rate", 2.0), lfo.get("depth", 0.5), sr)
        freq_env = freq_env * (2 ** (lfo_sig * 2 / 12))

    # Oscillator with pitch envelope
    phase = np.cumsum(2 * np.pi * freq_env / sr)
    if wave == "sawtooth":
        osc = 2 * (np.cumsum(freq_env / sr) % 1) - 1
    elif wave == "square":
        osc = np.sign(np.sin(phase))
    elif wave == "triangle":
        osc = 2 * np.abs(2 * (np.cumsum(freq_env / sr) % 1) - 1) - 1
    else:  # sine (default for bass)
        osc = np.sin(phase)

    # Filter with envelope
    effective_freq = filter_freq
    if filter_env:
        f_env = _filter_envelope(
            length,
            filter_env.get("amount", 2000),
            filter_env.get("attack", 0.01),
            filter_env.get("decay", 0.3),
            sr,
        )
        effective_freq = filter_freq + np.mean(f_env)

    if lfo and lfo.get("target") == "filter":
        lfo_sig = _lfo_signal(length, lfo.get("rate", 2.0), lfo.get("depth", 0.5), sr)
        effective_freq = effective_freq + np.mean(lfo_sig) * effective_freq * 0.5

    # Apply filter
    filtered = apply_filter(osc, filter_type, effective_freq, filter_res, sr)

    # ADSR
    env = _adsr_envelope(length, attack, decay, sustain, release, sr)
    result = filtered * env * 0.7

    # Volume LFO
    if lfo and lfo.get("target") == "volume":
        lfo_sig = _lfo_signal(length, lfo.get("rate", 2.0), lfo.get("depth", 0.5), sr)
        result *= (1 - lfo_sig * 0.5)

    return result


# ── Render Functions ──

def render_bass(
    track: Any,
    bpm: float = 120,
    swing: float = 0,
    mixer_channel: dict | None = None,
    length: int = 16,
    loops: int = 2,
    is_fill: bool = False,
    quantize: bool = False,
    global_key: str = "C",
    global_scale: str = "chromatic",
) -> np.ndarray:
    """Render bass track to stereo numpy array."""
    return _render_melodic(track, bpm, swing, mixer_channel, length, loops,
                           is_bass=True, is_fill=is_fill,
                           quantize=quantize, global_key=global_key, global_scale=global_scale)


def render_synth(
    track: Any,
    bpm: float = 120,
    swing: float = 0,
    mixer_channel: dict | None = None,
    length: int = 16,
    loops: int = 2,
    is_fill: bool = False,
    quantize: bool = False,
    global_key: str = "C",
    global_scale: str = "chromatic",
) -> np.ndarray:
    """Render synth track to stereo numpy array."""
    return _render_melodic(track, bpm, swing, mixer_channel, length, loops,
                           is_bass=False, is_fill=is_fill,
                           quantize=quantize, global_key=global_key, global_scale=global_scale)


def _render_melodic(
    track: Any,
    bpm: float,
    swing: float,
    mixer_channel: dict | None,
    length: int,
    loops: int,
    is_bass: bool,
    is_fill: bool = False,
    quantize: bool = False,
    global_key: str = "C",
    global_scale: str = "chromatic",
) -> np.ndarray:
    step_duration = 60 / bpm / 4
    total_steps = length * loops
    total_duration = total_steps * step_duration
    total_samples = int(total_duration * SR) + SR

    mix = np.zeros((total_samples, 2))

    # Track properties
    chord_type = get(track, "chord", "none")
    wave = get(track, "wave", "sine" if is_bass else "sawtooth")
    filter_freq = get(track, "filterFreq", 800 if is_bass else 2000)
    filter_res = get(track, "filterRes", 0.0)
    filter_type = get(track, "filterType", "lowpass")
    steps = get(track, "steps", [])

    # ADSR from track (Phase 1b)
    t_attack = get(track, "attack", 0.005 if is_bass else 0.01)
    t_decay = get(track, "decay", 0.15 if is_bass else 0.08)
    t_sustain = get(track, "sustain", 0.4 if is_bass else 0.6)
    t_release = get(track, "release", 0.05)

    # Filter envelope (Phase 1c)
    filter_env_data = get(track, "filterEnv")
    filter_env = None
    if filter_env_data:
        filter_env = {
            "amount": get(filter_env_data, "amount", 2000),
            "attack": get(filter_env_data, "attack", 0.01),
            "decay": get(filter_env_data, "decay", 0.3),
        }

    # LFO (Phase 1c)
    lfo_data = get(track, "lfo")
    lfo = None
    if lfo_data:
        lfo = {
            "rate": get(lfo_data, "rate", 2.0),
            "depth": get(lfo_data, "depth", 0.5),
            "target": get(lfo_data, "target", "filter"),
        }

    # Per-track swing (Phase 2d)
    track_swing = get(track, "swing")
    effective_swing = track_swing if track_swing is not None else swing

    # Mixer
    ch_vol = 0.8
    ch_pan = 0.0
    ch_mute = False
    if mixer_channel:
        ch_vol = mixer_channel.get("volume", 0.8)
        ch_pan = mixer_channel.get("pan", 0.0)
        ch_mute = mixer_channel.get("mute", False)

    if ch_mute:
        return mix

    chord_intervals = CHORD_INTERVALS.get(chord_type, [0])

    # Arpeggiator (Phase 3c)
    arp_data = get(track, "arp")
    arp_on = False
    arp_mode = "up"
    arp_rate_frac = 1.0  # fraction of step_duration
    arp_gate = 0.5
    if arp_data:
        arp_on = get(arp_data, "on", False)
        arp_mode = get(arp_data, "mode", "up")
        rate_str = get(arp_data, "rate", "1/16")
        arp_rate_frac = {"1/4": 4.0, "1/8": 2.0, "1/16": 1.0, "1/32": 0.5}.get(rate_str, 1.0)
        gate_str = get(arp_data, "gate", "medium")
        arp_gate = {"short": 0.25, "medium": 0.5, "long": 0.75}.get(gate_str, 0.5)

    for loop_idx in range(loops):
        for step_idx in range(length):
            if step_idx >= len(steps):
                continue

            step = steps[step_idx]
            if not get(step, "on", False):
                continue
            note = get(step, "note")
            if note is None:
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
            plock_filter = get(plock, "filterFreq")
            plock_filter_res = get(plock, "filterRes")
            plock_pan = get(plock, "pan")

            vol = (plock_vol if plock_vol is not None else ch_vol) * vel_scale
            actual_note = note + (plock_pitch or 0)
            if quantize and global_scale != "chromatic":
                actual_note = quantize_note(actual_note, global_key, global_scale)
            actual_filter = plock_filter if plock_filter is not None else filter_freq
            actual_res = plock_filter_res if plock_filter_res is not None else filter_res
            pan = plock_pan if plock_pan is not None else ch_pan

            global_step = loop_idx * length + step_idx
            swing_offset = (effective_swing / 100) * step_duration * 0.5 if step_idx % 2 == 1 else 0
            micro = get(step, "microTiming", 0)
            micro_offset = (micro / 100) * step_duration
            step_time = global_step * step_duration + swing_offset + micro_offset

            sample_offset = max(0, int(step_time * SR))

            # Arpeggiator: spread chord notes over time
            if arp_on and len(chord_intervals) > 1:
                arp_notes = [actual_note + iv for iv in chord_intervals]

                # Order notes by mode
                if arp_mode == "up":
                    ordered = sorted(arp_notes)
                elif arp_mode == "down":
                    ordered = sorted(arp_notes, reverse=True)
                elif arp_mode == "updown":
                    up = sorted(arp_notes)
                    ordered = up + up[-2:0:-1]  # up then down (no repeat of top/bottom)
                else:  # random
                    ordered = list(arp_notes)
                    np.random.shuffle(ordered)

                arp_step_dur = step_duration * arp_rate_frac
                arp_note_dur = arp_step_dur * arp_gate

                for arp_idx, arp_note in enumerate(ordered):
                    arp_time_offset = arp_idx * arp_step_dur
                    arp_sample = sample_offset + int(arp_time_offset * SR)

                    note_dur = min(arp_note_dur, t_attack + t_decay + 0.1 + t_release)
                    if is_bass:
                        sound = synth_bass_note(
                            arp_note, wave=wave, filter_freq=actual_filter,
                            filter_res=actual_res, filter_type=filter_type,
                            duration=note_dur, attack=t_attack, decay=t_decay,
                            sustain=t_sustain, release=t_release,
                            filter_env=filter_env, lfo=lfo,
                        ) * vol
                    else:
                        sound = synth_note(
                            arp_note, wave=wave, filter_freq=actual_filter,
                            filter_res=actual_res, filter_type=filter_type,
                            duration=note_dur, attack=t_attack, decay=t_decay,
                            sustain=t_sustain, release=t_release,
                            filter_env=filter_env, lfo=lfo,
                        ) * vol

                    end = min(arp_sample + len(sound), total_samples)
                    slen = end - arp_sample
                    if slen > 0 and arp_sample >= 0:
                        l_gain = min(1.0, (1 - pan))
                        r_gain = min(1.0, (1 + pan))
                        mix[arp_sample:end, 0] += sound[:slen] * l_gain
                        mix[arp_sample:end, 1] += sound[:slen] * r_gain
            else:
                # Normal chord stacking
                note_dur = min(t_attack + t_decay + 0.1 + t_release, step_duration)
                for interval in chord_intervals:
                    chord_note = actual_note + interval
                    if is_bass:
                        sound = synth_bass_note(
                            chord_note, wave=wave, filter_freq=actual_filter,
                            filter_res=actual_res, filter_type=filter_type,
                            duration=note_dur, attack=t_attack, decay=t_decay,
                            sustain=t_sustain, release=t_release,
                            filter_env=filter_env, lfo=lfo,
                        ) * vol
                    else:
                        sound = synth_note(
                            chord_note, wave=wave, filter_freq=actual_filter,
                            filter_res=actual_res, filter_type=filter_type,
                            duration=note_dur, attack=t_attack, decay=t_decay,
                            sustain=t_sustain, release=t_release,
                            filter_env=filter_env, lfo=lfo,
                        ) * vol

                    end = min(sample_offset + len(sound), total_samples)
                    slen = end - sample_offset
                    if slen > 0:
                        l_gain = min(1.0, (1 - pan))
                        r_gain = min(1.0, (1 + pan))
                        mix[sample_offset:end, 0] += sound[:slen] * l_gain
                        mix[sample_offset:end, 1] += sound[:slen] * r_gain

    track_name = "bass" if is_bass else "synth"
    logger.info(f"Rendered {track_name}: {total_duration:.1f}s, chord={chord_type}, filter={filter_type}")
    return mix
