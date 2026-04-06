"""Stem mixer v3 — send/return effects routing, solo logic, master processing."""

import numpy as np
import soundfile as sf
import tempfile
import logging
from typing import Any

from effects_engine import apply_reverb, apply_delay, apply_chorus, apply_flanger

logger = logging.getLogger("elastic-groove.mixer")

SR = 44100


def mix_stems(
    stems: list[np.ndarray],
    master_vol: float = 0.8,
    mixer_channels: list[dict] | None = None,
    effects: Any | None = None,
    bpm: float = 120,
) -> str:
    """Mix multiple stereo stems with send/return effects routing.

    Flow:
      For each stem → apply channel vol/pan → dry bus + send to reverb/delay buses
      Reverb bus → apply reverb
      Delay bus → apply delay
      Dry + reverb return + delay return → master bus
      Master bus → chorus (if on) → flanger (if on) → master vol → normalize → WAV
    """
    if not stems:
        raise ValueError("No stems to mix")

    channels = mixer_channels or []
    max_len = max(s.shape[0] for s in stems)

    # ── Solo logic (Phase 2c) ──
    any_solo = any(ch.get("solo", False) for ch in channels)

    # ── Dry mix + send buses ──
    dry_mix = np.zeros((max_len, 2))
    reverb_bus = np.zeros((max_len, 2))
    delay_bus = np.zeros((max_len, 2))

    for i, stem in enumerate(stems):
        ch = channels[i] if i < len(channels) else {}

        # Solo/mute logic
        if any_solo and not ch.get("solo", False):
            continue
        if ch.get("mute", False):
            continue

        # Channel volume + pan (already applied during render, but mixer can override)
        ch_vol = ch.get("volume", 0.8)
        ch_pan = ch.get("pan", 0.0)

        slen = stem.shape[0]
        processed = np.zeros((max_len, 2))
        l_gain = min(1.0, (1 - ch_pan)) * ch_vol
        r_gain = min(1.0, (1 + ch_pan)) * ch_vol
        processed[:slen, 0] = stem[:slen, 0] * l_gain
        processed[:slen, 1] = stem[:slen, 1] * r_gain

        # Add to dry mix
        dry_mix += processed

        # Send to effect buses
        send_reverb = ch.get("sendReverb", 0) / 100
        send_delay = ch.get("sendDelay", 0) / 100

        if send_reverb > 0:
            reverb_bus += processed * send_reverb
        if send_delay > 0:
            delay_bus += processed * send_delay

    # ── Process effect buses ──
    fx = _parse_effects(effects)

    reverb_return = np.zeros((max_len, 2))
    delay_return = np.zeros((max_len, 2))

    if fx["reverb"]["on"] and np.any(reverb_bus):
        reverb_return = apply_reverb(
            reverb_bus,
            size=fx["reverb"].get("size", "medium"),
            mix=100,  # Full wet on return bus
        )

    if fx["delay"]["on"] and np.any(delay_bus):
        delay_return = apply_delay(
            delay_bus,
            bpm=bpm,
            time=fx["delay"].get("time", "1/8"),
            feedback=fx["delay"].get("feedback", 40),
            mix=100,  # Full wet on return bus
        )

    # ── Master bus ──
    master = dry_mix
    if fx["reverb"]["on"]:
        master = master + reverb_return * (fx["reverb"].get("mix", 30) / 100)
    if fx["delay"]["on"]:
        master = master + delay_return * (fx["delay"].get("mix", 25) / 100)

    # ── Master inserts: Chorus → Flanger ──
    if fx["chorus"]["on"]:
        master = apply_chorus(
            master,
            rate=fx["chorus"].get("rate", 1.5),
            depth=fx["chorus"].get("depth", 50),
            mix=fx["chorus"].get("mix", 30),
        )

    if fx["flanger"]["on"]:
        master = apply_flanger(
            master,
            rate=fx["flanger"].get("rate", 0.5),
            depth=fx["flanger"].get("depth", 60),
            feedback=fx["flanger"].get("feedback", 50),
            mix=fx["flanger"].get("mix", 20),
        )

    # ── Master volume ──
    master *= master_vol

    # ── Normalize to prevent clipping ──
    peak = np.max(np.abs(master))
    if peak > 0.95:
        master = master * (0.95 / peak)

    # ── Write WAV ──
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = output.name
    output.close()
    sf.write(output_path, master, SR, subtype="PCM_24")
    logger.info(f"Mixed {len(stems)} stems, {max_len / SR:.1f}s, peak={peak:.3f}, effects={'on' if any(fx[k]['on'] for k in fx) else 'off'}")

    return output_path


def _parse_effects(effects: Any | None) -> dict:
    """Parse effects data into a consistent dict with defaults."""
    defaults = {
        "reverb": {"on": False, "mix": 30, "size": "medium"},
        "delay": {"on": False, "mix": 25, "time": "1/8", "feedback": 40},
        "chorus": {"on": False, "mix": 30, "rate": 1.5, "depth": 50},
        "flanger": {"on": False, "mix": 20, "rate": 0.5, "depth": 60, "feedback": 50},
    }
    if effects is None:
        return defaults

    result = {}
    for key in defaults:
        effect_data = None
        if hasattr(effects, key):
            effect_data = getattr(effects, key)
        elif isinstance(effects, dict):
            effect_data = effects.get(key)

        if effect_data is None:
            result[key] = defaults[key]
        elif isinstance(effect_data, dict):
            result[key] = {**defaults[key], **effect_data}
        else:
            # Pydantic model → convert to dict
            d = defaults[key].copy()
            for field in defaults[key]:
                val = getattr(effect_data, field, None)
                if val is not None:
                    d[field] = val
            result[key] = d

    return result
