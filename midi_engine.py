"""MIDI export engine v2 — 9-track MIDI with drums, bass, synth 1 & 2, chords."""

from midiutil import MIDIFile
import tempfile
import logging
import random
from typing import Any

from utils import get, should_trig_fire, CHORD_INTERVALS

logger = logging.getLogger("elastic-groove.midi")

# General MIDI Drum Map (Channel 10)
GM_DRUM_MAP = {
    0: 36,  # Kick → Bass Drum 1
    1: 38,  # Snare → Acoustic Snare
    2: 42,  # HH Closed → Closed Hi-Hat
    3: 46,  # HH Open → Open Hi-Hat
    4: 39,  # Clap → Hand Clap
    5: 56,  # Perc → Cowbell
    6: 37,  # Rimshot → Side Stick
    7: 49,  # Crash → Crash Cymbal 1
}


def render_midi(
    pattern: Any,
    bpm: float = 120,
    swing: float = 0,
    mixer_channels: list[dict] | None = None,
    loops: int = 4,
    is_fill: bool = False,
) -> str:
    """Render full pattern to a multi-track MIDI file."""

    length = get(pattern, "length", 16)
    step_duration = 0.25  # 16th note in beats

    # 4 tracks: Drums (ch10), Bass (ch1), Synth1 (ch2), Synth2 (ch3)
    midi = MIDIFile(4)
    midi.addTrackName(0, 0, "Drums")
    midi.addTrackName(1, 0, "Bass")
    midi.addTrackName(2, 0, "Synth 1")
    midi.addTrackName(3, 0, "Synth 2")
    for t in range(4):
        midi.addTempo(t, 0, bpm)

    drums = get(pattern, "drums", [])
    bass = get(pattern, "bass")
    synth = get(pattern, "synth")
    synth2 = get(pattern, "synth2")

    for loop_idx in range(loops):
        for step_idx in range(length):
            global_step = loop_idx * length + step_idx
            swing_offset = (swing / 100) * step_duration * 0.5 if step_idx % 2 == 1 else 0
            beat_time = global_step * step_duration + swing_offset

            # ── Drums (Track 0, Channel 9) ──
            for voice_idx, track in enumerate(drums):
                if voice_idx >= len(GM_DRUM_MAP):
                    break
                steps = get(track, "steps", [])
                if step_idx >= len(steps):
                    continue
                step = steps[step_idx]
                if not get(step, "on", False):
                    continue
                if not should_trig_fire(get(step, "condition", "always"), loop_idx, is_fill):
                    continue

                if random.random() * 100 >= get(step, "probability", 100):
                    continue

                # Per-track swing
                track_swing = get(track, "swing")
                effective_swing = track_swing if track_swing is not None else swing
                drum_swing_offset = (effective_swing / 100) * step_duration * 0.5 if step_idx % 2 == 1 else 0
                drum_beat_time = global_step * step_duration + drum_swing_offset

                vel = get(step, "velocity", 100)
                note = GM_DRUM_MAP[voice_idx]
                dur = step_duration * 0.8

                micro = get(step, "microTiming", 0)
                micro_offset = (micro / 100) * step_duration
                t = max(0, drum_beat_time + micro_offset)

                retrig = get(step, "retrig", 0)
                if retrig > 0:
                    retrig_interval = step_duration / retrig
                    for r in range(retrig):
                        rv = max(1, int(vel * (1 - r * 0.1)))
                        midi.addNote(0, 9, note, t + r * retrig_interval, dur / retrig, rv)
                else:
                    midi.addNote(0, 9, note, t, dur, vel)

            # ── Bass (Track 1, Channel 0) ──
            if bass:
                _add_melodic_notes(midi, 1, 0, bass, step_idx, beat_time, step_duration, loop_idx, swing, global_step, is_fill)

            # ── Synth 1 (Track 2, Channel 1) ──
            if synth:
                _add_melodic_notes(midi, 2, 1, synth, step_idx, beat_time, step_duration, loop_idx, swing, global_step, is_fill)

            # ── Synth 2 (Track 3, Channel 2) ──
            if synth2:
                _add_melodic_notes(midi, 3, 2, synth2, step_idx, beat_time, step_duration, loop_idx, swing, global_step, is_fill)

    output = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    output_path = output.name
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    total_beats = loops * length * step_duration
    logger.info(f"Rendered MIDI: {total_beats} beats, {loops} loops, 4 tracks")
    return output_path


def _add_melodic_notes(
    midi: MIDIFile,
    track_idx: int,
    channel: int,
    track: Any,
    step_idx: int,
    beat_time: float,
    step_duration: float,
    loop_idx: int,
    global_swing: float = 0,
    global_step: int = 0,
    is_fill: bool = False,
):
    """Add melodic notes (with chord expansion and per-track swing) to MIDI."""
    steps = get(track, "steps", [])
    if step_idx >= len(steps):
        return
    step = steps[step_idx]
    if not get(step, "on", False):
        return
    note = get(step, "note")
    if note is None:
        return
    if not should_trig_fire(get(step, "condition", "always"), loop_idx, is_fill):
        return

    if random.random() * 100 >= get(step, "probability", 100):
        return

    # Per-track swing
    track_swing = get(track, "swing")
    effective_swing = track_swing if track_swing is not None else global_swing
    swing_offset = (effective_swing / 100) * step_duration * 0.5 if step_idx % 2 == 1 else 0
    beat_time = global_step * step_duration + swing_offset

    vel = get(step, "velocity", 100)
    plock = get(step, "plock", {})
    pitch_offset = get(plock, "pitch", 0) or 0

    chord_type = get(track, "chord", "none")
    intervals = CHORD_INTERVALS.get(chord_type, [0])

    micro = get(step, "microTiming", 0)
    micro_offset = (micro / 100) * step_duration
    t = max(0, beat_time + micro_offset)
    dur = step_duration * 0.9

    # Arpeggiator (Phase 3c)
    arp_data = get(track, "arp")
    arp_on = arp_data and get(arp_data, "on", False)

    if arp_on and len(intervals) > 1:
        arp_mode = get(arp_data, "mode", "up")
        rate_str = get(arp_data, "rate", "1/16")
        arp_rate_frac = {"1/4": 4.0, "1/8": 2.0, "1/16": 1.0, "1/32": 0.5}.get(rate_str, 1.0)
        gate_str = get(arp_data, "gate", "medium")
        arp_gate = {"short": 0.25, "medium": 0.5, "long": 0.75}.get(gate_str, 0.5)

        arp_notes = sorted([note + pitch_offset + iv for iv in intervals])
        if arp_mode == "down":
            arp_notes = list(reversed(arp_notes))
        elif arp_mode == "updown":
            up = list(arp_notes)
            arp_notes = up + up[-2:0:-1]
        elif arp_mode == "random":
            random.shuffle(arp_notes)

        arp_step_dur = step_duration * arp_rate_frac
        arp_note_dur = arp_step_dur * arp_gate

        for arp_idx, arp_note in enumerate(arp_notes):
            arp_time = t + arp_idx * arp_step_dur
            midi.addNote(track_idx, channel, arp_note, arp_time, arp_note_dur, vel)
    else:
        for interval in intervals:
            midi.addNote(track_idx, channel, note + pitch_offset + interval, t, dur, vel)
