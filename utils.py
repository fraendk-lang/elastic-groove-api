"""Shared utilities for Elastic Groove API engines."""

from typing import Any


# ── Chord Intervals (matching frontend CHORD_INTERVALS) ──

CHORD_INTERVALS = {
    "none": [0],
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "7th": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "power": [0, 7],
}


def get(obj: Any, attr: str, default=None):
    """Get attribute from Pydantic model or dict, with fallback to default."""
    if hasattr(obj, attr):
        val = getattr(obj, attr)
        return val if val is not None else default
    if isinstance(obj, dict):
        val = obj.get(attr, default)
        return val if val is not None else default
    return default


def note_to_freq(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440 * (2 ** ((note - 69) / 12))


# ── Scale Definitions (matching frontend scales.ts) ──

SCALES = {
    "chromatic":    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major":        [0, 2, 4, 5, 7, 9, 11],
    "minor":        [0, 2, 3, 5, 7, 8, 10],
    "dorian":       [0, 2, 3, 5, 7, 9, 10],
    "phrygian":     [0, 1, 3, 5, 7, 8, 10],
    "lydian":       [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":   [0, 2, 4, 5, 7, 9, 10],
    "aeolian":      [0, 2, 3, 5, 7, 8, 10],
    "locrian":      [0, 1, 3, 5, 6, 8, 10],
    "pentatonic":   [0, 2, 4, 7, 9],
    "blues":        [0, 3, 5, 6, 7, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
}

NOTE_NAME_TO_OFFSET = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def quantize_note(midi_note: int, key: str = "C", scale: str = "chromatic") -> int:
    """Quantize a MIDI note to the nearest note in the given scale + key."""
    if scale == "chromatic":
        return midi_note

    key_offset = NOTE_NAME_TO_OFFSET.get(key, 0)
    intervals = SCALES.get(scale, SCALES["chromatic"])

    # Build set of valid pitch classes
    valid_pcs = set((key_offset + i) % 12 for i in intervals)

    note_pc = midi_note % 12
    if note_pc in valid_pcs:
        return midi_note

    # Find nearest valid pitch class
    best_dist = 12
    best_note = midi_note
    for offset in range(1, 7):
        if (note_pc + offset) % 12 in valid_pcs:
            best_note = midi_note + offset
            best_dist = offset
            break
    for offset in range(1, 7):
        if (note_pc - offset) % 12 in valid_pcs:
            if offset < best_dist:
                best_note = midi_note - offset
            break

    return best_note


def should_trig_fire(condition: str, repeat_count: int, is_fill: bool = False) -> bool:
    """Check if a conditional trig should fire."""
    if condition == "always":
        return True
    if condition == "1:2":
        return repeat_count % 2 == 0
    if condition == "2:2":
        return repeat_count % 2 == 1
    if condition == "1:3":
        return repeat_count % 3 == 0
    if condition == "1:4":
        return repeat_count % 4 == 0
    if condition == "fill":
        return is_fill
    if condition == "!fill":
        return not is_fill
    if condition == "pre":
        return repeat_count == 0
    if condition == "!pre":
        return repeat_count > 0
    return True
