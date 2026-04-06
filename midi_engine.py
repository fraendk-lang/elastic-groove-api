"""MIDI export engine — generates Standard MIDI File from pattern data."""

from midiutil import MIDIFile
import tempfile
import logging

logger = logging.getLogger("elastic-groove.midi")

# General MIDI Drum Map (Channel 10)
GM_DRUM_MAP = {
    0: 36,  # Kick → Bass Drum 1
    1: 38,  # Snare → Acoustic Snare
    2: 42,  # HH Closed → Closed Hi-Hat
    3: 46,  # HH Open → Open Hi-Hat
    4: 39,  # Clap → Hand Clap
    5: 56,  # Perc → Cowbell (or 37 for Side Stick)
}

VOICE_NAMES = ["Kick", "Snare", "HH-C", "HH-O", "Clap", "Perc"]


def render_midi(
    pattern: list[list[bool]],
    bpm: float = 120,
    swing: float = 0,
    track_vols: list[float] | None = None,
    loops: int = 4,
) -> str:
    """Render a drum pattern to a MIDI file. Returns the file path."""

    if track_vols is None:
        track_vols = [0.9, 0.8, 0.6, 0.6, 0.7, 0.5]

    midi = MIDIFile(1)  # 1 track
    track = 0
    channel = 9  # MIDI channel 10 (0-indexed = 9) for drums
    time = 0

    midi.addTrackName(track, time, "Elastic Groove Drums")
    midi.addTempo(track, time, bpm)

    step_duration = 0.25  # 16th note in beats (1 beat = quarter note)

    for loop in range(loops):
        for step in range(16):
            global_step = loop * 16 + step
            # Swing offset in beats
            swing_offset = (swing / 100) * step_duration * 0.5 if step % 2 == 1 else 0
            beat_time = global_step * step_duration + swing_offset

            for voice in range(min(len(pattern), len(GM_DRUM_MAP))):
                if voice < len(pattern) and step < len(pattern[voice]) and pattern[voice][step]:
                    note = GM_DRUM_MAP[voice]
                    velocity = int(track_vols[voice] * 127)
                    velocity = max(1, min(127, velocity))
                    duration = step_duration * 0.8  # slightly shorter than full step
                    midi.addNote(track, channel, note, beat_time, duration, velocity)

    # Write to temp file
    output_path = tempfile.mktemp(suffix=".mid")
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    total_beats = loops * 16 * step_duration
    logger.info(f"Rendered MIDI: {total_beats} beats, {loops} loops, {bpm} BPM")

    return output_path
