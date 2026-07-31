# Elastic Groove API — Improvement Plan (v2 → v3)

## Context

The Elastic Groove API is a Python/FastAPI backend that synthesizes audio from step-sequencer patterns (Elektron Digitakt/Digitone inspired). It renders WAV and MIDI exports from JSON pattern data.

**Current state (v2.0.0):**
- 2 endpoints: `POST /export/wav` (24-bit stereo WAV), `POST /export/midi` (4-track MIDI)
- 6 drum synths (kick, snare, closed HH, open HH, clap, perc) — all pure NumPy synthesis
- 808-style sine bass (fixed waveform, no modulation)
- Subtractive synth (saw/square/tri/sine → 1st-order IIR filter → hardcoded ADSR)
- Optional synth2 track
- Step features: velocity, probability, conditional trigs, retrig, micro-timing, swing, p-lock (volume/pitch/filter/pan), chord expansion (11 types)
- Mixer: stem summing + master volume + peak normalize. No effects.

**Frontend** (`elastic-universe-landing/src/app/tools/groove/`) already defines types for features the API doesn't support yet: Effects (Reverb/Delay/Chorus/Flanger), Mixer Sends, Solo, Fill Trigs, Per-Track Swing, Arpeggiator.

**Tech stack:** Python 3.11, FastAPI 0.115, NumPy 1.26, soundfile 0.12, midiutil 1.2. Deployed on Railway.

**Constraint:** All new request fields MUST be Optional with sensible defaults so existing frontend requests continue to work without changes.

---

## Project Structure (current)

```
elastic-groove-api/
  main.py           # FastAPI app, routes, Pydantic models
  drum_engine.py    # 6 drum synth functions + step sequencer rendering
  synth_engine.py   # Bass + synth synthesis, chord expansion, filter, ADSR
  midi_engine.py    # 4-track MIDI export
  mixer.py          # Stem summing, master volume, WAV output
  requirements.txt
  Procfile           # Railway deployment
  runtime.txt        # Python 3.11.8
```

---

## Phase 0: Cleanup & Architecture

**Goal:** Clean foundation before adding features. Do this phase first, completely, before moving to Phase 1.

### 0a) Shared Utilities — new file `utils.py`

Extract duplicated code from all three engines:

- `should_trig_fire(condition, repeat_count, is_fill=False)` — currently duplicated in `drum_engine.py`, `synth_engine.py`, `midi_engine.py`. Add `is_fill` parameter (needed for Phase 3).
- `note_to_freq(note)` — currently in `synth_engine.py`, also needed elsewhere.
- `CHORD_INTERVALS` dict — currently duplicated in all three engine files.
- `_get(obj, attr, default)` helper — currently only in `midi_engine.py`, needed in all engines for consistent Pydantic/dict property access.

All engines should import from `utils.py` instead of defining their own copies.

### 0b) Consistent Property Access

Replace the `hasattr`/`getattr`/`.get()` pattern scattered throughout `drum_engine.py` and `synth_engine.py` with the `_get()` helper from `utils.py`. Example of current inconsistent pattern:

```python
# Current (ugly, repeated everywhere):
on = step.on if hasattr(step, "on") else step.get("on", False)

# Target (clean):
on = _get(step, "on", False)
```

### 0c) Tempfile Fix

Replace `tempfile.mktemp()` (race condition) with `tempfile.NamedTemporaryFile(delete=False, suffix=".wav")` in `mixer.py` and `midi_engine.py`.

### 0d) Request Model Preparation

Add new Optional fields to the Pydantic models in `main.py` (all with defaults matching current behavior). This prepares the data model for Phases 1–3 without breaking existing requests. The specific fields are listed in their respective phases below.

---

## Phase 1: Sound Engine Upgrade

**Goal:** Dramatic sound quality improvement. This is the highest priority phase.

### 1a) Resonant Biquad Filter — `synth_engine.py`

Replace the 1st-order IIR filter (Python for-loop, no resonance) with a proper 2-pole biquad filter.

**Implementation:**
- Use `scipy.signal.sosfilt` with second-order sections for vectorized processing (no Python for-loop)
- Add `scipy` to `requirements.txt`
- Support filter types: `lowpass` (default), `highpass`, `bandpass`
- Parameters: `filterFreq` (Hz, already exists) + new `filterRes` (0.0–1.0, maps to Q range 0.5–15)
- New optional fields on `MelodicTrackData`: `filterRes: float | None = None`, `filterType: str | None = None`
- P-lock support: `PLockData` gets new optional `filterRes: float | None = None`
- Apply to both synth AND bass tracks

**Biquad coefficient calculation:**
- Lowpass: standard cookbook formula with frequency and Q
- Q mapping: `filterRes` 0→Q=0.707 (Butterworth, no resonance), 1→Q=15 (heavy resonance)

### 1b) Configurable ADSR Envelope — `synth_engine.py`

Current hardcoded values: attack=10ms, decay=80ms, sustain=0.6, release=50ms.

**New optional fields on `MelodicTrackData`:**
```python
attack: float | None = None    # seconds, default 0.01
decay: float | None = None     # seconds, default 0.08
sustain: float | None = None   # 0-1, default 0.6
release: float | None = None   # seconds, default 0.05
```

These defaults preserve exact current behavior. The ADSR envelope generator in `synth_note()` reads from track params instead of hardcoded constants.

**Note duration:** Currently fixed at 0.3s (synth) / 0.4s (bass). Should be derived from `min(attack + decay + 0.1 + release, step_duration)` — i.e. the ADSR timing determines the actual sound length, capped at one step. This makes ADSR settings actually audible (a pad with 0.5s attack won't get cut off at 0.3s).

### 1c) Bass Modulation — `synth_engine.py`

Three improvements to `synth_bass_note()` / bass rendering:

**1. `wave` support:** Bass currently always uses sine, ignoring the `wave` field. Route through the same oscillator selection as synth (saw/square/tri/sine). Keep sine as default for backwards compatibility.

**2. Filter Envelope:** New optional parameter block on `MelodicTrackData`:
```python
filterEnv: dict | None = None
# Structure: { "amount": float (Hz), "attack": float (s), "decay": float (s) }
# Defaults: None (no filter envelope, current behavior)
```

When set, the filter cutoff is modulated over time:
- `cutoff(t) = baseFreq + amount * envelope(t)`
- Envelope shape: linear attack to peak, then exponential decay back to baseFreq
- This creates the classic "wah" / acid bass character

**3. LFO:** New optional parameter block on `MelodicTrackData`:
```python
lfo: dict | None = None
# Structure: { "rate": float (Hz), "depth": float (0-1), "target": str ("pitch"|"filter"|"volume") }
# Defaults: None (no LFO, current behavior)
```

- `pitch`: modulates frequency by `depth * 2` semitones (vibrato)
- `filter`: modulates filter cutoff by `depth * cutoff_range` (wah)
- `volume`: modulates amplitude by `depth` (tremolo)
- Waveform: sine (simple, predictable)
- Applied per-sample during synthesis

### 1d) Drum Improvements — `drum_engine.py`

**Pitch P-Lock:** Currently `plock.pitch` is ignored for drums. Implementation:
- Read `plock.pitch` (semitones) per step
- Resample the pre-rendered drum sound using `numpy.interp` to shift pitch
- Pitch shift ratio: `2 ** (pitch / 12)`
- This affects all 6 drum voices (tune kicks lower, hihats higher, etc.)

**Decay P-Lock:** Currently `plock.decay` exists in the model but is never read. Implementation:
- Read `plock.decay` (0.0–1.0) per step
- Scale the drum sound length: `sound[:int(len(sound) * decay)]` with a fade-out
- 0.0 = very short (10% of original), 1.0 = full length
- Default (None) = full length (current behavior)

### 1e) Velocity Fix — all engines

**Bug:** Frontend sends velocity 0–100, engines divide by 127. Max velocity (100) only reaches 78.7% amplitude.

**Fix:** Divide by 100 instead of 127. This is a one-line change in `drum_engine.py` and `synth_engine.py`:
```python
# Before:
vel_scale = vel / 127
# After:
vel_scale = vel / 100
```

---

## Phase 2: Effects & Mixer

**Goal:** Effects chain matching the frontend type system. Send-based routing.

### 2a) Effects Engine — new file `effects_engine.py`

All effects operate on stereo NumPy arrays. All processing vectorized (no Python for-loops).

**Reverb** (Schroeder algorithm):
- 4 parallel comb filters + 2 series allpass filters
- `size` parameter maps to preset values:
  - `"small"`: room (short delays, fast decay ~0.3s)
  - `"medium"`: studio (moderate delays, ~0.7s decay)
  - `"large"`: hall-ish (~1.2s decay)
  - `"hall"`: cathedral (~2.0s decay)
- `mix`: 0–100 dry/wet balance
- Pre-delay derived from size

**Delay** (BPM-synced buffer):
- Circular buffer with feedback loop
- `time` maps to beat divisions: `"1/4"` = quarter note, `"1/8"`, `"1/16"`, `"dotted"` = dotted 1/8
- Delay time in samples: `(60 / bpm) * beat_fraction * SR`
- `feedback`: 0–100 (percentage fed back into buffer)
- `mix`: 0–100 dry/wet
- Stereo: slight L/R offset for width

**Chorus** (modulated delay):
- Base delay: ~7ms
- LFO (sine) modulates delay time by `depth`
- `rate`: LFO speed in Hz
- `depth`: modulation depth 0–100
- `mix`: 0–100

**Flanger** (short modulated delay + feedback):
- Same principle as chorus but shorter base delay (~1–3ms)
- Added `feedback` parameter (0–100) for the metallic resonance character
- `rate`, `depth`, `mix` same as chorus

### 2b) Send Routing — `mixer.py`

Current flow: `stems → sum → master vol → normalize → WAV`

New flow:
```
For each stem:
  1. Apply channel volume + pan → dry signal
  2. Calculate send amounts (sendReverb, sendDelay) → add to send buses

Reverb bus → apply reverb effect
Delay bus → apply delay effect

Dry mix + reverb return + delay return → master bus
Apply chorus (if on) → post-insert
Apply flanger (if on) → post-insert
Apply master volume
Normalize → WAV
```

**New fields in `ExportRequest`:**
```python
effects: dict | None = None
# Structure matches frontend EffectsState:
# {
#   "reverb": { "on": bool, "mix": int, "size": str },
#   "delay": { "on": bool, "mix": int, "time": str, "feedback": int },
#   "chorus": { "on": bool, "mix": int, "rate": float, "depth": int },
#   "flanger": { "on": bool, "mix": int, "rate": float, "depth": int, "feedback": int }
# }
```

**MixerChannel fields already in frontend type but missing in API:**
- `solo: bool` (default false)
- `sendReverb: int` (0–100, default 0)
- `sendDelay: int` (0–100, default 0)

### 2c) Solo Logic — `mixer.py` or pre-render

If any mixer channel has `solo: true`:
- Only render channels where `solo == true`
- Muted channels stay muted even if solo'd
- Implementation: filter the stem list before mixing, or set non-solo channels to mute before rendering

### 2d) Per-Track Swing — all engines

Frontend already has `swing?: number` on both `DrumTrack` and `MelodicTrack`.

**New optional fields:**
- `DrumTrackData.swing: float | None = None`
- `MelodicTrackData.swing: float | None = None`

Logic in each engine: `effective_swing = track.swing if track.swing is not None else global_swing`

One-line change per engine in the swing offset calculation.

---

## Phase 3: Feature Parity & Extras

**Goal:** Implement remaining features the frontend already has types for.

### 3a) Fill Trigs

Frontend has `"fill"` and `"!fill"` trig conditions. API currently doesn't handle them.

**New field in `ExportRequest`:**
```python
fill: bool = False
```

**Change in `utils.py`:** `should_trig_fire` already gets `is_fill` param from Phase 0. Fill logic:
- `"fill"` condition: fires only when `is_fill == True`
- `"!fill"` condition: fires only when `is_fill == False`

All engines pass `data.fill` through to the trig evaluation.

### 3b) globalKey / globalScale Quantization

Currently accepted but ignored. The frontend does its own quantization, so this must be opt-in.

**New field in `ExportRequest`:**
```python
quantize: bool = False
```

When `quantize == True`:
- After applying p-lock pitch offset, before synthesis
- Round each MIDI note to the nearest note in the given scale + key
- Scale definitions: port from frontend `scales.ts` into `utils.py`
- Common scales: chromatic (no-op), major, minor, dorian, mixolydian, pentatonic, blues, etc.

When `quantize == False` (default): current behavior, no server-side quantization.

### 3c) Arpeggiator

Frontend defines: `ArpMode` (up/down/updown/random), `ArpRate` (1/4, 1/8, 1/16, 1/32), `ArpGate` (short/medium/long).

**New optional field on `MelodicTrackData`:**
```python
arp: dict | None = None
# Structure: { "on": bool, "mode": str, "rate": str, "gate": str }
# Defaults: None (no arpeggiator, current behavior)
```

**Logic (in `_render_melodic`):**
When `arp.on == True` and chord intervals > 1:
1. Instead of stacking all chord notes simultaneously, spread them over time
2. `rate` determines the time between arp notes: `"1/16"` = one step duration, `"1/32"` = half step, etc.
3. `mode` determines note order:
   - `"up"`: low to high
   - `"down"`: high to low
   - `"updown"`: low→high→low (ping-pong)
   - `"random"`: random order each cycle
4. `gate` determines note length relative to arp rate:
   - `"short"` = 25%
   - `"medium"` = 50%
   - `"long"` = 75%

When `arp` is None or `arp.on == False`: current chord behavior (all notes stacked).

---

## Dependency: `requirements.txt` Update

Add `scipy` for the biquad filter implementation in Phase 1a:

```
fastapi==0.115.0
uvicorn==0.32.0
numpy==1.26.4
soundfile==0.12.1
midiutil==1.2.1
scipy>=1.11.0
```

---

## Implementation Order

Execute strictly in this order. Each phase should be fully complete and testable before starting the next.

```
Phase 0: Cleanup
  0a → 0b → 0c → 0d

Phase 1: Sound Engine (highest priority)
  1e (velocity fix, quick win)
  → 1a (biquad filter)
  → 1b (configurable ADSR)
  → 1c (bass modulation: wave → filter env → LFO)
  → 1d (drum pitch + decay p-lock)

Phase 2: Effects & Mixer
  2a (effects engine)
  → 2b (send routing)
  → 2c (solo)
  → 2d (per-track swing)

Phase 3: Feature Parity
  3a (fill trigs)
  → 3b (scale quantization)
  → 3c (arpeggiator)
```

---

## Files Modified / Created

| File | Action | Phases |
|------|--------|--------|
| `utils.py` | **NEW** — shared utilities | 0, 3 |
| `main.py` | MODIFY — new Optional fields in models | 0, 1, 2, 3 |
| `synth_engine.py` | MODIFY — biquad filter, ADSR, bass modulation, LFO | 0, 1 |
| `drum_engine.py` | MODIFY — pitch/decay p-lock, use shared utils | 0, 1 |
| `midi_engine.py` | MODIFY — use shared utils, fill trigs | 0, 3 |
| `mixer.py` | MODIFY — send routing, solo, tempfile fix | 0, 2 |
| `effects_engine.py` | **NEW** — reverb, delay, chorus, flanger | 2 |
| `requirements.txt` | MODIFY — add scipy | 1 |

---

## Testing Strategy

After each phase, test with a minimal curl request to verify:

```bash
# Health check
curl http://localhost:8002/

# Minimal WAV export (should still work after every phase)
curl -X POST http://localhost:8002/export/wav \
  -H "Content-Type: application/json" \
  -d '{"pattern":{"drums":[{"name":"KICK","steps":[{"on":true,"velocity":100,"probability":100,"retrig":0,"microTiming":0,"plock":{},"condition":"always"},{"on":false,"velocity":100,"probability":100,"retrig":0,"microTiming":0,"plock":{},"condition":"always"}]}],"bass":{"name":"BASS","steps":[{"on":false,"velocity":100,"probability":100,"retrig":0,"microTiming":0,"plock":{},"condition":"always"}],"octave":2,"chord":"none"},"synth":{"name":"SYNTH","steps":[{"on":false,"velocity":100,"probability":100,"retrig":0,"microTiming":0,"plock":{},"condition":"always"}],"octave":4,"chord":"none","wave":"sawtooth","filterFreq":2000},"length":2},"bpm":120,"swing":0,"masterVol":0.8,"loops":1}' \
  --output test.wav
```

Backwards compatibility is verified when this request produces a valid WAV after every phase without modification.
