# Elastic Groove — Sound Expansion & 8-Slot Drums

## Overview

Expand the Elastic Groove sound engine: 8 drum slots (up from 6), alternative drum kits, and better synth sounds. Changes affect both the frontend (elastic-universe-landing) and backend API (elastic-groove-api).

## Current State

**Frontend drums** (`drums.ts`): 6 oscillator-based voices: Kick, Snare, HH-C, HH-O, Clap, Perc
**Frontend synths** (`soundengine.ts`): 15 Soundfont programs + 1 OSC fallback via smplr CDN
**Frontend bass** (`soundengine.ts`): 7 Soundfont programs + 1 OSC (808 sine)
**Backend drums** (`drum_engine.py`): 6 synthesized voices, hardcoded in `SYNTHS` list
**Backend MIDI** (`midi_engine.py`): `GM_DRUM_MAP` with 6 entries

**Key files:**
- `elastic-universe-landing/src/app/tools/groove/_lib/drums.ts` — Drum oscillator synthesis
- `elastic-universe-landing/src/app/tools/groove/_lib/soundengine.ts` — Soundfont engine, program catalogs
- `elastic-universe-landing/src/app/tools/groove/_lib/types.ts` — Type definitions, `DrumTrack`
- `elastic-universe-landing/src/app/tools/groove/_lib/presets.ts` — Genre presets
- `elastic-universe-landing/src/app/tools/groove/page.tsx` — Main UI, drum grid, channel strips
- `elastic-groove-api/drum_engine.py` — Backend drum synthesis
- `elastic-groove-api/midi_engine.py` — MIDI export drum map
- `elastic-groove-api/main.py` — Pydantic models

---

## Feature 1: Expand to 8 Drum Slots

### 1a) Two New Drum Voices

Add 2 new voices to reach 8 slots:

**Slot 7: RIMSHOT**
- Frontend (`drums.ts`): Short sine burst at 800Hz with bandpass filter, fast decay (50ms). Bright, cutting transient.
```typescript
export function playRimshot(ctx: AudioContext, dest: AudioNode, time: number, vol: number) {
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.type = "triangle";
  osc.frequency.setValueAtTime(800, time);
  osc.frequency.exponentialRampToValueAtTime(400, time + 0.03);
  g.gain.setValueAtTime(vol * 0.7, time);
  g.gain.exponentialRampToValueAtTime(0.001, time + 0.05);
  const bp = ctx.createBiquadFilter(); bp.type = "bandpass"; bp.frequency.value = 1200; bp.Q.value = 3;
  osc.connect(bp).connect(g).connect(dest);
  osc.start(time); osc.stop(time + 0.06);
}
```

**Slot 8: CRASH**
- Frontend (`drums.ts`): Long noise burst with high-pass filter, slow decay (800ms). Wide, shimmery.
```typescript
export function playCrash(ctx: AudioContext, dest: AudioNode, time: number, vol: number) {
  const len = ctx.sampleRate * 0.8;
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  const src = ctx.createBufferSource(); src.buffer = buf;
  const g = ctx.createGain();
  g.gain.setValueAtTime(vol * 0.35, time);
  g.gain.exponentialRampToValueAtTime(0.001, time + 0.8);
  const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 5000;
  const bp = ctx.createBiquadFilter(); bp.type = "bandpass"; bp.frequency.value = 8000; bp.Q.value = 0.5;
  src.connect(hp).connect(bp).connect(g).connect(dest);
  src.start(time);
}
```

### 1b) Frontend Changes

**`drums.ts`** — Add `playRimshot` and `playCrash` to `DRUM_SYNTHS` array (now 8 entries).

**`soundengine.ts`** — Extend `DRUM_PROGRAMS` array:
```typescript
export const DRUM_PROGRAMS = [
  { id: "kick",    name: "KICK",    gmNote: 36 },
  { id: "snare",   name: "SNARE",   gmNote: 38 },
  { id: "hhc",     name: "HH-C",    gmNote: 42 },
  { id: "hho",     name: "HH-O",    gmNote: 46 },
  { id: "clap",    name: "CLAP",    gmNote: 39 },
  { id: "perc",    name: "PERC",    gmNote: 56 },
  { id: "rim",     name: "RIM",     gmNote: 37 },  // NEW — Side Stick
  { id: "crash",   name: "CRASH",   gmNote: 49 },  // NEW — Crash Cymbal
];
```

**`types.ts`** — No changes needed. `DrumTrack[]` is already dynamic length.

**`page.tsx`** — The drum grid needs to show 8 rows instead of 6. Check how the grid is rendered:
- If it loops over `DRUM_PROGRAMS` or `pattern.drums` → automatically picks up the new entries
- If rows are hardcoded → needs updating
- Keyboard shortcuts: add U and I for the 2 new voices (Q W E R T Y already used for 6)
- Mixer channel strips: need 2 more channels (now 10 total: 8 drums + bass + synth1 + synth2 = 11)

**`presets.ts`** — All presets need 2 more entries in their `drums` array (steps for RIM and CRASH). Default: empty steps (all off). Some presets could have patterns:
- Hip-Hop: Rimshot on beats 5 and 13
- Techno: Crash on beat 1 every 4 bars (via conditional trig "1:4")

### 1c) Backend Changes

**`drum_engine.py`** — Add 2 new synth functions:

```python
def synth_rimshot(sr: int = SR) -> np.ndarray:
    length = int(sr * 0.06)
    t = np.linspace(0, 0.06, length, endpoint=False)
    freq = 800 * np.exp(-t * 40) + 400
    phase = np.cumsum(2 * np.pi * freq / sr)
    osc = np.sin(phase) * 0.7
    env = np.exp(-t * 60)
    return osc * env * 0.7

def synth_crash(sr: int = SR) -> np.ndarray:
    length = int(sr * 0.8)
    t = np.linspace(0, 0.8, length, endpoint=False)
    noise = np.random.randn(length) * 0.35
    env = np.exp(-t * 4)
    return noise * env * 0.5
```

Update `SYNTHS` list to include `synth_rimshot` and `synth_crash` (now 8 entries).

**`midi_engine.py`** — Extend `GM_DRUM_MAP`:
```python
GM_DRUM_MAP = {
    0: 36,  # Kick
    1: 38,  # Snare
    2: 42,  # HH Closed
    3: 46,  # HH Open
    4: 39,  # Clap
    5: 56,  # Perc (Cowbell)
    6: 37,  # Rimshot (Side Stick)
    7: 49,  # Crash Cymbal
}
```

**`main.py`** — Update mixer channel slicing from `[:6]` to `[:8]`:
```python
mixer_channels=mixer_channels[:8],  # was [:6]
```

---

## Feature 2: Alternative Drum Kits

Instead of always using the same oscillator drums, offer selectable drum kits with different character.

### 2a) Drum Kit Definitions

**Frontend** — New file `_lib/drumkits.ts`:

Each kit provides parameter overrides for the 8 drum voices. The oscillator synthesis functions in `drums.ts` get additional parameters (freq, decay, tone, noise amount) so the same functions produce different characters.

```typescript
export interface DrumKitParams {
  kick:  { freq: number; decay: number; pitchDrop: number };
  snare: { freq: number; decay: number; noiseAmount: number; hpFreq: number };
  hhc:   { hpFreq: number; decay: number };
  hho:   { hpFreq: number; decay: number };
  clap:  { freq: number; decay: number };
  perc:  { freq: number; decay: number; wave: OscillatorType };
  rim:   { freq: number; decay: number };
  crash: { hpFreq: number; decay: number };
}

export const DRUM_KITS: { id: string; name: string; params: DrumKitParams }[] = [
  {
    id: "default",
    name: "Electronic",
    params: {
      kick:  { freq: 150, decay: 0.4, pitchDrop: 30 },
      snare: { freq: 200, decay: 0.15, noiseAmount: 0.8, hpFreq: 1000 },
      hhc:   { hpFreq: 7000, decay: 0.05 },
      hho:   { hpFreq: 7000, decay: 0.3 },
      clap:  { freq: 1500, decay: 0.08 },
      perc:  { freq: 800, decay: 0.1, wave: "triangle" },
      rim:   { freq: 800, decay: 0.05 },
      crash: { hpFreq: 5000, decay: 0.8 },
    },
  },
  {
    id: "808",
    name: "808",
    params: {
      kick:  { freq: 60, decay: 0.8, pitchDrop: 20 },    // Deep sub kick, long tail
      snare: { freq: 180, decay: 0.2, noiseAmount: 0.6, hpFreq: 800 },  // Tight 808 snare
      hhc:   { hpFreq: 8000, decay: 0.03 },               // Short, tight
      hho:   { hpFreq: 6000, decay: 0.4 },                // Longer open hat
      clap:  { freq: 1200, decay: 0.12 },                  // Wider clap
      perc:  { freq: 600, decay: 0.08, wave: "sine" },    // Cowbell-like
      rim:   { freq: 1000, decay: 0.04 },
      crash: { hpFreq: 4000, decay: 1.0 },
    },
  },
  {
    id: "909",
    name: "909",
    params: {
      kick:  { freq: 180, decay: 0.3, pitchDrop: 40 },    // Punchy 909 kick
      snare: { freq: 250, decay: 0.18, noiseAmount: 1.0, hpFreq: 1200 },  // Bright, snappy
      hhc:   { hpFreq: 9000, decay: 0.04 },               // Metallic
      hho:   { hpFreq: 8000, decay: 0.25 },
      clap:  { freq: 1800, decay: 0.06 },                  // Tight clap
      perc:  { freq: 700, decay: 0.12, wave: "triangle" }, // Ride-like
      rim:   { freq: 900, decay: 0.04 },
      crash: { hpFreq: 6000, decay: 0.7 },
    },
  },
  {
    id: "lofi",
    name: "Lo-Fi",
    params: {
      kick:  { freq: 100, decay: 0.25, pitchDrop: 35 },   // Muted, warm
      snare: { freq: 160, decay: 0.12, noiseAmount: 0.5, hpFreq: 600 },  // Dusty
      hhc:   { hpFreq: 4000, decay: 0.04 },               // Muffled
      hho:   { hpFreq: 3000, decay: 0.2 },
      clap:  { freq: 1000, decay: 0.1 },
      perc:  { freq: 500, decay: 0.15, wave: "sine" },    // Soft bell
      rim:   { freq: 600, decay: 0.06 },
      crash: { hpFreq: 3000, decay: 0.6 },
    },
  },
  {
    id: "acoustic",
    name: "Acoustic",
    params: {
      kick:  { freq: 120, decay: 0.35, pitchDrop: 50 },   // Boomy, natural
      snare: { freq: 220, decay: 0.2, noiseAmount: 0.9, hpFreq: 900 },  // Snare wires
      hhc:   { hpFreq: 6000, decay: 0.06 },
      hho:   { hpFreq: 5000, decay: 0.35 },
      clap:  { freq: 1400, decay: 0.1 },
      perc:  { freq: 1200, decay: 0.08, wave: "triangle" },  // Woodblock
      rim:   { freq: 1000, decay: 0.03 },
      crash: { hpFreq: 4500, decay: 0.9 },
    },
  },
];
```

### 2b) Parameterized Drum Synthesis

**`drums.ts`** — Refactor all drum functions to accept kit parameters:

```typescript
export function playKick(ctx: AudioContext, dest: AudioNode, time: number, vol: number,
                          freq = 150, decay = 0.4, pitchDrop = 30) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.setValueAtTime(freq, time);
  osc.frequency.exponentialRampToValueAtTime(pitchDrop, time + decay * 0.3);
  gain.gain.setValueAtTime(vol, time);
  gain.gain.exponentialRampToValueAtTime(0.001, time + decay);
  osc.connect(gain).connect(dest);
  osc.start(time); osc.stop(time + decay);
}
// ... same pattern for all 8 voices
```

### 2c) Kit Selector in UI

**`page.tsx`** — Add a dropdown in the Drums tab header:
```
Drums   Kit: [Electronic ▾]   Copy   Paste   Clear
```

The selected kit is stored in the pattern state. When the kit changes, the drum synthesis calls pass the kit's parameters.

### 2d) Backend Kit Support

**`drum_engine.py`** — The drum synth functions already accept `sr` parameter. Add optional frequency/decay overrides:

```python
def synth_kick(sr: int = SR, freq: float = 150, decay: float = 0.4, pitch_drop: float = 30) -> np.ndarray:
    t = np.linspace(0, decay, int(sr * decay), endpoint=False)
    freq_env = freq * np.exp(-t * 12) + pitch_drop
    phase = np.cumsum(2 * np.pi * freq_env / sr)
    env = np.exp(-t * (3 / decay))
    return np.sin(phase) * env * 0.9
```

**`main.py`** — Add optional `drumKit` field to `ExportRequest`:
```python
drumKit: str | None = None  # "default", "808", "909", "lofi", "acoustic"
```

Kit parameters are defined as a dict in `drum_engine.py` (matching the frontend kits). If `drumKit` is set, the corresponding parameters are passed to each synth function.

---

## Feature 3: Better Synth Sounds

### 3a) More Soundfont Programs

**`soundengine.ts`** — Extend `SYNTH_PROGRAMS` with additional GM instruments:

```typescript
// Additional Pads
{ id: "pad_4_choir",         name: "Choir Pad",       category: "Pad" },
{ id: "pad_5_bowed",         name: "Bowed Pad",       category: "Pad" },
{ id: "pad_8_sweep",         name: "Sweep Pad",       category: "Pad" },

// Additional Leads
{ id: "lead_3_calliope",     name: "Calliope Lead",   category: "Lead" },
{ id: "lead_4_chiff",        name: "Chiff Lead",      category: "Lead" },
{ id: "lead_6_voice",        name: "Voice Lead",      category: "Lead" },
{ id: "lead_8_bass_lead",    name: "Bass + Lead",     category: "Lead" },

// Synth Effects
{ id: "fx_1_rain",           name: "Rain FX",         category: "FX" },
{ id: "fx_3_crystal",        name: "Crystal",         category: "FX" },
{ id: "fx_4_atmosphere",     name: "Atmosphere",      category: "FX" },
{ id: "fx_7_echoes",         name: "Echoes",          category: "FX" },

// Additional Keys
{ id: "acoustic_grand_piano", name: "Grand Piano",    category: "Keys" },
{ id: "bright_acoustic_piano", name: "Bright Piano",  category: "Keys" },
{ id: "harpsichord",          name: "Harpsichord",    category: "Keys" },
{ id: "celesta",              name: "Celesta",        category: "Keys" },

// Plucked
{ id: "acoustic_guitar_nylon", name: "Nylon Guitar",  category: "Plucked" },
{ id: "acoustic_guitar_steel", name: "Steel Guitar",  category: "Plucked" },
{ id: "sitar",                 name: "Sitar",         category: "Plucked" },
{ id: "kalimba",               name: "Kalimba",       category: "Plucked" },

// Brass & Wind
{ id: "trumpet",              name: "Trumpet",        category: "Brass" },
{ id: "french_horn",          name: "French Horn",    category: "Brass" },
{ id: "flute",                name: "Flute",          category: "Wind" },
{ id: "shakuhachi",           name: "Shakuhachi",     category: "Wind" },
```

This brings the total from 15 to ~35 synth programs. All these are standard General MIDI instruments available through the smplr Soundfont CDN — no additional hosting needed.

### 3b) More Bass Programs

```typescript
// Additional bass
{ id: "contrabass",           name: "Contrabass" },
{ id: "tuba",                 name: "Tuba" },
{ id: "synth_bass_3",        name: "Synth Bass 3" },  // if available in GM
```

### 3c) Program Category Grouping in UI

The synth program dropdown in `page.tsx` should group programs by category with headers:

```
─── Oscillator ───
  OSC (Subtractive)
─── Pad ───
  New Age Pad
  Warm Pad
  Polysynth
  Halo Pad
  Choir Pad
  ...
─── Lead ───
  Square Lead
  Saw Lead
  ...
─── Keys ───
  Grand Piano
  E-Piano 1
  ...
```

Use `<optgroup label="Pad">` in the `<select>` element.

---

## Files Modified / Created

### Frontend (elastic-universe-landing)

| File | Action |
|------|--------|
| `src/app/tools/groove/_lib/drums.ts` | MODIFY — Add rimshot + crash, parameterize all voices |
| `src/app/tools/groove/_lib/drumkits.ts` | **CREATE** — 5 drum kit definitions |
| `src/app/tools/groove/_lib/soundengine.ts` | MODIFY — 8 drum programs, ~20 more synth programs, more bass |
| `src/app/tools/groove/_lib/types.ts` | MODIFY — Add drumKit field to Pattern type |
| `src/app/tools/groove/_lib/presets.ts` | MODIFY — All presets get 8 drum tracks, kit selection |
| `src/app/tools/groove/page.tsx` | MODIFY — 8-row drum grid, kit selector dropdown, grouped synth dropdown |

### Backend (elastic-groove-api)

| File | Action |
|------|--------|
| `drum_engine.py` | MODIFY — Add rimshot + crash, parameterize synths, kit support |
| `midi_engine.py` | MODIFY — Extend GM_DRUM_MAP to 8 entries |
| `main.py` | MODIFY — Update mixer slice [:8], add drumKit to ExportRequest |

---

## Implementation Order

```
1. drums.ts — Add rimshot + crash functions, parameterize existing voices
2. drumkits.ts — Create kit definitions file
3. soundengine.ts — Extend DRUM_PROGRAMS to 8, add synth/bass programs
4. types.ts — Add drumKit to Pattern
5. presets.ts — Update all presets to 8 drums + kit
6. page.tsx — 8-row grid, kit selector, grouped synth dropdown, keyboard shortcuts U/I
7. drum_engine.py — Backend: 8 voices + kit params
8. midi_engine.py — Backend: 8-entry GM_DRUM_MAP
9. main.py — Backend: mixer[:8], drumKit field
10. Test + commit + push
```

---

## Backwards Compatibility

- `drumKit` is Optional in the API (default = current "Electronic" sound)
- Frontend patterns with only 6 drums load fine — empty RIM and CRASH tracks are added on load
- All existing presets continue to work (just get 2 extra empty tracks)
- MIDI export works with old 6-track patterns (extra voices only fire if steps are active)
