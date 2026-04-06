"""Elastic Groove API — WAV + MIDI Export with full Pattern support."""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from drum_engine import render_drums
from synth_engine import render_bass, render_synth
from midi_engine import render_midi
from mixer import mix_stems

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("elastic-groove")

app = FastAPI(title="Elastic Groove API", version="3.0.0")

_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
ALLOWED_ORIGINS = []
for o in _origins:
    o = o.strip()
    ALLOWED_ORIGINS.append(o)
    if o.startswith("https://www."):
        ALLOWED_ORIGINS.append(o.replace("https://www.", "https://"))
    elif o.startswith("https://") and "//www." not in o and o != "*":
        ALLOWED_ORIGINS.append(o.replace("https://", "https://www."))
if "*" in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Data Models (matching the frontend Pattern type) ──
# Allow extra fields so frontend can send e.g. "key" on DrumTrack without 422


class FlexModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PLockData(FlexModel):
    volume: float | None = None
    pitch: int | None = None
    filterFreq: float | None = None
    filterRes: float | None = None        # Phase 1a: biquad resonance (0-1)
    decay: float | None = None
    pan: float | None = None


class StepData(FlexModel):
    on: bool = False
    note: int | None = None
    velocity: int = 100
    probability: int = 100
    retrig: int = 0
    microTiming: int = 0
    plock: PLockData = PLockData()
    condition: str = "always"


class FilterEnvData(FlexModel):
    amount: float = 2000       # Hz sweep range
    attack: float = 0.01       # seconds
    decay: float = 0.3         # seconds


class LFOData(FlexModel):
    rate: float = 2.0          # Hz
    depth: float = 0.5         # 0-1
    target: str = "filter"     # "pitch" | "filter" | "volume"


class ArpData(FlexModel):
    on: bool = False
    mode: str = "up"           # up | down | updown | random
    rate: str = "1/16"         # 1/4 | 1/8 | 1/16 | 1/32
    gate: str = "medium"       # short | medium | long


class DrumTrackData(FlexModel):
    name: str
    steps: list[StepData]
    swing: float | None = None             # Phase 2d: per-track swing (overrides global)


class MelodicTrackData(FlexModel):
    name: str
    steps: list[StepData]
    octave: int = 4
    chord: str = "none"
    wave: str | None = None
    filterFreq: float | None = None
    filterRes: float | None = None         # Phase 1a: resonance (0-1 → Q 0.5-15)
    filterType: str | None = None          # Phase 1a: "lowpass" | "highpass" | "bandpass"
    attack: float | None = None            # Phase 1b: ADSR attack (seconds)
    decay: float | None = None             # Phase 1b: ADSR decay (seconds)
    sustain: float | None = None           # Phase 1b: ADSR sustain level (0-1)
    release: float | None = None           # Phase 1b: ADSR release (seconds)
    filterEnv: FilterEnvData | None = None # Phase 1c: filter envelope
    lfo: LFOData | None = None             # Phase 1c: LFO modulation
    arp: ArpData | None = None             # Phase 3c: arpeggiator
    swing: float | None = None             # Phase 2d: per-track swing


class EffectsData(FlexModel):
    reverb: dict | None = None   # {on, mix, size}
    delay: dict | None = None    # {on, mix, time, feedback}
    chorus: dict | None = None   # {on, mix, rate, depth}
    flanger: dict | None = None  # {on, mix, rate, depth, feedback}


class PatternData(FlexModel):
    drums: list[DrumTrackData]
    bass: MelodicTrackData
    synth: MelodicTrackData
    synth2: MelodicTrackData | None = None
    length: int = 16


class ExportRequest(FlexModel):
    pattern: PatternData
    bpm: float = 120
    swing: float = 0
    masterVol: float = 0.8
    mixer: list[dict] | None = None
    effects: EffectsData | None = None     # Phase 2: effects chain
    globalKey: str = "C"
    globalScale: str = "chromatic"
    quantize: bool = False                 # Phase 3b: server-side scale quantization
    fill: bool = False                     # Phase 3a: fill trig mode
    loops: int = 2


@app.get("/")
async def health():
    return {"status": "online", "service": "Elastic Groove API", "version": "2.0.0"}


@app.post("/export/wav")
async def export_wav(data: ExportRequest):
    logger.info(f"WAV export: {data.bpm} BPM, swing {data.swing}%, {data.loops} loops")
    try:
        mixer_channels = data.mixer or []

        # Render each stem
        drum_stem = render_drums(
            tracks=data.pattern.drums,
            bpm=data.bpm,
            swing=data.swing,
            mixer_channels=mixer_channels[:6],
            length=data.pattern.length,
            loops=data.loops,
            is_fill=data.fill,
        )

        bass_stem = render_bass(
            track=data.pattern.bass,
            bpm=data.bpm,
            swing=data.swing,
            mixer_channel=mixer_channels[6] if len(mixer_channels) > 6 else None,
            length=data.pattern.length,
            loops=data.loops,
            is_fill=data.fill,
            quantize=data.quantize,
            global_key=data.globalKey,
            global_scale=data.globalScale,
        )

        synth_stem = render_synth(
            track=data.pattern.synth,
            bpm=data.bpm,
            swing=data.swing,
            mixer_channel=mixer_channels[7] if len(mixer_channels) > 7 else None,
            length=data.pattern.length,
            loops=data.loops,
            is_fill=data.fill,
            quantize=data.quantize,
            global_key=data.globalKey,
            global_scale=data.globalScale,
        )

        synth2_stem = None
        if data.pattern.synth2:
            synth2_stem = render_synth(
                track=data.pattern.synth2,
                bpm=data.bpm,
                swing=data.swing,
                mixer_channel=mixer_channels[8] if len(mixer_channels) > 8 else None,
                length=data.pattern.length,
                loops=data.loops,
                is_fill=data.fill,
                quantize=data.quantize,
                global_key=data.globalKey,
                global_scale=data.globalScale,
            )

        # Mix all stems
        stems = [drum_stem, bass_stem, synth_stem]
        if synth2_stem is not None:
            stems.append(synth2_stem)

        path = mix_stems(
            stems,
            master_vol=data.masterVol,
            mixer_channels=mixer_channels,
            effects=data.effects,
            bpm=data.bpm,
        )

        return FileResponse(
            path,
            media_type="audio/wav",
            filename="elastic-groove.wav",
        )
    except Exception as e:
        logger.exception("WAV export failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/midi")
async def export_midi(data: ExportRequest):
    logger.info(f"MIDI export: {data.bpm} BPM, swing {data.swing}%")
    try:
        path = render_midi(
            pattern=data.pattern,
            bpm=data.bpm,
            swing=data.swing,
            mixer_channels=data.mixer or [],
            loops=data.loops,
            is_fill=data.fill,
        )
        return FileResponse(
            path,
            media_type="audio/midi",
            filename="elastic-groove.mid",
        )
    except Exception as e:
        logger.exception("MIDI export failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
