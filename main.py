import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from drum_engine import render_pattern
from midi_engine import render_midi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("elastic-groove")

app = FastAPI(title="Elastic Groove API")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatternData(BaseModel):
    pattern: list[list[bool]]
    bpm: float = 120
    swing: float = 0
    trackVols: list[float] = [0.9, 0.8, 0.6, 0.6, 0.7, 0.5]
    masterVol: float = 0.8


@app.get("/")
async def health():
    return {"status": "online", "service": "Elastic Groove API", "version": "1.0.0"}


@app.post("/export/wav")
async def export_wav(data: PatternData):
    logger.info(f"WAV export: {data.bpm} BPM, swing {data.swing}%")
    try:
        path = render_pattern(
            pattern=data.pattern,
            bpm=data.bpm,
            swing=data.swing,
            track_vols=data.trackVols,
            master_vol=data.masterVol,
            loops=2,
        )
        return FileResponse(
            path,
            media_type="audio/wav",
            filename="elastic-groove-pattern.wav",
        )
    except Exception as e:
        logger.exception("WAV export failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/midi")
async def export_midi(data: PatternData):
    logger.info(f"MIDI export: {data.bpm} BPM, swing {data.swing}%")
    try:
        path = render_midi(
            pattern=data.pattern,
            bpm=data.bpm,
            swing=data.swing,
            track_vols=data.trackVols,
            loops=4,
        )
        return FileResponse(
            path,
            media_type="audio/midi",
            filename="elastic-groove-pattern.mid",
        )
    except Exception as e:
        logger.exception("MIDI export failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
