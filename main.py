# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import torch
from asr_engine import asr_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bengali Speech-to-Text API",
    description="High-accuracy Bengali ASR using ai4bharat Conformer model (RNNT decoding)",
    version="1.0.0"
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def root():
    return {
        "service": "Bengali ASR API",
        "status": "running",
        "endpoints": {
            "transcribe": "POST /transcribe (upload audio file)",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and load balancers"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_available": torch.cuda.is_available(),
        "model": "ai4bharat/indic-conformer-600m-multilingual"
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and receive Bengali transcription
    Supported formats: WAV, MP3, M4A
    """
    allowed_extensions = {".wav", ".mp3", ".m4a"}
    if not Path(file.filename).suffix.lower() in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Supported: WAV, MP3, M4A"
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Received file: {file.filename}")

        # Perform transcription
        transcription = asr_engine.transcribe(str(file_path))

        return JSONResponse({
            "filename": file.filename,
            "transcription": transcription,
            "language": "Bengali (bn)",
            "model": "ai4bharat/indic-conformer-600m-multilingual",
            "decoding": "rnnt"
        })

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()
            logger.info("Temporary file cleaned up")