# asr_engine.py
from transformers import AutoModel
from huggingface_hub import login
import torch
import torchaudio
from pathlib import Path
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BengaliASREngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Login to Hugging Face if token provided
        if settings.HF_TOKEN:
            login(token=settings.HF_TOKEN)
            logger.info("Logged in to Hugging Face Hub")

        # Set device
        self.device = "cuda" if settings.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model '{settings.MODEL_NAME}' on {self.device}...")

        # Load model
        self.model = AutoModel.from_pretrained(
            settings.MODEL_NAME,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()
        logger.info("Model loaded successfully and ready for inference")

        self._initialized = True

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to Bengali text
        Supports WAV, MP3, M4A (thanks to soundfile backend)
        """
        try:
            path = Path(audio_path).resolve()
            logger.info(f"Loading audio file: {path}")

            waveform, sample_rate = torchaudio.load(path)
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

            # Resample if necessary
            if sample_rate != settings.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, settings.SAMPLE_RATE)
                waveform = resampler(waveform)

            waveform = waveform.to(self.device)

            # Perform inference
            result = self.model(waveform, "bn", settings.DECODING_METHOD)
            transcription = str(result).strip()

            logger.info("Transcription completed successfully")
            return transcription

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Audio transcription error: {str(e)}")

# Global singleton instance
asr_engine = BengaliASREngine()