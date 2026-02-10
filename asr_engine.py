# asr_engine.py
import logging
import threading
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from huggingface_hub import login
from transformers import AutoModel

from config import settings

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BengaliASREngine:
    """
    Singleton ASR Engine for Bengali speech-to-text transcription.
    Supports long-form audio through memory-efficient sliding window chunking.
    """
    _instance: Optional["BengaliASREngine"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialize_model()
        self._initialized = True

    def _initialize_model(self):
        """Authenticates with HF and loads the model to the target device."""
        if settings.HF_TOKEN:
            login(token=settings.HF_TOKEN)
            logger.info("Authenticated with Hugging Face Hub")

        self.device = torch.device(
            "cuda" if settings.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Initializing model '{settings.MODEL_NAME}' on {self.device}")

        try:
            # Load model with remote code execution enabled for custom architectures
            self.model = AutoModel.from_pretrained(
                settings.MODEL_NAME,
                trust_remote_code=True
            ).to(self.device)

            self.model.eval()
            logger.info("ASR Model loaded successfully in evaluation mode")
        except Exception as e:
            logger.critical(f"Critical failure during model initialization: {e}")
            raise RuntimeError(f"Failed to load ASR engine: {e}")

    @torch.inference_mode()
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes audio files with OOM protection via temporal chunking.
        
        Args:
            audio_path: Local path to the audio file.
            
        Returns:
            Joined string of transcribed text.
        """
        try:
            path = Path(audio_path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")

            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(str(path))
            
            # 1. Standardize Sample Rate (Critical for model accuracy)
            if sample_rate != settings.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, settings.SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # 2. Downmix to Mono if Stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            total_samples = waveform.shape[1]
            duration_sec = total_samples / settings.SAMPLE_RATE
            logger.info(f"Processing audio: {duration_sec:.2f}s duration")

            # 3. Chunking Configuration (20s windows for VRAM safety)
            CHUNK_DURATION_SEC = 20  
            chunk_samples = CHUNK_DURATION_SEC * settings.SAMPLE_RATE
            full_transcription = []

            # 4. Iterative Inference Loop
            for start in range(0, total_samples, chunk_samples):
                end = min(start + chunk_samples, total_samples)
                
                # Drop segments shorter than 0.5s to filter out padding/noise
                if (end - start) < (0.5 * settings.SAMPLE_RATE):
                    continue

                # Slice segment and transfer to target device
                chunk_waveform = waveform[:, start:end].to(self.device)

                try:
                    # Model-specific inference call
                    # Parameters: (waveform, language_code, decoding_strategy)
                    result = self.model(chunk_waveform, "bn", settings.DECODING_METHOD)
                    text = str(result).strip()
                    
                    if text:
                        full_transcription.append(text)
                            
                except Exception as e:
                    logger.error(f"Inference failed for chunk {start//settings.SAMPLE_RATE}s: {e}")
                finally:
                    # Proactive VRAM management for production stability
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

            final_text = " ".join(full_transcription)
            logger.info("Transcription pipeline completed successfully")
            return final_text

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            raise RuntimeError(f"ASR Pipeline Failure: {str(e)}")

# Export singleton instance for app-wide use
asr_engine = BengaliASREngine()