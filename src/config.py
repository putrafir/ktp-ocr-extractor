from pathlib import Path
from dataclasses import dataclass
import os

@dataclass
class Config:
    # OCR & Parser settings
    OCR_ENABLED: bool = True
    OCR_LANGUAGE: str = "ind+eng"
    OCR_SERVER_URL: str | None = os.getenv("OCR_SERVER_URL", None)
    DPI: int = 200
    OUTPUT_FORMAT: str = "json"
    
    # Path settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    SAMPLES_DIR: Path = BASE_DIR / "data" / "samples"
    
    # Preprocessing thresholds
    MAX_IMAGE_WIDTH: int = 1200
    MAX_IMAGE_HEIGHT: int = 1200
    
    # Matching thresholds
    FUZZY_SCORE_THRESHOLD: float = 65.0
