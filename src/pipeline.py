from pathlib import Path
from typing import Union, Dict, Any, Optional
import os
import numpy as np

from src.config import Config
from src.preprocessor import ImagePreprocessor
from src.extractor import KTPExtractor
from src.validator import KTPValidator
from src.models import KTPData, KTPValidationResult

class KTPExtractionPipeline:
    """
    End-to-end pipeline:
    1. Preprocessing (OpenCV deskew + resolution optimization + HEIC support)
    2. Document & Layout OCR Parsing (PaddleOCR / LiteParse)
    3. Document Validation & Classification (Two-Tier Hybrid KTPValidator)
    4. Key-Value Extraction (KTPExtractor with spatial pairing & regex validation)
    """

    def __init__(self, config: Config | None = None, engine: str = "paddle", strict_validation: bool = True):
        self.config = config or Config()
        self.engine_name = engine.lower()
        self.strict_validation = strict_validation
        self.preprocessor = ImagePreprocessor(target_height=900, max_width=self.config.MAX_IMAGE_WIDTH)
        
        if self.engine_name == "liteparse":
            from src.parser import LiteParseKTPParser
            self.parser = LiteParseKTPParser(config=self.config)
        else:
            from src.paddle_parser import PaddleKTPParser
            self.parser = PaddleKTPParser(config=self.config)

        self.extractor = KTPExtractor(config=self.config)
        self.validator = KTPValidator(config=self.config)

    def process(self, image_input: Union[str, Path, bytes, np.ndarray], strict_validation: Optional[bool] = None) -> Dict[str, Any]:
        """
        Executes end-to-end validation and extraction.
        
        Returns:
            Dict containing:
            - 'status': 'success' or 'rejected'
            - 'is_ktp': bool
            - 'validation': Dict of validation details
            - 'data': Dict of extracted key-value fields (or None if rejected)
            - 'raw_text': Reconstructed layout text from OCR
            - 'latency_ms': Processing time in ms
            - 'engine': Name of OCR engine used
            - 'text_items_count': Number of detected text boxes
            - 'preprocessed_img': Preprocessed cv2 image
        """
        # Step 1: Preprocess image (EXIF orientation, deskew, and adaptive resize)
        preprocessed_img, temp_file_path = self.preprocessor.process(image_input)

        try:
            # Step 2: Parse layout and text with chosen engine
            parse_result = self.parser.parse_document(temp_file_path)

            # Step 3: Two-Tier Hybrid KTP Validation
            validation_result = self.validator.validate(preprocessed_img, parse_result)

            is_strict = self.strict_validation if strict_validation is None else strict_validation

            # If rejected under strict validation, abort extraction to prevent hallucinated data
            if not validation_result.is_ktp and is_strict:
                return {
                    "status": "rejected",
                    "is_ktp": False,
                    "validation": validation_result.to_dict(),
                    "data": None,
                    "raw_text": parse_result.get("raw_text", ""),
                    "text_items_count": len(parse_result.get("text_items", [])),
                    "latency_ms": parse_result.get("latency_ms", 0.0),
                    "engine": self.engine_name,
                    "preprocessed_img": preprocessed_img
                }

            # Step 4: Extract structured fields if validated (or non-strict mode)
            ktp_data: KTPData = self.extractor.extract(parse_result)

            return {
                "status": "success",
                "is_ktp": validation_result.is_ktp,
                "validation": validation_result.to_dict(),
                "data": ktp_data.to_dict(),
                "raw_text": parse_result.get("raw_text", ""),
                "text_items_count": len(parse_result.get("text_items", [])),
                "latency_ms": parse_result.get("latency_ms", 0.0),
                "engine": self.engine_name,
                "preprocessed_img": preprocessed_img
            }
        finally:
            if temp_file_path.exists():
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
