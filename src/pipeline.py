from pathlib import Path
from typing import Union, Dict, Any
import os
import numpy as np

from src.config import Config
from src.preprocessor import ImagePreprocessor
from src.extractor import KTPExtractor
from src.models import KTPData

class KTPExtractionPipeline:
    """
    End-to-end pipeline:
    1. Preprocessing (OpenCV deskew + resolution optimization + HEIC support)
    2. Document & Layout OCR Parsing (PaddleOCR / LiteParse)
    3. Key-Value Extraction (KTPExtractor with spatial pairing & regex validation)
    """

    def __init__(self, config: Config | None = None, engine: str = "paddle"):
        self.config = config or Config()
        self.engine_name = engine.lower()
        self.preprocessor = ImagePreprocessor(target_height=900, max_width=self.config.MAX_IMAGE_WIDTH)
        
        if self.engine_name == "liteparse":
            from src.parser import LiteParseKTPParser
            self.parser = LiteParseKTPParser(config=self.config)
        else:
            from src.paddle_parser import PaddleKTPParser
            self.parser = PaddleKTPParser(config=self.config)

        self.extractor = KTPExtractor(config=self.config)

    def process(self, image_input: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Executes end-to-end extraction.
        
        Returns:
            Dict containing:
            - 'data': Dict of extracted key-value fields
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

            # Step 3: Extract structured fields
            ktp_data: KTPData = self.extractor.extract(parse_result)

            return {
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
