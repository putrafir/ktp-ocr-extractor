from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import time
import logging
from liteparse import LiteParse
from liteparse.types import ParseResult, ParsedPage, TextItem

from src.config import Config

logger = logging.getLogger(__name__)

class LiteParseKTPParser:
    """Wrapper around LiteParse engine for spatial layout and OCR extraction."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize LiteParse instance
        # Supports both built-in OCR and HTTP OCR servers (e.g. PaddleOCR/RapidOCR)
        self.parser = LiteParse(
            ocr_enabled=self.config.OCR_ENABLED,
            ocr_language=self.config.OCR_LANGUAGE,
            ocr_server_url=self.config.OCR_SERVER_URL,
            output_format=self.config.OUTPUT_FORMAT,
            dpi=self.config.DPI
        )

    def parse_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parses document/image using LiteParse.
        
        Returns:
            Dict containing:
            - 'raw_text': Full reconstructed text preserving layout
            - 'pages': List of parsed page info
            - 'text_items': List of individual text items with coordinates
            - 'latency_ms': Processing time in milliseconds
        """
        start_time = time.time()
        file_path_str = str(file_path)
        
        try:
            result: ParseResult = self.parser.parse(file_path_str)
        except Exception as e:
            logger.error(f"LiteParse error on {file_path_str}: {e}")
            raise RuntimeError(f"Gagal melakukan parsing dokumen: {e}") from e

        latency_ms = (time.time() - start_time) * 1000.0

        # Collect text items with bounding coordinates
        all_text_items: List[Dict[str, Any]] = []
        full_text = result.text or ""

        page_width = 0.0
        page_height = 0.0
        if result.pages:
            page_width = result.pages[0].width
            page_height = result.pages[0].height
            for page in result.pages:
                if not full_text and page.text:
                    full_text += page.text + "\n"
                if page.text_items:
                    for item in page.text_items:
                        all_text_items.append({
                            "text": item.text,
                            "x": item.x,
                            "y": item.y,
                            "width": item.width,
                            "height": item.height,
                            "confidence": item.confidence
                        })

        return {
            "raw_text": full_text.strip(),
            "total_pages": result.total_pages,
            "page_width": page_width,
            "page_height": page_height,
            "text_items": all_text_items,
            "latency_ms": round(latency_ms, 2)
        }
