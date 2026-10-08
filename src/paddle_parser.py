from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import time
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from src.config import Config

class PaddleKTPParser:
    """
    PaddleOCR Engine wrapper using RapidOCR ONNX Runtime.
    Provides deep-learning text detection (DBNet) and recognition (SVTR/CRNN).
    Extracts text, bounding boxes, and confidence scores locally with sub-second latency.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        # Initialize RapidOCR (PaddleOCR models)
        self.engine = RapidOCR()
        # Expand bounding boxes slightly to prevent ascender/descender clipping on KTP cards
        if hasattr(self.engine, "text_det") and hasattr(self.engine.text_det, "postprocess_op"):
            self.engine.text_det.postprocess_op.unclip_ratio = getattr(self.config, "OCR_DET_UNCLIP_RATIO", 2.2)

    def parse_document(self, file_path: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Parses image using PaddleOCR.
        
        Returns:
            Dict containing:
            - 'raw_text': Full reconstructed text
            - 'page_width': Image width
            - 'page_height': Image height
            - 'text_items': List of individual text items with coordinates & confidence
            - 'latency_ms': Processing time in milliseconds
        """
        start_time = time.time()
        
        # Load image if file path
        if isinstance(file_path, (str, Path)):
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"Could not read image from {file_path}")
        else:
            img = file_path

        h, w = img.shape[:2]
        
        # Run PaddleOCR inference with explicit unclip_ratio to prevent default 1.6 override
        unclip_ratio = getattr(self.config, "OCR_DET_UNCLIP_RATIO", 2.2)
        use_cls = getattr(self.config, "OCR_USE_TEXT_CLS", False)
        result, elapse = self.engine(img, unclip_ratio=unclip_ratio, use_cls=use_cls)
        latency_ms = (time.time() - start_time) * 1000.0

        all_text_items: List[Dict[str, Any]] = []
        text_lines: List[str] = []

        if result:
            # Sort items by Y first, then X
            # box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            for item in result:
                box, text, score = item
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                item_w = x_max - x_min
                item_h = y_max - y_min

                clean_text = str(text).strip()
                if clean_text:
                    text_lines.append(clean_text)
                    all_text_items.append({
                        "text": clean_text,
                        "x": float(x_min),
                        "y": float(y_min),
                        "width": float(item_w),
                        "height": float(item_h),
                        "confidence": float(score)
                    })

        return {
            "raw_text": "\n".join(text_lines),
            "total_pages": 1,
            "page_width": float(w),
            "page_height": float(h),
            "text_items": all_text_items,
            "latency_ms": round(latency_ms, 2)
        }
