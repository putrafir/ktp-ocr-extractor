import unittest
from pathlib import Path
import cv2
import numpy as np

from src.validator import KTPValidator
from src.pipeline import KTPExtractionPipeline
from src.models import KTPValidationResult

class TestKTPValidator(unittest.TestCase):
    def setUp(self):
        self.validator = KTPValidator()
        self.pipeline = KTPExtractionPipeline(engine="paddle")
        self.samples_dir = Path("data/samples")

    def test_valid_ktp_sample_2026(self):
        img_path = self.samples_dir / "2026.jpg"
        if not img_path.exists():
            self.skipTest("Sample 2026.jpg not found")

        result = self.pipeline.process(img_path)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_ktp"])
        self.assertGreaterEqual(result["validation"]["confidence_score"], 0.70)
        self.assertIsNotNone(result["data"])
        self.assertEqual(result["data"]["nik"], "3602141204920003")

    def test_valid_ktp_mira_setiawan(self):
        img_path = self.samples_dir / "mira_setiawan_ktp.jpg"
        if not img_path.exists():
            self.skipTest("Sample mira_setiawan_ktp.jpg not found")

        result = self.pipeline.process(img_path)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_ktp"])
        self.assertGreaterEqual(result["validation"]["confidence_score"], 0.70)
        self.assertIsNotNone(result["data"])

    def test_non_ktp_dummy_receipt_rejected(self):
        receipt_path = self.samples_dir / "dummy_receipt.jpg"
        if not receipt_path.exists():
            self.skipTest("dummy_receipt.jpg not found")

        # Test under strict validation
        result = self.pipeline.process(receipt_path, strict_validation=True)
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["is_ktp"])
        self.assertLess(result["validation"]["confidence_score"], 0.50)
        self.assertIsNone(result["data"])
        self.assertGreater(len(result["validation"]["rejection_reasons"]), 0)

    def test_non_ktp_dummy_receipt_non_strict(self):
        receipt_path = self.samples_dir / "dummy_receipt.jpg"
        if not receipt_path.exists():
            self.skipTest("dummy_receipt.jpg not found")

        # Test under non-strict validation
        result = self.pipeline.process(receipt_path, strict_validation=False)
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["is_ktp"])  # is_ktp flag is still accurately False
        self.assertIsNotNone(result["data"])

    def test_semantic_validator_pure_text(self):
        # Invoice text
        invoice_ocr = {
            "raw_text": """
            STARBUCKS COFFEE
            INVOICE # 98124
            1x CAFE LATTE 55.000
            TOTAL: 55.000
            CASHIER: JANE
            THANK YOU
            """,
            "text_items": [],
            "page_width": 400
        }
        dummy_img = np.ones((600, 350, 3), dtype=np.uint8) * 255
        val_res: KTPValidationResult = self.validator.validate(dummy_img, invoice_ocr)
        self.assertFalse(val_res.is_ktp)
        self.assertLess(val_res.confidence_score, 0.40)
        self.assertTrue(any("struk" in r.lower() or "faktur" in r.lower() for r in val_res.rejection_reasons))

    def test_blur_detection_visual_penalty(self):
        # Create an extremely blurry/uniform image
        blank_img = np.zeros((600, 950, 3), dtype=np.uint8)
        score, details, penalties = self.validator.validate_visual(blank_img)
        self.assertTrue(any("buram" in p.lower() or "kabur" in p.lower() for p in penalties))
        self.assertLess(score, 0.70)

if __name__ == "__main__":
    unittest.main()
