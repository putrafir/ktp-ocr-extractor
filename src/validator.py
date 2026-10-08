import re
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

from src.models import KTPValidationResult
from src.config import Config

class KTPValidator:
    """
    Two-Tier Hybrid KTP Document Validator & Classifier.
    - Tier 1: Pre-OCR Visual Sanity (aspect ratio, blur score, face presence indicator).
    - Tier 2: Post-OCR Semantic Structure (headers, NIK signature, canonical label density, layout topology).
    """

    # KTP standard keywords
    HEADER_KEYWORDS = [
        "REPUBLIK INDONESIA", "PROVINSI", "KABUPATEN", "KOTA",
        "PROVINSIBANTEN", "JAWA TIMUR", "JAWA BARAT", "DKI JAKARTA", "BALI"
    ]

    CANONICAL_LABELS = [
        "NIK", "NAMA", "TEMPAT", "LAHIR", "JENIS KELAMIN", "GOL. DARAH",
        "ALAMAT", "RT/RW", "KEL/DESA", "KECAMATAN", "AGAMA",
        "STATUS PERKAWINAN", "PEKERJAAN", "KEWARGANEGARAAN", "BERLAKU HINGGA"
    ]

    NON_KTP_KEYWORDS = [
        "TOTAL", "SUBTOTAL", "INVOICE", "FAKTUR", "RECEIPT", "STRUK", "KASIR",
        "CASHIER", "CHANGE", "KEMBALIAN", "PAYMENT", "TUNAI", "ITEM", "HARGA",
        "TAX", "PPN", "QTY", "THANK YOU", "TERIMA KASIH", "MENU"
    ]

    def __init__(self, config: Optional[Config] = None, min_confidence: Optional[float] = None):
        self.config = config or Config()
        self.min_confidence = min_confidence if min_confidence is not None else self.config.MIN_KTP_CONFIDENCE_THRESHOLD

        # Load face detector once (OpenCV Haar Cascade)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def validate_visual(self, image: np.ndarray) -> Tuple[float, Dict[str, Any], List[str]]:
        """
        Tier 1: Evaluates physical visual characteristics of the card.
        """
        h, w = image.shape[:2]
        ratio = max(w, h) / max(min(w, h), 1)  # Landscape aspect ratio
        
        penalties = []
        visual_score = 1.0
        details: Dict[str, Any] = {
            "image_dimensions": f"{w}x{h}",
            "aspect_ratio": round(ratio, 3)
        }

        # 1. Aspect ratio check: ID-1 nominal is ~1.586. Tolerant crop range is 1.15 to 2.10
        if 1.25 <= ratio <= 1.95:
            details["aspect_ratio_valid"] = True
        elif 1.10 <= ratio <= 2.25:
            details["aspect_ratio_valid"] = True
            visual_score -= 0.15
        else:
            details["aspect_ratio_valid"] = False
            visual_score -= 0.40
            penalties.append(f"Rasio aspek ({ratio:.2f}) tidak lazim untuk kartu identitas")

        # 2. Blur / sharpness check
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        details["blur_laplacian_variance"] = round(laplacian_var, 1)

        if laplacian_var < self.config.BLUR_VARIANCE_THRESHOLD:
            visual_score -= 0.35
            penalties.append(f"Gambar terlalu buram/kabur (variance: {laplacian_var:.1f})")

        # 3. Face Presence Check (Soft Indicator)
        # On Indonesian KTP, the portrait photo is strictly on the right side (x > w * 0.40)
        has_face = False
        face_on_right = False
        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                # Downscale for fast face detection (< 10 ms)
                small_gray = cv2.resize(gray, (400, int(400 * h / w)))
                sw, sh = small_gray.shape[1], small_gray.shape[0]
                faces = self.face_cascade.detectMultiScale(small_gray, scaleFactor=1.15, minNeighbors=4, minSize=(25, 25))
                if len(faces) > 0:
                    has_face = True
                    for (fx, fy, fw, fh) in faces:
                        if (fx + fw / 2) > sw * 0.38:
                            face_on_right = True
                            break
            except Exception:
                pass

        details["face_detected"] = has_face
        details["face_on_right_side"] = face_on_right

        if face_on_right:
            visual_score = min(1.0, visual_score + 0.10)
        elif not has_face:
            # Soft penalty only (portrait may be obscured/cropped/grayscale)
            visual_score -= 0.10

        visual_score = max(0.0, min(1.0, visual_score))
        return visual_score, details, penalties

    def validate_semantics(self, ocr_result: Dict[str, Any]) -> Tuple[float, Dict[str, Any], List[str]]:
        """
        Tier 2: Evaluates text density, presence of official KTP headers, NIK pattern, and layout.
        """
        raw_text = (ocr_result.get("raw_text") or "").upper()
        text_items = ocr_result.get("text_items", [])
        page_width = float(ocr_result.get("page_width", 1.0))

        penalties = []
        details: Dict[str, Any] = {}

        if not raw_text.strip():
            return 0.0, {"error": "Tidak ada teks yang terdeteksi"}, ["Tidak ada teks terdeteksi pada citra"]

        # Check for Strong Negative Indicators (e.g. Receipt / Faktur / Struk)
        non_ktp_matches = [w for w in self.NON_KTP_KEYWORDS if re.search(rf'\b{w}\b', raw_text)]
        if len(non_ktp_matches) >= 2:
            penalties.append(f"Terdeteksi kata kunci struk/faktur: {', '.join(non_ktp_matches[:3])}")
            penalty_score = min(0.60, len(non_ktp_matches) * 0.20)
        else:
            penalty_score = 0.0

        # 1. Header Score (25%)
        has_header = any(h in raw_text for h in ["PROVINSI", "KABUPATEN", "KOTA", "REPUBLIK"])
        header_score = 0.25 if has_header else 0.0
        details["has_official_header"] = has_header
        if not has_header:
            penalties.append("Header resmi (Provinsi/Kabupaten/Kota) tidak terdeteksi")

        # 2. NIK Signature Score (35%)
        has_nik_keyword = bool(re.search(r'\b(NIK|N1K|NOMOR INDUK)\b', raw_text))
        has_16_digits = bool(re.search(r'\b\d{16}\b', raw_text))
        has_near_digits = bool(re.search(r'\b\d{14,17}\b', raw_text))

        if has_16_digits:
            nik_score = 0.35
        elif has_nik_keyword and has_near_digits:
            nik_score = 0.30
        elif has_near_digits:
            nik_score = 0.20
        elif has_nik_keyword:
            nik_score = 0.15
        else:
            nik_score = 0.0
            penalties.append("Pola NIK (16 digit) tidak ditemukan")
        details["nik_detected"] = (has_16_digits or (has_nik_keyword and has_near_digits))

        # 3. Canonical Labels Density (30%)
        matched_labels = []
        for lbl in self.CANONICAL_LABELS:
            if lbl in raw_text:
                matched_labels.append(lbl)

        label_count = len(matched_labels)
        details["matched_labels_count"] = label_count
        details["matched_labels"] = matched_labels

        if label_count >= 6:
            label_score = 0.30
        elif label_count >= 4:
            label_score = 0.22
        elif label_count >= 2:
            label_score = 0.12
        else:
            label_score = 0.0
            penalties.append(f"Label standar KTP terlalu sedikit ({label_count} label)")

        # 4. Topological Layout Score (10%)
        left_items_count = sum(1 for it in text_items if it.get("x", 0) < page_width * 0.45)
        total_items = max(len(text_items), 1)
        left_ratio = left_items_count / total_items
        details["left_items_ratio"] = round(left_ratio, 2)

        if left_ratio >= 0.30:
            layout_score = 0.10
        else:
            layout_score = 0.04

        raw_semantic_score = header_score + nik_score + label_score + layout_score
        semantic_score = max(0.0, raw_semantic_score - penalty_score)
        return round(semantic_score, 3), details, penalties

    def validate(self, image: np.ndarray, ocr_result: Dict[str, Any]) -> KTPValidationResult:
        """
        Runs both tiers and produces a composite decision.
        """
        visual_score, visual_details, visual_penalties = self.validate_visual(image)
        semantic_score, semantic_details, semantic_penalties = self.validate_semantics(ocr_result)

        # Composite score formula: 20% visual, 80% semantic
        confidence = (0.20 * visual_score) + (0.80 * semantic_score)
        confidence = round(max(0.0, min(1.0, confidence)), 3)

        is_ktp = confidence >= self.min_confidence

        all_penalties = []
        if not is_ktp:
            all_penalties = semantic_penalties + visual_penalties

        detected_features = {
            **visual_details,
            **semantic_details,
            "min_confidence_threshold": self.min_confidence
        }

        return KTPValidationResult(
            is_ktp=is_ktp,
            confidence_score=confidence,
            visual_score=round(visual_score, 3),
            semantic_score=round(semantic_score, 3),
            rejection_reasons=all_penalties,
            detected_features=detected_features
        )
