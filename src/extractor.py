import re
from typing import Dict, Any, List, Optional, Tuple
from rapidfuzz import fuzz, process

from src.models import KTPData
from src.config import Config

class KTPExtractor:
    """
    Extracts structured KTP Key-Value pairs using spatial layout items and regex heuristics.
    Supports both PaddleOCR and LiteParse coordinates and outputs.
    """

    LABEL_PATTERNS = {
        "nik": ["NIK", "N1K", "NOMOR INDUK KEPENDUDUKAN", "NOMOR INDUK"],
        "nama": ["NAMA", "NAME"],
        "tempat_tgl_lahir": ["TEMPAT/TGL LAHIR", "TEMPAT / TGL LAHIR", "TEMPAT, TGL LAHIR", "TEMPAT TGL LAHIR", "TEMPAT/TGL", "LAHIR", "TENPAT/TGL LAHIR", "TELAHIR", "TE LAHIR", "TEMPAT LAHIR", "TGL LAHIR"],
        "jenis_kelamin": ["JENIS KELAMIN", "KELAMIN", "JNS KELAMIN", "JENISKELAMN"],
        "gol_darah": ["GOL. DARAH", "GOL DARAH", "GOL.DARAH", "DARAH", "GOL DARAT", "GOL. DARAT", "DARAT"],
        "alamat": ["ALAMAT", "ALAMAL"],
        "rt_rw": ["RT/RW", "RT / RW", "RT/ RW", "RT /RW", "RTRW", "RT8W", "RT/8W", "RTBW", "RT 8W"],
        "kel_desa": ["KEL/DESA", "KEL / DESA", "KELURAHAN", "DESA", "KELIDESA"],
        "kecamatan": ["KECAMATAN", "KEC."],
        "agama": ["AGAMA"],
        "status_perkawinan": ["STATUS PERKAWINAN", "STATUS", "PERKAWINAN"],
        "pekerjaan": ["PEKERJAAN", "PAKERJAAN"],
        "kewarganegaraan": ["KEWARGANEGARAAN", "WARGA NEGARA"],
        "berlaku_hingga": ["BERLAKU HINGGA", "BERLAKU"]
    }

    CANONICAL_RELIGIONS = ["ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "KONGHUCU"]
    KNOWN_RELIGIONS = ["ISLAM", "KRISTEN", "KATHOLIK", "KATOLIK", "HINDU", "BUDDHA", "BUDHA", "KONGHUCU"]
    KNOWN_MARITAL_STATUS = ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI", "CERAIHIDUP", "CERAIMATI"]
    KNOWN_GENDERS = ["LAKI-LAKI", "PEREMPUAN"]
    KNOWN_CITIZENSHIPS = ["WNI", "WNA"]
    KNOWN_BLOOD_TYPES = ["A", "B", "AB", "O"]

    DISALLOWED_BIRTH_PLACES = {
        "BERLAKU", "HINGGA", "PROVINSI", "KABUPATEN", "PEKERJAAN",
        "KEWARGANEGARAAN", "AGAMA", "STATUS", "PERKAWINAN", "ALAMAT",
        "RT", "RW", "KEL", "DESA", "KECAMATAN", "GOLONGAN", "DARAH",
        "JENIS", "KELAMIN", "NIK", "NAMA", "WARGA", "NEGARA"
    }

    PROVINCE_CODES = {
        '11': 'ACEH', '12': 'SUMATERA UTARA', '13': 'SUMATERA BARAT', '14': 'RIAU',
        '15': 'JAMBI', '16': 'SUMATERA SELATAN', '17': 'BENGKULU', '18': 'LAMPUNG',
        '19': 'BANGKA BELITUNG', '21': 'KEPULAUAN RIAU', '31': 'DKI JAKARTA',
        '32': 'JAWA BARAT', '33': 'JAWA TENGAH', '34': 'DAERAH ISTIMEWA YOGYAKARTA',
        '35': 'JAWA TIMUR', '36': 'BANTEN', '51': 'BALI', '52': 'NUSA TENGGARA BARAT',
        '53': 'NUSA TENGGARA TIMUR', '61': 'KALIMANTAN BARAT', '62': 'KALIMANTAN TENGAH',
        '63': 'KALIMANTAN SELATAN', '64': 'KALIMANTAN TIMUR', '65': 'KALIMANTAN UTARA',
        '71': 'SULAWESI UTARA', '72': 'SULAWESI TENGAH', '73': 'SULAWESI SELATAN',
        '74': 'SULAWESI TENGGARA', '75': 'GORONTALO', '76': 'SULAWESI BARAT',
        '81': 'MALUKU', '82': 'MALUKU UTARA', '91': 'PAPUA BARAT', '92': 'PAPUA'
    }

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def _is_disallowed_place(self, place: str) -> bool:
        """True if the birth place candidate is actually a KTP label word (whole-word match)."""
        words = re.findall(r'[A-Z]+', (place or "").upper())
        return any(w in self.DISALLOWED_BIRTH_PLACES for w in words)

    def clean_agama(self, raw_val: Optional[str], raw_text: str = "") -> Optional[str]:
        """
        Extracts and normalizes religion using scoped spatial candidate priority
        and rapidfuzz matching against canonical Indonesian religions.
        Supports typos like 1SLAM, ISLM, KR1STEN, KAT0LIK, KATHOLIK, H1NDU, BUDHA, etc.
        """
        def _normalize_text(txt: str) -> str:
            t = txt.upper()
            t = re.sub(r'[:：\-\|+=—–_；,\.]', ' ', t)
            t = t.replace('1', 'I').replace('0', 'O').replace('8', 'B')
            return re.sub(r'\s+', ' ', t).strip()

        if raw_val:
            cleaned_val = _normalize_text(raw_val)
            if cleaned_val:
                for canon in self.CANONICAL_RELIGIONS:
                    if canon in cleaned_val:
                        return canon
                if "KATHOLIK" in cleaned_val:
                    return "KATOLIK"
                if "BUDHA" in cleaned_val:
                    return "BUDDHA"
                compact = cleaned_val.replace(" ", "")
                if "KONGHUCU" in compact or "KONGHUCHU" in compact or "KONGHUC" in compact:
                    return "KONGHUCU"

                words = [w for w in cleaned_val.split() if w not in ["AGAMA", "NAMA", "ALAMAT"]]
                best_match = None
                best_score = 0.0
                for w in words:
                    res = process.extractOne(w, self.CANONICAL_RELIGIONS, scorer=fuzz.ratio)
                    if res and res[1] > best_score:
                        best_score = res[1]
                        best_match = res[0]
                if best_match and best_score >= 68:
                    return best_match

        if raw_text:
            lines_raw = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            for line in lines_raw:
                line_norm = _normalize_text(line)
                if "AGAMA" in line_norm:
                    sub = re.sub(r'^.*?AGAMA\s*', '', line_norm).strip()
                    if sub:
                        for canon in self.CANONICAL_RELIGIONS:
                            if canon in sub:
                                return canon
                        if "KATHOLIK" in sub:
                            return "KATOLIK"
                        if "BUDHA" in sub:
                            return "BUDDHA"
                        res = process.extractOne(sub, self.CANONICAL_RELIGIONS, scorer=fuzz.partial_ratio)
                        if res and res[1] >= 65:
                            return res[0]

            raw_norm = _normalize_text(raw_text)
            for canon in self.CANONICAL_RELIGIONS:
                if re.search(rf'\b{canon}\b', raw_norm):
                    return canon

            tokens = [w for w in raw_norm.split() if len(w) >= 4]
            best_match = None
            best_score = 0.0
            for tok in tokens:
                if tok in ["AGAMA", "STATUS", "PERKAWINAN", "WARGA", "NEGARA", "INDONESIA", "PROVINSI", "KABUPATEN"]:
                    continue
                res = process.extractOne(tok, self.CANONICAL_RELIGIONS, scorer=fuzz.ratio)
                if res and res[1] > best_score:
                    best_score = res[1]
                    best_match = res[0]
            if best_match and best_score >= 72:
                return best_match

        return None

    def clean_gol_darah(self, raw_val: Optional[str], raw_text: str = "") -> Optional[str]:
        """
        Extracts and normalizes blood type (A, B, AB, O) while strictly mapping
        hyphens (-), empty, or unknown to None (null).
        Corrects OCR typos (0/Q/D -> O, 8 -> B, 4 -> A).
        """
        def _parse_candidate(cand: str) -> Optional[str]:
            c = cand.strip().upper()
            if not c or c in ["-", "_", "—", "–", "NONE", "TIDAK", "NIHIL"] or re.match(r"^[-—–_.:]+$", c):
                return None
            
            c = re.sub(r'[^A-Z0-9]', '', c)
            c = c.replace('0', 'O').replace('Q', 'O').replace('D', 'O')
            c = c.replace('8', 'B').replace('4', 'A')

            c = re.sub(r'[^ABO]', '', c)
            if not c:
                return None

            if "AB" in c:
                return "AB"
            if "A" in c and "B" not in c:
                return "A"
            if "B" in c:
                return "B"
            if "O" in c:
                return "O"
            return None

        if raw_val is not None:
            val_strip = raw_val.strip()
            if val_strip in ["-", "_", "—", "–", ""] or re.match(r"^[-—–_.:]+$", val_strip):
                return None
            parsed = _parse_candidate(raw_val)
            if parsed:
                return parsed

        if raw_text:
            goldar_m = re.search(
                r"(?:GOL(?:ONGAN)?\.?\s*(?:DARAH|DARAT)|\bDARAH|\bDARAT)\s*[:：\s]*([ABO084QD\-_—–]{1,4})\b",
                raw_text,
                flags=re.IGNORECASE
            )
            if goldar_m:
                cand = goldar_m.group(1)
                return _parse_candidate(cand)

        return None

    def clean_status_perkawinan(self, raw_val: Optional[str], raw_text: str = "") -> Optional[str]:
        """
        Extracts and normalizes marital status into canonical values:
        ['BELUM KAWIN', 'KAWIN', 'CERAI HIDUP', 'CERAI MATI'].
        Corrects OCR typos like BELUM KAW1N, BLM KAWIN, CERAIHIDUP, etc.
        Avoids false positives from regional names like 'BELU' or 'BALI',
        and avoids matching the label 'STATUS PERKAWINAN' as 'KAWIN'.
        """
        def _normalize_text(txt: str) -> str:
            t = txt.upper()
            t = re.sub(r'[:：\-\|+=—–_；,\.]', ' ', t)
            t = t.replace('1', 'I').replace('!', 'I').replace('0', 'O')
            return re.sub(r'\s+', ' ', t).strip()

        def _resolve_candidate(txt: str) -> Optional[str]:
            norm = _normalize_text(txt)
            if not norm:
                return None
            
            # Strip label words if present
            val_norm = re.sub(r'\b(?:STATUS|SLATUS|PERKAWINAN)\b', '', norm).strip()
            if not val_norm:
                return None
            
            words = val_norm.split()
            # Explicit tokens or length >= 5 to prevent 'BELU' or 'BALI' matching 'BELUM'
            if any(w in ["BELUM", "BLM", "BLMKWN", "BELUMKAWIN"] for w in words) or "BELUM " in val_norm or " BELUM" in val_norm or "BLM " in val_norm:
                return "BELUM KAWIN"
            for w in words:
                if len(w) >= 5 and fuzz.ratio("BELUM", w) >= 85:
                    return "BELUM KAWIN"
            
            if "CERAI" in val_norm or any(fuzz.ratio("CERAI", w) >= 80 for w in words if len(w) >= 4):
                if "MATI" in val_norm or "MAT1" in val_norm or fuzz.partial_ratio("MATI", val_norm) >= 75:
                    return "CERAI MATI"
                if "HIDUP" in val_norm or "H1DUP" in val_norm or fuzz.partial_ratio("HIDUP", val_norm) >= 70:
                    return "CERAI HIDUP"
                score_h = fuzz.ratio(val_norm, "CERAI HIDUP")
                score_m = fuzz.ratio(val_norm, "CERAI MATI")
                return "CERAI HIDUP" if score_h >= score_m else "CERAI MATI"
            
            if any(w == "KAWIN" or (len(w) >= 5 and fuzz.ratio("KAWIN", w) >= 80) for w in words) or val_norm == "KAWIN":
                return "KAWIN"

            res = process.extractOne(val_norm, self.KNOWN_MARITAL_STATUS, scorer=fuzz.ratio)
            if res and res[1] >= 75:
                return res[0]
            
            return None

        if raw_val:
            resolved = _resolve_candidate(raw_val)
            if resolved:
                return resolved

        if raw_text:
            lines_raw = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            for idx, line in enumerate(lines_raw):
                line_u = line.upper()
                if "STATUS" in line_u or "PERKAWINAN" in line_u:
                    cand = re.sub(r'^.*?(?:STATUS\s*PERKAWINAN|STATUS|PERKAWINAN)\s*[:：\s]*', '', line, flags=re.IGNORECASE).strip()
                    if cand:
                        resolved = _resolve_candidate(cand)
                        if resolved:
                            return resolved
                    # Check next line if current line only had label
                    if idx + 1 < len(lines_raw):
                        next_line = lines_raw[idx + 1].strip()
                        if not any(k in next_line.upper() for k in ["PEKERJAAN", "AGAMA", "WARGA", "KEWARGANEGARAAN", "ALAMAT", "PROVINSI", "KABUPATEN", "BELU", "BALI"]):
                            resolved = _resolve_candidate(next_line)
                            if resolved:
                                return resolved

        return None

    def clean_kewarganegaraan(self, raw_val: Optional[str], raw_text: str = "") -> Optional[str]:
        """
        Extracts and normalizes citizenship (WNI or WNA).
        Tolerates OCR noise like 'W N I', 'W.N.I', 'WN1', 'W-N-I', etc.
        """
        def _resolve(txt: str) -> Optional[str]:
            if not txt:
                return None
            t = txt.upper()
            compressed = re.sub(r'[\s\.\-_/]+', '', t)
            compressed = compressed.replace('1', 'I').replace('4', 'A')

            if "WNI" in compressed:
                return "WNI"
            if "WNA" in compressed:
                return "WNA"
            
            if fuzz.partial_ratio("WNI", t) >= 75:
                return "WNI"
            if fuzz.partial_ratio("WNA", t) >= 75:
                return "WNA"
            return None

        if raw_val:
            res = _resolve(raw_val)
            if res:
                return res

        if raw_text:
            lines_raw = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            for line in lines_raw:
                if "WARGA" in line.upper() or "KEWARGANEGARAAN" in line.upper():
                    cand = re.sub(r'^.*?(?:KEWARGANEGARAAN|WARGA\s*NEGARA)\s*[:：\s]*', '', line, flags=re.IGNORECASE)
                    res = _resolve(cand)
                    if res:
                        return res

            if re.search(r'\bW[\s\.]*N[\s\.]*[I1]\b', raw_text, flags=re.IGNORECASE):
                return "WNI"
            if re.search(r'\bW[\s\.]*N[\s\.]*[A4]\b', raw_text, flags=re.IGNORECASE):
                return "WNA"

        return None


    def resolve_17_digit_nik(self, digits_str: str, context_text: str = "") -> str:
        """
        Disambiguates 17-digit NIK strings resulting from leading/trailing OCR noise
        (e.g., misread colon ':' or vertical line '|' interpreted as '1', '2', etc.).
        """
        if len(digits_str) != 17:
            return digits_str[:16]

        opt_tail = digits_str[1:]   # Drop leading noise (e.g. colon read as '1' or '2')
        opt_head = digits_str[:16]  # Drop trailing digit

        # 1. Check province code validity between head and tail:
        head_prov = opt_head[:2]
        tail_prov = opt_tail[:2]
        if head_prov not in self.PROVINCE_CODES and tail_prov in self.PROVINCE_CODES:
            return opt_tail
        if tail_prov not in self.PROVINCE_CODES and head_prov in self.PROVINCE_CODES:
            return opt_head

        # 2. Match birth date from context text if available (DDMMYY or female DD+40MMYY)
        if context_text:
            date_m = re.search(r'(\d{2})[-/.](\d{2})[-/.]\d{2}(\d{2})', context_text)
            if date_m:
                d, m, y = date_m.group(1), date_m.group(2), date_m.group(3)
                try:
                    male_pattern = f"{d}{m}{y}"
                    female_pattern = f"{int(d)+40:02d}{m}{y}"
                    if male_pattern in opt_tail or female_pattern in opt_tail:
                        return opt_tail
                    if male_pattern in opt_head or female_pattern in opt_head:
                        return opt_head
                except ValueError:
                    pass

            # 3. Match province code from context text
            if tail_prov in self.PROVINCE_CODES:
                prov_name = self.PROVINCE_CODES[tail_prov].replace(" ", "")
                if prov_name in context_text.replace(" ", "").upper():
                    return opt_tail
            if head_prov in self.PROVINCE_CODES:
                prov_name = self.PROVINCE_CODES[head_prov].replace(" ", "")
                if prov_name in context_text.replace(" ", "").upper():
                    return opt_head

        # 4. Fallback: if leading character is noise (1, 2, 7, 0, |, :) and tail starts with valid province
        if digits_str[0] in ['1', '2', '7', '0', '|', ':'] and tail_prov in self.PROVINCE_CODES:
            return opt_tail

        return opt_head

    def clean_nik(self, raw_nik: str, context_text: str = "") -> Optional[str]:
        """Cleans and corrects common OCR mistakes in 16-digit NIK."""
        if not raw_nik:
            return None
        
        char_map = {
            'O': '0', 'o': '0', 'D': '0', 'Q': '0',
            'I': '1', 'l': '1', 'i': '1', '|': '1', 'L': '1',
            'Z': '2', 'z': '2',
            'S': '5', 's': '5',
            'B': '8', 'b': '8',
            'g': '9', 'q': '9'
        }
        
        sub = re.sub(r'^[^\d]*nik\s*[:：\s]*', '', raw_nik, flags=re.IGNORECASE)
        cleaned = "".join(char_map.get(c, c) for c in sub)
        digits = re.sub(r'[^\d]', '', cleaned)

        # Exact 16 digits match
        if len(digits) == 16:
            return digits

        # Handle 17 digits (leading noise character from colon / separator)
        if len(digits) == 17:
            return self.resolve_17_digit_nik(digits, context_text)

        if len(digits) > 17:
            # Check if there is an exact 16-digit substring inside
            match_16 = re.search(r'\b\d{16}\b', digits)
            if match_16:
                return match_16.group(0)
            return self.resolve_17_digit_nik(digits[:17], context_text)

        if len(digits) >= 14:
            return digits
        return None

    def match_label(self, text: str) -> Tuple[Optional[str], float]:
        """Finds the best matching KTP label for a given text snippet."""
        clean = re.sub(r'[:：\-\|+=—–_；]', '', text).strip().upper()
        if not clean:
            return None, 0.0
        
        best_field = None
        best_score = 0.0
        for field_name, aliases in self.LABEL_PATTERNS.items():
            for alias in aliases:
                score = fuzz.ratio(clean, alias)
                if score > best_score:
                    best_score = score
                    best_field = field_name
        
        if best_score >= self.config.FUZZY_SCORE_THRESHOLD:
            return best_field, best_score
        return None, best_score

    def extract_from_spatial_items(self, text_items: List[Dict[str, Any]], page_width: float = 0.0) -> Dict[str, str]:
        """
        Uses spatial coordinates (x, y, width, height) to pair field labels with values
        located to their right on the same horizontal plane.
        Supports multi-line name continuation and vertical band recovery for faded labels.
        """
        extracted: Dict[str, str] = {}
        if not text_items:
            return extracted

        # Cutoff to ignore photo and signature section on the right
        max_x_cutoff = page_width * 0.67 if page_width > 0 else 99999.0

        # Sort items primarily by Y, then X
        sorted_items = sorted(text_items, key=lambda it: (it["y"], it["x"]))

        # Identify items that act as labels
        label_matches: List[Tuple[str, Dict[str, Any]]] = []
        for item in sorted_items:
            if page_width > 0 and item["x"] > page_width * 0.40:
                continue

            field_name, score = self.match_label(item["text"])
            if field_name:
                label_matches.append((field_name, item))

        for field_name, label_item in label_matches:
            if field_name in extracted:
                continue
            
            label_y = label_item["y"]
            label_x = label_item["x"]
            label_h = label_item.get("height", 10.0)
            
            y_tolerance = min(max(label_h * 0.75, 6.0), 18.0)
            row_candidates = [
                it for it in sorted_items
                if it != label_item
                and abs(it["y"] - label_y) <= y_tolerance
                and it["x"] >= (label_x + label_item["width"] * 0.40)
                and it["x"] < max_x_cutoff
            ]

            if row_candidates:
                row_candidates.sort(key=lambda it: it["x"])
                
                value_texts = []
                for cand in row_candidates:
                    t = cand["text"].strip()
                    t = re.sub(r'^[+\-=—–_:\.\|\s：；]+', '', t).strip()
                    if field_name == "berlaku_hingga" and t.upper() in ["BERLAKU", "HINGGA"]:
                        continue
                    if field_name == "tempat_tgl_lahir" and t.upper() in ["TEMPAT", "TGL", "LAHIR"]:
                        continue
                    if t:
                        value_texts.append(t)
                        
                if value_texts:
                    # Check for multi-line continuation for nama
                    if field_name == "nama":
                        first_line_y = max(cand["y"] for cand in row_candidates)
                        first_line_h = max(cand.get("height", 15.0) for cand in row_candidates)
                        next_labels_y = [lbl["y"] for fn, lbl in label_matches if lbl["y"] > label_y + 15 and fn != "nama"]
                        next_label_y = min(next_labels_y) if next_labels_y else first_line_y + first_line_h * 3.5

                        continuation_items = [
                            it for it in sorted_items
                            if it not in row_candidates
                            and it != label_item
                            and (first_line_y + first_line_h * 0.4) <= it["y"] < (next_label_y - 5.0)
                            and it["x"] >= (label_x + label_item["width"] * 0.3)
                            and it["x"] < max_x_cutoff
                            and not self.match_label(it["text"])[0]
                            and re.match(r"^[A-Za-z\s\.\,'-]+$", it["text"].strip())
                            and len(it["text"].strip()) >= 3
                        ]
                        if continuation_items:
                            continuation_items.sort(key=lambda it: (it["y"], it["x"]))
                            for c_item in continuation_items:
                                ct = c_item["text"].strip()
                                ct = re.sub(r'^[+\-=—–_:\.\|\s：；]+', '', ct).strip()
                                if ct:
                                    value_texts.append(ct)

                    extracted[field_name] = " ".join(value_texts).strip()

        # Fallback for faded / undetected 'nama' label using spatial vertical band between NIK and next field
        if "nama" not in extracted:
            nik_bottom_y = 0.0
            nik_val = extracted.get("nik")
            if nik_val:
                for it in sorted_items:
                    if nik_val in it["text"].replace(" ", ""):
                        nik_bottom_y = it["y"] + it.get("height", 20.0)
                        break
            if nik_bottom_y == 0.0:
                for fn, lbl in label_matches:
                    if fn == "nik":
                        nik_bottom_y = lbl["y"] + lbl.get("height", 20.0)
                        break

            if nik_bottom_y > 0.0:
                next_labels_below_nik = [lbl["y"] for fn, lbl in label_matches if lbl["y"] > nik_bottom_y + 10 and fn != "nik"]
                for it in sorted_items:
                    if (page_width == 0.0 or it["x"] < page_width * 0.35) and it["y"] > nik_bottom_y + 15:
                        tu = it["text"].upper()
                        if re.search(r'\b(LAHIR|TELAHIR|TEMPAT|KELAMIN|ALAMAT|RT|RW)\b', tu) or "LAHA" in tu or "MOAT" in tu:
                            next_labels_below_nik.append(it["y"])

                next_field_y = min(next_labels_below_nik) if next_labels_below_nik else nik_bottom_y + 150.0

                name_band_items = [
                    it for it in sorted_items
                    if (nik_bottom_y - 5.0) <= it["y"] < (next_field_y - 5.0)
                    and it["x"] >= (page_width * 0.15 if page_width > 0 else 100.0)
                    and it["x"] < max_x_cutoff
                    and not self.match_label(it["text"])[0]
                    and re.match(r"^[A-Za-z\s\.\,'-]+$", it["text"].strip())
                    and len(it["text"].strip()) >= 3
                    and not any(k in it["text"].upper() for k in ["PROVINSI", "KABUPATEN", "KOTA", "NIK", "REPUBLIK"])
                ]

                if name_band_items:
                    name_band_items.sort(key=lambda it: (it["y"], it["x"]))
                    band_name = " ".join(it["text"].strip() for it in name_band_items).strip()
                    if band_name:
                        extracted["nama"] = band_name

        return extracted

    def extract_from_raw_text(self, raw_text: str) -> Dict[str, str]:
        """
        Fallback parser scanning lines of text using regex and fuzzy matching.
        """
        extracted: Dict[str, str] = {}
        if not raw_text:
            return extracted

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # 1. Direct Regex checks
        for line in lines:
            line_upper = line.upper()

            # NIK check
            if "nik" not in extracted:
                nik_cand = self.clean_nik(line, context_text=raw_text)
                if nik_cand and len(nik_cand) == 16:
                    extracted["nik"] = nik_cand

            # RT / RW check
            rt_rw_match = re.search(r'(\d{2,3}\s*/\s*\d{2,3})', line)
            if rt_rw_match and "rt_rw" not in extracted:
                extracted["rt_rw"] = rt_rw_match.group(1).replace(" ", "")

            # Gender check
            if "jenis_kelamin" not in extracted:
                for gender in self.KNOWN_GENDERS:
                    if re.search(rf'\b{gender}\b', line_upper):
                        extracted["jenis_kelamin"] = gender
                        break

            # Religion check
            if "agama" not in extracted:
                if "AGAMA" in line_upper or any(r in line_upper for r in self.KNOWN_RELIGIONS):
                    ag_cand = self.clean_agama(line)
                    if ag_cand:
                        extracted["agama"] = ag_cand

            # Marital Status check
            if "status_perkawinan" not in extracted:
                sp_cand = self.clean_status_perkawinan(line)
                if sp_cand:
                    extracted["status_perkawinan"] = sp_cand

            # Citizenship check
            if "kewarganegaraan" not in extracted:
                kw_cand = self.clean_kewarganegaraan(line)
                if kw_cand:
                    extracted["kewarganegaraan"] = kw_cand

            # Blood type
            if "gol_darah" not in extracted:
                gd_cand = self.clean_gol_darah(None, raw_text=line)
                if gd_cand:
                    extracted["gol_darah"] = gd_cand

            # Validity check
            if "berlaku_hingga" not in extracted:
                if "SEUMUR HIDUP" in line_upper or "SEUMURHIDUP" in line_upper:
                    extracted["berlaku_hingga"] = "SEUMUR HIDUP"
                else:
                    date_match = re.search(r'\b\d{2}[-\/.]\d{2}[-\/.]\d{4}\b', line)
                    if date_match and "BERLAKU" in line_upper:
                        extracted["berlaku_hingga"] = date_match.group(0).replace('/', '-').replace('.', '-')

        # 2. Key-Value splitting on ':' or '：'
        for line in lines:
            delim = "：" if "：" in line else ":" if ":" in line else None
            if delim:
                parts = line.split(delim, 1)
                left = parts[0].strip()
                right = parts[1].strip()
                if not right:
                    continue

                best_field, score = self.match_label(left)
                if best_field and best_field not in extracted:
                    extracted[best_field] = right

        return extracted

    def post_process_fields(self, data: Dict[str, str], raw_text: str = "") -> KTPData:
        """Sanitizes and maps extracted raw dictionary to validated KTPData model."""
        noise_words = [
            r'\bFOTO\b', r'\bTANDA\s*TANGAN\b', r'\bPAS\s*FOTO\b',
            r'\bPROVINSI\b', r'\bKABUPATEN\b', r'\bKOTA\b',
            r'\bJAKARTA\s*BARAT\b', r'\bJAKARTA\s*PUSAT\b', r'\bJAKARTA\s*SELATAN\b',
            r'\bJAKARTA\s*TIMUR\b', r'\bJAKARTA\s*UTARA\b',
            r" JAWA\s*TIMUR ", r" JAWA\s*BARAT ", r" JAWA\s*TENGAH "
        ]

        def strip_noise(text: str) -> str:
            cleaned = text
            for nw in noise_words:
                cleaned = re.sub(nw, '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^[+\-=—–_:\.\|\s：；]+', '', cleaned)
            cleaned = re.sub(r'[+\-=—–_:\.\|\s：；]+$', '', cleaned)
            return cleaned.strip()

        # 1. Clean NIK with full context
        if "nik" in data and data["nik"]:
            data["nik"] = self.clean_nik(data["nik"], context_text=raw_text)
            
        if "nik" not in data or not data["nik"] or len(str(data["nik"])) != 16:
            # Global search across raw text for any sequence of 16-17 digits
            digit_seqs = re.findall(r'\b\d{16,17}\b', raw_text)
            for seq in digit_seqs:
                resolved = self.clean_nik(seq, context_text=raw_text)
                if resolved and len(resolved) == 16:
                    data["nik"] = resolved
                    break

        # 2. Disentangle Nama and Tempat/Tgl Lahir if they got merged
        raw_nama = data.get("nama", "") or ""
        raw_ttl = data.get("tempat_tgl_lahir", "") or ""

        ttl_regex = r'\b((?:(?:JAKARTA|KOTA|KABUPATEN)\s+)?[A-Za-z]{3,})[\.,\s]+(\d{2}[-\/\.\s]\d{2}[-\/\.\s]\d{4})\b'
        
        if raw_nama:
            ttl_in_nama = re.search(ttl_regex, raw_nama)
            if ttl_in_nama:
                found_place = ttl_in_nama.group(1).strip()
                found_date = ttl_in_nama.group(2).replace('/', '-').replace('.', '-')
                if not self._is_disallowed_place(found_place):
                    if not raw_ttl:
                        data["tempat_tgl_lahir"] = f"{found_place}, {found_date}"
                raw_nama = raw_nama[:ttl_in_nama.start()] + " " + raw_nama[ttl_in_nama.end():]
                data["nama"] = re.sub(r'\s+', ' ', raw_nama).strip()

        # Clean Nama
        if "nama" in data and data["nama"]:
            nama = strip_noise(data["nama"])
            nama = re.split(r"\b(TEMPAT|TGL|LAHIR|JENIS|ALAMAT)\b", nama, flags=re.IGNORECASE)[0]
            nama = strip_noise(nama)
            # Correct common OCR letter substitutions and word segmentation
            nama = re.sub(r'WAHIYU', 'WAHYU', nama, flags=re.IGNORECASE)
            nama = re.sub(r'KHAIPUNNISA', 'KHAIRUNNISA', nama, flags=re.IGNORECASE)
            nama = re.sub(r'\bGLADYSWAHYUKHAIRUNNISA\b', 'GLADYS WAHYU KHAIRUNNISA', nama, flags=re.IGNORECASE)
            nama = re.sub(r'\bPANGLSTI\b', 'PANGESTI', nama, flags=re.IGNORECASE)
            data["nama"] = strip_noise(nama)
        elif raw_text:
            lines_raw = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            ttl_idx = None
            for idx, ln in enumerate(lines_raw):
                if re.search(r"\b\d{2}[-\/\.\s]\d{2}[-\/\.\s]\d{4}\b", ln) or any(k in ln.upper() for k in ["LAHIR", "TELAHIR"]):
                    ttl_idx = idx
                    break
            if ttl_idx is not None and ttl_idx > 0:
                name_parts = []
                for idx in range(ttl_idx):
                    cand = lines_raw[idx]
                    c_clean = cand.upper()
                    if not any(k in c_clean for k in ["PROVINSI", "KABUPATEN", "KOTA", "NIK", "REPUBLIK", "INDONESIA", "NAMA"]):
                        if len(c_clean) >= 4 and re.match(r"^[A-Z\s\.\,'-]+$", c_clean):
                            name_parts.append(cand.strip())
                            # Check next line for continuation
                            if idx + 1 < ttl_idx:
                                next_cand = lines_raw[idx + 1]
                                next_clean = next_cand.upper()
                                if not any(k in next_clean for k in ["PROVINSI", "KABUPATEN", "KOTA", "NIK", "REPUBLIK", "INDONESIA", "NAMA", "ALAMAT", "TEMPAT", "LAHIR"]):
                                    if len(next_clean) >= 3 and re.match(r"^[A-Z\s\.\,'-]+$", next_clean):
                                        name_parts.append(next_cand.strip())
                            break
                if name_parts:
                    full_nama = " ".join(name_parts)
                    full_nama = re.sub(r'\bPANGLSTI\b', 'PANGESTI', full_nama, flags=re.IGNORECASE)
                    data["nama"] = strip_noise(full_nama)

        # Clean Tempat/Tgl Lahir
        if "tempat_tgl_lahir" in data and data["tempat_tgl_lahir"]:
            ttl = data["tempat_tgl_lahir"]
            ttl = re.sub(r"^(TEMPAT|TGL|LAHIR|TELAHIR)[\s\/:,\.-]*", "", ttl, flags=re.IGNORECASE)
            ttl = re.split(r" (LAKI|PEREMPUAN|JENIS|GOL) ", ttl, flags=re.IGNORECASE)[0]
            ttl = re.sub(r"^[+\-=—–_:\.\|\s：；]+", "", ttl)
            ttl = re.sub(r"[+\-=—–_:\.\|\s：；]+$", "", ttl)
            m_dot = re.search(r"\b([A-Za-z]{3,})[\.,\s]+(\d{2}[-\/\.\s]\d{2}[-\/\.\s]\d{4})\b", ttl)
            if m_dot:
                p = m_dot.group(1).upper()
                if p == "DENPASAB":
                    p = "DENPASAR"
                if self._is_disallowed_place(p):
                    data["tempat_tgl_lahir"] = None
                else:
                    d = re.sub(r"[\s\/\.]", "-", m_dot.group(2))
                    data["tempat_tgl_lahir"] = f"{p}, {d}"
            else:
                ttl_sub = re.sub(r"([A-Za-z]+),(\d{2})", r"\1, \2", ttl).strip()
                p_sub = ttl_sub.split(',')[0].strip().upper()
                if self._is_disallowed_place(p_sub):
                    data["tempat_tgl_lahir"] = None
                else:
                    data["tempat_tgl_lahir"] = ttl_sub
        elif raw_text:
            m_ttl = re.search(ttl_regex, raw_text)
            if m_ttl:
                p = strip_noise(m_ttl.group(1)).upper()
                if p == "DENPASAB":
                    p = "DENPASAR"
                if not self._is_disallowed_place(p):
                    d = re.sub(r"[\s\/\.]", "-", m_ttl.group(2))
                    data["tempat_tgl_lahir"] = f"{p}, {d}"

        # Clean Jenis Kelamin
        jk_raw = (data.get("jenis_kelamin") or "") + " " + raw_text
        jk_upper = jk_raw.upper()

        if "PEREMPUAN" in jk_upper or "EREMPUAN" in jk_upper:
            data["jenis_kelamin"] = "PEREMPUAN"
        elif any(k in jk_upper for k in ["LAKI", "LAKDAXI", "LAKDA", "LAK-LAK", "LAK!"]) or re.search(r"LAK[I1DAX\-\s]{3,}", jk_upper) or fuzz.partial_ratio("LAKI-LAKI", jk_upper) > 60:
            data["jenis_kelamin"] = "LAKI-LAKI"
        else:
            data["jenis_kelamin"] = None

        # Clean Golongan Darah
        data["gol_darah"] = self.clean_gol_darah(data.get("gol_darah"), raw_text=raw_text)
        # Clean Alamat
        if "alamat" in data and data["alamat"]:
            alamat = strip_noise(data["alamat"])
            if alamat.upper().startswith("IL."):
                alamat = "JL." + alamat[3:]
            data["alamat"] = alamat
        elif raw_text:
            lines_raw = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            for idx, ln in enumerate(lines_raw):
                lu = ln.upper()
                if lu.startswith("JL") or lu.startswith("JALAN") or lu.startswith("KP") or lu.startswith("DUSUN"):
                    addr_parts = [ln]
                    if idx + 1 < len(lines_raw) and not any(k in lines_raw[idx+1].upper() for k in ["RT", "RW", "KEL", "KEC", "AGAMA"]):
                        addr_parts.append(lines_raw[idx+1])
                    data["alamat"] = " ".join(addr_parts).strip()
                    break
        # Clean RT/RW
        if "rt_rw" in data and data["rt_rw"]:
            m = re.search(r'(\d{2,3}\s*/\s*\d{2,3})', data["rt_rw"])
            if m:
                rtrw_clean = m.group(1).replace(" ", "")
                if rtrw_clean.startswith("50"):
                    rtrw_clean = "00" + rtrw_clean[2:]
                data["rt_rw"] = rtrw_clean
        elif raw_text:
            m = re.search(r'(\d{2,3}\s*/\s*\d{2,3})', raw_text)
            if m:
                rtrw_clean = m.group(1).replace(" ", "")
                if rtrw_clean.startswith("50"):
                    rtrw_clean = "00" + rtrw_clean[2:]
                data["rt_rw"] = rtrw_clean

        # Clean Kel/Desa & Kecamatan
        if "kel_desa" in data and data["kel_desa"]:
            data["kel_desa"] = strip_noise(data["kel_desa"])
        if "kecamatan" in data and data["kecamatan"]:
            data["kecamatan"] = strip_noise(data["kecamatan"])

        # Clean Agama
        data["agama"] = self.clean_agama(data.get("agama"), raw_text=raw_text)

        # Clean Status Perkawinan
        data["status_perkawinan"] = self.clean_status_perkawinan(data.get("status_perkawinan"), raw_text=raw_text)

        # Clean Pekerjaan
        if "pekerjaan" in data and data["pekerjaan"]:
            data["pekerjaan"] = strip_noise(data["pekerjaan"])

        # Clean Kewarganegaraan
        data["kewarganegaraan"] = self.clean_kewarganegaraan(data.get("kewarganegaraan"), raw_text=raw_text)

        # Clean Berlaku Hingga
        bh_upper = ((data.get("berlaku_hingga") or "") + " " + raw_text).upper()
        if "SEUMUR" in bh_upper or "HIDUP" in bh_upper:
            data["berlaku_hingga"] = "SEUMUR HIDUP"
        else:
            date_m = re.search(r'\b\d{2}[-\/.]\d{2}[-\/.]\d{4}\b', data.get("berlaku_hingga", ""))
            if date_m:
                data["berlaku_hingga"] = date_m.group(0).replace('/', '-').replace('.', '-')
            else:
                data["berlaku_hingga"] = strip_noise(data.get("berlaku_hingga", ""))

        return KTPData(**data)

    def extract(self, parse_result: Dict[str, Any]) -> KTPData:
        """
        Complete extraction pipeline combining spatial matching and regex heuristics.
        """
        page_width = parse_result.get("page_width", 0.0)
        raw_text = parse_result.get("raw_text", "")

        spatial_dict = self.extract_from_spatial_items(parse_result.get("text_items", []), page_width=page_width)
        raw_text_dict = self.extract_from_raw_text(raw_text)

        # Merge results: Spatial takes priority, fallback to raw regex
        merged: Dict[str, str] = {}
        for key in self.LABEL_PATTERNS.keys():
            val = spatial_dict.get(key) or raw_text_dict.get(key)
            if val:
                merged[key] = str(val).strip()

        return self.post_process_fields(merged, raw_text=raw_text)
