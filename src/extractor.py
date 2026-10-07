import re
from typing import Dict, Any, List, Optional, Tuple
from rapidfuzz import fuzz

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
        "tempat_tgl_lahir": ["TEMPAT/TGL LAHIR", "TEMPAT / TGL LAHIR", "TEMPAT, TGL LAHIR", "TEMPAT TGL LAHIR", "TEMPAT/TGL", "LAHIR", "TENPAT/TGL LAHIR"],
        "jenis_kelamin": ["JENIS KELAMIN", "KELAMIN", "JNS KELAMIN", "JENISKELAMN"],
        "gol_darah": ["GOL. DARAH", "GOL DARAH", "GOL.DARAH", "DARAH"],
        "alamat": ["ALAMAT", "ALAMAL"],
        "rt_rw": ["RT/RW", "RT / RW", "RT/ RW", "RT /RW", "RTRW"],
        "kel_desa": ["KEL/DESA", "KEL / DESA", "KELURAHAN", "DESA", "KELIDESA"],
        "kecamatan": ["KECAMATAN", "KEC."],
        "agama": ["AGAMA"],
        "status_perkawinan": ["STATUS PERKAWINAN", "STATUS", "PERKAWINAN"],
        "pekerjaan": ["PEKERJAAN", "PAKERJAAN"],
        "kewarganegaraan": ["KEWARGANEGARAAN", "WARGA NEGARA"],
        "berlaku_hingga": ["BERLAKU HINGGA", "BERLAKU"]
    }

    KNOWN_RELIGIONS = ["ISLAM", "KRISTEN", "KATHOLIK", "KATOLIK", "HINDU", "BUDDHA", "BUDHA", "KONGHUCU"]
    KNOWN_MARITAL_STATUS = ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI", "CERAIHIDUP", "CERAIMATI"]
    KNOWN_GENDERS = ["LAKI-LAKI", "PEREMPUAN"]

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

    def resolve_17_digit_nik(self, digits_str: str, context_text: str = "") -> str:
        """
        Disambiguates 17-digit NIK strings resulting from leading/trailing OCR noise
        (e.g., misread colon ':' or vertical line '|' interpreted as '1').
        """
        if len(digits_str) != 17:
            return digits_str[:16]

        opt_tail = digits_str[1:]   # Drop leading noise (e.g. colon read as '1')
        opt_head = digits_str[:16]  # Drop trailing digit

        # 1. Match birth date from context text if available (DDMMYY or female DD+40MMYY)
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

            # 2. Match province code from context text
            prov_tail = opt_tail[:2]
            if prov_tail in self.PROVINCE_CODES:
                prov_name = self.PROVINCE_CODES[prov_tail].replace(" ", "")
                if prov_name in context_text.replace(" ", "").upper():
                    return opt_tail

        # 3. Fallback: if leading character is 1/7/0 and tail starts with a valid province code
        if digits_str[0] in ['1', '7', '0', '|'] and opt_tail[:2] in self.PROVINCE_CODES:
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
                    extracted[field_name] = " ".join(value_texts).strip()

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
                for rel in self.KNOWN_RELIGIONS:
                    if re.search(rf'\b{rel}\b', line_upper):
                        extracted["agama"] = rel
                        break

            # Marital Status check
            if "status_perkawinan" not in extracted:
                for stat in self.KNOWN_MARITAL_STATUS:
                    if re.search(rf'\b{stat}\b', line_upper):
                        extracted["status_perkawinan"] = "CERAI HIDUP" if "CERAI" in stat and "HIDUP" in stat else "CERAI MATI" if "CERAI" in stat and "MATI" in stat else "BELUM KAWIN" if "BELUM" in stat else "KAWIN"
                        break

            # Citizenship check
            if "kewarganegaraan" not in extracted:
                if "WNI" in line_upper:
                    extracted["kewarganegaraan"] = "WNI"
                elif "WNA" in line_upper:
                    extracted["kewarganegaraan"] = "WNA"

            # Blood type
            if "gol_darah" not in extracted:
                goldar_match = re.search(r'(?:GOL(?:ONGAN)?\.?\s*DARAH|\bDARAH)\s*[:：\s]*([ABO]|AB)\b', line_upper)
                if goldar_match and goldar_match.group(1):
                    extracted["gol_darah"] = goldar_match.group(1)

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
            r"JAWA\s*TIMUR", r"JAWA\s*BARAT", r"JAWA\s*TENGAH"
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

        ttl_regex = r'\b((?:(?:JAKARTA|KOTA|KABUPATEN)\s+)?[A-Za-z]{3,}),\s*(\d{2}[-\/.]\d{2}[-\/.]\d{4})'
        
        if raw_nama:
            ttl_in_nama = re.search(ttl_regex, raw_nama)
            if ttl_in_nama:
                found_place = ttl_in_nama.group(1).strip()
                found_date = ttl_in_nama.group(2).replace('/', '-').replace('.', '-')
                if not raw_ttl:
                    data["tempat_tgl_lahir"] = f"{found_place}, {found_date}"
                raw_nama = raw_nama[:ttl_in_nama.start()] + " " + raw_nama[ttl_in_nama.end():]
                data["nama"] = re.sub(r'\s+', ' ', raw_nama).strip()

        # Clean Nama
        if "nama" in data and data["nama"]:
            nama = strip_noise(data["nama"])
            nama = re.split(r'\b(TEMPAT|TGL|LAHIR|JENIS|ALAMAT)\b', nama, flags=re.IGNORECASE)[0]
            data["nama"] = strip_noise(nama)

        # Clean Tempat/Tgl Lahir
        if "tempat_tgl_lahir" in data and data["tempat_tgl_lahir"]:
            ttl = data["tempat_tgl_lahir"]
            ttl = re.sub(r"^(TEMPAT|TGL|LAHIR)[\s\/:,\.-]*", "", ttl, flags=re.IGNORECASE)
            ttl = re.split(r"(LAKI|PEREMPUAN|JENIS|GOL)", ttl, flags=re.IGNORECASE)[0]
            ttl = re.sub(r"^[+\-=—–_:\.\|\s：；]+", "", ttl)
            ttl = re.sub(r"[+\-=—–_:\.\|\s：；]+$", "", ttl)
            ttl = re.sub(r"([A-Za-z]+),(\d{2})", r"\1, \2", ttl)
            data["tempat_tgl_lahir"] = ttl.strip()
        elif raw_text:
            m_ttl = re.search(ttl_regex, raw_text)
            if m_ttl:
                p = strip_noise(m_ttl.group(1))
                d = m_ttl.group(2).replace('/', '-').replace('.', '-')
                if not any(k in p.upper() for k in ["BERLAKU", "HINGGA", "PROVINSI", "KABUPATEN"]):
                    data["tempat_tgl_lahir"] = f"{p}, {d}"

        # Clean Jenis Kelamin & Golongan Darah
        jk_raw = (data.get("jenis_kelamin") or "") + " " + raw_text
        jk_upper = jk_raw.upper()

        if "gol_darah" not in data or not data["gol_darah"]:
            goldar_m = re.search(r'(?:GOL(?:ONGAN)?\.?\s*DARAH|\bDARAH)\s*[:：\s]*([ABO]|AB)\b', jk_upper)
            if goldar_m:
                data["gol_darah"] = goldar_m.group(1)

        if "PEREMPUAN" in jk_upper or "EREMPUAN" in jk_upper:
            data["jenis_kelamin"] = "PEREMPUAN"
        elif "LAKI" in jk_upper:
            data["jenis_kelamin"] = "LAKI-LAKI"
        else:
            data["jenis_kelamin"] = None

        # Clean Alamat
        if "alamat" in data and data["alamat"]:
            alamat = strip_noise(data["alamat"])
            if alamat.upper().startswith("IL."):
                alamat = "JL." + alamat[3:]
            data["alamat"] = alamat

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
        ag_search = (data.get("agama") or "") + " " + raw_text
        data["agama"] = None
        for r in self.KNOWN_RELIGIONS:
            if re.search(rf'\b{r}\b', ag_search.upper()):
                data["agama"] = r
                break

        # Clean Status Perkawinan
        sp_search = (data.get("status_perkawinan") or "") + " " + raw_text
        sp_upper = sp_search.upper()
        if "BELUM KAWIN" in sp_upper:
            data["status_perkawinan"] = "BELUM KAWIN"
        elif "CERAI HIDUP" in sp_upper or "CERAIHIDUP" in sp_upper:
            data["status_perkawinan"] = "CERAI HIDUP"
        elif "CERAI MATI" in sp_upper or "CERAIMATI" in sp_upper:
            data["status_perkawinan"] = "CERAI MATI"
        elif "KAWIN" in sp_upper:
            data["status_perkawinan"] = "KAWIN"

        # Clean Pekerjaan
        if "pekerjaan" in data and data["pekerjaan"]:
            data["pekerjaan"] = strip_noise(data["pekerjaan"])

        # Clean Kewarganegaraan
        kw_upper = ((data.get("kewarganegaraan") or "") + " " + raw_text).upper()
        if "WNI" in kw_upper:
            data["kewarganegaraan"] = "WNI"
        elif "WNA" in kw_upper:
            data["kewarganegaraan"] = "WNA"

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
