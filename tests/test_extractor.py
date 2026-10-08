import unittest
from src.extractor import KTPExtractor
from src.models import KTPData

class TestKTPExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = KTPExtractor()

    def test_clean_nik_with_ocr_typos(self):
        # Testing substitution of 'O' with '0', 'l' with '1'
        raw = "NIK : 32O123456789OOOl"
        cleaned = self.extractor.clean_nik(raw)
        self.assertEqual(cleaned, "3201234567890001")

    def test_clean_nik_valid_16_digits(self):
        raw = "3171010101900001"
        cleaned = self.extractor.clean_nik(raw)
        self.assertEqual(cleaned, "3171010101900001")

    def test_extract_from_raw_text(self):
        raw_text = """
PROVINSI DKI JAKARTA
JAKARTA PUSAT
NIK : 3171010101900001
Nama : AHMAD SANTOSO
Tempat/Tgl Lahir : JAKARTA, 12-08-1994
Jenis Kelamin : LAKI-LAKI  Gol. Darah : O
Alamat : JL. MERDEKA NO. 45
RT/RW : 003/005
Kel/Desa : GAMBIR
Kecamatan : GAMBIR
Agama : ISLAM
Status Perkawinan: BELUM KAWIN
Pekerjaan : KARYAWAN SWASTA
Kewarganegaraan : WNI
Berlaku Hingga : SEUMUR HIDUP
        """
        parse_result = {
            "raw_text": raw_text,
            "text_items": []
        }
        ktp_data: KTPData = self.extractor.extract(parse_result)
        
        self.assertEqual(ktp_data.nik, "3171010101900001")
        self.assertEqual(ktp_data.nama, "AHMAD SANTOSO")
        self.assertEqual(ktp_data.jenis_kelamin, "LAKI-LAKI")
        self.assertEqual(ktp_data.gol_darah, "O")
        self.assertEqual(ktp_data.rt_rw, "003/005")
        self.assertEqual(ktp_data.kel_desa, "GAMBIR")
        self.assertEqual(ktp_data.kecamatan, "GAMBIR")
        self.assertEqual(ktp_data.agama, "ISLAM")
        self.assertEqual(ktp_data.status_perkawinan, "BELUM KAWIN")
        self.assertEqual(ktp_data.pekerjaan, "KARYAWAN SWASTA")
        self.assertEqual(ktp_data.kewarganegaraan, "WNI")
        self.assertEqual(ktp_data.berlaku_hingga, "SEUMUR HIDUP")

    def test_spatial_pairing(self):
        # Simulate spatial layout items
        text_items = [
            {"text": "NIK", "x": 30.0, "y": 80.0, "width": 40.0, "height": 10.0, "confidence": 0.95},
            {"text": "3171010101900001", "x": 100.0, "y": 80.0, "width": 120.0, "height": 10.0, "confidence": 0.95},
            {"text": "Nama", "x": 30.0, "y": 100.0, "width": 50.0, "height": 10.0, "confidence": 0.95},
            {"text": "BUDI UTOMO", "x": 100.0, "y": 100.0, "width": 100.0, "height": 10.0, "confidence": 0.95},
        ]
        parse_result = {
            "raw_text": "",
            "text_items": text_items
        }
        ktp_data: KTPData = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nik, "3171010101900001")
        self.assertEqual(ktp_data.nama, "BUDI UTOMO")

    def test_mira_setiawan_extraction(self):
        raw_text = """
PROVINSI DKI JAKARTA
JAKARTA BARAT
NIK : 3171234567890123
Nama : MIRA SETIAWAN
Tempat/Tgl Lahir : JAKARTA, 18-02-1986
Jenis Kelamin : PEREMPUAN Gol. Darah : B
Alamat : JL. PASTI CEPAT A7/66
RT/RW : 007/008
Kel/Desa : PEGADUNGAN
Kecamatan : KALIDERES
Agama : ISLAM
Status Perkawinan: KAWIN
Pekerjaan : PEGAWAI SWASTA
Kewarganegaraan : WNI JAKARTA BARAT
Berlaku Hingga : 22-02-2017 02-12-2012
        """
        parse_result = {
            "raw_text": raw_text,
            "text_items": []
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nik, "3171234567890123")
        self.assertEqual(ktp_data.nama, "MIRA SETIAWAN")
        self.assertEqual(ktp_data.jenis_kelamin, "PEREMPUAN")
        self.assertEqual(ktp_data.gol_darah, "B")
        self.assertEqual(ktp_data.rt_rw, "007/008")
        self.assertEqual(ktp_data.kel_desa, "PEGADUNGAN")
        self.assertEqual(ktp_data.kecamatan, "KALIDERES")
        self.assertEqual(ktp_data.berlaku_hingga, "22-02-2017")

    def test_user_ktp_scenario(self):
        parse_result = {
            "raw_text": """
PROVINSI JAWA TIMUR
KABUPATEN BANYUWANGI
NIK : 3510120212040001
Nama : += AHMAD BANYUWANGI, 02-12-2004 PUTRA FIRDAUS
Tempat/Tgl Lahir : 
Jenis Kelamin : LAKI-LAKI Gol. Darah : -
Alamat : DUSUN KRAJAN BARAT
RT/RW : 502/003
Kel/Desa : LABANASEM
Kecamatan : —KABAT
Agama : ISLAM
Status Perkawinan: BELUM KAWIN
Pekerjaan : PELAJAR/MAHASISWA
Kewarganegaraan : WNI
Berlaku Hingga : SEUMUR HIDUP
            """,
            "text_items": []
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nik, "3510120212040001")
        self.assertEqual(ktp_data.nama, "AHMAD PUTRA FIRDAUS")
        self.assertEqual(ktp_data.tempat_tgl_lahir, "BANYUWANGI, 02-12-2004")
        self.assertEqual(ktp_data.jenis_kelamin, "LAKI-LAKI")
        self.assertEqual(ktp_data.rt_rw, "002/003")
        self.assertEqual(ktp_data.kecamatan, "KABAT")
        self.assertEqual(ktp_data.kel_desa, "LABANASEM")
        self.assertEqual(ktp_data.agama, "ISLAM")
        self.assertEqual(ktp_data.status_perkawinan, "BELUM KAWIN")
        self.assertEqual(ktp_data.pekerjaan, "PELAJAR/MAHASISWA")
        self.assertEqual(ktp_data.berlaku_hingga, "SEUMUR HIDUP")
    def test_2026_andri_papriana_17_digit_nik(self):
        # Case where OCR misreads colon as leading 1: 13602141204920003
        raw_text = """
PROVINSIBANTEN
KABUPATENLEBAK
NIK : 13602141204920003
Nama : ANDRIPAPRIANA
Tempat/Tgl Lahir : LEBAK,12-04-1992
Jenis Kelamin : LAKI-LAKI
Gol. Darah : 
Alamat : KP.SELAHAUR
RT/RW : 001/009
Kel/Desa : CJOROLEBAK
Kecamatan : RANGKASBITUNG
Agama : ISLAM
Status Perkawinan : CERAIHIDUP
Pekerjaan : KARYAWANSWASTA
Kewarganegaraan : WNI
Berlaku Hingga : SEUMURHIDUP
        """
        parse_result = {
            "raw_text": raw_text,
            "text_items": []
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nik, "3602141204920003")
        self.assertEqual(ktp_data.nama, "ANDRIPAPRIANA")
        self.assertEqual(ktp_data.tempat_tgl_lahir, "LEBAK, 12-04-1992")
        self.assertEqual(ktp_data.status_perkawinan, "CERAI HIDUP")
        self.assertEqual(ktp_data.berlaku_hingga, "SEUMUR HIDUP")

    def test_mentor_satya_smartphone_photo(self):
        # Case from mentor screenshot with real smartphone camera noise/typos:
        # LAKDAXI, TeLahir, DENPASAB.23-03 2003, RT8W, missing Nama label
        raw_text = """
IGEDESATYANANDA GAUTAMA
TeLahir
DENPASAB.23-03 2003
LAKDAXI
Gol Darat
JLSIULANGG SEKARSARI
XH25XMEPIASARI
RT8W
000/000
        """
        parse_result = {
            "raw_text": raw_text,
            "text_items": []
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nama, "IGEDESATYANANDA GAUTAMA")
        self.assertEqual(ktp_data.tempat_tgl_lahir, "DENPASAR, 23-03-2003")
        self.assertEqual(ktp_data.jenis_kelamin, "LAKI-LAKI")
        self.assertEqual(ktp_data.rt_rw, "000/000")
        self.assertIn("JLSIULANGG SEKARSARI", ktp_data.alamat)


    def test_clean_agama_fuzzy_and_typos(self):
        # Test various OCR typos in religion
        self.assertEqual(self.extractor.clean_agama("1SLAM"), "ISLAM")
        self.assertEqual(self.extractor.clean_agama("ISLM"), "ISLAM")
        self.assertEqual(self.extractor.clean_agama("ISLAN"), "ISLAM")
        self.assertEqual(self.extractor.clean_agama("KR1STEN"), "KRISTEN")
        self.assertEqual(self.extractor.clean_agama("KRIS TEN"), "KRISTEN")
        self.assertEqual(self.extractor.clean_agama("KAT0LIK"), "KATOLIK")
        self.assertEqual(self.extractor.clean_agama("KATHOLIK"), "KATOLIK")
        self.assertEqual(self.extractor.clean_agama("H1NDU"), "HINDU")
        self.assertEqual(self.extractor.clean_agama("BUDHA"), "BUDDHA")
        self.assertEqual(self.extractor.clean_agama("KONGHUCU"), "KONGHUCU")
        self.assertEqual(self.extractor.clean_agama("KONG HU CU"), "KONGHUCU")

    def test_clean_gol_darah_normalization_and_hyphen(self):
        # Hyphens and unknown should strictly map to None
        self.assertIsNone(self.extractor.clean_gol_darah("-"))
        self.assertIsNone(self.extractor.clean_gol_darah(" - "))
        self.assertIsNone(self.extractor.clean_gol_darah("--"))
        self.assertIsNone(self.extractor.clean_gol_darah("_"))
        self.assertIsNone(self.extractor.clean_gol_darah(""))

        # Digit substitutions
        self.assertEqual(self.extractor.clean_gol_darah("0"), "O")
        self.assertEqual(self.extractor.clean_gol_darah("Q"), "O")
        self.assertEqual(self.extractor.clean_gol_darah("8"), "B")
        self.assertEqual(self.extractor.clean_gol_darah("4"), "A")
        self.assertEqual(self.extractor.clean_gol_darah("A8"), "AB")
        self.assertEqual(self.extractor.clean_gol_darah("4B"), "AB")

        # In context text
        self.assertEqual(self.extractor.clean_gol_darah(None, raw_text="Gol Darat : 0"), "O")
        self.assertEqual(self.extractor.clean_gol_darah(None, raw_text="Gol. Darah : B"), "B")
        self.assertIsNone(self.extractor.clean_gol_darah(None, raw_text="Gol. Darah : -"))

    def test_clean_status_perkawinan_fuzzy(self):
        self.assertEqual(self.extractor.clean_status_perkawinan("BELUM KAW1N"), "BELUM KAWIN")
        self.assertEqual(self.extractor.clean_status_perkawinan("BLM KAWIN"), "BELUM KAWIN")
        self.assertEqual(self.extractor.clean_status_perkawinan("BELUMKAWIN"), "BELUM KAWIN")
        self.assertEqual(self.extractor.clean_status_perkawinan("KAW1N"), "KAWIN")
        self.assertEqual(self.extractor.clean_status_perkawinan("CERAIHIDUP"), "CERAI HIDUP")
        self.assertEqual(self.extractor.clean_status_perkawinan("CERAI H1DUP"), "CERAI HIDUP")
        self.assertEqual(self.extractor.clean_status_perkawinan("CERAIMATI"), "CERAI MATI")
        self.assertEqual(self.extractor.clean_status_perkawinan("CERAI MAT1"), "CERAI MATI")

    def test_clean_kewarganegaraan_fuzzy(self):
        self.assertEqual(self.extractor.clean_kewarganegaraan("W N I"), "WNI")
        self.assertEqual(self.extractor.clean_kewarganegaraan("W.N.I"), "WNI")
        self.assertEqual(self.extractor.clean_kewarganegaraan("WN1"), "WNI")
        self.assertEqual(self.extractor.clean_kewarganegaraan("W-N-I"), "WNI")
        self.assertEqual(self.extractor.clean_kewarganegaraan("W N A"), "WNA")
        self.assertEqual(self.extractor.clean_kewarganegaraan("WN4"), "WNA")

    def test_end_to_end_fuzzy_categorical_extraction(self):
        parse_result = {
            "raw_text": """
PROVINSI JAWA BARAT
KABUPATEN BANDUNG
NIK : 3204123456780001
Nama : BUDI SETIAWAN
Tempat/Tgl Lahir : BANDUNG, 10-05-1995
Jenis Kelamin : LAKI-LAKI Gol. Darah : 0
Alamat : JL. ASIA AFRIKA NO 10
RT/RW : 001/002
Kel/Desa : BRAGA
Kecamatan : SUMUR BANDUNG
Agama : 1SLAM
Status Perkawinan: BELUM KAW1N
Pekerjaan : WIRASWASTA
Kewarganegaraan : W N I
Berlaku Hingga : SEUMUR HIDUP
            """,
            "text_items": []
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.agama, "ISLAM")
        self.assertEqual(ktp_data.gol_darah, "O")
        self.assertEqual(ktp_data.status_perkawinan, "BELUM KAWIN")
        self.assertEqual(ktp_data.kewarganegaraan, "WNI")

    def test_end_to_end_hyphen_blood_and_katholik(self):
        parse_result = {
            "raw_text": """
PROVINSI NTT
KABUPATEN SIKKA
NIK : 5304123456780002
Nama : MARIA FRANSISKA
Tempat/Tgl Lahir : MAUMERE, 15-08-1998
Jenis Kelamin : PEREMPUAN Gol. Darah : -
Alamat : JL. FLORES INDAH NO 5
RT/RW : 002/001
Kel/Desa : KOTA UNENG
Kecamatan : ALOK
Agama : KATHOLIK
Status Perkawinan: CERAI H1DUP
Pekerjaan : PEGAWAI SWASTA
Kewarganegaraan : WNI
Berlaku Hingga : SEUMUR HIDUP
            """,
            "text_items": []
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.agama, "KATOLIK")
        self.assertIsNone(ktp_data.gol_darah)
        self.assertEqual(ktp_data.status_perkawinan, "CERAI HIDUP")
        self.assertEqual(ktp_data.kewarganegaraan, "WNI")

    def test_tuban_gladys_nik_and_nama(self):
        # Case from censored Tuban KTP with dot-matrix font & 17-digit leading colon noise:
        # NIK: 23523165706980004 (23 invalid province, 35 valid Jatim)
        # Nama: GLADYSWAHYUKHAIPUNNISA (P->R dot matrix error and joined words)
        raw_text = """
PROVINSI JAWA TIMUR
KABUPATEN TUBAN
NIK : 23523165706980004
Nama : GLADYSWAHYUKHAIPUNNISA
Tempat/Tgl Lahir : 
Jenis kelamin : 
Alamat : 
RT/RW : 
Kel/Desa : 
Kecamatan : 
Agama : KATOLIK
Status Perkawinan: KAWIN
Pekerjaan : 
Kewarganegaraan: WNI
Berlaku Hingga : 
        """
        parse_result = {
            "raw_text": raw_text,
            "text_items": []
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nik, "3523165706980004")
        self.assertEqual(ktp_data.nama, "GLADYS WAHYU KHAIRUNNISA")
        self.assertEqual(ktp_data.agama, "KATOLIK")
        self.assertEqual(ktp_data.status_perkawinan, "KAWIN")
        self.assertEqual(ktp_data.kewarganegaraan, "WNI")

    def test_multiline_nama_spatial_faded_label(self):
        # KTP Belu: long name wraps to 2 lines, 'Nama' label not detected by OCR
        items = [
            {"text": "PROVINSINUSA TENGGARA TIMUR", "x": 354.0, "y": 11.0, "width": 787.0, "height": 60.0},
            {"text": "KABUPATENBELU", "x": 543.0, "y": 62.0, "width": 416.0, "height": 54.0},
            {"text": "NIK", "x": 43.0, "y": 119.0, "width": 136.0, "height": 70.0},
            {"text": "5304046805980001", "x": 370.0, "y": 129.0, "width": 615.0, "height": 62.0},
            {"text": "MARIA ALBERTINE F INTAN", "x": 396.0, "y": 214.0, "width": 461.0, "height": 43.0},
            {"text": "PANGLSTI", "x": 402.0, "y": 256.0, "width": 178.0, "height": 38.0},
            {"text": "moat/lolLaha", "x": 99.0, "y": 303.0, "width": 207.0, "height": 27.0},
            {"text": "Kecamatan", "x": 126.0, "y": 505.0, "width": 187.0, "height": 42.0},
            {"text": "Slalus Perkawinan", "x": 35.0, "y": 588.0, "width": 303.0, "height": 43.0},
            {"text": "BELU", "x": 1170.0, "y": 606.0, "width": 102.0, "height": 47.0},
            {"text": "Pekerjaan", "x": 32.0, "y": 628.0, "width": 172.0, "height": 50.0},
            {"text": "22-00-2018", "x": 1129.0, "y": 653.0, "width": 184.0, "height": 45.0},
        ]
        parse_result = {
            "raw_text": "\n".join(it["text"] for it in items),
            "text_items": items,
            "page_width": 1433.0,
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nik, "5304046805980001")
        self.assertEqual(ktp_data.nama, "MARIA ALBERTINE F INTAN PANGESTI")
        # Issue date next to 'Pekerjaan' must not become birth place/date
        self.assertIsNone(ktp_data.tempat_tgl_lahir)
        # 'BELU' (regency) must not be read as 'BELUM KAWIN'
        self.assertIsNone(ktp_data.status_perkawinan)

    def test_multiline_nama_spatial_with_label(self):
        items = [
            {"text": "NIK", "x": 40.0, "y": 120.0, "width": 130.0, "height": 40.0},
            {"text": "5304046805980001", "x": 370.0, "y": 120.0, "width": 600.0, "height": 40.0},
            {"text": "Nama", "x": 40.0, "y": 214.0, "width": 90.0, "height": 35.0},
            {"text": "MARIA ALBERTINE F INTAN", "x": 396.0, "y": 214.0, "width": 461.0, "height": 40.0},
            {"text": "PANGESTI", "x": 402.0, "y": 256.0, "width": 178.0, "height": 38.0},
            {"text": "Tempat/Tgl Lahir", "x": 40.0, "y": 303.0, "width": 250.0, "height": 35.0},
            {"text": "ATAMBUA, 28-05-1998", "x": 396.0, "y": 303.0, "width": 300.0, "height": 35.0},
        ]
        parse_result = {
            "raw_text": "\n".join(it["text"] for it in items),
            "text_items": items,
            "page_width": 1433.0,
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nama, "MARIA ALBERTINE F INTAN PANGESTI")
        self.assertEqual(ktp_data.tempat_tgl_lahir, "ATAMBUA, 28-05-1998")

    def test_status_perkawinan_rejects_regional_false_positive(self):
        self.assertIsNone(self.extractor.clean_status_perkawinan("BELU"))
        self.assertIsNone(self.extractor.clean_status_perkawinan("Slatus Perkawinan"))
        self.assertEqual(self.extractor.clean_status_perkawinan("BELUM KAWIN"), "BELUM KAWIN")

    def test_disallowed_birth_place_whole_word(self):
        self.assertTrue(self.extractor._is_disallowed_place("PEKERJAAN"))
        self.assertFalse(self.extractor._is_disallowed_place("JAKARTA"))  # contains 'RT' substring
        self.assertFalse(self.extractor._is_disallowed_place("KOTA BIMA"))

    def test_ktp_belu_male_nik_and_goldar(self):
        items = [
            {"text": "PROVINSI NUSA TENGGARA TIMUR", "x": 354.0, "y": 11.0, "width": 787.0, "height": 60.0},
            {"text": "KABUPATEN BELU", "x": 543.0, "y": 62.0, "width": 416.0, "height": 54.0},
            {"text": "NIK", "x": 43.0, "y": 119.0, "width": 136.0, "height": 70.0},
            {"text": ": 5304040304040005", "x": 370.0, "y": 129.0, "width": 615.0, "height": 62.0},
            {"text": "Gol. Darah : O", "x": 500.0, "y": 340.0, "width": 200.0, "height": 35.0},
            {"text": "Kewarganegaraan", "x": 33.0, "y": 672.0, "width": 300.0, "height": 53.0},
            {"text": "WNI", "x": 370.0, "y": 672.0, "width": 100.0, "height": 53.0},
        ]
        parse_result = {
            "raw_text": "\n".join(it["text"] for it in items),
            "text_items": items,
            "page_width": 1433.0,
            "page_height": 900.0
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nik, "5304040304040005")
        self.assertEqual(ktp_data.gol_darah, "O")
        self.assertEqual(ktp_data.kewarganegaraan, "WNI")

    def test_geometric_zone_nik_recovery_without_label(self):
        # Case where 'NIK' label is totally missed by OCR, but 16 digits sit in NIK geometric zone
        items = [
            {"text": "5304040304040005", "x": 370.0, "y": 130.0, "width": 600.0, "height": 50.0},
            {"text": "Nama", "x": 40.0, "y": 214.0, "width": 90.0, "height": 35.0},
            {"text": "BUDI SANTOSO", "x": 370.0, "y": 214.0, "width": 350.0, "height": 35.0},
        ]
        parse_result = {
            "raw_text": "\n".join(it["text"] for it in items),
            "text_items": items,
            "page_width": 1400.0,
            "page_height": 900.0
        }
        ktp_data = self.extractor.extract(parse_result)
        self.assertEqual(ktp_data.nik, "5304040304040005")
        self.assertEqual(ktp_data.nama, "BUDI SANTOSO")

if __name__ == "__main__":
    unittest.main()
