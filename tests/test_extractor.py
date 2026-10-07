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


if __name__ == "__main__":
    unittest.main()
