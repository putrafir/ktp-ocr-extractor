from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class KTPData(BaseModel):
    nik: Optional[str] = Field(None, description="Nomor Induk Kependudukan (16 digit)")
    nama: Optional[str] = Field(None, description="Nama lengkap")
    tempat_tgl_lahir: Optional[str] = Field(None, description="Tempat dan tanggal lahir (e.g. JAKARTA, 01-01-1990)")
    jenis_kelamin: Optional[str] = Field(None, description="LAKI-LAKI atau PEREMPUAN")
    gol_darah: Optional[str] = Field(None, description="Golongan darah (A, B, AB, O, atau -)")
    alamat: Optional[str] = Field(None, description="Alamat jalan / nomor rumah")
    rt_rw: Optional[str] = Field(None, description="RT dan RW (e.g. 001/002)")
    kel_desa: Optional[str] = Field(None, description="Kelurahan atau Desa")
    kecamatan: Optional[str] = Field(None, description="Kecamatan")
    agama: Optional[str] = Field(None, description="Agama")
    status_perkawinan: Optional[str] = Field(None, description="BELUM KAWIN, KAWIN, CERAI HIDUP, CERAI MATI")
    pekerjaan: Optional[str] = Field(None, description="Pekerjaan")
    kewarganegaraan: Optional[str] = Field(None, description="WNI atau WNA")
    berlaku_hingga: Optional[str] = Field(None, description="Masa berlaku KTP (e.g. SEUMUR HIDUP)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary format."""
        return self.model_dump()
