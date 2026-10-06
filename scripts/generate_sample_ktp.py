from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_sample_ktp():
    # KTP dimensions (ratio approx 85.6 x 53.98 mm -> e.g. 856 x 540 px)
    w, h = 856, 540
    img = Image.new("RGB", (w, h), color=(180, 220, 245))  # Light KTP-blue
    draw = ImageDraw.Draw(img)

    # Subtly draw header background
    draw.rectangle([0, 0, w, 70], fill=(160, 205, 235))

    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        font_header = ImageFont.truetype(font_path, 22)
        font_nik = ImageFont.truetype(font_path, 24)
        font_body = ImageFont.truetype(font_path, 17)
    except Exception:
        font_header = font_nik = font_body = ImageFont.load_default()

    # Headers
    draw.text((w // 2, 20), "PROVINSI DKI JAKARTA", fill=(0, 0, 0), font=font_header, anchor="mm")
    draw.text((w // 2, 48), "JAKARTA PUSAT", fill=(0, 0, 0), font=font_header, anchor="mm")

    # NIK
    draw.text((40, 90), "NIK", fill=(0, 0, 0), font=font_nik)
    draw.text((180, 90), ": 3171010101900001", fill=(0, 0, 0), font=font_nik)

    # Body Fields
    fields = [
        ("Nama", ": FULAN BIN FULAN"),
        ("Tempat/Tgl Lahir", ": JAKARTA, 01-01-1990"),
        ("Jenis Kelamin", ": LAKI-LAKI        Gol. Darah : O"),
        ("Alamat", ": JL. GAJAH MADA NO. 10"),
        ("   RT/RW", ": 002/004"),
        ("   Kel/Desa", ": PETOJO UTARA"),
        ("   Kecamatan", ": GAMBIR"),
        ("Agama", ": ISLAM"),
        ("Status Perkawinan", ": BELUM KAWIN"),
        ("Pekerjaan", ": KARYAWAN SWASTA"),
        ("Kewarganegaraan", ": WNI"),
        ("Berlaku Hingga", ": SEUMUR HIDUP")
    ]

    start_y = 135
    line_spacing = 30
    for i, (label, val) in enumerate(fields):
        y = start_y + (i * line_spacing)
        draw.text((40, y), label, fill=(0, 0, 0), font=font_body)
        draw.text((220, y), val, fill=(0, 0, 0), font=font_body)

    # Photo placeholder on the right
    photo_box = [670, 120, 810, 310]
    draw.rectangle(photo_box, fill=(190, 200, 210), outline=(80, 80, 80), width=2)
    draw.text((740, 215), "FOTO", fill=(80, 80, 80), font=font_body, anchor="mm")

    # Signature placeholder
    sig_box = [670, 360, 810, 440]
    draw.rectangle(sig_box, fill=(210, 225, 235), outline=(150, 150, 150), width=1)
    draw.text((740, 400), "TANDA TANGAN", fill=(100, 100, 100), font=font_body, anchor="mm")

    output_dir = Path(__file__).resolve().parent.parent / "data" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "sample_ktp_dummy.png"
    img.save(out_file)
    print(f"Sample KTP generated successfully at: {out_file}")

if __name__ == "__main__":
    create_sample_ktp()
