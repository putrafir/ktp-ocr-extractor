#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import KTPExtractionPipeline

def main():
    parser = argparse.ArgumentParser(
        description="KTP Extractor Prototype (Local Model -> OCR + Layout Extractor)"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        type=str,
        help="Path to KTP image file (JPG, PNG, HEIC, WEBP)"
    )
    parser.add_argument(
        "--engine", "-e",
        type=str,
        choices=["paddle", "liteparse"],
        default="paddle",
        help="OCR Engine: 'paddle' (Recommended, deep learning) or 'liteparse' (experimental)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional path to save JSON output"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output"
    )

    args = parser.parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Error: File '{image_path}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Memproses gambar KTP: {image_path} [Engine: {args.engine}]...")
    pipeline = KTPExtractionPipeline(engine=args.engine)

    try:
        result = pipeline.process(image_path)
    except Exception as e:
        print(f"[!] Error saat ekstraksi: {e}", file=sys.stderr)
        sys.exit(1)

    data = result["data"]
    latency = result["latency_ms"]

    indent = 2 if args.pretty else None
    json_str = json.dumps(data, ensure_ascii=False, indent=indent)

    print("\n[+] Hasil Ekstraksi Key-Value:")
    print(json_str)
    print(f"\n[+] Total Latensi: {latency} ms (Engine: {result['engine']})")
    print(f"[+] Total Text Elements: {result['text_items_count']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"[+] Output tersimpan di: {out_path}")

if __name__ == "__main__":
    main()
