
import os, sys
# add this at the very top
try:
    import Crypto
except ImportError:
    print("⚠️ Missing pycryptodome. Please install with: pip install pycryptodome")

from PyPDF2 import PdfReader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TEXT_DIR = os.path.join(DATA_DIR, "texts")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\\n".join(texts)

def ocr_pdf(pdf_path):
    # Optional OCR fallback using pdf2image + pytesseract
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as e:
        print("OCR dependencies not installed (pdf2image/pytesseract). Install them if you need OCR.", file=sys.stderr)
        return ""
    images = convert_from_path(pdf_path)
    parts = []
    for img in images:
        parts.append(pytesseract.image_to_string(img))
    return "\\n".join(parts)

def ensure_dirs():
    if not os.path.exists(TEXT_DIR):
        os.makedirs(TEXT_DIR)

def ingest_all(ocr_fallback=False):
    ensure_dirs()
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in {DATA_DIR}")
    for pdf in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf)
        txt = extract_text_from_pdf(pdf_path)
        if (not txt.strip()) and ocr_fallback:
            print(f"No text extracted from {pdf} with PDF parser — trying OCR...")
            txt = ocr_pdf(pdf_path)
        out_name = os.path.splitext(pdf)[0] + ".txt"
        out_path = os.path.join(TEXT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print(f"Extracted: {pdf} -> texts/{out_name} (chars: {len(txt)})")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ocr", action="store_true", help="Enable OCR fallback (requires pdf2image + pytesseract + poppler)")
    args = p.parse_args()
    ingest_all(ocr_fallback=args.ocr)
