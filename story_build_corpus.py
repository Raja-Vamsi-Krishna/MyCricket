import fitz  # PyMuPDF
import re
import os
from tqdm import tqdm

PDF_DIR = "data/pdfs"
OUT_FILE = "data/corpus.txt"


def clean_text(text: str) -> str:
    # remove page numbers
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # fix broken lines
    text = re.sub(r"-\n", "", text)

    # normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # remove weird unicode junk
    text = text.replace("\ufeff", "").replace("\u200b", "")

    return text.strip()


def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)

    doc.close()
    return "\n".join(pages)


def main():
    all_text = []

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    for pdf in tqdm(pdf_files, desc="📄 Processing PDFs"):
        path = os.path.join(PDF_DIR, pdf)
        raw_text = extract_pdf_text(path)
        cleaned = clean_text(raw_text)
        all_text.append(cleaned)

    final_text = "\n\n".join(all_text)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"✅ Corpus saved to {OUT_FILE}")
    print(f"📏 Total characters: {len(final_text):,}")


if __name__ == "__main__":
    main()
