import re
import fitz  # PyMuPDF
from config import PDF_PATH

def clean_page_numbers(text: str) -> str:
    """
    Remove standalone page numbers and line-end page numbers in the text.
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Remove lines that contain only numbers (isolated page numbers)
        if re.fullmatch(r'\s*\d{1,3}\s*', line):
            continue
        # Remove page numbers at the end of a line (e.g., "Document。 38" -> "Document。")
        line = re.sub(r"(。|：)\s*\d{1,3}\b", r"\1", line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def load_pdf(file_path=PDF_PATH, footer_ratio=0.04):
    """
    Read PDF and remove text blocks located in the bottom 'footer_ratio' of the page (usually page numbers).
    Also applies regex-based cleaning for any remaining isolated page numbers.
    """
    doc = fitz.open(file_path)
    full_text = []

    for page in doc:
        page_height = page.rect.height
        cutoff_y = page_height * (1 - footer_ratio)  # bottom footer threshold

        # Extract text blocks from the page
        blocks = page.get_text("blocks")  # each block: (x0, y0, x1, y1, "text", ...)
        page_text = []

        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            # Skip blocks located in the bottom footer area
            if y0 >= cutoff_y:
                continue
            page_text.append(text)

        # Combine text blocks and apply regex cleaning for isolated page numbers
        page_text_str = "\n".join(page_text)
        page_text_str = clean_page_numbers(page_text_str)
        full_text.append(page_text_str)

    return "\n".join(full_text)

if __name__ == "__main__":
    text = load_pdf()
    print(text[:1500])
