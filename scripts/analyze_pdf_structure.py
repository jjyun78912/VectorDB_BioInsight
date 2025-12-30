#!/usr/bin/env python3
"""PDF 구조 분석 - 섹션 헤더 패턴 파악"""
import fitz
from pathlib import Path

def analyze_pdf(pdf_path: str):
    """PDF에서 텍스트 블록과 폰트 정보 추출"""
    doc = fitz.open(pdf_path)
    print(f"\n{'='*70}")
    print(f"분석: {Path(pdf_path).name}")
    print(f"{'='*70}")

    potential_headers = []

    for page_num in range(min(5, len(doc))):  # 처음 5페이지만
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    font_size = span["size"]
                    font_name = span["font"]
                    flags = span["flags"]  # bold, italic 등

                    # 볼드 체크 (flags의 bit 4)
                    is_bold = bool(flags & 2**4)

                    # 잠재적 섹션 헤더 감지
                    header_keywords = [
                        "abstract", "introduction", "background",
                        "method", "material", "result", "discussion",
                        "conclusion", "reference", "acknowledgment",
                        "experimental", "procedure", "patient",
                        "statistical", "data", "study", "treatment"
                    ]

                    text_lower = text.lower()
                    if any(kw in text_lower for kw in header_keywords):
                        if len(text) < 80 and (font_size > 10 or is_bold):
                            potential_headers.append({
                                "page": page_num + 1,
                                "text": text,
                                "font_size": round(font_size, 1),
                                "font": font_name,
                                "is_bold": is_bold
                            })

    doc.close()

    # 결과 출력
    print("\n잠재적 섹션 헤더:")
    print("-" * 70)
    seen = set()
    for h in potential_headers:
        key = h["text"][:50]
        if key not in seen:
            seen.add(key)
            bold_mark = "🔷" if h["is_bold"] else "  "
            print(f"  {bold_mark} [p{h['page']:2}] (size:{h['font_size']:5}) {h['text'][:60]}")

    return potential_headers


def main():
    papers_dir = Path("./data/papers")
    for pdf_file in papers_dir.glob("*.pdf"):
        analyze_pdf(str(pdf_file))


if __name__ == "__main__":
    main()
