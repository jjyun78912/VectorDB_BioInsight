"""PDF Parser for Bio Papers - Extracts text and identifies sections."""
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import fitz  # PyMuPDF

from .config import BIO_PAPER_SECTIONS, METHODS_SUBSECTIONS, EXCLUDE_SECTIONS


@dataclass
class PaperMetadata:
    """Metadata extracted from a bio paper."""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    journal: str = ""
    year: str = ""
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    file_path: str = ""


@dataclass
class PaperSection:
    """A section from a bio paper."""
    name: str
    content: str
    subsections: list["PaperSection"] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0


@dataclass
class TextBlock:
    """A block of text with formatting info."""
    text: str
    font_size: float
    is_bold: bool
    page_num: int
    position: int  # Character position in full text


class BioPaperParser:
    """Parser for biomedical research papers in PDF format."""

    # 섹션 키워드 (소문자)
    SECTION_KEYWORDS = {
        # Primary sections
        "abstract": "Abstract",
        "introduction": "Introduction",
        "background": "Background",
        "methods": "Methods",
        "method": "Methods",
        "materials and methods": "Materials and Methods",
        "material and methods": "Materials and Methods",
        "experimental procedures": "Methods",
        "experimental": "Methods",
        "patients and methods": "Methods",
        "study design": "Methods",
        "results": "Results",
        "findings": "Results",
        "discussion": "Discussion",
        "conclusion": "Conclusion",
        "conclusions": "Conclusion",
        "summary": "Conclusion",
        "references": "References",
        "acknowledgments": "Acknowledgments",
        "acknowledgements": "Acknowledgments",
        "supplementary": "Supplementary",
        "appendix": "Supplementary",
        # Extended sections
        "background and relevance": "Background",
        "clinical implications": "Discussion",
        "future directions": "Discussion",
        "limitations": "Discussion",
        "statistical analysis": "Methods",
        "data analysis": "Methods",
        "study population": "Methods",
        "participants": "Methods",
        "trial design": "Methods",
        "treatment": "Methods",
    }

    # Methods 하위 섹션 키워드
    METHODS_KEYWORDS = [
        "trial design", "study design", "study population",
        "participants", "patients", "treatment", "intervention",
        "statistical analysis", "data analysis", "outcomes",
        "rna extraction", "dna extraction", "sequencing",
        "library preparation", "immunohistochemistry",
        "western blot", "pcr", "cell culture", "animal",
        "endpoints", "assessments", "measurements"
    ]

    def __init__(self):
        self.exclude_patterns = self._build_exclude_patterns()

    def _build_exclude_patterns(self) -> list[re.Pattern]:
        """Build patterns for content to exclude."""
        return [
            re.compile(r"^\s*\d+\s*$"),
            re.compile(r"^[\s\-_=]+$"),
            re.compile(r"©|Copyright|All rights reserved", re.IGNORECASE),
            re.compile(r"^\s*Downloaded from", re.IGNORECASE),
        ]

    def parse_pdf(self, pdf_path: str | Path) -> tuple[PaperMetadata, list[PaperSection]]:
        """Parse a PDF file and extract structured content."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)

        # 1. 폰트 기반으로 섹션 헤더 감지
        section_headers = self._detect_section_headers_by_font(doc)

        # 2. 전체 텍스트 추출
        full_text, text_blocks = self._extract_text_with_positions(doc)

        doc.close()

        # 3. 메타데이터 추출
        metadata = self._extract_metadata(full_text, pdf_path)

        # 4. 섹션 추출 (폰트 기반 헤더 사용)
        if section_headers:
            sections = self._extract_sections_by_headers(full_text, section_headers)
        else:
            # 폴백: 텍스트 패턴 기반
            sections = self._extract_sections_by_pattern(full_text)

        # 5. 제외 섹션 필터링
        sections = [s for s in sections if s.name not in EXCLUDE_SECTIONS]

        return metadata, sections

    def _detect_section_headers_by_font(self, doc) -> list[dict]:
        """폰트 크기/볼드 기반으로 섹션 헤더 감지."""
        headers = []
        char_position = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    line_text = ""
                    line_font_size = 0
                    line_is_bold = False

                    for span in line["spans"]:
                        text = span["text"]
                        line_text += text
                        line_font_size = max(line_font_size, span["size"])
                        # Bold 체크 (flags bit)
                        if span["flags"] & 2**4:
                            line_is_bold = True

                    line_text = line_text.strip()
                    line_lower = line_text.lower()

                    # 섹션 헤더 조건:
                    # 1. 키워드 포함
                    # 2. 짧은 텍스트 (< 80자)
                    # 3. 볼드이거나 큰 폰트
                    if len(line_text) < 80 and (line_is_bold or line_font_size >= 10):
                        matched_section = self._match_section_keyword(line_lower)
                        if matched_section:
                            headers.append({
                                "text": line_text,
                                "section_name": matched_section,
                                "page": page_num,
                                "position": char_position,
                                "font_size": line_font_size,
                                "is_bold": line_is_bold
                            })

                    char_position += len(line_text) + 1

        # 중복 제거 (같은 섹션이 여러 번 감지될 수 있음)
        seen_sections = set()
        unique_headers = []
        for h in headers:
            key = (h["section_name"], h["page"])
            if key not in seen_sections:
                seen_sections.add(key)
                unique_headers.append(h)

        return unique_headers

    def _match_section_keyword(self, text: str) -> Optional[str]:
        """텍스트에서 섹션 키워드 매칭."""
        text = text.strip().lower()

        # 정확한 매칭 우선
        if text in self.SECTION_KEYWORDS:
            return self.SECTION_KEYWORDS[text]

        # 부분 매칭 (키워드로 시작하는 경우)
        for keyword, section_name in self.SECTION_KEYWORDS.items():
            if text.startswith(keyword) or text == keyword:
                return section_name
            # "1. Introduction" 같은 번호 패턴
            if re.match(rf"^\d+\.?\s*{re.escape(keyword)}", text):
                return section_name

        return None

    def _extract_text_with_positions(self, doc) -> tuple[str, list[TextBlock]]:
        """전체 텍스트와 위치 정보 추출."""
        full_text = ""
        text_blocks = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text = self._clean_text(text)

            full_text += f"\n[PAGE_{page_num}]\n{text}"

        return full_text, text_blocks

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        for pattern in self.exclude_patterns:
            text = pattern.sub("", text)

        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        return text.strip()

    def _extract_metadata(self, text: str, pdf_path: Path) -> PaperMetadata:
        """Extract paper metadata from text."""
        metadata = PaperMetadata(file_path=str(pdf_path))

        lines = text.split("\n")
        for line in lines[:20]:
            line = line.strip()
            if len(line) > 20 and not any(x in line.lower() for x in ["doi:", "http", "journal", "volume", "[page"]):
                metadata.title = line
                break

        doi_match = re.search(r"(?:doi[:\s]*)?10\.\d{4,}/[^\s]+", text, re.IGNORECASE)
        if doi_match:
            metadata.doi = doi_match.group().strip()

        year_match = re.search(r"\b(19|20)\d{2}\b", text[:2000])
        if year_match:
            metadata.year = year_match.group()

        abstract_match = re.search(
            r"abstract[:\s]*\n?(.*?)(?=\n\s*(?:introduction|background|keywords|1\.|graphical))",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            metadata.abstract = abstract_match.group(1).strip()[:2000]

        keywords_match = re.search(
            r"keywords?[:\s]*([^\n]+(?:\n[^\n]+)?)",
            text,
            re.IGNORECASE
        )
        if keywords_match:
            keywords_text = keywords_match.group(1)
            metadata.keywords = [k.strip() for k in re.split(r"[,;]", keywords_text) if k.strip()]

        return metadata

    def _extract_sections_by_headers(self, full_text: str, headers: list[dict]) -> list[PaperSection]:
        """폰트 기반 헤더를 사용해 섹션 추출."""
        sections = []

        # 텍스트에서 헤더 위치 찾기
        header_positions = []
        for header in headers:
            # 헤더 텍스트를 전체 텍스트에서 찾기
            pattern = re.escape(header["text"])
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            if matches:
                header_positions.append({
                    "name": header["section_name"],
                    "start": matches[0].start(),
                    "end": matches[0].end(),
                    "original_text": header["text"]
                })

        # 위치순 정렬
        header_positions.sort(key=lambda x: x["start"])

        # 섹션 내용 추출
        for i, hp in enumerate(header_positions):
            start = hp["end"]
            end = header_positions[i + 1]["start"] if i + 1 < len(header_positions) else len(full_text)

            content = full_text[start:end].strip()
            content = re.sub(r"\[PAGE_\d+\]", "", content).strip()

            if content and len(content) > 50:  # 최소 내용이 있어야 함
                section = PaperSection(
                    name=hp["name"],
                    content=content
                )

                # Methods 하위 섹션 추출
                if hp["name"] in ["Methods", "Materials and Methods"]:
                    section.subsections = self._extract_methods_subsections(content)

                sections.append(section)

        # 섹션이 없으면 전체 텍스트를 하나의 섹션으로
        if not sections:
            clean_text = re.sub(r"\[PAGE_\d+\]", "", full_text).strip()
            sections.append(PaperSection(name="Full Text", content=clean_text))

        return sections

    def _extract_sections_by_pattern(self, text: str) -> list[PaperSection]:
        """텍스트 패턴 기반 섹션 추출 (폴백)."""
        sections = []

        # 패턴 매칭으로 섹션 찾기
        section_pattern = re.compile(
            r"^[\s]*(?:\d+\.?\s*)?(Abstract|Introduction|Background|Methods?|Materials?\s+and\s+Methods?|"
            r"Results?|Discussion|Conclusions?|References|Acknowledgm?ents?)\s*$",
            re.IGNORECASE | re.MULTILINE
        )

        matches = list(section_pattern.finditer(text))

        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            content = text[start:end].strip()
            content = re.sub(r"\[PAGE_\d+\]", "", content).strip()

            if content:
                section_name = self._normalize_section_name(match.group(1))
                sections.append(PaperSection(name=section_name, content=content))

        if not sections:
            clean_text = re.sub(r"\[PAGE_\d+\]", "", text).strip()
            sections.append(PaperSection(name="Full Text", content=clean_text))

        return sections

    def _extract_methods_subsections(self, methods_text: str) -> list[PaperSection]:
        """Methods 섹션에서 하위 섹션 추출."""
        subsections = []

        for keyword in self.METHODS_KEYWORDS:
            pattern = rf"({re.escape(keyword)})\s*\n(.*?)(?=\n[A-Z][a-z]+|\n\d+\.\s+[A-Z]|\Z)"
            matches = re.finditer(pattern, methods_text, re.IGNORECASE | re.DOTALL)

            for match in matches:
                content = match.group(2).strip()
                if content and len(content) > 30:
                    subsections.append(PaperSection(
                        name=keyword.title(),
                        content=content
                    ))

        return subsections

    def _normalize_section_name(self, name: str) -> str:
        """Normalize section name to standard format."""
        name = name.strip()

        # 대문자면 Title Case로
        if name.isupper():
            name = name.title()

        # 변형 매핑
        mappings = {
            "Material And Methods": "Materials and Methods",
            "Materials And Methods": "Materials and Methods",
            "Experimental Procedures": "Methods",
            "Conclusions": "Conclusion",
            "Method": "Methods",
            "Result": "Results",
        }

        return mappings.get(name, name)


def parse_paper(pdf_path: str | Path) -> tuple[PaperMetadata, list[PaperSection]]:
    """Convenience function to parse a bio paper PDF."""
    parser = BioPaperParser()
    return parser.parse_pdf(pdf_path)
