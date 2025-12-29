"""PDF Parser for Bio Papers - Extracts text and identifies sections."""
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import fitz  # PyMuPDF

# Optional OCR support
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

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

    def _is_garbled_line(self, line: str) -> bool:
        """
        Detect if a line contains garbled math formula text.
        These come from PDFs with embedded TeX fonts that don't extract properly.
        """
        if len(line.strip()) < 10:
            return False

        # Count indicators of garbled text
        garbled_score = 0
        words = line.split()

        for word in words:
            word_lower = word.lower().strip('.,;:()[]{}')
            if len(word_lower) < 2:
                continue

            # Check for nonsensical consonant-heavy words
            vowels = sum(1 for c in word_lower if c in 'aeiou')
            consonants = sum(1 for c in word_lower if c.isalpha() and c not in 'aeiou')

            # Very low vowel ratio suggests garbled text
            if len(word_lower) >= 4 and consonants > 0:
                vowel_ratio = vowels / len(word_lower)
                if vowel_ratio < 0.15:  # Less than 15% vowels
                    garbled_score += 2

            # Specific garbled patterns from TeX fonts
            garbled_patterns = [
                'lfs', 'pdb', 'bgk', 'hth', 'lfsr', 'pdbgk',
                'jcj', 'clfs', 'fsrj', 'fsrg', 'sosrh',
                'map of', '1/4 p', 'j2c'
            ]
            if any(p in word_lower for p in garbled_patterns):
                garbled_score += 3

            # Mixed case in middle of word (like "hTh")
            if len(word) >= 3 and any(word[i].isupper() and word[i-1].islower() and word[i+1].islower()
                                       for i in range(1, len(word)-1) if word[i].isalpha()):
                garbled_score += 2

        # If more than 30% of meaningful content is garbled, mark the line
        if len(words) > 0 and garbled_score / max(len(words), 1) > 0.3:
            return True

        # Also check for high density of unusual character sequences
        unusual_count = len(re.findall(r'[;:]\s*[a-z]\s*[;:]|1/4\s+[a-z]|[a-z]\s*<\s*[a-z]{2,}\s*>', line))
        if unusual_count >= 2:
            return True

        return False

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

        # 2.5. 텍스트가 거의 없으면 OCR 시도
        if len(full_text.strip()) < 500:
            print(f"  [!] Low text content ({len(full_text)} chars), attempting OCR...")
            ocr_text = self._extract_text_with_ocr(pdf_path)
            if ocr_text and len(ocr_text) > len(full_text):
                full_text = ocr_text
                print(f"  [+] OCR extracted {len(full_text)} chars")

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

        # 5.5. Abstract를 섹션 맨 앞에 추가 (메타데이터에서 추출된 경우)
        if metadata.abstract and len(metadata.abstract) > 50:
            # Abstract가 이미 섹션에 없으면 추가
            has_abstract = any(s.name.lower() == "abstract" for s in sections)
            if not has_abstract:
                abstract_section = PaperSection(name="Abstract", content=metadata.abstract)
                sections.insert(0, abstract_section)

        # 6. 섹션이 없거나 내용이 거의 없으면 전체 텍스트를 하나의 섹션으로
        total_content = sum(len(s.content) for s in sections)
        if not sections or total_content < 500:
            clean_text = re.sub(r"\[PAGE_\d+\]", "", full_text).strip()
            if clean_text:
                sections = [PaperSection(name="Full Text", content=clean_text)]

        return metadata, sections

    def _extract_text_with_ocr(self, pdf_path: Path) -> str:
        """Extract text from PDF using OCR (for scanned/image PDFs)."""
        if not OCR_AVAILABLE:
            print("  [!] OCR not available. Install pytesseract and pdf2image.")
            return ""

        try:
            # Convert PDF pages to images
            images = convert_from_path(str(pdf_path), dpi=200)

            # OCR each page
            full_text = ""
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image, lang='eng')
                full_text += f"\n[PAGE_{i}]\n{text}"

            return self._clean_text(full_text)
        except Exception as e:
            print(f"  [!] OCR failed: {e}")
            return ""

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
        """Clean extracted text and fix common encoding issues."""
        # 1. Fix common PDF encoding issues for mathematical symbols
        # These are ligatures and special characters that get mangled
        encoding_fixes = {
            # Ligatures
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            # Common math symbols that get corrupted
            '¼': '1/4',
            '½': '1/2',
            '¾': '3/4',
            'þ': 'th',  # thorn character often misread
            'ð': 'd',   # eth character
            'Þ': 'Th',
            'Ð': 'D',
            # Greek letters that may appear garbled
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            'ε': 'epsilon',
            'μ': 'mu',
            'σ': 'sigma',
            'π': 'pi',
            'λ': 'lambda',
            'Δ': 'Delta',
            'Σ': 'Sigma',
            # Common mathematical operators
            '×': 'x',
            '÷': '/',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '±': '+/-',
            '∞': 'infinity',
            '√': 'sqrt',
            '∑': 'sum',
            '∏': 'product',
            '∫': 'integral',
            # Superscripts/subscripts
            '²': '^2',
            '³': '^3',
            '¹': '^1',
            '⁰': '^0',
            '⁴': '^4',
            '⁵': '^5',
            '⁶': '^6',
            '⁷': '^7',
            '⁸': '^8',
            '⁹': '^9',
            '₀': '_0',
            '₁': '_1',
            '₂': '_2',
            '₃': '_3',
            '₄': '_4',
            '₅': '_5',
            '₆': '_6',
            '₇': '_7',
            '₈': '_8',
            '₉': '_9',
        }

        for bad_char, replacement in encoding_fixes.items():
            text = text.replace(bad_char, replacement)

        # 2. Remove garbled math formula text (from TeX/LaTeX fonts)
        # These appear when PDF uses custom math fonts (CMSY, CMMI, etc.)
        # Pattern examples: "lfsosrh g 1/4 pdbgk", "hTh MAP of bgk"

        # 2a. Remove sequences with nonsensical consonant clusters
        # Real English rarely has patterns like "lfs", "pdb", "bgk", "hTh"
        nonsense_patterns = [
            r'\b[bcdfghjklmnpqrstvwxz]{3,}\b',  # 3+ consonants in a row
            r'\blfs\w*\b',  # lfs... patterns
            r'\bpdb\w*\b',  # pdb... patterns
            r'\bbgk\w*\b',  # bgk... patterns
            r'\bhTh\b',     # hTh
            r'\blfsr\w*\b', # lfsr... patterns
            r'\bMLE\w*\s+of\s+LFC\w*',  # MLEs of LFCs (garbled)
            r'\bLFC\w*\s+for\s+gene\w*\s+with',  # LFCs for genes with
        ]

        for pattern in nonsense_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 2b. Remove lines that look like garbled math formulas
        # These often have unusual character distributions
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that are mostly garbled (high ratio of unusual patterns)
            if self._is_garbled_line(line):
                cleaned_lines.append('[mathematical formula]')
            else:
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)

        # 3. Clean up multiple spaces and newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # 4. Apply exclude patterns
        for pattern in self.exclude_patterns:
            text = pattern.sub("", text)

        # 5. Fix hyphenated words split across lines
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # 6. Remove null bytes and other control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

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

        # 출판년도 추출 (더 정확한 패턴 우선)
        # 1. "Published: 2024" 또는 "© 2024" 패턴
        year_patterns = [
            r"(?:published|received|accepted|copyright|©)[:\s]*(\d{4})",
            r"(\d{4})\s*(?:Endocrine|Oxford|Elsevier|Springer|Wiley|Nature)",
            r"(?:Volume|Vol\.?)\s*\d+.*?(\d{4})",
            r",\s*(202[0-5]|201\d)\b",  # 최근 년도 우선
        ]

        for pattern in year_patterns:
            year_match = re.search(pattern, text[:3000], re.IGNORECASE)
            if year_match:
                year = year_match.group(1) if year_match.lastindex else year_match.group()
                # 합리적인 출판년도 범위 (2000-2030)
                if year.isdigit() and 2000 <= int(year) <= 2030:
                    metadata.year = year
                    break

        # 위 패턴으로 못 찾으면 2010년 이후 년도 중 첫 번째
        if not metadata.year:
            year_match = re.search(r"\b(20[1-2]\d)\b", text[:2000])
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
