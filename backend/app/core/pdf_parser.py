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


# ============================================================================
# Constants - Encoding fixes and patterns
# ============================================================================

ENCODING_FIXES = {
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
    'þ': 'th',
    'ð': 'd',
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

NONSENSE_PATTERNS = [
    r'\b[bcdfghjklmnpqrstvwxz]{3,}\b',  # 3+ consonants in a row
    r'\blfs\w*\b',
    r'\bpdb\w*\b',
    r'\bbgk\w*\b',
    r'\bhTh\b',
    r'\blfsr\w*\b',
    r'\bMLE\w*\s+of\s+LFC\w*',
    r'\bLFC\w*\s+for\s+gene\w*\s+with',
]

METADATA_SKIP_PATTERNS = [
    "doi:", "http", "https", "://", "journal", "volume", "[page",
    "issn", "isbn", "copyright", "©", "published", "article",
    "open access", "creative commons", "licence", "license",
    "academic.oup", "elsevier", "springer", "wiley", "nature.com",
    "frontiersin", "received:", "accepted:", "printed:",
]

YEAR_EXTRACTION_PATTERNS = [
    r"(?:published|received|accepted|copyright|©)[:\s]*(\d{4})",
    r"(\d{4})\s*(?:Endocrine|Oxford|Elsevier|Springer|Wiley|Nature)",
    r"(?:Volume|Vol\.?)\s*\d+.*?(\d{4})",
    r",\s*(202[0-5]|201\d)\b",
]


# ============================================================================
# Helper Classes
# ============================================================================

class GarbledTextDetector:
    """Detects garbled mathematical formula text from PDFs with embedded TeX fonts."""

    # Thresholds
    MIN_LINE_LENGTH = 10
    LOW_VOWEL_RATIO = 0.15
    VERY_LOW_VOWEL_RATIO = 0.15
    MIN_WORD_LENGTH = 4
    GARBLED_SCORE_THRESHOLD = 0.3
    MIN_UNUSUAL_SEQUENCES = 2

    # Known garbled patterns from TeX fonts
    GARBLED_PATTERNS = [
        'lfs', 'pdb', 'bgk', 'hth', 'lfsr', 'pdbgk',
        'jcj', 'clfs', 'fsrj', 'fsrg', 'sosrh',
        'map of', '1/4 p', 'j2c'
    ]

    def is_garbled(self, line: str) -> bool:
        """Check if a line contains garbled text."""
        if len(line.strip()) < self.MIN_LINE_LENGTH:
            return False

        words = line.split()
        if not words:
            return False

        garbled_score = sum(self._score_word(word) for word in words)

        # Check score threshold
        if garbled_score / max(len(words), 1) > self.GARBLED_SCORE_THRESHOLD:
            return True

        # Check for unusual character sequences
        unusual_count = len(re.findall(
            r'[;:]\s*[a-z]\s*[;:]|1/4\s+[a-z]|[a-z]\s*<\s*[a-z]{2,}\s*>',
            line
        ))
        if unusual_count >= self.MIN_UNUSUAL_SEQUENCES:
            return True

        return False

    def _score_word(self, word: str) -> int:
        """Calculate garbled score for a single word."""
        word_lower = word.lower().strip('.,;:()[]{}')
        if len(word_lower) < 2:
            return 0

        score = 0
        score += self._check_vowel_ratio(word_lower)
        score += self._check_garbled_patterns(word_lower)
        score += self._check_mixed_case(word)

        return score

    def _check_vowel_ratio(self, word: str) -> int:
        """Check for abnormally low vowel ratio."""
        if len(word) < self.MIN_WORD_LENGTH:
            return 0

        vowels = sum(1 for c in word if c in 'aeiou')
        consonants = sum(1 for c in word if c.isalpha() and c not in 'aeiou')

        if consonants > 0:
            vowel_ratio = vowels / len(word)
            if vowel_ratio < self.LOW_VOWEL_RATIO:
                return 2

        return 0

    def _check_garbled_patterns(self, word: str) -> int:
        """Check for known garbled patterns from TeX fonts."""
        if any(pattern in word for pattern in self.GARBLED_PATTERNS):
            return 3
        return 0

    def _check_mixed_case(self, word: str) -> int:
        """Check for unusual mixed case patterns (e.g., 'hTh')."""
        if len(word) < 3:
            return 0

        for i in range(1, len(word) - 1):
            if (word[i].isupper() and
                word[i].isalpha() and
                word[i-1].islower() and
                word[i+1].islower()):
                return 2

        return 0


class MetadataExtractor:
    """Extracts metadata (title, DOI, year, etc.) from paper text."""

    def __init__(self):
        self.skip_patterns = METADATA_SKIP_PATTERNS
        self.year_patterns = YEAR_EXTRACTION_PATTERNS

    def extract(self, text: str, pdf_path: Path) -> "PaperMetadata":
        """Extract all metadata from paper text."""
        metadata = PaperMetadata(file_path=str(pdf_path))

        metadata.title = self._find_title(text)
        metadata.doi = self._find_doi(text)
        metadata.year = self._find_year(text)
        metadata.abstract = self._find_abstract(text)
        metadata.keywords = self._find_keywords(text)

        return metadata

    def _find_title(self, text: str) -> str:
        """Find paper title from first substantial line."""
        lines = text.split("\n")

        for line in lines[:30]:
            line = line.strip()

            # Too short
            if len(line) < 20:
                continue

            # Skip metadata patterns
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in self.skip_patterns):
                continue

            # Skip number-only lines
            if re.match(r'^[\d\-/.,\s]+$', line):
                continue

            # Skip non-alphabetic lines
            alpha_ratio = sum(1 for c in line if c.isalpha()) / len(line)
            if alpha_ratio < 0.5:
                continue

            return line

        return ""

    def _find_doi(self, text: str) -> str:
        """Extract DOI from text."""
        doi_match = re.search(r"(?:doi[:\s]*)?(10\.\d{4,}/[^\s]+)", text, re.IGNORECASE)
        if doi_match:
            # Return just the DOI part (group 1), not the "DOI:" prefix
            return doi_match.group(1).strip()
        return ""

    def _find_year(self, text: str) -> str:
        """Extract publication year from text."""
        # Try specific patterns first
        for pattern in self.year_patterns:
            year_match = re.search(pattern, text[:3000], re.IGNORECASE)
            if year_match:
                year = year_match.group(1) if year_match.lastindex else year_match.group()
                # Validate year range
                if year.isdigit() and 2000 <= int(year) <= 2030:
                    return year

        # Fallback: find first 2010+ year
        year_match = re.search(r"\b(20[1-2]\d)\b", text[:2000])
        if year_match:
            return year_match.group()

        return ""

    def _find_abstract(self, text: str) -> str:
        """Extract abstract from text."""
        abstract_match = re.search(
            r"abstract[:\s]*\n?(.*?)(?=\n\s*(?:introduction|background|keywords|1\.|graphical))",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            return abstract_match.group(1).strip()[:2000]
        return ""

    def _find_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        keywords_match = re.search(
            r"keywords?[:\s]*([^\n]+(?:\n[^\n]+)?)",
            text,
            re.IGNORECASE
        )
        if keywords_match:
            keywords_text = keywords_match.group(1)
            return [k.strip() for k in re.split(r"[,;]", keywords_text) if k.strip()]
        return []


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
        "treatment": "Treatment",
        # Bio/Medical specific sections
        "epidemiology": "Epidemiology",
        "etiology": "Etiology",
        "pathophysiology": "Pathophysiology",
        "pathogenesis": "Pathogenesis",
        "diagnosis": "Diagnosis",
        "diagnostic": "Diagnosis",
        "clinical presentation": "Clinical Presentation",
        "clinical features": "Clinical Features",
        "clinical manifestations": "Clinical Presentation",
        "management": "Management",
        "management and treatment": "Management",
        "therapeutic": "Treatment",
        "therapy": "Treatment",
        "prognosis": "Prognosis",
        "outcome": "Outcomes",
        "outcomes": "Outcomes",
        "complications": "Complications",
        "prevention": "Prevention",
        "screening": "Screening",
        "surveillance": "Surveillance",
        "follow-up": "Follow-up",
        "case report": "Case Report",
        "case presentation": "Case Report",
        "patient characteristics": "Patient Characteristics",
        "hereditary": "Hereditary Patterns",
        "genetic": "Genetics",
        "genetics": "Genetics",
        "molecular": "Molecular",
        "imaging": "Imaging",
        "laboratory": "Laboratory",
        "histopathology": "Histopathology",
        "immunohistochemistry": "Immunohistochemistry",
        "biomarkers": "Biomarkers",
        "surgical": "Surgical Treatment",
        "surgery": "Surgical Treatment",
        "pharmacotherapy": "Pharmacotherapy",
        "chemotherapy": "Chemotherapy",
        "radiotherapy": "Radiotherapy",
        "immunotherapy": "Immunotherapy",
        "targeted therapy": "Targeted Therapy",
        "clinical trial": "Clinical Trials",
        "novel therapies": "Novel Therapies",
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
        self.garbled_detector = GarbledTextDetector()
        self.metadata_extractor = MetadataExtractor()

    def _build_exclude_patterns(self) -> list[re.Pattern]:
        """Build patterns for content to exclude."""
        return [
            re.compile(r"^\s*\d+\s*$"),
            re.compile(r"^[\s\-_=]+$"),
            re.compile(r"©|Copyright|All rights reserved", re.IGNORECASE),
            re.compile(r"^\s*Downloaded from", re.IGNORECASE),
        ]

    def _is_garbled_line(self, line: str) -> bool:
        """Detect if a line contains garbled math formula text."""
        return self.garbled_detector.is_garbled(line)

    def parse_pdf(self, pdf_path: str | Path) -> tuple[PaperMetadata, list[PaperSection]]:
        """Parse a PDF file and extract structured content."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)

        # 1. 폰트 기반으로 섹션 헤더 감지
        section_headers = self._detect_section_headers_by_font(doc)

        # 1.5. 폰트 기반 타이틀 추출 (가장 큰 폰트 사용)
        font_based_title = self._extract_title_by_font(doc)

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

        # 3.5. 폰트 기반 타이틀이 있으면 우선 사용
        if font_based_title and len(font_based_title) > len(metadata.title or ""):
            metadata.title = font_based_title

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
                    # 1. 키워드 포함 OR numbered subsection pattern
                    # 2. 짧은 텍스트 (< 80자)
                    # 3. 볼드이거나 큰 폰트 (>=9 for subsections, >=10 for main)
                    if len(line_text) < 80 and (line_is_bold or line_font_size >= 9):
                        matched_section = self._match_section_keyword(line_lower)

                        # Check if it's a numbered subsection (e.g., "3.1 Overview")
                        subsection_match = re.match(r'^([\d.]+)\s+(.+)', line_text)
                        subsection_num = subsection_match.group(1) if subsection_match else None
                        subsection_title = line_text  # Keep original for display

                        # If no keyword match but has numbered pattern, infer parent section
                        if not matched_section and subsection_num and "." in subsection_num:
                            # Get the main section number (e.g., "3" from "3.1")
                            main_num = subsection_num.split(".")[0]
                            # Try to find the parent section from previous headers
                            for prev_h in reversed(headers):
                                prev_num = prev_h.get("subsection_num", "")
                                if prev_num == main_num:
                                    matched_section = prev_h["section_name"]
                                    break

                        if matched_section:
                            headers.append({
                                "text": line_text,
                                "section_name": matched_section,
                                "subsection_num": subsection_num,
                                "subsection_title": subsection_title,
                                "page": page_num,
                                "position": char_position,
                                "font_size": line_font_size,
                                "is_bold": line_is_bold
                            })

                    char_position += len(line_text) + 1

        # 중복 제거 (같은 섹션이 여러 번 감지될 수 있음)
        # But allow different subsections (e.g., 3.1, 3.2, 3.3 are all under Results)
        seen_sections = set()
        unique_headers = []
        for h in headers:
            # Use section_name + page + subsection_num as key to allow different subsections
            subsec = h.get("subsection_num") or ""
            key = (h["section_name"], h["page"], subsec)
            if key not in seen_sections:
                seen_sections.add(key)
                unique_headers.append(h)

        return unique_headers

    def _extract_title_by_font(self, doc) -> str:
        """Extract paper title using font size (largest text on first page)."""
        if len(doc) == 0:
            return ""

        page = doc[0]
        blocks = page.get_text("dict")["blocks"]

        # Collect text spans with font info
        title_candidates = []

        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = ""
                max_font_size = 0
                is_bold = False

                for span in line["spans"]:
                    line_text += span["text"]
                    max_font_size = max(max_font_size, span["size"])
                    if span["flags"] & 16:  # Bold flag
                        is_bold = True

                line_text = line_text.strip()

                # Skip empty or very short lines
                if len(line_text) < 5:
                    continue

                # Skip metadata patterns
                skip_patterns = ["http", "://", "doi:", "issn", "copyright", "©",
                               "journal", "volume", "published", "@", "review"]
                if any(p in line_text.lower() for p in skip_patterns):
                    continue

                title_candidates.append({
                    "text": line_text,
                    "font_size": max_font_size,
                    "is_bold": is_bold
                })

        if not title_candidates:
            return ""

        # Find the largest font size (likely title)
        max_font = max(c["font_size"] for c in title_candidates)

        # Collect all lines with the largest font (title might span multiple lines)
        title_parts = []
        for c in title_candidates:
            if c["font_size"] >= max_font - 1:  # Allow small variation
                title_parts.append(c["text"])
            elif title_parts:  # Stop once we've passed the title
                break

        # Combine title parts
        title = " ".join(title_parts)

        # Clean up title
        title = re.sub(r"\s+", " ", title).strip()

        return title

    def _match_section_keyword(self, text: str) -> Optional[str]:
        """텍스트에서 섹션 키워드 매칭."""
        text = text.strip().lower()

        # Remove leading numbers and punctuation (e.g., "1 ", "1. ", "2.1 ", "4.2.1 ")
        # This handles formats like "1 Introduction", "1. Introduction", "2.1 Methods"
        stripped_text = re.sub(r'^[\d.]+\s*', '', text).strip()

        # 정확한 매칭 우선 (번호 제거 후)
        if stripped_text in self.SECTION_KEYWORDS:
            return self.SECTION_KEYWORDS[stripped_text]

        # Also check original text
        if text in self.SECTION_KEYWORDS:
            return self.SECTION_KEYWORDS[text]

        # 부분 매칭 (키워드로 시작하는 경우)
        for keyword, section_name in self.SECTION_KEYWORDS.items():
            # Check stripped text first
            if stripped_text.startswith(keyword) or stripped_text == keyword:
                return section_name
            # Check original text
            if text.startswith(keyword) or text == keyword:
                return section_name
            # "1. Introduction" 같은 번호 패턴 (more flexible regex)
            if re.match(rf"^[\d.]+\s*{re.escape(keyword)}", text):
                return section_name

        # Special handling for compound section names
        compound_sections = [
            ("clinical predictor", "Clinical Predictors"),
            ("predictor of metastasis", "Clinical Predictors"),
            ("hereditary and phenotypic", "Hereditary Patterns"),
            ("susceptibility genes", "Genetics"),
            ("inheritance pattern", "Genetics"),
            ("molecular phenotype", "Molecular"),
            ("laboratory tests", "Laboratory"),
            ("imaging study", "Imaging"),
            ("genetic testing", "Genetics"),
            ("surgical resection", "Surgical Treatment"),
            ("preoperative stabilization", "Preoperative Management"),
            ("non-surgical", "Non-surgical Treatment"),
            ("novel therapies", "Novel Therapies"),
            ("restaging", "Surveillance"),
            ("metastatic ppgl", "Metastatic Disease"),
        ]
        for pattern, section_name in compound_sections:
            if pattern in stripped_text or pattern in text:
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
        text = self._fix_encoding_issues(text)
        text = self._remove_garbled_patterns(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common PDF encoding issues (ligatures, symbols, Greek letters)."""
        for bad_char, replacement in ENCODING_FIXES.items():
            text = text.replace(bad_char, replacement)
        return text

    def _remove_garbled_patterns(self, text: str) -> str:
        """Remove garbled math formula text from TeX/LaTeX fonts."""
        # Remove nonsensical consonant clusters
        for pattern in NONSENSE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove garbled lines
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if self._is_garbled_line(line):
                cleaned_lines.append('[mathematical formula]')
            else:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Clean up whitespace, line breaks, and control characters."""
        # Consolidate multiple newlines and spaces
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Apply exclude patterns
        for pattern in self.exclude_patterns:
            text = pattern.sub("", text)

        # Fix hyphenated words split across lines
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        return text

    def _extract_metadata(self, text: str, pdf_path: Path) -> PaperMetadata:
        """Extract paper metadata from text."""
        return self.metadata_extractor.extract(text, pdf_path)

    def _extract_sections_by_headers(self, full_text: str, headers: list[dict]) -> list[PaperSection]:
        """폰트 기반 헤더를 사용해 섹션 추출."""
        sections = []

        # 텍스트에서 헤더 위치 찾기
        header_positions = []
        for header in headers:
            # 헤더 텍스트를 전체 텍스트에서 찾기
            # Use partial matching (first 25 chars) to handle special characters
            search_text = header["text"][:25] if len(header["text"]) > 25 else header["text"]
            pattern = re.escape(search_text)
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            if matches:
                # Create display name: "Results - 3.1 Overview..." or just "Results"
                subsection_title = header.get("subsection_title", "")
                subsection_num = header.get("subsection_num", "")

                # If it's a subsection, create a combined display name
                # Only show subsection number if it's a real subsection (e.g., "3.1", "2.2")
                # Don't show for main sections like "3" or "3."
                if subsection_num:
                    # Clean up: remove trailing dots
                    clean_num = subsection_num.rstrip(".")
                    # Check if it's a real subsection (has format like "3.1", "2.2")
                    if "." in clean_num:
                        display_name = f"{header['section_name']} ({clean_num})"
                    else:
                        display_name = header["section_name"]
                else:
                    display_name = header["section_name"]

                header_positions.append({
                    "name": display_name,
                    "parent_section": header["section_name"],
                    "subsection_num": subsection_num,
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
