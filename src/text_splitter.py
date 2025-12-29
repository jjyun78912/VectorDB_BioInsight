"""
Bio Paper Text Splitter - Section-aware chunking for biomedical papers.

Strategy:
1. Split by section (Abstract, Introduction, Methods, Results, etc.)
2. Further split large sections using RecursiveCharacterTextSplitter
3. Preserve section/subsection context in metadata
"""
import re
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .pdf_parser import PaperSection, PaperMetadata

# Patterns to filter out from chunks (non-scientific content)
JUNK_PATTERNS = [
    # Author contributions - various formats
    r"(?:Author\s+)?Contributions?[:.].*?(?:Writing|Review|Editing|Conceptualization|Methodology|Validation|Investigation|Supervision|Data\s+curation)",
    r"[A-Z]{1,3}[-–][A-Z]:\s*(?:Writing|Review|Conceptualization|Methodology|Investigation|Supervision)",  # e.g., "HN-C: Writing"
    r"[A-Z][A-Za-z\-]+\s+[A-Z][A-Za-z\-]+:\s*(?:Writing|Review|Conceptualization|Validation)",  # e.g., "John Smith: Writing"
    # Funding statements
    r"(?:Funding|Financial\s+Support)[:.].*?(?:grant|funding|supported|funded)",
    # Conflicts of interest
    r"(?:Conflicts?\s+of\s+Interest|Competing\s+Interests?|Declaration)[:.].*?(?:declare|conflict|interest)",
    # Acknowledgments
    r"Acknowledgm?ents?[:.].*",
    # References list items
    r"^\d+\.\s+[A-Z][a-z]+\s+[A-Z]{1,2}[,.].*?(?:\d{4}|\(\d{4}\))",
]

# Sections that should be completely excluded from embedding
EXCLUDE_SECTION_NAMES = [
    "author contributions",
    "author contribution",
    "authors' contributions",
    "acknowledgments",
    "acknowledgements",
    "competing interests",
    "conflict of interest",
    "conflicts of interest",
    "funding",
    "financial support",
    "references",
    "bibliography",
    "abbreviations",
    "supplementary material",
    "supplementary information",
    "data availability",
    "ethical approval",
    "ethics statement",
]


@dataclass
class TextChunk:
    """A chunk of text with associated metadata."""
    content: str
    metadata: dict

    def __post_init__(self):
        """Validate and clean chunk."""
        self.content = self.content.strip()
        # Allow empty chunks to be filtered later instead of raising error
        # This prevents pipeline failures on edge cases


class BioPaperSplitter:
    """
    Section-aware text splitter for biomedical papers.

    Splits papers hierarchically:
    1. Section level (Abstract, Introduction, Methods, etc.)
    2. Subsection level (for Methods: RNA extraction, Library prep, etc.)
    3. Chunk level (using RecursiveCharacterTextSplitter)
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] | None = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Bio-specific separators (paragraph, sentence, clause boundaries)
        self.separators = separators or [
            "\n\n",      # Paragraph break
            "\n",        # Line break
            ". ",        # Sentence end
            "; ",        # Clause separator
            ", ",        # Phrase separator
            " ",         # Word boundary
            ""           # Character fallback
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split_paper(
        self,
        metadata: PaperMetadata,
        sections: list[PaperSection]
    ) -> list[TextChunk]:
        """
        Split a parsed paper into chunks with rich metadata.

        Args:
            metadata: Paper metadata (title, DOI, etc.)
            sections: List of paper sections

        Returns:
            List of TextChunk objects with metadata
        """
        chunks = []
        chunk_index = 0

        for section in sections:
            # Skip sections that are completely excluded
            section_lower = section.name.lower().strip()
            should_skip = False
            for exclude_name in EXCLUDE_SECTION_NAMES:
                if exclude_name in section_lower or section_lower in exclude_name:
                    should_skip = True
                    break
            if should_skip:
                continue

            section_chunks = self._split_section(
                section=section,
                paper_metadata=metadata,
                chunk_start_index=chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Filter out empty chunks and junk content
        chunks = [c for c in chunks if c.content.strip() and not self._is_junk_content(c.content, c.metadata.get("section", ""))]

        return chunks

    def _is_junk_content(self, text: str, section_name: str = "") -> bool:
        """Check if chunk contains mostly non-scientific content."""
        # Check if section name indicates excluded content
        section_lower = section_name.lower().strip()
        for exclude_name in EXCLUDE_SECTION_NAMES:
            if exclude_name in section_lower or section_lower in exclude_name:
                return True

        # Check for junk patterns
        for pattern in JUNK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                # If pattern matches more than 30% of content, it's junk
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match and len(match.group()) > len(text) * 0.3:
                    return True

        # Check if chunk is mostly author contribution style text
        contrib_keywords = ["writing", "editing", "review", "conceptualization",
                          "methodology", "validation", "investigation", "supervision",
                          "data curation", "visualization", "funding acquisition",
                          "original draft", "formal analysis", "resources",
                          "project administration", "writing–review"]
        keyword_count = sum(1 for kw in contrib_keywords if kw.lower() in text.lower())
        if keyword_count >= 3:  # If 3+ contribution keywords, likely junk
            return True

        # Check for CRediT author contribution format: "XX-Y: Writing" patterns
        credit_pattern = r"[A-Z]{1,4}[-–]?[A-Z]?:\s*(?:Writing|Review|Conceptualization|Methodology|Investigation|Supervision|Validation|Funding|Resources)"
        credit_matches = re.findall(credit_pattern, text, re.IGNORECASE)
        if len(credit_matches) >= 2:  # Multiple CRediT role assignments
            return True

        return False

    def _split_section(
        self,
        section: PaperSection,
        paper_metadata: PaperMetadata,
        chunk_start_index: int,
        parent_section: str | None = None
    ) -> list[TextChunk]:
        """Split a single section into chunks."""
        chunks = []

        # Base metadata for all chunks from this section
        base_metadata = {
            "paper_title": paper_metadata.title,
            "doi": paper_metadata.doi,
            "year": paper_metadata.year,
            "keywords": paper_metadata.keywords,
            "source_file": paper_metadata.file_path,
            "section": section.name,
            "parent_section": parent_section,
        }

        # If section has subsections, process them separately
        if section.subsections:
            for subsection in section.subsections:
                sub_chunks = self._split_section(
                    section=subsection,
                    paper_metadata=paper_metadata,
                    chunk_start_index=chunk_start_index + len(chunks),
                    parent_section=section.name
                )
                chunks.extend(sub_chunks)

        # Split the main section content
        if section.content:
            text_chunks = self.text_splitter.split_text(section.content)

            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": chunk_start_index + len(chunks),
                    "chunk_in_section": i,
                    "total_chunks_in_section": len(text_chunks),
                }

                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata=chunk_metadata
                ))

        return chunks

    def split_text_simple(
        self,
        text: str,
        metadata: dict | None = None
    ) -> list[TextChunk]:
        """
        Simple text splitting without section awareness.
        Useful for plain text or when section parsing fails.

        Args:
            text: Raw text to split
            metadata: Optional metadata to attach to all chunks

        Returns:
            List of TextChunk objects
        """
        metadata = metadata or {}
        text_chunks = self.text_splitter.split_text(text)

        return [
            TextChunk(
                content=chunk,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                }
            )
            for i, chunk in enumerate(text_chunks)
        ]


def split_bio_paper(
    metadata: PaperMetadata,
    sections: list[PaperSection],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list[TextChunk]:
    """Convenience function to split a bio paper."""
    splitter = BioPaperSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_paper(metadata, sections)
