"""
Tests for the BioPaperSplitter text splitter.
"""
import pytest
from backend.app.core.text_splitter import BioPaperSplitter, TextChunk, PaperMetadata, PaperSection


class TestBioPaperSplitter:
    """Test suite for BioPaperSplitter."""

    @pytest.fixture
    def splitter(self):
        """Create a splitter instance."""
        return BioPaperSplitter(chunk_size=500, chunk_overlap=50)

    @pytest.fixture
    def sample_metadata(self):
        """Create sample paper metadata."""
        return PaperMetadata(
            title="Test Paper on KRAS",
            doi="10.1000/test",
            authors=["Smith J", "Jones A"],
            journal="Test Journal",
            year="2024",
            keywords=["cancer", "KRAS"]
        )

    @pytest.fixture
    def sample_sections(self):
        """Create sample paper sections."""
        return [
            PaperSection(
                name="Abstract",
                content="We investigated KRAS mutations in pancreatic cancer. Our results show significant gene expression changes."
            ),
            PaperSection(
                name="Introduction",
                content="Pancreatic cancer is one of the most lethal cancers. KRAS mutations are found in over 90% of cases."
            ),
            PaperSection(
                name="Methods",
                content="RNA was extracted using TRIzol. Libraries were prepared with Illumina kits. Sequencing was performed on NovaSeq."
            ),
            PaperSection(
                name="Results",
                content="We identified 256 differentially expressed genes. Pathway analysis revealed MAPK signaling enrichment."
            ),
            PaperSection(
                name="References",
                content="1. Smith et al. (2020) Cancer Cell. 2. Jones et al. (2021) Nature Medicine."
            )
        ]

    def test_splitter_initialization(self, splitter):
        """Test that splitter initializes correctly."""
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50

    def test_split_paper_returns_chunks(self, splitter, sample_metadata, sample_sections):
        """Test that split_paper returns TextChunk objects."""
        chunks = splitter.split_paper(sample_metadata, sample_sections)

        # Should produce multiple chunks
        assert len(chunks) > 0

        # All should be TextChunk instances
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert chunk.content
            assert chunk.metadata is not None

    def test_chunks_have_section_metadata(self, splitter, sample_metadata, sample_sections):
        """Test that chunks have section information in metadata."""
        chunks = splitter.split_paper(sample_metadata, sample_sections)

        sections_found = set()
        for chunk in chunks:
            if "section" in chunk.metadata:
                sections_found.add(chunk.metadata["section"])

        # Should find at least some sections
        assert len(sections_found) > 0

    def test_excludes_references_section(self, splitter, sample_metadata, sample_sections):
        """Test that References section is excluded."""
        chunks = splitter.split_paper(sample_metadata, sample_sections)

        for chunk in chunks:
            section = chunk.metadata.get("section", "").lower()
            assert "reference" not in section
            # Content from References should not appear
            assert "Smith et al. (2020) Cancer Cell" not in chunk.content

    def test_chunks_have_paper_metadata(self, splitter, sample_metadata, sample_sections):
        """Test that chunks include paper metadata."""
        chunks = splitter.split_paper(sample_metadata, sample_sections)

        for chunk in chunks:
            # Should have paper title
            assert chunk.metadata.get("paper_title") == sample_metadata.title

    def test_custom_chunk_size(self, sample_metadata, sample_sections):
        """Test splitter with custom chunk size."""
        splitter = BioPaperSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_paper(sample_metadata, sample_sections)

        # With smaller chunks, should have more of them
        assert len(chunks) >= 2


class TestTextChunk:
    """Test suite for TextChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a TextChunk."""
        chunk = TextChunk(
            content="Sample text content",
            metadata={"section": "Abstract", "chunk_index": 0}
        )

        assert chunk.content == "Sample text content"
        assert chunk.metadata["section"] == "Abstract"
        assert chunk.metadata["chunk_index"] == 0

    def test_chunk_with_empty_metadata(self):
        """Test creating a TextChunk with empty metadata."""
        chunk = TextChunk(content="Sample text", metadata={})

        assert chunk.content == "Sample text"
        assert chunk.metadata == {}

    def test_chunk_strips_whitespace(self):
        """Test that chunk content is stripped."""
        chunk = TextChunk(content="  content with spaces  ", metadata={})

        assert chunk.content == "content with spaces"


class TestPaperMetadata:
    """Test suite for PaperMetadata."""

    def test_metadata_creation(self):
        """Test creating paper metadata."""
        metadata = PaperMetadata(
            title="Test Title",
            doi="10.1000/test",
            authors=["Author1"],
            journal="Journal",
            year="2024"
        )

        assert metadata.title == "Test Title"
        assert metadata.doi == "10.1000/test"

    def test_metadata_optional_fields(self):
        """Test metadata with minimal fields."""
        metadata = PaperMetadata(
            title="Test"
        )

        assert metadata.title == "Test"
        assert metadata.authors == []  # default empty list


class TestPaperSection:
    """Test suite for PaperSection."""

    def test_section_creation(self):
        """Test creating a paper section."""
        section = PaperSection(
            name="Introduction",
            content="This is the introduction text."
        )

        assert section.name == "Introduction"
        assert section.content == "This is the introduction text."
