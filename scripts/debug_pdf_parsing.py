#!/usr/bin/env python3
"""Debug script to analyze PDF parsing and chunk content."""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_parser import BioPaperParser
from src.text_splitter import BioPaperSplitter

def debug_pdf(pdf_path: str):
    """Analyze PDF parsing in detail."""
    print(f"\n{'='*60}")
    print(f"Debugging PDF: {pdf_path}")
    print('='*60)

    parser = BioPaperParser()
    splitter = BioPaperSplitter()

    # Parse PDF
    print("\n[1] Parsing PDF...")
    metadata, sections = parser.parse_pdf(pdf_path)

    # Show metadata
    print(f"\n[2] Metadata:")
    print(f"   Title: {metadata.title[:80]}..." if len(metadata.title) > 80 else f"   Title: {metadata.title}")
    print(f"   DOI: {metadata.doi}")
    print(f"   Year: {metadata.year}")
    print(f"   Keywords: {metadata.keywords[:5]}" if metadata.keywords else "   Keywords: None")
    print(f"   Abstract length: {len(metadata.abstract)} chars")

    # Show sections
    print(f"\n[3] Extracted Sections ({len(sections)} total):")
    total_content = 0
    for i, section in enumerate(sections):
        content_len = len(section.content)
        total_content += content_len
        preview = section.content[:100].replace('\n', ' ')
        print(f"   {i+1}. {section.name}: {content_len} chars")
        print(f"      Preview: {preview}...")

        if section.subsections:
            for sub in section.subsections:
                print(f"      - {sub.name}: {len(sub.content)} chars")

    print(f"\n   Total content: {total_content} chars")

    # Split into chunks
    print(f"\n[4] Splitting into chunks...")
    chunks = splitter.split_paper(metadata, sections)

    print(f"   Total chunks: {len(chunks)}")

    # Analyze chunks
    section_counts = {}
    for chunk in chunks:
        section = chunk.metadata.get("section", "Unknown")
        section_counts[section] = section_counts.get(section, 0) + 1

    print(f"\n[5] Chunks by section:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"   {section}: {count} chunks")

    # Show sample chunks
    print(f"\n[6] Sample chunks content:")
    for i, chunk in enumerate(chunks[:5]):
        section = chunk.metadata.get("section", "Unknown")
        preview = chunk.content[:150].replace('\n', ' ')
        print(f"\n   Chunk {i+1} [{section}]:")
        print(f"   {preview}...")

    # Check for potential issues
    print(f"\n[7] Quality check:")

    # Check if content is mostly numbers/symbols
    empty_chunks = sum(1 for c in chunks if len(c.content.strip()) < 50)
    print(f"   Empty/short chunks: {empty_chunks}")

    # Check for junk content ratio
    junk_sections = ["References", "Acknowledgments", "Author Contributions"]
    junk_chunks = sum(1 for c in chunks if c.metadata.get("section") in junk_sections)
    print(f"   Junk section chunks: {junk_chunks}")

    # Check for scientific content indicators
    scientific_keywords = ["study", "result", "method", "patient", "treatment", "analysis",
                          "data", "significant", "p-value", "protein", "gene", "cell"]
    scientific_chunks = 0
    for chunk in chunks:
        content_lower = chunk.content.lower()
        if any(kw in content_lower for kw in scientific_keywords):
            scientific_chunks += 1
    print(f"   Scientific content chunks: {scientific_chunks}/{len(chunks)}")

    return chunks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Look for test PDFs
        test_dirs = [
            Path("/Users/admin/VectorDB_BioInsight/data/papers"),
            Path("/Users/admin/VectorDB_BioInsight/input"),
            Path.home() / "Downloads"
        ]

        pdf_found = None
        for d in test_dirs:
            if d.exists():
                pdfs = list(d.glob("*.pdf"))
                if pdfs:
                    pdf_found = pdfs[0]
                    break

        if pdf_found:
            print(f"Using found PDF: {pdf_found}")
            debug_pdf(str(pdf_found))
        else:
            print("Usage: python debug_pdf_parsing.py <path_to_pdf>")
            print("\nNo test PDFs found in common locations.")
    else:
        debug_pdf(sys.argv[1])
