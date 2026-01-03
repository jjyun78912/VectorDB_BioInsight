#!/usr/bin/env python3
"""
PubMed Full Text Crawler
Extracts full text from Open Access articles via PubMed.

Usage:
    python pubmed_crawler.py <PMID> [--output <path>]

Example:
    python pubmed_crawler.py 41454916
    python pubmed_crawler.py 41454916 --output ./papers/
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Required packages missing. Install with:")
    print("  pip install requests beautifulsoup4 lxml")
    sys.exit(1)


@dataclass
class PaperMetadata:
    pmid: str
    title: str
    authors: list[str]
    journal: str
    publication_date: str
    doi: Optional[str]
    pmc_id: Optional[str]
    abstract: str


@dataclass
class PaperFullText:
    metadata: PaperMetadata
    sections: dict[str, str]
    references: list[str]
    source_url: str
    crawl_timestamp: str


class PubMedCrawler:
    """Crawls PubMed and extracts full text from Open Access articles."""
    
    PUBMED_BASE = "https://pubmed.ncbi.nlm.nih.gov"
    PMC_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles"
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; AcademicCrawler/1.0; +research)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    # Common section name patterns
    SECTION_PATTERNS = {
        "abstract": r"^abstract",
        "introduction": r"^(introduction|background)",
        "methods": r"^(methods?|materials?\s*(and|&)\s*methods?|experimental|procedures?)",
        "results": r"^results?",
        "discussion": r"^discussion",
        "conclusion": r"^(conclusion|concluding\s*remarks?|summary)",
        "acknowledgements": r"^(acknowledg|funding|support)",
        "references": r"^(references?|bibliography|citations?)",
    }
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize crawler with polite delay between requests.
        
        Args:
            delay: Seconds to wait between requests (default 1.0 for politeness)
        """
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.delay = delay
        self._last_request_time = 0
    
    def _polite_request(self, url: str, **kwargs) -> requests.Response:
        """Make a request with polite delay."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        
        response = self.session.get(url, **kwargs)
        self._last_request_time = time.time()
        return response
    
    def get_metadata_from_pubmed(self, pmid: str) -> PaperMetadata:
        """Fetch paper metadata from PubMed page."""
        url = f"{self.PUBMED_BASE}/{pmid}/"
        response = self._polite_request(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        
        # Extract title
        title_elem = soup.select_one("h1.heading-title")
        title = title_elem.get_text(strip=True) if title_elem else ""
        
        # Extract authors
        authors = []
        author_elems = soup.select("span.authors-list-item a.full-name")
        for elem in author_elems:
            authors.append(elem.get_text(strip=True))
        
        # Extract journal info
        journal_elem = soup.select_one("button.journal-actions-trigger")
        journal = journal_elem.get_text(strip=True) if journal_elem else ""
        
        # Extract publication date
        date_elem = soup.select_one("span.cit")
        pub_date = date_elem.get_text(strip=True) if date_elem else ""
        
        # Extract DOI
        doi = None
        doi_elem = soup.select_one("span.citation-doi")
        if doi_elem:
            doi_text = doi_elem.get_text(strip=True)
            doi_match = re.search(r"10\.\d{4,}/[^\s]+", doi_text)
            if doi_match:
                doi = doi_match.group(0).rstrip(".")
        
        # Extract PMC ID if available
        pmc_id = None
        pmc_link = soup.select_one("a.id-link[href*='pmc/articles']")
        if pmc_link:
            pmc_match = re.search(r"PMC\d+", pmc_link.get("href", ""))
            if pmc_match:
                pmc_id = pmc_match.group(0)
        
        # Extract abstract
        abstract = ""
        abstract_elem = soup.select_one("div.abstract-content")
        if abstract_elem:
            abstract = abstract_elem.get_text(separator=" ", strip=True)
        
        return PaperMetadata(
            pmid=pmid,
            title=title,
            authors=authors,
            journal=journal,
            publication_date=pub_date,
            doi=doi,
            pmc_id=pmc_id,
            abstract=abstract,
        )
    
    def get_fulltext_link(self, pmid: str) -> Optional[str]:
        """Get the full text link from PubMed page."""
        url = f"{self.PUBMED_BASE}/{pmid}/"
        response = self._polite_request(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Method 1: Look for PMC link in full-text-links section
        full_text_div = soup.select_one("div.full-text-links-list")
        if full_text_div:
            # Find all links
            all_links = full_text_div.find_all("a", href=True)
            for link in all_links:
                href = link.get("href", "")
                if "pmc.ncbi.nlm.nih.gov" in href or "pmc/articles" in href:
                    return href

        # Method 2: Look for any anchor with pmc URL
        pmc_links = soup.find_all("a", href=re.compile(r"pmc\.ncbi\.nlm\.nih\.gov|pmc/articles"))
        if pmc_links:
            return pmc_links[0].get("href")

        # Method 3: Construct PMC URL from meta tags
        keywords = soup.find("meta", {"name": "keywords"})
        if keywords:
            content = keywords.get("content", "")
            pmc_match = re.search(r"PMC\d+", content)
            if pmc_match:
                pmc_id = pmc_match.group(0)
                return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"

        # Method 4: Try any free full text link
        free_links = soup.select("div.full-text-links-list a")
        for link in free_links:
            href = link.get("href", "")
            if href and href.startswith("http"):
                return href

        return None
    
    def _normalize_section_name(self, raw_name: str) -> str:
        """Normalize section name to standard format."""
        raw_lower = raw_name.lower().strip()
        
        for standard_name, pattern in self.SECTION_PATTERNS.items():
            if re.match(pattern, raw_lower, re.IGNORECASE):
                return standard_name
        
        # Return original if no match, cleaned up
        return re.sub(r"[^\w\s]", "", raw_name).strip().lower().replace(" ", "_")
    
    def extract_pmc_fulltext(self, pmc_url: str) -> dict[str, str]:
        """Extract full text from PMC article."""
        response = self._polite_request(pmc_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        sections = {}
        
        # Find main article content
        article = soup.select_one("article.main-content, div.article")
        if not article:
            article = soup
        
        # Extract sections by heading
        current_section = "preamble"
        current_content = []
        
        for elem in article.find_all(["h1", "h2", "h3", "h4", "p", "div"]):
            if elem.name in ["h1", "h2", "h3", "h4"]:
                # Save previous section
                if current_content:
                    text = " ".join(current_content)
                    if text.strip():
                        normalized = self._normalize_section_name(current_section)
                        if normalized in sections:
                            sections[normalized] += " " + text
                        else:
                            sections[normalized] = text
                
                current_section = elem.get_text(strip=True)
                current_content = []
            elif elem.name == "p":
                text = elem.get_text(separator=" ", strip=True)
                if text and len(text) > 20:  # Filter out very short elements
                    current_content.append(text)
        
        # Save last section
        if current_content:
            text = " ".join(current_content)
            if text.strip():
                normalized = self._normalize_section_name(current_section)
                if normalized in sections:
                    sections[normalized] += " " + text
                else:
                    sections[normalized] = text
        
        return sections
    
    def extract_publisher_fulltext(self, url: str) -> dict[str, str]:
        """Extract full text from publisher site (generic extractor)."""
        response = self._polite_request(url, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        sections = {}
        
        # Try common article content selectors
        content_selectors = [
            "article",
            "div.article-content",
            "div.fulltext-view",
            "div.NLM_sec",
            "main",
            "div#article-content",
        ]
        
        article = None
        for selector in content_selectors:
            article = soup.select_one(selector)
            if article:
                break
        
        if not article:
            article = soup.body or soup
        
        # Extract text by sections
        for section_div in article.find_all(["section", "div"], class_=re.compile(r"(sec|section)", re.I)):
            heading = section_div.find(["h1", "h2", "h3", "h4"])
            if heading:
                section_name = heading.get_text(strip=True)
                paragraphs = section_div.find_all("p")
                content = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
                
                if content.strip():
                    normalized = self._normalize_section_name(section_name)
                    if normalized in sections:
                        sections[normalized] += " " + content
                    else:
                        sections[normalized] = content
        
        # If no sections found, try to extract all paragraphs
        if not sections:
            all_paragraphs = article.find_all("p")
            full_text = " ".join(p.get_text(separator=" ", strip=True) for p in all_paragraphs)
            if full_text.strip():
                sections["full_text"] = full_text
        
        return sections
    
    def extract_references(self, soup: BeautifulSoup) -> list[str]:
        """Extract references from article."""
        references = []
        
        # Common reference selectors
        ref_selectors = [
            "div.ref-cit-blk",
            "li.ref",
            "div.citation",
            "ol.references li",
        ]
        
        for selector in ref_selectors:
            refs = soup.select(selector)
            if refs:
                for ref in refs:
                    text = ref.get_text(separator=" ", strip=True)
                    if text and len(text) > 10:
                        references.append(text)
                break
        
        return references
    
    def crawl(self, pmid: str) -> PaperFullText:
        """
        Crawl a PubMed article and extract full text.
        
        Args:
            pmid: PubMed ID (numeric string)
            
        Returns:
            PaperFullText object with metadata and sections
            
        Raises:
            ValueError: If article is not Open Access
            requests.HTTPError: If network request fails
        """
        print(f"Fetching metadata for PMID: {pmid}")
        metadata = self.get_metadata_from_pubmed(pmid)
        
        print(f"Looking for full text link...")
        fulltext_url = self.get_fulltext_link(pmid)
        
        if not fulltext_url:
            raise ValueError(f"No Open Access full text available for PMID {pmid}")
        
        print(f"Full text URL: {fulltext_url}")
        
        # Determine extraction method based on URL
        if "pmc" in fulltext_url.lower() or "ncbi.nlm.nih.gov" in fulltext_url:
            print("Extracting from PMC...")
            sections = self.extract_pmc_fulltext(fulltext_url)
        else:
            print("Extracting from publisher site...")
            sections = self.extract_publisher_fulltext(fulltext_url)
        
        # Add abstract if not already present
        if "abstract" not in sections and metadata.abstract:
            sections["abstract"] = metadata.abstract
        
        # Try to get references
        references = []
        
        return PaperFullText(
            metadata=metadata,
            sections=sections,
            references=references,
            source_url=fulltext_url,
            crawl_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    
    def to_json(self, paper: PaperFullText) -> str:
        """Convert PaperFullText to JSON string."""
        data = {
            "metadata": asdict(paper.metadata),
            "sections": paper.sections,
            "references": paper.references,
            "source_url": paper.source_url,
            "crawl_timestamp": paper.crawl_timestamp,
        }
        return json.dumps(data, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Extract full text from Open Access PubMed articles"
    )
    parser.add_argument("pmid", help="PubMed ID (e.g., 41454916)")
    parser.add_argument(
        "--output", "-o",
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Validate PMID
    pmid = args.pmid.strip()
    if not pmid.isdigit():
        print(f"Error: Invalid PMID '{pmid}'. Must be numeric.", file=sys.stderr)
        sys.exit(1)
    
    crawler = PubMedCrawler(delay=args.delay)
    
    try:
        paper = crawler.crawl(pmid)
        
        # Generate output filename
        import os
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, f"PMID_{pmid}.json")
        
        # Write JSON
        json_content = crawler.to_json(paper)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_content)
        
        print(f"\n✅ Successfully extracted full text!")
        print(f"   Title: {paper.metadata.title[:60]}...")
        print(f"   Sections: {', '.join(paper.sections.keys())}")
        print(f"   Output: {output_file}")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"Network error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()