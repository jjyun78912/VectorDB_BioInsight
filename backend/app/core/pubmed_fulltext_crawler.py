"""
PubMed Full Text Crawler Module
Extracts full text from Open Access articles via PubMed/PMC.

Integrated into the backend for automatic full text retrieval.
"""

import re
import time
from dataclasses import dataclass, asdict
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup


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

    def get_full_text(self) -> str:
        """Get all sections combined as a single text string."""
        # Order sections logically
        section_order = [
            "abstract", "introduction", "background", "methods",
            "results", "discussion", "conclusion", "preamble", "full_text"
        ]

        ordered_text = []
        used_sections = set()

        # Add sections in order
        for section in section_order:
            if section in self.sections:
                ordered_text.append(f"[{section.upper()}]\n{self.sections[section]}")
                used_sections.add(section)

        # Add any remaining sections
        for section, content in self.sections.items():
            if section not in used_sections and section not in ["acknowledgements", "references"]:
                ordered_text.append(f"[{section.upper()}]\n{content}")

        return "\n\n".join(ordered_text)


class PubMedFullTextCrawler:
    """Crawls PubMed and extracts full text from Open Access articles."""

    PUBMED_BASE = "https://pubmed.ncbi.nlm.nih.gov"
    PMC_BASE = "https://pmc.ncbi.nlm.nih.gov/articles"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; BioInsightCrawler/1.0; +research)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    # Section name patterns for normalization
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

    def __init__(self, delay: float = 0.5):
        """
        Initialize crawler with polite delay between requests.

        Args:
            delay: Seconds to wait between requests (default 0.5)
        """
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.delay = delay
        self._last_request_time = 0
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _polite_request(self, url: str, **kwargs) -> requests.Response:
        """Make a request with polite delay."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        response = self.session.get(url, timeout=30, **kwargs)
        self._last_request_time = time.time()
        return response

    def get_fulltext_link(self, pmid: str) -> Optional[str]:
        """Get the full text link from PubMed page."""
        url = f"{self.PUBMED_BASE}/{pmid}/"

        try:
            response = self._polite_request(url)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch PubMed page: {e}")
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Method 1: Look for PMC link in full-text-links section
        full_text_div = soup.select_one("div.full-text-links-list")
        if full_text_div:
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

        # Return cleaned up version
        return re.sub(r"[^\w\s]", "", raw_name).strip().lower().replace(" ", "_")

    def extract_pmc_fulltext(self, pmc_url: str) -> dict[str, str]:
        """Extract full text from PMC article."""
        try:
            response = self._polite_request(pmc_url)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch PMC page: {e}")
            return {}

        soup = BeautifulSoup(response.text, "lxml")
        sections = {}

        # Find main article content
        article = soup.select_one("article.main-content, div.article, main")
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
                    if text.strip() and len(text) > 50:  # Skip very short sections
                        normalized = self._normalize_section_name(current_section)
                        if normalized in sections:
                            sections[normalized] += " " + text
                        else:
                            sections[normalized] = text

                current_section = elem.get_text(strip=True)
                current_content = []
            elif elem.name == "p":
                text = elem.get_text(separator=" ", strip=True)
                if text and len(text) > 20:
                    current_content.append(text)

        # Save last section
        if current_content:
            text = " ".join(current_content)
            if text.strip() and len(text) > 50:
                normalized = self._normalize_section_name(current_section)
                if normalized in sections:
                    sections[normalized] += " " + text
                else:
                    sections[normalized] = text

        return sections

    def extract_publisher_fulltext(self, url: str) -> dict[str, str]:
        """Extract full text from publisher site (generic extractor)."""
        try:
            response = self._polite_request(url, allow_redirects=True)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch publisher page: {e}")
            return {}

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

                if content.strip() and len(content) > 50:
                    normalized = self._normalize_section_name(section_name)
                    if normalized in sections:
                        sections[normalized] += " " + content
                    else:
                        sections[normalized] = content

        # If no sections found, try to extract all paragraphs
        if not sections:
            all_paragraphs = article.find_all("p")
            full_text = " ".join(p.get_text(separator=" ", strip=True) for p in all_paragraphs)
            if full_text.strip() and len(full_text) > 100:
                sections["full_text"] = full_text

        return sections

    def crawl_sync(self, pmid: str) -> Optional[str]:
        """
        Synchronous crawl method - returns combined full text or None.

        Args:
            pmid: PubMed ID (numeric string)

        Returns:
            Full text string or None if not available
        """
        # Get full text link
        fulltext_url = self.get_fulltext_link(pmid)

        if not fulltext_url:
            print(f"No Open Access full text link found for PMID {pmid}")
            return None

        print(f"Found full text URL: {fulltext_url}")

        # Determine extraction method based on URL
        if "pmc" in fulltext_url.lower() or "ncbi.nlm.nih.gov" in fulltext_url:
            sections = self.extract_pmc_fulltext(fulltext_url)
        else:
            sections = self.extract_publisher_fulltext(fulltext_url)

        if not sections:
            print(f"Failed to extract sections from {fulltext_url}")
            return None

        # Combine sections into full text
        section_order = [
            "abstract", "introduction", "background", "methods",
            "results", "discussion", "conclusion", "preamble", "full_text"
        ]

        ordered_text = []
        used_sections = set()

        for section in section_order:
            if section in sections:
                ordered_text.append(f"[{section.upper()}]\n{sections[section]}")
                used_sections.add(section)

        for section, content in sections.items():
            if section not in used_sections and section not in ["acknowledgements", "references"]:
                ordered_text.append(f"[{section.upper()}]\n{content}")

        return "\n\n".join(ordered_text)

    async def crawl(self, pmid: str) -> Optional[str]:
        """
        Async crawl method - wraps sync method in executor.

        Args:
            pmid: PubMed ID (numeric string)

        Returns:
            Full text string or None if not available
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.crawl_sync, pmid)


# Singleton instance
_crawler: Optional[PubMedFullTextCrawler] = None


def get_fulltext_crawler() -> PubMedFullTextCrawler:
    """Get or create the singleton crawler instance."""
    global _crawler
    if _crawler is None:
        _crawler = PubMedFullTextCrawler()
    return _crawler


async def fetch_fulltext_by_pmid(pmid: str) -> Optional[str]:
    """
    Convenience function to fetch full text by PMID.

    Args:
        pmid: PubMed ID

    Returns:
        Full text string or None if not available
    """
    crawler = get_fulltext_crawler()
    return await crawler.crawl(pmid)
