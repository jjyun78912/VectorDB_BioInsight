"""
Playwright-based Deep Crawler for BioInsight.

Provides advanced web crawling capabilities for extracting PDFs
and full-text content from journal websites that require JavaScript
rendering or complex navigation.

Features:
1. PDF download from publisher sites (Nature, Science, Cell, etc.)
2. Full-text extraction from HTML pages
3. Recursive crawling for related papers
4. Cookie/session handling for paywalled content
5. Rate limiting to respect robots.txt

Usage:
    from backend.app.core.playwright_crawler import PlaywrightDeepCrawler

    crawler = PlaywrightDeepCrawler()

    # Download PDF from publisher
    pdf_path = await crawler.download_pdf("https://www.nature.com/articles/...")

    # Extract full text from page
    text = await crawler.extract_full_text("https://www.cell.com/...")

    # Recursive crawl for references
    papers = await crawler.crawl_references("10.1038/s41586-023-06747-5", depth=2)
"""

import asyncio
import os
import re
import hashlib
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from urllib.parse import urljoin, urlparse

# Try to import playwright, handle if not installed
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not installed. Run: pip install playwright && playwright install chromium")

from .config import PAPERS_DIR
from .web_crawler_agent import FetchedPaper, WebCrawlerAgent


# Publisher-specific configurations
PUBLISHER_CONFIGS = {
    "nature.com": {
        "pdf_selector": "a[data-track-action='download pdf']",
        "pdf_fallback": "a[href*='.pdf']",
        "full_text_selector": "article.c-article-body",
        "title_selector": "h1.c-article-title",
        "abstract_selector": ".c-article-section__content",
        "wait_for": ".c-article-body",
    },
    "cell.com": {
        "pdf_selector": "a.article-tools__item__pdf",
        "pdf_fallback": "a[href*='pdf']",
        "full_text_selector": "div.article-content",
        "title_selector": "h1.article-header__title",
        "abstract_selector": ".article__abstract",
        "wait_for": ".article-content",
    },
    "science.org": {
        "pdf_selector": "a.btn-pdf",
        "pdf_fallback": "a[href*='.pdf']",
        "full_text_selector": ".article-content",
        "title_selector": "h1.article__headline",
        "abstract_selector": ".section.abstract",
        "wait_for": ".article-content",
    },
    "nih.gov": {
        "pdf_selector": "a.pdf-link",
        "pdf_fallback": "a[href*='pdf']",
        "full_text_selector": "#mc",
        "title_selector": "h1.content-title",
        "abstract_selector": "#abstract",
        "wait_for": "#mc",
    },
    "mdpi.com": {
        "pdf_selector": "a.download-pdf",
        "pdf_fallback": "a[href*='/pdf']",
        "full_text_selector": ".article-content",
        "title_selector": "h1.title",
        "abstract_selector": ".art-abstract",
        "wait_for": ".article-content",
    },
    "frontiersin.org": {
        "pdf_selector": "a.download-files-pdf",
        "pdf_fallback": "a[href*='.pdf']",
        "full_text_selector": ".article-content",
        "title_selector": "h1.JournalFullText",
        "abstract_selector": ".JournalAbstract",
        "wait_for": ".article-content",
    },
    "default": {
        "pdf_selector": "a[href*='.pdf']",
        "pdf_fallback": "a[href*='pdf']",
        "full_text_selector": "article, .article, .content, main",
        "title_selector": "h1",
        "abstract_selector": ".abstract, #abstract",
        "wait_for": "body",
    }
}


@dataclass
class CrawlResult:
    """Result from a deep crawl operation."""
    url: str
    success: bool
    pdf_path: Optional[str] = None
    full_text: str = ""
    title: str = ""
    abstract: str = ""
    references: List[str] = field(default_factory=list)
    error: Optional[str] = None
    crawled_at: str = ""

    def __post_init__(self):
        if not self.crawled_at:
            self.crawled_at = datetime.now().isoformat()


class PlaywrightDeepCrawler:
    """
    Deep crawler using Playwright for JavaScript-heavy sites.

    Handles:
    - PDF downloads from publisher sites
    - Full-text extraction
    - Reference harvesting
    - Rate limiting
    """

    def __init__(
        self,
        download_dir: Optional[Path] = None,
        headless: bool = True,
        rate_limit_delay: float = 2.0,
        max_retries: int = 3,
    ):
        """
        Initialize the deep crawler.

        Args:
            download_dir: Directory for downloaded PDFs
            headless: Run browser in headless mode
            rate_limit_delay: Delay between requests (seconds)
            max_retries: Maximum retry attempts
        """
        self.download_dir = download_dir or PAPERS_DIR
        self.headless = headless
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._last_request_time: float = 0
        self._visited_urls: Set[str] = set()

        # Web crawler agent for metadata
        self._web_agent = WebCrawlerAgent()

    async def _ensure_browser(self):
        """Ensure browser is initialized."""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is not installed. Run: pip install playwright && playwright install chromium")

        if self._browser is None:
            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context(
                accept_downloads=True,
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

    async def _rate_limit(self):
        """Apply rate limiting."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    def _get_publisher_config(self, url: str) -> Dict[str, str]:
        """Get publisher-specific configuration."""
        domain = urlparse(url).netloc.replace("www.", "")

        for publisher, config in PUBLISHER_CONFIGS.items():
            if publisher in domain:
                return config

        return PUBLISHER_CONFIGS["default"]

    async def download_pdf(self, url: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Download PDF from a publisher page.

        Args:
            url: URL of the article page (not the PDF URL directly)
            filename: Optional custom filename for the PDF

        Returns:
            Path to downloaded PDF or None if failed
        """
        await self._ensure_browser()
        await self._rate_limit()

        config = self._get_publisher_config(url)
        page = await self._context.new_page()

        try:
            # Navigate to page
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Wait for content to load
            try:
                await page.wait_for_selector(config["wait_for"], timeout=10000)
            except TimeoutError:
                pass  # Continue even if selector not found

            # Find PDF link
            pdf_url = None

            # Try primary selector
            pdf_link = await page.query_selector(config["pdf_selector"])
            if pdf_link:
                pdf_url = await pdf_link.get_attribute("href")

            # Try fallback selector
            if not pdf_url:
                pdf_link = await page.query_selector(config["pdf_fallback"])
                if pdf_link:
                    pdf_url = await pdf_link.get_attribute("href")

            if not pdf_url:
                print(f"No PDF link found on {url}")
                return None

            # Make URL absolute
            pdf_url = urljoin(url, pdf_url)

            # Generate filename
            if not filename:
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"paper_{url_hash}.pdf"

            if not filename.endswith(".pdf"):
                filename += ".pdf"

            pdf_path = self.download_dir / filename

            # Download with Playwright's download handling
            async with page.expect_download() as download_info:
                # Click the PDF link or navigate directly
                await page.goto(pdf_url, timeout=60000)

            download = await download_info.value
            await download.save_as(str(pdf_path))

            print(f"Downloaded PDF: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            print(f"Failed to download PDF from {url}: {e}")

            # Try direct download as fallback
            try:
                response = await page.request.get(pdf_url or url)
                if response.ok and "pdf" in (response.headers.get("content-type", "") or ""):
                    content = await response.body()
                    if not filename:
                        filename = f"paper_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
                    pdf_path = self.download_dir / filename
                    pdf_path.write_bytes(content)
                    return str(pdf_path)
            except (asyncio.TimeoutError, IOError, Exception) as e:
                pass  # PDF download failed

            return None
        finally:
            await page.close()

    async def extract_full_text(self, url: str) -> CrawlResult:
        """
        Extract full text content from an article page.

        Args:
            url: URL of the article

        Returns:
            CrawlResult with extracted content
        """
        await self._ensure_browser()
        await self._rate_limit()

        config = self._get_publisher_config(url)
        page = await self._context.new_page()

        try:
            # Navigate to page
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)

            # Wait for content
            try:
                await page.wait_for_selector(config["wait_for"], timeout=10000)
            except TimeoutError:
                pass  # Continue even if selector not found

            # Extract title
            title = ""
            title_elem = await page.query_selector(config["title_selector"])
            if title_elem:
                title = await title_elem.inner_text()

            # Extract abstract
            abstract = ""
            abstract_elem = await page.query_selector(config["abstract_selector"])
            if abstract_elem:
                abstract = await abstract_elem.inner_text()

            # Extract full text
            full_text = ""
            content_elem = await page.query_selector(config["full_text_selector"])
            if content_elem:
                full_text = await content_elem.inner_text()

            # Extract references (look for DOIs)
            references = []
            ref_links = await page.query_selector_all("a[href*='doi.org']")
            for ref in ref_links[:50]:  # Limit to 50 references
                href = await ref.get_attribute("href")
                if href and "doi.org" in href:
                    # Extract DOI
                    doi_match = re.search(r"10\.\d{4,}/[^\s\]\"'<>]+", href)
                    if doi_match:
                        references.append(doi_match.group())

            return CrawlResult(
                url=url,
                success=True,
                full_text=full_text,
                title=title.strip(),
                abstract=abstract.strip(),
                references=list(set(references)),
            )

        except Exception as e:
            return CrawlResult(
                url=url,
                success=False,
                error=str(e),
            )
        finally:
            await page.close()

    async def crawl_paper_with_pdf(self, doi_or_url: str) -> tuple[Optional[FetchedPaper], Optional[str]]:
        """
        Crawl a paper and download its PDF.

        Args:
            doi_or_url: DOI or URL of the paper

        Returns:
            Tuple of (FetchedPaper, pdf_path) or (None, None)
        """
        # First get metadata via API
        paper = None
        if doi_or_url.startswith("10."):
            paper = await self._web_agent.fetch_by_doi(doi_or_url)
            url = paper.url if paper else f"https://doi.org/{doi_or_url}"
        else:
            url = doi_or_url
            paper = await self._web_agent.fetch_by_url(url)

        # Try to download PDF
        pdf_path = await self.download_pdf(url)

        # If no PDF, try to extract full text
        if not pdf_path and paper:
            crawl_result = await self.extract_full_text(url)
            if crawl_result.success and crawl_result.full_text:
                paper.full_text = crawl_result.full_text

        return paper, pdf_path

    async def crawl_references(
        self,
        doi_or_url: str,
        depth: int = 1,
        max_papers: int = 20,
    ) -> List[FetchedPaper]:
        """
        Recursively crawl paper references.

        Args:
            doi_or_url: Starting DOI or URL
            depth: How many levels deep to crawl
            max_papers: Maximum total papers to fetch

        Returns:
            List of FetchedPaper objects
        """
        if depth < 1 or len(self._visited_urls) >= max_papers:
            return []

        papers = []

        # Get the starting paper
        if doi_or_url.startswith("10."):
            paper = await self._web_agent.fetch_by_doi(doi_or_url)
            url = paper.url if paper else f"https://doi.org/{doi_or_url}"
        else:
            url = doi_or_url
            paper = await self._web_agent.fetch_by_url(url)

        if paper and url not in self._visited_urls:
            self._visited_urls.add(url)
            papers.append(paper)

            # Get references from the page
            crawl_result = await self.extract_full_text(url)

            # Recursively crawl references
            if depth > 1 and crawl_result.references:
                for ref_doi in crawl_result.references[:10]:  # Limit per level
                    if len(papers) >= max_papers:
                        break

                    ref_papers = await self.crawl_references(
                        ref_doi,
                        depth=depth - 1,
                        max_papers=max_papers - len(papers)
                    )
                    papers.extend(ref_papers)

        return papers

    async def batch_download_pdfs(
        self,
        urls: List[str],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Download multiple PDFs.

        Args:
            urls: List of article URLs
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping URL to PDF path (or None if failed)
        """
        results = {}

        for i, url in enumerate(urls):
            if progress_callback:
                progress_callback(i, len(urls), url)

            pdf_path = await self.download_pdf(url)
            results[url] = pdf_path

        return results

    async def close(self):
        """Close browser and cleanup."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        self._browser = None
        self._context = None
        self._visited_urls.clear()


# Convenience function for one-off downloads
async def download_paper_pdf(url: str, output_dir: Optional[Path] = None) -> Optional[str]:
    """
    Download a single paper PDF.

    Args:
        url: URL of the paper
        output_dir: Output directory (default: PAPERS_DIR)

    Returns:
        Path to downloaded PDF or None
    """
    crawler = PlaywrightDeepCrawler(download_dir=output_dir)
    try:
        return await crawler.download_pdf(url)
    finally:
        await crawler.close()


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python playwright_crawler.py <url>")
        print("  Downloads PDF from the given paper URL")
        sys.exit(1)

    url = sys.argv[1]

    async def main():
        crawler = PlaywrightDeepCrawler(headless=True)
        try:
            print(f"Crawling: {url}")

            # Try to download PDF
            pdf_path = await crawler.download_pdf(url)

            if pdf_path:
                print(f"Downloaded: {pdf_path}")
            else:
                print("PDF download failed, extracting text...")
                result = await crawler.extract_full_text(url)
                print(f"Title: {result.title}")
                print(f"Abstract: {result.abstract[:500]}..." if result.abstract else "No abstract")
                print(f"References found: {len(result.references)}")
        finally:
            await crawler.close()

    asyncio.run(main())
