---
name: pubmed-fulltext-crawler
description: Extract full text from Open Access PubMed articles. Use when Claude needs to (1) crawl a PubMed article by PMID, (2) extract full paper text including Abstract, Introduction, Methods, Results, Discussion sections, (3) get paper metadata (authors, journal, DOI, PMC ID), or (4) save structured JSON output of academic papers. Only works with Open Access articles.
---

# PubMed Full Text Crawler

Extract full text and metadata from Open Access PubMed articles into structured JSON.

## Quick Start

```bash
python scripts/pubmed_crawler.py <PMID>
```

Example:
```bash
python scripts/pubmed_crawler.py 41454916 --output ./papers/
```

## Requirements

Install dependencies first:
```bash
pip install requests beautifulsoup4 lxml --break-system-packages
```

## Workflow

1. **Get PMID** from user (e.g., `41454916`)
2. **Run crawler**: `python scripts/pubmed_crawler.py <PMID>`
3. **Output**: JSON file saved as `PMID_<id>.json`

## Output Format

```json
{
  "metadata": {
    "pmid": "41454916",
    "title": "Paper title...",
    "authors": ["Author 1", "Author 2"],
    "journal": "Journal Name",
    "publication_date": "2026 Dec 31",
    "doi": "10.1080/...",
    "pmc_id": "PMC...",
    "abstract": "Abstract text..."
  },
  "sections": {
    "abstract": "...",
    "introduction": "...",
    "methods": "...",
    "results": "...",
    "discussion": "...",
    "conclusion": "..."
  },
  "references": [],
  "source_url": "https://...",
  "crawl_timestamp": "2026-01-02T12:00:00Z"
}
```

## Important Notes

- **Open Access only**: The crawler only works with Open Access articles that have free full text links
- **Rate limiting**: Default 1 second delay between requests for polite crawling
- **PMC preferred**: The crawler prioritizes PubMed Central (PMC) links for best extraction results

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| "No Open Access full text available" | Article is paywalled | Check if article has OA version or use different PMID |
| "Network error" | Connection failed | Check internet connection, retry |
| "Invalid PMID" | Non-numeric ID | Verify PMID is numeric |

## Command Options

```bash
python scripts/pubmed_crawler.py <PMID> [OPTIONS]

Options:
  --output, -o    Output directory (default: current directory)
  --delay         Delay between requests in seconds (default: 1.0)
```