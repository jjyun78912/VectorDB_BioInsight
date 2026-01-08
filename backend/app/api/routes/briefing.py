"""
BIO Daily Briefing API Routes

Provides endpoints to access the daily newsletter data.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter()

# Path to briefing output directory
BRIEFING_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "bio-daily-briefing" / "output"


class TrendItem(BaseModel):
    """Single trend item."""
    keyword: str
    count: int
    change_label: str
    trend_indicator: str
    category: Optional[str] = ""
    why_hot: Optional[str] = ""
    is_predefined: bool = True
    is_emerging: bool = False


class ArticleItem(BaseModel):
    """Single news article."""
    title: str
    summary: str
    source: Optional[str] = ""
    pmid: Optional[str] = ""
    doi: Optional[str] = ""
    journal: Optional[str] = ""
    pub_date: Optional[str] = ""


class BriefingData(BaseModel):
    """Daily briefing data structure."""
    issue_number: int
    date: str
    total_papers_analyzed: int
    trends: List[TrendItem]
    articles: Optional[dict] = {}  # articles grouped by trend keyword
    articles_by_trend: Optional[dict] = {}  # alias for compatibility
    editor_comment: str
    quick_news: Optional[List[str]] = []
    emerging_trends: Optional[List[TrendItem]] = []


class BriefingListItem(BaseModel):
    """Briefing list item for archive."""
    date: str
    issue_number: int
    filename: str


@router.get("/latest", response_model=BriefingData)
async def get_latest_briefing():
    """
    Get the latest daily briefing.

    Returns the most recent newsletter data including trends,
    articles, and editor comments.
    """
    try:
        # Find latest JSON file
        json_files = sorted(BRIEFING_OUTPUT_DIR.glob("briefing_*.json"), reverse=True)

        if not json_files:
            raise HTTPException(status_code=404, detail="No briefing found")

        latest_file = json_files[0]

        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return BriefingData(**data)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Briefing file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid briefing data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/date/{date_str}", response_model=BriefingData)
async def get_briefing_by_date(date_str: str):
    """
    Get briefing for a specific date.

    Args:
        date_str: Date in YYYYMMDD format
    """
    try:
        filepath = BRIEFING_OUTPUT_DIR / f"briefing_{date_str}.json"

        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"No briefing found for {date_str}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return BriefingData(**data)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Briefing file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/html/latest")
async def get_latest_briefing_html():
    """
    Get the latest briefing as raw HTML.

    Returns the HTML content of the newsletter for embedding.
    """
    try:
        # Try new v3 format first (in html/ subdirectory)
        html_dir = BRIEFING_OUTPUT_DIR / "html"
        if html_dir.exists():
            html_files = sorted(html_dir.glob("bio_daily_briefing_*.html"), reverse=True)
            if html_files:
                latest_file = html_files[0]
                with open(latest_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                # Extract issue number from filename
                filename = latest_file.stem  # bio_daily_briefing_16_2026-01-05
                parts = filename.split("_")
                issue_num = parts[3] if len(parts) > 3 else "0"
                date_str = parts[4] if len(parts) > 4 else ""
                return {
                    "html": html_content,
                    "date": date_str,
                    "issue_number": int(issue_num),
                    "version": "v3"
                }

        # Fallback to old format
        html_files = sorted(BRIEFING_OUTPUT_DIR.glob("briefing_*.html"), reverse=True)

        if not html_files:
            raise HTTPException(status_code=404, detail="No briefing found")

        latest_file = html_files[0]

        with open(latest_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        return {"html": html_content, "date": latest_file.stem.replace("briefing_", ""), "version": "v2"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/newsletter/generate")
async def generate_newsletter():
    """
    Generate a new newsletter with sample data (for testing).

    Returns the path to the generated HTML file.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "bio-daily-briefing"))

        from src.newsletter_generator import NewsletterGenerator, create_sample_data

        generator = NewsletterGenerator()
        sample_data = create_sample_data()

        # Get next issue number
        html_dir = BRIEFING_OUTPUT_DIR / "html"
        if html_dir.exists():
            existing = list(html_dir.glob("bio_daily_briefing_*.html"))
            issue_numbers = []
            for f in existing:
                parts = f.stem.split("_")
                if len(parts) > 3:
                    try:
                        issue_numbers.append(int(parts[3]))
                    except (ValueError, IndexError) as e:
                        pass  # Invalid filename format
            next_issue = max(issue_numbers) + 1 if issue_numbers else 1
        else:
            next_issue = 1

        filepath = generator.generate_and_save(
            data=sample_data,
            issue_number=next_issue
        )

        return {
            "success": True,
            "issue_number": next_issue,
            "filepath": filepath,
            "message": f"Newsletter #{next_issue} generated successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/archive", response_model=List[BriefingListItem])
async def get_briefing_archive():
    """
    Get list of all available briefings.

    Returns a list of all briefing dates and issue numbers.
    """
    try:
        json_files = sorted(BRIEFING_OUTPUT_DIR.glob("briefing_*.json"), reverse=True)

        archive = []
        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                date_str = filepath.stem.replace("briefing_", "")
                archive.append(BriefingListItem(
                    date=date_str,
                    issue_number=data.get("issue_number", 0),
                    filename=filepath.name
                ))
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                continue  # Skip invalid briefing files

        return archive

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/summary")
async def get_trends_summary():
    """
    Get a quick summary of current hot topics.

    Returns top 5 trends with their counts and changes.
    """
    try:
        json_files = sorted(BRIEFING_OUTPUT_DIR.glob("briefing_*.json"), reverse=True)

        if not json_files:
            raise HTTPException(status_code=404, detail="No briefing found")

        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        trends = data.get("trends", [])[:5]

        return {
            "date": data.get("date", ""),
            "issue_number": data.get("issue_number", 0),
            "total_papers": data.get("total_papers_analyzed", 0),
            "top_trends": [
                {
                    "keyword": t.get("keyword"),
                    "count": t.get("count"),
                    "change": t.get("change_label"),
                    "indicator": t.get("trend_indicator")
                }
                for t in trends
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
