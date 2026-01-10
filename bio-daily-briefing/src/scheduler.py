"""
Scheduler - Automation for daily newsletter generation and delivery
"""

import os
import sys
import json
import asyncio
import smtplib
import argparse
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import List, Optional

import schedule
from dotenv import load_dotenv

from .pubmed_fetcher import PubMedFetcher
from .trend_analyzer import TrendAnalyzer
from .ai_summarizer import AISummarizer
from .newsletter_generator import NewsletterGenerator
from .aggregator import NewsAggregator


class BriefingScheduler:
    """Manages automated newsletter generation and delivery."""

    def __init__(self, config_dir: Optional[Path] = None):
        # Load environment from multiple locations
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"

        # Try loading from VectorDB_BioInsight root first
        root_env = Path(__file__).parent.parent.parent / ".env"
        if root_env.exists():
            load_dotenv(root_env)
            print(f"Loaded .env from: {root_env}")

        # Also try config directory
        env_path = self.config_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"Loaded .env from: {env_path}")

        # Email settings
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.sender_email = os.getenv("SENDER_EMAIL", self.smtp_user)
        self.sender_name = os.getenv("SENDER_NAME", "BIO Daily Briefing")

        # Schedule settings
        self.generate_time = os.getenv("GENERATE_TIME", "06:00")
        self.send_time = os.getenv("SEND_TIME", "08:00")

        # Analysis settings
        self.papers_per_day = int(os.getenv("PAPERS_PER_DAY", "100"))
        self.top_trends = int(os.getenv("TOP_TRENDS", "5"))
        self.lookback_days = int(os.getenv("LOOKBACK_DAYS", "7"))

        # Slack (optional)
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")

        # Subscribers
        self.subscribers_file = self.config_dir / "subscribers.json"

        # Components
        self.fetcher = PubMedFetcher()
        self.analyzer = TrendAnalyzer()
        self.summarizer = AISummarizer(language="ko")
        self.generator = NewsletterGenerator()
        self.aggregator = NewsAggregator()  # Multi-source aggregator

        # State
        self._latest_html: Optional[str] = None
        self._latest_data: Optional[dict] = None

    def load_subscribers(self) -> List[str]:
        """Load subscriber email list."""
        if not self.subscribers_file.exists():
            return []

        with open(self.subscribers_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("subscribers", [])

    def save_subscribers(self, subscribers: List[str], unsubscribed: Optional[List[str]] = None):
        """Save subscriber list."""
        data = {
            "subscribers": subscribers,
            "unsubscribed": unsubscribed or [],
        }
        with open(self.subscribers_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_subscriber(self, email: str) -> bool:
        """Add a new subscriber."""
        subscribers = self.load_subscribers()
        if email not in subscribers:
            subscribers.append(email)
            self.save_subscribers(subscribers)
            print(f"Added subscriber: {email}")
            return True
        print(f"Subscriber already exists: {email}")
        return False

    def remove_subscriber(self, email: str) -> bool:
        """Remove a subscriber."""
        subscribers = self.load_subscribers()
        if email in subscribers:
            subscribers.remove(email)
            self.save_subscribers(subscribers)
            print(f"Removed subscriber: {email}")
            return True
        print(f"Subscriber not found: {email}")
        return False

    async def generate_briefing(self) -> Optional[str]:
        """
        Generate the daily briefing using multi-source aggregator.

        Returns:
            HTML string of the newsletter
        """
        print(f"\n[{datetime.now()}] Starting briefing generation...")

        try:
            # Get next issue number
            output_dir = Path(__file__).parent.parent / "output"
            issue_number_file = output_dir / "issue_number.txt"
            if issue_number_file.exists():
                try:
                    issue_number = int(issue_number_file.read_text().strip()) + 1
                except ValueError:
                    issue_number = 1
            else:
                existing = list(output_dir.glob("briefing_*.json"))
                issue_number = len(existing) + 1

            # 1. Use multi-source aggregator (FDA, ClinicalTrials, bioRxiv, PubMed)
            print("Using multi-source aggregator...")
            briefing_data = await self.aggregator.aggregate_daily(
                fda_hours=72,        # 3 days of FDA news
                trials_days=30,      # 30 days of clinical trials
                preprint_days=3,     # 3 days of preprints
                pubmed_days=2,       # 2 days of PubMed
                issue_number=issue_number,
            )

            # Get trends from aggregated data
            trends = briefing_data.hot_topics

            # 2. Generate AI summaries for hot topics
            print("Generating AI summaries...")
            # Get papers for summarization
            papers = await self.fetcher.fetch_comprehensive(days=2, max_total=300)
            articles_by_trend = self.summarizer.summarize_papers_by_trend(trends, max_per_trend=2)

            article_count = sum(len(a) for a in articles_by_trend.values())
            print(f"Generated {article_count} articles")

            # 3. Generate editor comment
            print("Generating editor comment...")
            editor_comment = self.summarizer.generate_editor_comment(trends)
            briefing_data.editor_comment = editor_comment

            # 4. Generate newsletter
            print("Generating newsletter HTML...")

            # Convert aggregated data to newsletter format
            agg_dict = briefing_data.to_dict()

            # Convert regulatory news to list format for newsletter generator
            regulatory_list = []
            for item in agg_dict.get("regulatory", [])[:5]:
                news_type = item.get("type", "")
                status = "approved" if "approval" in news_type else "warning" if "warning" in news_type or "safety" in news_type else "pending"
                regulatory_list.append({
                    "status": status,
                    "title": item.get("title", "")[:100],
                    "description": item.get("summary", "")[:200]
                })

            # Convert clinical_trials dict to list format for newsletter generator
            clinical_list = []
            ct_dict = agg_dict.get("clinical_trials", {})
            # Phase 3 results first
            for item in ct_dict.get("phase3_results", [])[:3]:
                clinical_list.append({
                    "type": "phase3_completed",
                    "title": item.get("title", "")[:100],
                    "description": item.get("summary", "")[:200],
                    "patients": item.get("metadata", {}).get("enrollment"),
                    "disease": ", ".join(item.get("metadata", {}).get("conditions", [])[:2])
                })
            # New trials
            for item in ct_dict.get("new_trials", [])[:2]:
                clinical_list.append({
                    "type": "new_trial",
                    "title": item.get("title", "")[:100],
                    "description": item.get("summary", "")[:200],
                    "patients": item.get("metadata", {}).get("enrollment"),
                    "disease": ", ".join(item.get("metadata", {}).get("conditions", [])[:2])
                })

            # Convert research dict to list format for newsletter generator
            research_list = []
            res_dict = agg_dict.get("research", {})
            # Preprints
            for item in res_dict.get("preprints", [])[:3]:
                research_list.append({
                    "source": item.get("source", "bioRxiv").lower(),
                    "journal": item.get("source", "bioRxiv"),
                    "title": item.get("title", "")[:100],
                    "summary": item.get("summary", "")[:200]
                })
            # High impact papers
            for item in res_dict.get("high_impact", [])[:3]:
                journal_name = item.get("metadata", {}).get("journal", "PubMed")
                research_list.append({
                    "source": journal_name.lower().split()[0] if journal_name else "pubmed",
                    "journal": journal_name,
                    "title": item.get("title", "")[:100],
                    "summary": item.get("summary", "")[:200]
                })

            # ========================================
            # DATA VALIDATION (재발 방지)
            # ========================================
            # 문제: newsletter_generator는 list 형식을 기대하지만,
            # aggregator는 dict 형식을 반환함 → 데이터 누락 발생
            # 해결: 명시적으로 list 형식으로 변환 후 검증
            # ========================================

            # Validate converted data (경고 출력)
            if not regulatory_list:
                print("⚠️ WARNING: regulatory_list is empty! Check FDA fetcher.")
            if not clinical_list:
                print("⚠️ WARNING: clinical_list is empty! Check ClinicalTrials fetcher.")
            if not research_list:
                print("⚠️ WARNING: research_list is empty! Check bioRxiv/PubMed fetcher.")

            print(f"[Data Validation] regulatory: {len(regulatory_list)}, clinical: {len(clinical_list)}, research: {len(research_list)}")

            # Build newsletter_data with all multi-source content (converted to lists)
            newsletter_data = {
                "total_papers": briefing_data.total_papers + briefing_data.total_preprints,
                "headline": agg_dict.get("headline") or {
                    "title": f"오늘의 바이오 트렌드: {trends[0].keyword if trends else '연구 동향'}",
                    "summary": editor_comment[:200] if editor_comment else "",
                    "why_important": ""
                },
                "regulatory": regulatory_list,  # FDA news as list (MUST be list, not dict)
                "clinical_trials": clinical_list,  # Clinical trials as list (MUST be list, not dict)
                "research": research_list,  # Research as list (MUST be list, not dict)
                "hot_topics": [
                    {
                        "keyword": t.keyword,
                        "count": t.count,
                        "trend_indicator": getattr(t, 'trend_indicator', '📊'),
                        "change_label": getattr(t, 'change_label', ''),
                        "why_hot": getattr(t, 'why_hot', ''),
                        "articles": [a.to_dict() if hasattr(a, 'to_dict') else a for a in articles_by_trend.get(t.keyword, [])]
                    }
                    for t in trends
                ],
                "editor": {
                    "quote": editor_comment[:150] if editor_comment else "",
                    "note": editor_comment[150:] if editor_comment and len(editor_comment) > 150 else ""
                },
                "stats": agg_dict.get("stats", {})
            }

            # Generate and save HTML
            filepath = self.generator.generate_and_save(newsletter_data, issue_number)
            print(f"Newsletter saved to: {filepath}")

            # Save issue number
            issue_number_file.write_text(str(issue_number))

            # Convert NewsArticle objects to dicts for JSON serialization
            articles_dict = {}
            for keyword, articles in articles_by_trend.items():
                articles_dict[keyword] = [
                    a.to_dict() if hasattr(a, 'to_dict') else a
                    for a in articles
                ]

            # Save JSON for API (includes all multi-source data)
            json_data = {
                "date": datetime.now().strftime("%Y%m%d"),
                "issue_number": issue_number,
                "total_papers_analyzed": briefing_data.total_papers + briefing_data.total_preprints,
                # Multi-source content
                "headline": agg_dict.get("headline"),
                "regulatory": agg_dict.get("regulatory", []),
                "clinical_trials": agg_dict.get("clinical_trials", {}),
                "research": agg_dict.get("research", {}),
                "hot_topics": [t.to_dict() for t in trends],
                "stats": agg_dict.get("stats", {}),
                # Trend details
                "trends": [
                    {
                        "keyword": t.keyword,
                        "count": t.count,
                        "change_label": getattr(t, 'change_label', ''),
                        "trend_indicator": getattr(t, 'trend_indicator', '📊'),
                        "category": getattr(t, 'category', ''),
                        "why_hot": getattr(t, 'why_hot', ''),
                        "is_predefined": getattr(t, 'is_predefined', True),
                        "is_emerging": getattr(t, 'is_emerging', False)
                    }
                    for t in trends
                ],
                "articles": articles_dict,
                "articles_by_trend": articles_dict,
                "editor_comment": editor_comment,
                "quick_news": [],
                "emerging_trends": [
                    {
                        "keyword": t.keyword,
                        "count": t.count,
                        "change_label": getattr(t, 'change_label', ''),
                        "trend_indicator": "🆕",
                        "category": "",
                        "why_hot": "",
                        "is_predefined": False,
                        "is_emerging": True
                    }
                    for t in trends if getattr(t, 'is_emerging', False)
                ]
            }
            json_path = output_dir / f"briefing_{datetime.now().strftime('%Y%m%d')}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"JSON saved to: {json_path}")

            # Read generated HTML
            html_path = Path(filepath)
            html = html_path.read_text(encoding="utf-8") if html_path.exists() else ""

            self._latest_html = html
            self._latest_data = json_data

            print(f"[{datetime.now()}] Briefing generation completed!")
            return html

        except Exception as e:
            print(f"Error generating briefing: {e}")
            import traceback
            traceback.print_exc()
            return None

    def send_email(
        self,
        html: str,
        recipient: str,
        subject: Optional[str] = None,
    ) -> bool:
        """
        Send newsletter email to a recipient.

        Args:
            html: HTML content
            recipient: Email address
            subject: Optional custom subject

        Returns:
            True if successful
        """
        if not self.smtp_user or not self.smtp_password:
            print("Email not configured. Set SMTP_USER and SMTP_PASSWORD.")
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")

            if subject is None:
                issue = self._latest_data.get("issue_number", 1) if self._latest_data else 1
                subject = f"BIO 데일리 브리핑 #{issue}"

            msg["Subject"] = subject
            msg["From"] = f"{self.sender_name} <{self.sender_email}>"
            msg["To"] = recipient

            # Attach HTML
            msg.attach(MIMEText(html, "html", "utf-8"))

            # Send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.sender_email, [recipient], msg.as_string())

            print(f"Email sent to: {recipient}")
            return True

        except Exception as e:
            print(f"Error sending email to {recipient}: {e}")
            return False

    def send_to_all_subscribers(self, html: str) -> int:
        """
        Send newsletter to all subscribers.

        Returns:
            Number of successful sends
        """
        subscribers = self.load_subscribers()

        if not subscribers:
            print("No subscribers found!")
            return 0

        success_count = 0
        for email in subscribers:
            if self.send_email(html, email):
                success_count += 1

        print(f"Sent to {success_count}/{len(subscribers)} subscribers")

        # Slack notification
        self._notify_slack(success_count, len(subscribers))

        return success_count

    def _notify_slack(self, success: int, total: int):
        """Send Slack notification."""
        if not self.slack_webhook:
            return

        try:
            import httpx

            message = {
                "text": f"BIO 데일리 브리핑 발송 완료: {success}/{total} 성공",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"📬 *BIO 데일리 브리핑 #{self._latest_data.get('issue_number', '?')}*\n"
                                    f"발송: {success}/{total} 성공\n"
                                    f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        }
                    }
                ]
            }

            httpx.post(self.slack_webhook, json=message)

        except Exception as e:
            print(f"Slack notification failed: {e}")

    async def run_once(self):
        """Run generation and send once."""
        html = await self.generate_briefing()
        if html:
            self.send_to_all_subscribers(html)

    def start_scheduler(self):
        """Start the scheduler daemon."""
        print(f"\nStarting scheduler...")
        print(f"  Generate time: {self.generate_time}")
        print(f"  Send time: {self.send_time}")
        print(f"  Subscribers: {len(self.load_subscribers())}")

        def run_generate():
            asyncio.run(self.generate_briefing())

        def run_send():
            if self._latest_html:
                self.send_to_all_subscribers(self._latest_html)
            else:
                print("No newsletter to send. Generate first.")

        # Schedule jobs
        schedule.every().day.at(self.generate_time).do(run_generate)
        schedule.every().day.at(self.send_time).do(run_send)

        print(f"\nScheduler started. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nScheduler stopped.")

    def check_config(self):
        """Check configuration."""
        print("\n=== BIO Daily Briefing Configuration ===\n")

        # API Keys
        print("[API Keys]")
        anthropic = os.getenv("ANTHROPIC_API_KEY")
        google = os.getenv("GOOGLE_API_KEY")
        print(f"  ANTHROPIC_API_KEY: {'✓ Set' if anthropic else '✗ Not set'}")
        print(f"  GOOGLE_API_KEY: {'✓ Set' if google else '✗ Not set'}")

        if not anthropic and not google:
            print("  ⚠️  Warning: At least one API key is required!")

        # Email
        print("\n[Email Settings]")
        print(f"  SMTP_HOST: {self.smtp_host}")
        print(f"  SMTP_PORT: {self.smtp_port}")
        print(f"  SMTP_USER: {'✓ Set' if self.smtp_user else '✗ Not set'}")
        print(f"  SMTP_PASSWORD: {'✓ Set' if self.smtp_password else '✗ Not set'}")

        # Schedule
        print("\n[Schedule]")
        print(f"  Generate time: {self.generate_time}")
        print(f"  Send time: {self.send_time}")

        # Subscribers
        subscribers = self.load_subscribers()
        print(f"\n[Subscribers]")
        print(f"  Total: {len(subscribers)}")
        for email in subscribers[:5]:
            print(f"    - {email}")
        if len(subscribers) > 5:
            print(f"    ... and {len(subscribers) - 5} more")

        print("")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BIO Daily Briefing - Newsletter Scheduler"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check configuration"
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run generation and send immediately"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate newsletter without sending"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon with scheduler"
    )
    parser.add_argument(
        "--add-subscriber",
        metavar="EMAIL",
        help="Add a subscriber"
    )
    parser.add_argument(
        "--remove-subscriber",
        metavar="EMAIL",
        help="Remove a subscriber"
    )
    parser.add_argument(
        "--list-subscribers",
        action="store_true",
        help="List all subscribers"
    )
    parser.add_argument(
        "--test-email",
        metavar="EMAIL",
        help="Send test email to address"
    )

    args = parser.parse_args()

    scheduler = BriefingScheduler()

    if args.check_config:
        scheduler.check_config()

    elif args.run_now:
        asyncio.run(scheduler.run_once())

    elif args.generate_only:
        asyncio.run(scheduler.generate_briefing())

    elif args.daemon:
        scheduler.start_scheduler()

    elif args.add_subscriber:
        scheduler.add_subscriber(args.add_subscriber)

    elif args.remove_subscriber:
        scheduler.remove_subscriber(args.remove_subscriber)

    elif args.list_subscribers:
        subscribers = scheduler.load_subscribers()
        print(f"Subscribers ({len(subscribers)}):")
        for email in subscribers:
            print(f"  - {email}")

    elif args.test_email:
        print("Generating test newsletter...")
        html = asyncio.run(scheduler.generate_briefing())
        if html:
            scheduler.send_email(html, args.test_email)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
