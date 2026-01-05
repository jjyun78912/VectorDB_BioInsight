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
from .newsletter_generator import NewsletterGenerator, NewsletterData


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
        Generate the daily briefing.

        Returns:
            HTML string of the newsletter
        """
        print(f"\n[{datetime.now()}] Starting briefing generation...")

        try:
            # 1. Fetch papers
            print(f"Fetching {self.papers_per_day} papers from last {self.lookback_days} days...")
            papers = await self.fetcher.fetch_recent_papers(
                max_results=self.papers_per_day,
                days=self.lookback_days,
            )
            print(f"Fetched {len(papers)} papers")

            if not papers:
                print("No papers found!")
                return None

            # 2. Analyze trends
            print(f"Analyzing trends (top {self.top_trends})...")
            trends = self.analyzer.get_hot_topics(papers, top_n=self.top_trends)
            print(f"Found {len(trends)} trending keywords")

            for t in trends:
                print(f"  {t.trend_indicator} {t.keyword}: {t.count} papers")

            # 3. Generate summaries
            print("Generating AI summaries...")
            articles_by_trend = self.summarizer.summarize_papers_by_trend(trends, max_per_trend=2)

            article_count = sum(len(a) for a in articles_by_trend.values())
            print(f"Generated {article_count} articles")

            # 4. Generate editor comment
            print("Generating editor comment...")
            editor_comment = self.summarizer.generate_editor_comment(trends)

            # 5. Generate newsletter
            print("Generating newsletter HTML...")
            issue_number = self.generator.get_issue_number()

            html = self.generator.generate_html(
                trends=trends,
                articles_by_trend=articles_by_trend,
                editor_comment=editor_comment,
                total_papers_analyzed=len(papers),
                issue_number=issue_number,
            )

            # Save HTML
            filepath = self.generator.save_html(html)
            print(f"Newsletter saved to: {filepath}")

            # Save JSON
            json_data = self.generator.generate_json(
                trends=trends,
                articles_by_trend=articles_by_trend,
                editor_comment=editor_comment,
                total_papers_analyzed=len(papers),
                issue_number=issue_number,
            )
            json_path = filepath.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

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
