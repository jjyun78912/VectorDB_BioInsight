"""
Newsletter Generator v2 - Multi-source Newsletter

New layout with:
- FDA approvals/warnings at top
- Clinical trial updates
- Research (High-impact journals + Preprints)
- Hot topic monitoring
- Editor's AI analysis
"""

import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

from jinja2 import Environment, BaseLoader

from .trend_analyzer import Trend
from .aggregator import DailyBriefingData, AggregatedNews


# ========================
# HTML Template v2
# ========================
HTML_TEMPLATE_V2 = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BIO 데일리 브리핑 #{{ issue_number }}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #1a1a2e; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Malgun Gothic', sans-serif; color: #e0e0e0;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #1a1a2e;">
        <tr>
            <td align="center" style="padding: 20px 10px;">
                <!-- Main Container -->
                <table role="presentation" width="650" cellspacing="0" cellpadding="0" style="background-color: #16213e; border-radius: 12px; overflow: hidden;">

                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #0f3460 0%, #16213e 100%); padding: 30px; border-bottom: 2px solid #4ecca3;">
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td>
                                        <p style="margin: 0; color: #4ecca3; font-size: 11px; font-weight: 600; letter-spacing: 3px; text-transform: uppercase;">
                                            BIO DAILY BRIEFING
                                        </p>
                                        <h1 style="margin: 8px 0 5px 0; color: #ffffff; font-size: 26px; font-weight: 700;">
                                            📰 {{ formatted_date }}
                                        </h1>
                                        <p style="margin: 0; color: #a0a0a0; font-size: 13px;">
                                            Issue #{{ issue_number }} | 멀티소스 분석 뉴스레터
                                        </p>
                                    </td>
                                    <td width="80" valign="top" align="right">
                                        <div style="background-color: #4ecca3; color: #16213e; border-radius: 8px; padding: 8px 12px; font-size: 11px; font-weight: 600;">
                                            v2.0
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Headline (if exists) -->
                    {% if headline %}
                    <tr>
                        <td style="padding: 25px 30px; background-color: #1e3a5f;">
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td>
                                        <p style="margin: 0 0 10px 0; color: #ff6b6b; font-size: 12px; font-weight: 700; letter-spacing: 2px;">
                                            🚨 오늘의 헤드라인
                                        </p>
                                        <h2 style="margin: 0 0 10px 0; color: #ffffff; font-size: 18px; font-weight: 600; line-height: 1.4;">
                                            {{ headline.title }}
                                        </h2>
                                        <p style="margin: 0; color: #a0a0a0; font-size: 13px;">
                                            → {{ headline.summary[:150] }}{% if headline.summary|length > 150 %}...{% endif %}
                                        </p>
                                        <p style="margin: 10px 0 0 0; color: #4ecca3; font-size: 12px;">
                                            📍 {{ headline.source }} | <a href="{{ headline.link }}" style="color: #4ecca3; text-decoration: none;">자세히 보기 →</a>
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    {% endif %}

                    <!-- Stats Bar -->
                    <tr>
                        <td style="padding: 20px 30px; background-color: #0f3460;">
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td width="25%" align="center">
                                        <p style="margin: 0; color: #4ecca3; font-size: 20px; font-weight: 700;">{{ stats.fda }}</p>
                                        <p style="margin: 0; color: #6c757d; font-size: 11px;">FDA 뉴스</p>
                                    </td>
                                    <td width="25%" align="center">
                                        <p style="margin: 0; color: #4ecca3; font-size: 20px; font-weight: 700;">{{ stats.trials }}</p>
                                        <p style="margin: 0; color: #6c757d; font-size: 11px;">임상시험</p>
                                    </td>
                                    <td width="25%" align="center">
                                        <p style="margin: 0; color: #4ecca3; font-size: 20px; font-weight: 700;">{{ stats.preprints }}</p>
                                        <p style="margin: 0; color: #6c757d; font-size: 11px;">프리프린트</p>
                                    </td>
                                    <td width="25%" align="center">
                                        <p style="margin: 0; color: #4ecca3; font-size: 20px; font-weight: 700;">{{ stats.papers }}</p>
                                        <p style="margin: 0; color: #6c757d; font-size: 11px;">논문 분석</p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- FDA / Regulatory Section -->
                    {% if regulatory and regulatory|length > 0 %}
                    <tr>
                        <td style="padding: 25px 30px 15px 30px;">
                            <h2 style="margin: 0 0 5px 0; color: #ffffff; font-size: 16px; font-weight: 600;">
                                ⚡ 규제/승인 소식 <span style="color: #ff6b6b; font-size: 12px;">(FDA)</span>
                            </h2>
                            <p style="margin: 0 0 15px 0; color: #6c757d; font-size: 12px;">
                                최근 72시간 FDA 발표
                            </p>
                        </td>
                    </tr>

                    {% for news in regulatory[:5] %}
                    <tr>
                        <td style="padding: 0 30px 15px 30px;">
                            <table width="100%" cellspacing="0" cellpadding="0" style="background-color: #1e3a5f; border-radius: 8px; border-left: 3px solid {% if 'approval' in news.type %}#4ecca3{% elif 'warning' in news.type or 'safety' in news.type %}#ff6b6b{% else %}#ffc107{% endif %};">
                                <tr>
                                    <td style="padding: 15px;">
                                        <p style="margin: 0 0 5px 0; color: {% if 'approval' in news.type %}#4ecca3{% elif 'warning' in news.type %}#ff6b6b{% else %}#ffc107{% endif %}; font-size: 11px; font-weight: 600;">
                                            {% if 'approval' in news.type %}✅ 승인{% elif 'rejection' in news.type %}❌ 거절{% elif 'warning' in news.type or 'safety' in news.type %}⚠️ 안전성 경고{% elif 'designation' in news.type %}🏷️ 지정{% else %}📋 발표{% endif %}
                                        </p>
                                        <p style="margin: 0 0 8px 0; color: #ffffff; font-size: 14px; font-weight: 500; line-height: 1.4;">
                                            {{ news.title }}
                                        </p>
                                        <p style="margin: 0; color: #a0a0a0; font-size: 12px;">
                                            {{ news.date }} | <a href="{{ news.link }}" style="color: #4ecca3; text-decoration: none;">상세 →</a>
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    {% endfor %}
                    {% endif %}

                    <!-- Clinical Trials Section -->
                    {% if clinical_trials %}
                    <tr>
                        <td style="padding: 25px 30px 15px 30px;">
                            <h2 style="margin: 0 0 5px 0; color: #ffffff; font-size: 16px; font-weight: 600;">
                                📊 임상시험 업데이트
                            </h2>
                            <p style="margin: 0 0 15px 0; color: #6c757d; font-size: 12px;">
                                ClinicalTrials.gov Phase 3 동향
                            </p>
                        </td>
                    </tr>

                    <!-- Phase 3 Results -->
                    {% if clinical_trials.phase3_results %}
                    <tr>
                        <td style="padding: 0 30px 10px 30px;">
                            <p style="margin: 0 0 10px 0; color: #4ecca3; font-size: 12px; font-weight: 600;">
                                [Phase 3 결과 발표]
                            </p>
                        </td>
                    </tr>
                    {% for trial in clinical_trials.phase3_results[:3] %}
                    <tr>
                        <td style="padding: 0 30px 10px 30px;">
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td style="padding: 12px 15px; background-color: #0f3460; border-radius: 6px;">
                                        <p style="margin: 0 0 5px 0; color: #ffffff; font-size: 13px; font-weight: 500;">
                                            • {{ trial.title[:80] }}{% if trial.title|length > 80 %}...{% endif %}
                                        </p>
                                        <p style="margin: 0; color: #a0a0a0; font-size: 11px;">
                                            {{ trial.metadata.sponsor }} | {{ trial.metadata.conditions[:2]|join(', ') }}
                                            {% if trial.metadata.has_results %}✅ 결과 발표{% endif %}
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    {% endfor %}
                    {% endif %}

                    <!-- New Trials -->
                    {% if clinical_trials.new_trials %}
                    <tr>
                        <td style="padding: 15px 30px 10px 30px;">
                            <p style="margin: 0 0 10px 0; color: #ffc107; font-size: 12px; font-weight: 600;">
                                [신규 임상 시작]
                            </p>
                        </td>
                    </tr>
                    {% for trial in clinical_trials.new_trials[:3] %}
                    <tr>
                        <td style="padding: 0 30px 8px 30px;">
                            <p style="margin: 0; color: #c0c0c0; font-size: 13px;">
                                • {{ trial.title[:70] }}{% if trial.title|length > 70 %}...{% endif %}
                                <span style="color: #6c757d;"> ({{ trial.metadata.sponsor }})</span>
                            </p>
                        </td>
                    </tr>
                    {% endfor %}
                    {% endif %}

                    <!-- Terminated -->
                    {% if clinical_trials.terminated %}
                    <tr>
                        <td style="padding: 15px 30px 10px 30px;">
                            <p style="margin: 0 0 10px 0; color: #ff6b6b; font-size: 12px; font-weight: 600;">
                                [임상 중단/실패]
                            </p>
                        </td>
                    </tr>
                    {% for trial in clinical_trials.terminated[:2] %}
                    <tr>
                        <td style="padding: 0 30px 8px 30px;">
                            <p style="margin: 0; color: #a0a0a0; font-size: 13px;">
                                ❌ {{ trial.title[:70] }}{% if trial.title|length > 70 %}...{% endif %}
                            </p>
                        </td>
                    </tr>
                    {% endfor %}
                    {% endif %}
                    {% endif %}

                    <!-- Research Section -->
                    {% if research %}
                    <tr>
                        <td style="padding: 25px 30px 15px 30px;">
                            <h2 style="margin: 0 0 5px 0; color: #ffffff; font-size: 16px; font-weight: 600;">
                                🔬 주목할 연구
                            </h2>
                            <p style="margin: 0 0 15px 0; color: #6c757d; font-size: 12px;">
                                고임팩트 저널 + 프리프린트
                            </p>
                        </td>
                    </tr>

                    <!-- High Impact Journals -->
                    {% if research.high_impact %}
                    <tr>
                        <td style="padding: 0 30px 15px 30px;">
                            <p style="margin: 0 0 10px 0; color: #4ecca3; font-size: 12px; font-weight: 600;">
                                [고임팩트 저널]
                            </p>
                        </td>
                    </tr>
                    {% for paper in research.high_impact[:4] %}
                    <tr>
                        <td style="padding: 0 30px 12px 30px;">
                            <table width="100%" cellspacing="0" cellpadding="0" style="background-color: #0f3460; border-radius: 8px;">
                                <tr>
                                    <td style="padding: 15px;">
                                        <p style="margin: 0 0 3px 0; color: #4ecca3; font-size: 11px; font-weight: 600;">
                                            📖 {{ paper.metadata.journal }}
                                        </p>
                                        <p style="margin: 0 0 8px 0; color: #ffffff; font-size: 13px; font-weight: 500; line-height: 1.4;">
                                            {{ paper.title[:100] }}{% if paper.title|length > 100 %}...{% endif %}
                                        </p>
                                        <p style="margin: 0; color: #a0a0a0; font-size: 12px; line-height: 1.5;">
                                            → {{ paper.summary[:120] }}{% if paper.summary|length > 120 %}...{% endif %}
                                        </p>
                                        <p style="margin: 8px 0 0 0; color: #6c757d; font-size: 11px;">
                                            <a href="{{ paper.link }}" style="color: #4ecca3; text-decoration: none;">PMID {{ paper.metadata.pmid }}</a>
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    {% endfor %}
                    {% endif %}

                    <!-- Preprints -->
                    {% if research.preprints %}
                    <tr>
                        <td style="padding: 15px 30px 10px 30px;">
                            <p style="margin: 0 0 10px 0; color: #ffc107; font-size: 12px; font-weight: 600;">
                                [속보 프리프린트] <span style="color: #6c757d; font-weight: 400;">⚠️ 피어리뷰 전</span>
                            </p>
                        </td>
                    </tr>
                    {% for preprint in research.preprints[:3] %}
                    <tr>
                        <td style="padding: 0 30px 10px 30px;">
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td style="padding: 12px 15px; background-color: #1e3a5f; border-radius: 6px; border-left: 3px solid #ffc107;">
                                        <p style="margin: 0 0 5px 0; color: #ffc107; font-size: 11px;">
                                            🔬 {{ preprint.source }} | {{ preprint.metadata.category }}
                                        </p>
                                        <p style="margin: 0 0 5px 0; color: #ffffff; font-size: 13px; font-weight: 500;">
                                            {{ preprint.title[:80] }}{% if preprint.title|length > 80 %}...{% endif %}
                                        </p>
                                        <p style="margin: 0; color: #a0a0a0; font-size: 11px;">
                                            {{ preprint.metadata.authors[:50] }}{% if preprint.metadata.authors|length > 50 %}...{% endif %}
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    {% endfor %}
                    {% endif %}
                    {% endif %}

                    <!-- Hot Topics Monitoring -->
                    {% if hot_topics %}
                    <tr>
                        <td style="padding: 25px 30px 15px 30px;">
                            <h2 style="margin: 0 0 5px 0; color: #ffffff; font-size: 16px; font-weight: 600;">
                                📈 핫토픽 모니터링
                            </h2>
                            <p style="margin: 0 0 15px 0; color: #6c757d; font-size: 12px;">
                                PubMed 논문 {{ stats.papers }}건 키워드 분석
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0 30px 15px 30px;">
                            <table width="100%" cellspacing="0" cellpadding="0" style="background-color: #0f3460; border-radius: 8px;">
                                <tr>
                                    <td style="padding: 15px;">
                                        <table width="100%" cellspacing="0" cellpadding="0">
                                            <!-- Header Row -->
                                            <tr>
                                                <td style="padding: 5px 0; border-bottom: 1px solid #1e3a5f; color: #6c757d; font-size: 11px;">키워드</td>
                                                <td width="60" align="center" style="padding: 5px 0; border-bottom: 1px solid #1e3a5f; color: #6c757d; font-size: 11px;">건수</td>
                                                <td width="80" align="center" style="padding: 5px 0; border-bottom: 1px solid #1e3a5f; color: #6c757d; font-size: 11px;">변동</td>
                                            </tr>
                                            {% for trend in hot_topics[:7] %}
                                            <tr>
                                                <td style="padding: 8px 0; border-bottom: 1px solid #1e3a5f; color: #ffffff; font-size: 13px;">
                                                    {% if trend.is_emerging %}🆕{% endif %} {{ trend.keyword }}
                                                </td>
                                                <td width="60" align="center" style="padding: 8px 0; border-bottom: 1px solid #1e3a5f; color: #a0a0a0; font-size: 13px;">
                                                    {{ trend.count }}
                                                </td>
                                                <td width="80" align="center" style="padding: 8px 0; border-bottom: 1px solid #1e3a5f; font-size: 12px;">
                                                    {% if trend.is_first_tracking %}
                                                    <span style="color: #4ecca3;">📊 신규</span>
                                                    {% elif trend.week_change >= 50 %}
                                                    <span style="color: #ff6b6b;">🔥 +{{ trend.week_change|round|int }}%</span>
                                                    {% elif trend.week_change >= 10 %}
                                                    <span style="color: #4ecca3;">⬆️ +{{ trend.week_change|round|int }}%</span>
                                                    {% elif trend.week_change <= -10 %}
                                                    <span style="color: #ff6b6b;">⬇️ {{ trend.week_change|round|int }}%</span>
                                                    {% else %}
                                                    <span style="color: #6c757d;">➡️ 유지</span>
                                                    {% endif %}
                                                </td>
                                            </tr>
                                            {% endfor %}
                                        </table>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    {% endif %}

                    <!-- Editor Comment -->
                    <tr>
                        <td style="padding: 25px 30px;">
                            <table width="100%" cellspacing="0" cellpadding="0" style="background: linear-gradient(135deg, #1e3a5f 0%, #0f3460 100%); border-radius: 12px; border: 1px solid #4ecca3;">
                                <tr>
                                    <td style="padding: 20px;">
                                        <h2 style="margin: 0 0 15px 0; color: #4ecca3; font-size: 14px; font-weight: 600;">
                                            💬 AI 에디터 코멘트
                                        </h2>
                                        <div style="margin: 0; color: #c0c0c0; font-size: 14px; line-height: 1.8;">
                                            {{ editor_comment | replace('\\n\\n', '</p><p style="margin: 12px 0; color: #c0c0c0; font-size: 14px; line-height: 1.8;">') | replace('\\n', '<br>') | replace('**', '<strong style="color: #ffffff;">') | safe }}
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 25px 30px; background-color: #0f3460; border-top: 1px solid #1e3a5f;">
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td align="center">
                                        <p style="margin: 0 0 10px 0; color: #4ecca3; font-size: 11px; letter-spacing: 2px;">
                                            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                        </p>
                                        <p style="margin: 0 0 5px 0; color: #ffffff; font-size: 13px; font-weight: 600;">
                                            BIO 데일리 브리핑 v2.0
                                        </p>
                                        <p style="margin: 0; color: #6c757d; font-size: 11px;">
                                            멀티소스 AI 기반 바이오 연구 뉴스레터<br>
                                            FDA | ClinicalTrials.gov | bioRxiv | PubMed
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                </table>
                <!-- End Main Container -->
            </td>
        </tr>
    </table>
</body>
</html>"""


class NewsletterGeneratorV2:
    """Generate HTML newsletters from multi-source aggregated data."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).parent.parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.issue_file = self.output_dir / "issue_number.txt"

    def get_issue_number(self) -> int:
        """Get and increment issue number."""
        if self.issue_file.exists():
            with open(self.issue_file, "r") as f:
                number = int(f.read().strip())
        else:
            number = 0
        number += 1
        with open(self.issue_file, "w") as f:
            f.write(str(number))
        return number

    def generate_html(self, briefing: DailyBriefingData) -> str:
        """
        Generate HTML newsletter from aggregated briefing data.

        Args:
            briefing: DailyBriefingData object from aggregator

        Returns:
            HTML string
        """
        # Format date
        weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        today = datetime.now()
        weekday = weekdays[today.weekday()]
        formatted_date = f"{today.strftime('%Y년 %m월 %d일')} {weekday}"

        # Setup Jinja
        env = Environment(loader=BaseLoader())
        template = env.from_string(HTML_TEMPLATE_V2)

        # Prepare data for template
        headline_data = None
        if briefing.headline:
            headline_data = briefing.headline.to_dict()

        regulatory_data = [n.to_dict() for n in briefing.regulatory_news]

        clinical_trials_data = {}
        for key, trials in briefing.clinical_trials.items():
            clinical_trials_data[key] = [t.to_dict() for t in trials]

        research_data = {}
        for key, items in briefing.research.items():
            research_data[key] = [i.to_dict() for i in items]

        hot_topics_data = [t.to_dict() for t in briefing.hot_topics]

        stats_data = {
            "fda": briefing.total_fda,
            "trials": briefing.total_trials,
            "preprints": briefing.total_preprints,
            "papers": briefing.total_papers,
        }

        # Render
        html = template.render(
            issue_number=briefing.issue_number,
            formatted_date=formatted_date,
            headline=headline_data,
            regulatory=regulatory_data,
            clinical_trials=clinical_trials_data,
            research=research_data,
            hot_topics=hot_topics_data,
            stats=stats_data,
            editor_comment=briefing.editor_comment,
        )

        return html

    def save_html(self, html: str, filename: Optional[str] = None) -> Path:
        """Save HTML to file."""
        if filename is None:
            filename = f"briefing_{datetime.now().strftime('%Y%m%d')}.html"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        return filepath

    def save_json(self, briefing: DailyBriefingData, filename: Optional[str] = None) -> Path:
        """Save briefing data as JSON."""
        if filename is None:
            filename = f"briefing_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.output_dir / filename

        data = briefing.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    def generate_and_save(self, briefing: DailyBriefingData) -> tuple:
        """
        Generate and save both HTML and JSON.

        Returns:
            Tuple of (html_path, json_path)
        """
        html = self.generate_html(briefing)
        html_path = self.save_html(html)
        json_path = self.save_json(briefing)
        return html_path, json_path


async def main():
    """Test v2 newsletter generation with real data."""
    from .aggregator import NewsAggregator

    print("=" * 60)
    print("Newsletter Generator v2 Test")
    print("=" * 60)

    # Aggregate data
    aggregator = NewsAggregator()
    briefing = await aggregator.aggregate_daily(
        fda_hours=168,  # 1 week for testing
        trials_days=60,
        preprint_days=7,
        pubmed_days=3,
        issue_number=1,
    )

    # Add editor comment placeholder
    briefing.editor_comment = """오늘의 주요 포인트를 정리해드립니다.

멀티소스 분석을 통해 FDA, 임상시험, 프리프린트, 학술 논문을 종합했습니다.

**주목 키워드**: 상단의 핫토픽 모니터링을 참고하세요.

내일도 바이오 인사이트와 함께하세요! 🏥"""

    # Generate newsletter
    generator = NewsletterGeneratorV2()
    html_path, json_path = generator.generate_and_save(briefing)

    print(f"\n✅ Newsletter generated:")
    print(f"   HTML: {html_path}")
    print(f"   JSON: {json_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
