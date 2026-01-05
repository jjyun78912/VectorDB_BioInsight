"""
Newsletter Generator - HTML email template generation

New format with hybrid hot topics:
- Predefined hot topics with why_hot explanation
- Emerging trends detection
- Week-over-week change tracking
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

from jinja2 import Environment, BaseLoader

from .trend_analyzer import Trend
from .ai_summarizer import NewsArticle


@dataclass
class NewsletterData:
    """Data for newsletter generation."""
    issue_number: int
    date: str
    trends: List[Trend]
    articles_by_trend: Dict[str, List[NewsArticle]]
    editor_comment: str
    quick_news: List[str] = None
    total_papers_analyzed: int = 0


# New HTML Template - Hybrid Hot Topic Layout
HTML_TEMPLATE = """<!DOCTYPE html>
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
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #16213e; border-radius: 8px;">

                    <!-- Header -->
                    <tr>
                        <td style="padding: 30px 30px 20px 30px; border-bottom: 1px solid #0f3460;">
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td>
                                        <p style="margin: 0; color: #4ecca3; font-size: 12px; font-weight: 600; letter-spacing: 2px;">
                                            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                        </p>
                                        <h1 style="margin: 10px 0 5px 0; color: #ffffff; font-size: 22px; font-weight: 700;">
                                            📰 BIO 데일리 브리핑
                                        </h1>
                                        <p style="margin: 0; color: #a0a0a0; font-size: 13px;">
                                            {{ date }} | Issue #{{ issue_number }}
                                        </p>
                                        <p style="margin: 10px 0 0 0; color: #4ecca3; font-size: 12px; font-weight: 600; letter-spacing: 2px;">
                                            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Hot Keywords Section -->
                    <tr>
                        <td style="padding: 25px 30px;">
                            <h2 style="margin: 0 0 8px 0; color: #ffffff; font-size: 16px; font-weight: 600;">
                                📊 오늘의 핫 키워드
                            </h2>
                            <p style="margin: 0 0 20px 0; color: #a0a0a0; font-size: 13px;">
                                최근 48시간 PubMed 논문 {{ total_papers_analyzed }}건 분석 결과
                            </p>

                            <!-- Predefined Hot Topics -->
                            {% set predefined = trends | selectattr('is_predefined', 'equalto', true) | list %}
                            {% if predefined %}
                            <p style="margin: 0 0 10px 0; color: #4ecca3; font-size: 12px; font-weight: 600;">
                                [업계 주목 키워드]
                            </p>
                            <table width="100%" cellspacing="0" cellpadding="0">
                                {% for trend in predefined %}
                                <tr>
                                    <td style="padding: 10px 0; border-bottom: 1px solid #0f3460;">
                                        <table width="100%" cellspacing="0" cellpadding="0">
                                            <tr>
                                                <td width="35" style="color: #6c757d; font-size: 13px; font-weight: 600;">
                                                    {{ loop.index }}위
                                                </td>
                                                <td style="color: #ffffff; font-size: 14px; font-weight: 500;">
                                                    {{ trend.keyword | title }}
                                                </td>
                                                <td width="60" align="right" style="color: #a0a0a0; font-size: 13px;">
                                                    {{ trend.count }}건
                                                </td>
                                                <td width="100" align="right">
                                                    {% if trend.is_first_tracking %}
                                                    <span style="color: #4ecca3; font-size: 13px;">
                                                        📊 추적 시작
                                                    </span>
                                                    {% elif trend.week_change >= 50 %}
                                                    <span style="color: #ff6b6b; font-size: 13px; font-weight: 500;">
                                                        🔥 +{{ trend.week_change|round|int }}%
                                                    </span>
                                                    {% elif trend.week_change >= 10 %}
                                                    <span style="color: #4ecca3; font-size: 13px;">
                                                        ⬆️ +{{ trend.week_change|round|int }}%
                                                    </span>
                                                    {% elif trend.week_change <= -10 %}
                                                    <span style="color: #ff6b6b; font-size: 13px;">
                                                        ⬇️ {{ trend.week_change|round|int }}%
                                                    </span>
                                                    {% else %}
                                                    <span style="color: #6c757d; font-size: 13px;">
                                                        ➡️ 유지
                                                    </span>
                                                    {% endif %}
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                                {% endfor %}
                            </table>
                            {% endif %}

                            <!-- Emerging Trends -->
                            {% set emerging = trends | selectattr('is_emerging', 'equalto', true) | list %}
                            {% if emerging %}
                            <p style="margin: 20px 0 10px 0; color: #ff6b6b; font-size: 12px; font-weight: 600;">
                                [🆕 급상승 감지]
                            </p>
                            <table width="100%" cellspacing="0" cellpadding="0">
                                {% for trend in emerging %}
                                <tr>
                                    <td style="padding: 10px 0; border-bottom: 1px solid #0f3460;">
                                        <table width="100%" cellspacing="0" cellpadding="0">
                                            <tr>
                                                <td width="20" style="color: #ff6b6b; font-size: 13px;">
                                                    •
                                                </td>
                                                <td style="color: #ffffff; font-size: 14px; font-weight: 500;">
                                                    {{ trend.keyword | title }}
                                                </td>
                                                <td width="60" align="right" style="color: #a0a0a0; font-size: 13px;">
                                                    {{ trend.count }}건
                                                </td>
                                                <td width="100" align="right">
                                                    <span style="color: #ff6b6b; font-size: 13px; font-weight: 500;">
                                                        {{ trend.change_label }}
                                                    </span>
                                                </td>
                                            </tr>
                                        </table>
                                        <p style="margin: 5px 0 0 20px; color: #6c757d; font-size: 11px;">
                                            └ 고정 목록 외 키워드 중 상위 급상승 감지
                                        </p>
                                    </td>
                                </tr>
                                {% endfor %}
                            </table>
                            {% endif %}
                        </td>
                    </tr>

                    <!-- Divider -->
                    <tr>
                        <td style="padding: 0 30px;">
                            <p style="margin: 0; color: #4ecca3; font-size: 12px; text-align: center; letter-spacing: 2px;">
                                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            </p>
                        </td>
                    </tr>

                    <!-- Main Research Section -->
                    <tr>
                        <td style="padding: 25px 30px 10px 30px;">
                            <h2 style="margin: 0 0 5px 0; color: #ffffff; font-size: 16px; font-weight: 600;">
                                ✨ 오늘의 주요 연구
                            </h2>
                            <p style="margin: 0; color: #6c757d; font-size: 12px;">
                                [상위 핫토픽 기반 대표 논문]
                            </p>
                        </td>
                    </tr>

                    <!-- Articles by Trend -->
                    {% set article_num = namespace(value=1) %}
                    {% for trend in trends %}
                    {% set articles = articles_by_trend.get(trend.keyword, []) %}
                    {% if articles %}
                    <tr>
                        <td style="padding: 15px 30px;">
                            <!-- Trend Header -->
                            <table width="100%" cellspacing="0" cellpadding="0" style="border-top: 1px solid #0f3460;">
                                <tr>
                                    <td style="padding-top: 15px;">
                                        <p style="margin: 0; color: #4ecca3; font-size: 12px; font-weight: 600;">
                                            ───────────────────────────────
                                        </p>
                                        <p style="margin: 8px 0; color: #ffffff; font-size: 14px; font-weight: 600;">
                                            {% if trend.is_emerging %}
                                            🆕 | {{ trend.keyword | title }}
                                            {% else %}
                                            {{ "%02d"|format(article_num.value) }} | {{ trend.keyword | title }} ({{ trend.change_label }})
                                            {% endif %}
                                        </p>
                                        <p style="margin: 0 0 15px 0; color: #4ecca3; font-size: 12px; font-weight: 600;">
                                            ───────────────────────────────
                                        </p>
                                    </td>
                                </tr>
                            </table>

                            <!-- Why Hot -->
                            {% if trend.why_hot %}
                            <p style="margin: 0 0 15px 0; padding: 10px 15px; background-color: #0f3460; border-radius: 6px; color: #a0a0a0; font-size: 13px;">
                                {% if trend.is_emerging %}
                                <strong style="color: #ff6b6b;">왜 갑자기 급상승?</strong><br>
                                {% else %}
                                <strong style="color: #4ecca3;">이 키워드가 왜 요즘 핫한가요?</strong><br>
                                {% endif %}
                                → {{ trend.why_hot }}
                            </p>
                            {% endif %}

                            {% for article in articles[:1] %}
                            <!-- Article Content -->
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td>
                                        <!-- Context Question -->
                                        <p style="margin: 0 0 12px 0; color: #a0a0a0; font-size: 13px; font-style: italic;">
                                            💬 {{ article.hook }}
                                        </p>

                                        <!-- Title -->
                                        <p style="margin: 0 0 15px 0; color: #ffffff; font-size: 15px; font-weight: 600; line-height: 1.5;">
                                            "{{ article.title }}"
                                        </p>

                                        <!-- Content -->
                                        <p style="margin: 0 0 15px 0; color: #c0c0c0; font-size: 14px; line-height: 1.7;">
                                            {{ article.content }}
                                        </p>

                                        <!-- Insight -->
                                        {% if article.insight %}
                                        <p style="margin: 0 0 12px 0; color: #4ecca3; font-size: 13px;">
                                            💡 {{ article.insight }}
                                        </p>
                                        {% endif %}

                                        <!-- Source -->
                                        <p style="margin: 0; color: #6c757d; font-size: 12px;">
                                            📍 {{ article.source_institution }} | 📖 {{ article.source_journal }}
                                            {% if article.pmid %}
                                            | <a href="https://pubmed.ncbi.nlm.nih.gov/{{ article.pmid }}" style="color: #4ecca3; text-decoration: none;">PMID</a>
                                            {% endif %}
                                        </p>
                                    </td>
                                </tr>
                            </table>
                            {% set article_num.value = article_num.value + 1 %}
                            {% endfor %}
                        </td>
                    </tr>
                    {% endif %}
                    {% endfor %}

                    <!-- Quick News Section -->
                    {% if quick_news %}
                    <tr>
                        <td style="padding: 10px 30px;">
                            <p style="margin: 0; color: #4ecca3; font-size: 12px; text-align: center; letter-spacing: 2px;">
                                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 20px 30px;">
                            <h2 style="margin: 0 0 15px 0; color: #ffffff; font-size: 16px; font-weight: 600;">
                                ⚡ 한눈에 보는 소식 <span style="color: #6c757d; font-size: 12px; font-weight: 400;">(자동 수집)</span>
                            </h2>
                            {% for news in quick_news %}
                            <p style="margin: 0 0 10px 0; color: #c0c0c0; font-size: 13px; line-height: 1.5;">
                                {{ news }}
                            </p>
                            {% endfor %}
                        </td>
                    </tr>
                    {% endif %}

                    <!-- Editor Comment -->
                    <tr>
                        <td style="padding: 10px 30px;">
                            <p style="margin: 0; color: #4ecca3; font-size: 12px; text-align: center; letter-spacing: 2px;">
                                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 20px 30px;">
                            <h2 style="margin: 0 0 15px 0; color: #ffffff; font-size: 16px; font-weight: 600;">
                                💬 AI 에디터 코멘트
                            </h2>
                            <div style="margin: 0; color: #c0c0c0; font-size: 14px; line-height: 1.8;">
                                {{ editor_comment | replace('\n\n', '</p><p style="margin: 15px 0; color: #c0c0c0; font-size: 14px; line-height: 1.8;">') | replace('\n', '<br>') | safe }}
                            </div>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 25px 30px; border-top: 1px solid #0f3460;">
                            <p style="margin: 0; color: #4ecca3; font-size: 12px; text-align: center; letter-spacing: 2px;">
                                ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            </p>
                            <table width="100%" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td style="padding-top: 15px; color: #6c757d; font-size: 11px; text-align: center;">
                                        BIO 데일리 브리핑 | AI 기반 바이오 연구 뉴스레터<br>
                                        <a href="#" style="color: #4ecca3; text-decoration: none;">구독 해지</a>
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


class NewsletterGenerator:
    """Generate HTML email newsletters."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).parent.parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Issue number tracking
        self.issue_file = self.output_dir / "issue_number.txt"

    def get_issue_number(self) -> int:
        """Get and increment issue number."""
        if self.issue_file.exists():
            with open(self.issue_file, "r") as f:
                number = int(f.read().strip())
        else:
            number = 0

        # Increment
        number += 1
        with open(self.issue_file, "w") as f:
            f.write(str(number))

        return number

    def generate_html(
        self,
        trends: List[Trend],
        articles_by_trend: Dict[str, List[NewsArticle]],
        editor_comment: str,
        quick_news: List[str] = None,
        total_papers_analyzed: int = 0,
        issue_number: Optional[int] = None,
    ) -> str:
        """
        Generate HTML newsletter.

        Args:
            trends: List of Trend objects
            articles_by_trend: Dictionary mapping keyword to articles
            editor_comment: Editor's comment
            quick_news: List of quick news items
            total_papers_analyzed: Total papers analyzed
            issue_number: Optional specific issue number

        Returns:
            HTML string
        """
        if issue_number is None:
            issue_number = self.get_issue_number()

        # Get weekday name in Korean
        weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        today = datetime.now()
        weekday = weekdays[today.weekday()]
        date_str = f"{today.strftime('%Y년 %m월 %d일')} {weekday}"

        # Setup Jinja environment
        env = Environment(loader=BaseLoader())
        template = env.from_string(HTML_TEMPLATE)

        # Render template
        html = template.render(
            issue_number=issue_number,
            date=date_str,
            trends=trends,
            articles_by_trend=articles_by_trend,
            editor_comment=editor_comment,
            quick_news=quick_news or [],
            total_papers_analyzed=total_papers_analyzed,
        )

        return html

    def save_html(
        self,
        html: str,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save HTML to file.

        Args:
            html: HTML string
            filename: Optional filename

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"briefing_{datetime.now().strftime('%Y%m%d')}.html"

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return filepath

    def generate_and_save(
        self,
        data: NewsletterData,
    ) -> Path:
        """
        Generate and save newsletter.

        Args:
            data: NewsletterData object

        Returns:
            Path to saved file
        """
        html = self.generate_html(
            trends=data.trends,
            articles_by_trend=data.articles_by_trend,
            editor_comment=data.editor_comment,
            quick_news=data.quick_news,
            total_papers_analyzed=data.total_papers_analyzed,
            issue_number=data.issue_number,
        )

        return self.save_html(html)

    def generate_json(
        self,
        trends: List[Trend],
        articles_by_trend: Dict[str, List[NewsArticle]],
        editor_comment: str,
        quick_news: List[str] = None,
        total_papers_analyzed: int = 0,
        issue_number: Optional[int] = None,
    ) -> dict:
        """Generate JSON data for the newsletter."""
        if issue_number is None:
            issue_number = self.get_issue_number()

        return {
            "issue_number": issue_number,
            "date": datetime.now().strftime("%Y년 %m월 %d일"),
            "total_papers_analyzed": total_papers_analyzed,
            "trends": [t.to_dict() for t in trends],
            "articles": {
                keyword: [a.to_dict() for a in articles]
                for keyword, articles in articles_by_trend.items()
            },
            "quick_news": quick_news or [],
            "editor_comment": editor_comment,
        }


def main():
    """Test newsletter generation with sample data."""
    from .trend_analyzer import Trend
    from .ai_summarizer import NewsArticle

    # Sample data - new format with predefined/emerging
    trends = [
        Trend(
            keyword="GLP-1",
            count=42,
            previous_count=35,
            week_ago_count=31,
            category="치료제",
            why_hot="비만→당뇨→심장→알츠하이머 적응증 확대 경쟁",
            is_predefined=True,
            is_emerging=False,
        ),
        Trend(
            keyword="CAR-T",
            count=28,
            previous_count=25,
            week_ago_count=25,
            category="세포치료",
            why_hot="고형암 적용, 자가면역질환 확장",
            is_predefined=True,
            is_emerging=False,
        ),
        Trend(
            keyword="CRISPR",
            count=25,
            previous_count=25,
            week_ago_count=26,
            category="유전자편집",
            why_hot="최초 유전자치료제 카스게비 승인",
            is_predefined=True,
            is_emerging=False,
        ),
        Trend(
            keyword="Ferroptosis",
            count=18,
            previous_count=5,
            week_ago_count=0,
            category="",
            why_hot="급상승 감지된 키워드",
            is_predefined=False,
            is_emerging=True,
        ),
    ]

    articles_by_trend = {
        "GLP-1": [
            NewsArticle(
                pmid="12345678",
                hook="왜 이 연구가 오늘 주목받나요?",
                title="위고비, 심부전 환자에서도 심혈관 보호 효과 확인",
                content="세마글루타이드(위고비)가 비만 동반 심부전 환자에서 심혈관 사건을 20% 감소시켰다는 대규모 임상 결과가 발표되었습니다.",
                insight="GLP-1 적응증 확대 경쟁 가속화 예상",
                source_journal="NEJM",
                source_institution="Novo Nordisk",
            )
        ],
        "CAR-T": [
            NewsArticle(
                pmid="11111111",
                hook="이번 주 계속 상승세",
                title="고형암 침투력 높인 차세대 CAR-T 개발",
                content="종양 미세환경을 뚫고 들어가는 새로운 CAR-T가 개발되었습니다.",
                insight="CAR-T의 고형암 적용 가능성 한 걸음 더",
                source_journal="Cell",
                source_institution="MIT",
            )
        ],
        "Ferroptosis": [
            NewsArticle(
                pmid="99999999",
                hook="왜 갑자기 급상승?",
                title="페롭토시스 유도 항암제, 전임상 돌파",
                content="기존 항암제 내성을 극복하는 새로운 세포사멸 메커니즘으로 주목받고 있습니다.",
                insight="새로운 항암 전략으로 부상",
                source_journal="Nature",
                source_institution="Harvard",
            )
        ],
    }

    editor_comment = """이번 주는 **GLP-1의 주간**입니다.
비만 → 당뇨 → 심장 → 알츠하이머까지, 적응증 확장 속도가 정말 빠르네요.

**주목 포인트**: Ferroptosis가 급상승 키워드로 감지되었습니다.
기존 항암제 내성 극복의 새로운 대안으로 떠오르고 있습니다."""

    # Generate newsletter
    generator = NewsletterGenerator()
    html = generator.generate_html(
        trends=trends,
        articles_by_trend=articles_by_trend,
        editor_comment=editor_comment,
        total_papers_analyzed=100,
    )

    # Save
    filepath = generator.save_html(html, "test_newsletter.html")
    print(f"Newsletter saved to: {filepath}")


if __name__ == "__main__":
    main()
