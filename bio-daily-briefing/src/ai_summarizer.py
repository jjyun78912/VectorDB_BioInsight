"""
AI Summarizer - Generate news-style summaries using Claude/Gemini
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass

from .pubmed_fetcher import Paper
from .trend_analyzer import Trend


@dataclass
class NewsArticle:
    """A news-style article generated from a paper."""
    pmid: str
    hook: str  # Engaging question or statement
    title: str  # Korean news title
    content: str  # Main content
    insight: str  # Why this matters
    source_journal: str
    source_institution: str
    doi: Optional[str] = None
    pub_date: str = ""

    def to_dict(self) -> Dict:
        return {
            "pmid": self.pmid,
            "hook": self.hook,
            "title": self.title,
            "content": self.content,
            "insight": self.insight,
            "source_journal": self.source_journal,
            "source_institution": self.source_institution,
            "doi": self.doi,
            "pub_date": self.pub_date,
        }


class AISummarizer:
    """Generate news-style summaries using AI."""

    KOREAN_PROMPT = """당신은 생명과학 뉴스 전문 기자입니다.
주어진 논문을 일반 독자가 이해할 수 있는 뉴스 기사로 변환해주세요.

**논문 정보:**
제목: {title}
초록: {abstract}
저널: {journal}
키워드: {keywords}

**출력 형식 (정확히 이 형식으로):**

HOOK: [독자의 호기심을 자극하는 질문, 30자 이내]

TITLE: [한글 뉴스 제목, 핵심 발견을 담은 25자 이내]

CONTENT: [연구 배경 → 핵심 발견 → 의미를 설명하는 3-4문장. 전문용어는 쉽게 풀어 설명. 총 150-200자]

INSIGHT: [이 연구가 환자/의료계에 미칠 영향을 1문장으로. 50자 이내]

INSTITUTION: [연구 수행 기관명, 없으면 저널명]

예시:
HOOK: 암세포는 어떻게 면역 공격을 피할까요?
TITLE: 암세포 면역회피 새 메커니즘 규명
CONTENT: 암세포는 주변 면역세포의 공격을 피하기 위해 다양한 전략을 사용합니다. 이번 연구에서는 암세포가 특정 단백질을 분비하여 면역세포의 활성을 억제한다는 사실이 밝혀졌습니다. 연구팀은 이 단백질을 차단하면 면역치료 효과가 2배 이상 증가함을 확인했습니다.
INSIGHT: 면역항암제 내성 극복을 위한 새로운 병용치료 전략 제시
INSTITUTION: Stanford University"""

    ENGLISH_PROMPT = """You are a science journalist specializing in life sciences.
Transform the given paper into a news article for general readers.

**Paper Information:**
Title: {title}
Abstract: {abstract}
Journal: {journal}
Keywords: {keywords}

**Output Format (use this exact format):**

HOOK: [Engaging question to grab reader attention, max 50 chars]

TITLE: [News headline capturing key finding, max 60 chars]

CONTENT: [3-4 sentences explaining background → discovery → significance. Use plain language. 150-200 words total]

INSIGHT: [One sentence on impact for patients/medicine, max 100 chars]

INSTITUTION: [Research institution name, or journal if not available]"""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        language: str = "ko",
    ):
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        # Support both GOOGLE_API_KEY and GEMINI_API_KEY
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.language = language

        self._llm = None

    def _get_llm(self, prefer_claude: bool = True):
        """Get LLM instance, preferring Claude Sonnet for quality."""
        if self._llm is not None:
            return self._llm

        # Try Claude first (preferred for Korean quality)
        if prefer_claude and self.anthropic_api_key:
            try:
                from langchain_anthropic import ChatAnthropic
                self._llm = ChatAnthropic(
                    model="claude-sonnet-4-20250514",
                    api_key=self.anthropic_api_key,
                    temperature=0.7,
                )
                print("Using Claude Sonnet API")
                return self._llm
            except Exception as e:
                print(f"Claude init failed: {e}")

        # Fallback to Gemini
        if self.google_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    google_api_key=self.google_api_key,
                    temperature=0.7,
                )
                print("Using Gemini API (fallback)")
                return self._llm
            except Exception as e:
                print(f"Gemini init failed: {e}")

        # Retry with Claude if Gemini failed
        if not prefer_claude and self.anthropic_api_key:
            try:
                from langchain_anthropic import ChatAnthropic
                self._llm = ChatAnthropic(
                    model="claude-sonnet-4-20250514",
                    api_key=self.anthropic_api_key,
                    temperature=0.7,
                )
                print("Using Claude Sonnet API (fallback)")
                return self._llm
            except Exception as e:
                print(f"Claude fallback init failed: {e}")

        raise RuntimeError("No LLM API key configured. Set GOOGLE_API_KEY or ANTHROPIC_API_KEY.")

    def _get_prompt_template(self) -> str:
        """Get prompt template based on language."""
        return self.KOREAN_PROMPT if self.language == "ko" else self.ENGLISH_PROMPT

    def summarize_paper(self, paper: Paper) -> Optional[NewsArticle]:
        """
        Generate a news article from a paper.

        Args:
            paper: Paper object

        Returns:
            NewsArticle object or None if failed
        """
        if not paper.abstract:
            return None

        try:
            llm = self._get_llm()
            prompt_template = self._get_prompt_template()

            # Format prompt
            keywords = ", ".join(paper.keywords[:5]) if paper.keywords else ""
            if not keywords and paper.mesh_terms:
                keywords = ", ".join(paper.mesh_terms[:5])

            prompt = prompt_template.format(
                title=paper.title,
                abstract=paper.abstract[:2000],
                journal=paper.journal,
                keywords=keywords,
            )

            # Call LLM
            response = llm.invoke(prompt)
            response_text = response.content

            # Parse response
            return self._parse_response(response_text, paper)

        except Exception as e:
            print(f"Error summarizing paper {paper.pmid}: {e}")
            return self._create_fallback_article(paper)

    def _parse_response(self, text: str, paper: Paper) -> NewsArticle:
        """Parse LLM response into NewsArticle."""
        hook = ""
        title = ""
        content = ""
        insight = ""
        institution = ""

        for line in text.split("\n"):
            line = line.strip()

            if line.startswith("HOOK:"):
                hook = line.replace("HOOK:", "").strip()
            elif line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("CONTENT:"):
                content = line.replace("CONTENT:", "").strip()
            elif line.startswith("INSIGHT:"):
                insight = line.replace("INSIGHT:", "").strip()
            elif line.startswith("INSTITUTION:"):
                institution = line.replace("INSTITUTION:", "").strip()

        # Fallbacks
        if not hook:
            hook = "최신 연구 결과가 발표되었습니다" if self.language == "ko" else "New research findings published"
        if not title:
            title = paper.title[:50]
        if not content:
            content = paper.abstract[:300] + "..."
        if not insight:
            insight = "향후 연구에 중요한 시사점 제공" if self.language == "ko" else "Important implications for future research"
        if not institution:
            institution = paper.journal

        return NewsArticle(
            pmid=paper.pmid,
            hook=hook,
            title=title,
            content=content,
            insight=insight,
            source_journal=paper.journal,
            source_institution=institution,
            doi=paper.doi,
            pub_date=paper.pub_date,
        )

    def _create_fallback_article(self, paper: Paper) -> NewsArticle:
        """Create a fallback article without LLM."""
        return NewsArticle(
            pmid=paper.pmid,
            hook="새로운 연구 결과" if self.language == "ko" else "New research",
            title=paper.title[:50],
            content=paper.abstract[:300] + "..." if paper.abstract else "",
            insight="",
            source_journal=paper.journal,
            source_institution=paper.journal,
            doi=paper.doi,
            pub_date=paper.pub_date,
        )

    def summarize_papers_by_trend(
        self,
        trends: List[Trend],
        max_per_trend: int = 2,
    ) -> Dict[str, List[NewsArticle]]:
        """
        Generate news articles grouped by trend.

        Args:
            trends: List of Trend objects with representative papers
            max_per_trend: Maximum articles per trend

        Returns:
            Dictionary mapping trend keyword to list of NewsArticle
        """
        result = {}
        used_pmids = set()  # Track already used papers to prevent duplicates

        for trend in trends:
            articles = []
            for paper in trend.representative_papers:
                # Skip if this paper was already used in another trend
                if paper.pmid in used_pmids:
                    continue

                article = self.summarize_paper(paper)
                if article:
                    articles.append(article)
                    used_pmids.add(paper.pmid)

                # Stop if we have enough articles for this trend
                if len(articles) >= max_per_trend:
                    break

            result[trend.keyword] = articles

        return result

    def generate_editor_comment(self, trends: List[Trend]) -> str:
        """Generate an editor's comment summarizing the day's trends."""
        try:
            llm = self._get_llm()

            # Separate predefined and emerging trends
            predefined = [t for t in trends if getattr(t, 'is_predefined', True) and not getattr(t, 'is_emerging', False)]
            emerging = [t for t in trends if getattr(t, 'is_emerging', False)]

            predefined_summary = "\n".join([
                f"- {t.keyword} ({t.count}건, {getattr(t, 'change_label', t.trend_indicator)}, 카테고리: {getattr(t, 'category', 'N/A')}, 왜 핫함: {getattr(t, 'why_hot', 'N/A')})"
                for t in predefined
            ])

            emerging_summary = "\n".join([
                f"- {t.keyword} ({t.count}건, 급상승 감지)"
                for t in emerging
            ]) if emerging else "없음"

            if self.language == "ko":
                prompt = f"""당신은 바이오/제약 업계 전문 에디터입니다.
오늘의 PubMed 논문 트렌드를 분석하여 에디터 코멘트를 작성해주세요.

[업계 주목 키워드 - 고정 핫토픽]
{predefined_summary}

[급상승 감지 키워드 - 신규]
{emerging_summary}

다음 형식으로 작성해주세요:
1. 첫 문단: 오늘 가장 주목할 키워드와 그 의미 (왜 이 키워드가 업계에서 중요한지)
2. 두 번째 문단: 급상승 키워드가 있다면 왜 갑자기 주목받는지 분석
3. 마지막: 연구자/투자자가 주목해야 할 포인트

마크다운 볼드(**키워드**) 사용 가능. 전체 200자 내외로 작성."""
            else:
                prompt = f"""You are a biotech/pharma industry editor.
Analyze today's PubMed trends and write an editor's comment.

[Industry Hot Topics - Predefined]
{predefined_summary}

[Emerging Keywords - New]
{emerging_summary}

Write a 2-3 paragraph comment covering key trends and what researchers/investors should note."""

            response = llm.invoke(prompt)
            return response.content.strip()

        except Exception as e:
            print(f"Error generating editor comment: {e}")
            if self.language == "ko":
                return "오늘의 주요 연구 동향을 확인하세요."
            return "Check out today's key research trends."


async def main():
    """Test the AI summarizer."""
    from .pubmed_fetcher import PubMedFetcher
    from .trend_analyzer import TrendAnalyzer

    # Fetch papers
    fetcher = PubMedFetcher()
    print("Fetching papers...")
    papers = await fetcher.fetch_recent_papers(max_results=30, days=3)

    # Analyze trends
    analyzer = TrendAnalyzer()
    trends = analyzer.get_hot_topics(papers, top_n=3)

    # Summarize
    summarizer = AISummarizer(language="ko")
    articles_by_trend = summarizer.summarize_papers_by_trend(trends)

    for keyword, articles in articles_by_trend.items():
        print(f"\n=== {keyword} ===")
        for article in articles:
            print(f"\n[HOOK] {article.hook}")
            print(f"[TITLE] {article.title}")
            print(f"[CONTENT] {article.content[:100]}...")
            print(f"[INSIGHT] {article.insight}")

    # Editor comment
    comment = summarizer.generate_editor_comment(trends)
    print(f"\n[EDITOR] {comment}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
