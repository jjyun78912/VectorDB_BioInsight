"""
논문 심층 평가 시스템 (Paper Deep Evaluator).

PubMed/DOI에서 논문을 가져와 Gemini API를 사용하여
장점, 한계점, 개선 방향을 포함한 종합적인 평가를 생성합니다.

Usage:
    # DOI로 논문 평가
    python scripts/paper_evaluator.py --doi "10.1038/s41586-025-09896-x"

    # PMID로 논문 평가
    python scripts/paper_evaluator.py --pmid "41501451"

    # PubMed 검색 후 최신 논문 평가
    python scripts/paper_evaluator.py --search "mitochondrial transfer neuropathy 2026"

    # 결과를 마크다운 파일로 저장
    python scripts/paper_evaluator.py --doi "10.1038/..." --output evaluation.md

비용 비교 (1M tokens 기준):
    - Claude Opus: $15 input / $75 output
    - Gemini 2.0 Flash: $0.10 input / $0.40 output (150x 저렴!)
"""

import os
import sys
import json
import asyncio
import aiohttp
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.config import DATA_DIR, setup_logging
from backend.app.core.web_crawler_agent import WebCrawlerAgent, FetchedPaper, FullTextFetcher

logger = setup_logging(__name__)

# Gemini API 설정
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.0-flash"  # 비용 효율적인 모델


@dataclass
class PaperEvaluation:
    """논문 평가 결과."""
    # 논문 기본 정보
    title: str
    authors: List[str]
    journal: str
    year: int
    doi: str
    pmid: str
    url: str

    # 연구 개요
    research_summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    methodology: str = ""

    # 평가 결과
    strengths: List[Dict[str, str]] = field(default_factory=list)  # {"title": "", "description": ""}
    limitations: List[Dict[str, str]] = field(default_factory=list)
    future_directions: List[Dict[str, Any]] = field(default_factory=list)  # {"area": "", "suggestion": "", "timeline": "", "impact": ""}

    # 적합성 평가
    relevance_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {"field": {"score": 5, "reason": ""}}

    # 메타데이터
    citation_count: int = 0
    altmetric_score: float = 0.0
    evaluated_at: str = ""
    model_used: str = ""

    def __post_init__(self):
        if not self.evaluated_at:
            self.evaluated_at = datetime.now().isoformat()

    def to_markdown(self) -> str:
        """평가 결과를 마크다운 형식으로 변환."""
        md = []

        # 논문 정보
        md.append(f"## 📄 {self.year}년 논문 분석\n")
        md.append(f"### **논문 정보**")
        md.append(f"- **제목**: {self.title}")
        md.append(f"- **저널**: {self.journal} ({self.year})")
        md.append(f"- **저자**: {', '.join(self.authors[:5])}{'...' if len(self.authors) > 5 else ''}")
        if self.doi:
            md.append(f"- **DOI**: [{self.doi}](https://doi.org/{self.doi})")
        if self.pmid:
            md.append(f"- **PMID**: [{self.pmid}](https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/)")
        if self.citation_count > 0:
            md.append(f"- **인용 수**: {self.citation_count}")
        md.append("")

        # 연구 개요
        md.append("---\n")
        md.append("## 🔬 연구 개요\n")
        md.append(self.research_summary)
        md.append("")

        # 주요 발견
        if self.key_findings:
            md.append("### 주요 발견")
            for i, finding in enumerate(self.key_findings, 1):
                md.append(f"{i}. {finding}")
            md.append("")

        # 방법론
        if self.methodology:
            md.append("### 연구 방법론")
            md.append(self.methodology)
            md.append("")

        # 장점
        md.append("---\n")
        md.append("## ✅ 장점 (Strengths)\n")
        for i, strength in enumerate(self.strengths, 1):
            md.append(f"### {i}. **{strength.get('title', '')}**")
            md.append(strength.get('description', ''))
            md.append("")

        # 한계점
        md.append("---\n")
        md.append("## ⚠️ 한계점 (Limitations)\n")
        for i, limitation in enumerate(self.limitations, 1):
            md.append(f"### {i}. **{limitation.get('title', '')}**")
            md.append(limitation.get('description', ''))
            md.append("")

        # 향후 개선 방향
        md.append("---\n")
        md.append("## 🚀 향후 개선 방향 (Future Directions)\n")
        md.append("| 영역 | 제안 | 예상 기간 | 잠재적 영향 |")
        md.append("|------|------|-----------|-------------|")
        for direction in self.future_directions:
            area = direction.get('area', '')
            suggestion = direction.get('suggestion', '')
            timeline = direction.get('timeline', '')
            impact = direction.get('impact', '')
            md.append(f"| **{area}** | {suggestion} | {timeline} | {impact} |")
        md.append("")

        # 적합성 평가
        if self.relevance_scores:
            md.append("---\n")
            md.append("## 📊 유저 적합성 평가\n")
            md.append("| 연구 분야 | 적합도 | 이유 |")
            md.append("|-----------|--------|------|")
            for field_name, score_info in self.relevance_scores.items():
                stars = "⭐" * score_info.get('score', 0)
                reason = score_info.get('reason', '')
                md.append(f"| **{field_name}** | {stars} | {reason} |")
            md.append("")

        # 결론
        md.append("---\n")
        md.append("## 🎯 결론\n")
        md.append(f"*평가 일시: {self.evaluated_at[:10]}*")
        md.append(f"*사용 모델: {self.model_used}*")

        return "\n".join(md)

    def to_dict(self) -> Dict:
        """딕셔너리로 변환."""
        return asdict(self)


class GeminiClient:
    """Gemini API 클라이언트."""

    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        """
        Initialize Gemini client.

        Args:
            api_key: Google AI API key
            model: Gemini model name
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경 변수가 필요합니다")

        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=120)

    async def generate(self, prompt: str, system_instruction: str = None) -> str:
        """
        Generate content using Gemini API.

        Args:
            prompt: User prompt
            system_instruction: System instruction (optional)

        Returns:
            Generated text
        """
        url = f"{GEMINI_API_URL}/{self.model}:generateContent?key={self.api_key}"

        # Build request body
        contents = [{"parts": [{"text": prompt}]}]

        body = {"contents": contents}

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        # Generation config
        body["generationConfig"] = {
            "temperature": 0.7,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": 8192,
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(url, json=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error: {response.status} - {error_text}")

                data = await response.json()

                # Extract generated text
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text", "")

                return ""


class PaperEvaluator:
    """논문 평가 시스템."""

    SYSTEM_INSTRUCTION = """당신은 바이오메디컬 연구 논문을 분석하고 평가하는 전문가입니다.

주어진 논문 정보를 바탕으로 다음을 수행합니다:
1. 연구 내용을 한국어로 요약
2. 연구의 장점을 5가지 이상 분석
3. 연구의 한계점을 5가지 이상 분석
4. 향후 개선 방향을 구체적으로 제안
5. 다양한 연구 분야별 적합성 평가

응답은 반드시 유효한 JSON 형식이어야 합니다.
모든 텍스트는 한국어로 작성하되, 전문 용어는 영문 병기 가능합니다.
구체적인 근거와 함께 분석하며, 과학적 정확성을 유지합니다."""

    EVALUATION_PROMPT_TEMPLATE = """다음 논문을 분석하고 평가해주세요.

## 논문 정보
- 제목: {title}
- 저널: {journal} ({year})
- 저자: {authors}
- DOI: {doi}
- PMID: {pmid}
- 인용 수: {citations}

## 초록
{abstract}

## 본문 (가능한 경우)
{full_text}

---

다음 JSON 형식으로 응답해주세요:

```json
{{
    "research_summary": "연구 내용 2-3문장 요약",
    "key_findings": [
        "주요 발견 1",
        "주요 발견 2",
        "주요 발견 3"
    ],
    "methodology": "사용된 연구 방법론 설명",
    "strengths": [
        {{"title": "장점 제목", "description": "장점 상세 설명"}},
        {{"title": "장점 제목", "description": "장점 상세 설명"}}
    ],
    "limitations": [
        {{"title": "한계점 제목", "description": "한계점 상세 설명"}},
        {{"title": "한계점 제목", "description": "한계점 상세 설명"}}
    ],
    "future_directions": [
        {{"area": "연구 영역", "suggestion": "개선 제안", "timeline": "예상 기간", "impact": "잠재적 영향도 (높음/중간/낮음)"}},
        {{"area": "연구 영역", "suggestion": "개선 제안", "timeline": "예상 기간", "impact": "잠재적 영향도"}}
    ],
    "relevance_scores": {{
        "연구 분야명": {{"score": 5, "reason": "적합한 이유"}},
        "다른 분야명": {{"score": 3, "reason": "적합한 이유"}}
    }}
}}
```

참고: score는 1-5 사이 정수 (5=매우 적합, 1=적합하지 않음)"""

    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        """
        Initialize paper evaluator.

        Args:
            api_key: Gemini API key
            model: Gemini model name
        """
        self.gemini = GeminiClient(api_key=api_key, model=model)
        self.model = model
        self.crawler = WebCrawlerAgent()
        self.fulltext_fetcher = FullTextFetcher()

    async def fetch_paper(self, doi: str = None, pmid: str = None, url: str = None) -> Optional[FetchedPaper]:
        """
        Fetch paper information from various sources.

        Args:
            doi: DOI string
            pmid: PubMed ID
            url: Paper URL

        Returns:
            FetchedPaper object or None
        """
        paper = None

        if doi:
            paper = await self.crawler.fetch_by_doi(doi)
        elif url:
            paper = await self.crawler.fetch_by_url(url)
        elif pmid:
            papers = await self.crawler.search_pubmed(f"{pmid}[uid]", max_results=1)
            if papers:
                paper = papers[0]

        # Try to get full text
        if paper:
            try:
                full_text = await self.fulltext_fetcher.fetch(
                    pmid=paper.pmid,
                    pmcid=paper.pmcid,
                    doi=paper.doi,
                    use_playwright=False  # 속도를 위해 비활성화
                )
                if full_text:
                    paper.full_text = full_text
            except Exception as e:
                logger.warning(f"Full text fetch failed: {e}")

        return paper

    async def search_and_get_latest(self, query: str, max_results: int = 5) -> List[FetchedPaper]:
        """
        Search PubMed and return latest papers.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of papers
        """
        return await self.crawler.search_pubmed(
            query=query,
            max_results=max_results,
            sort="pub_date"
        )

    async def evaluate(self, paper: FetchedPaper) -> PaperEvaluation:
        """
        Evaluate a paper using Gemini API.

        Args:
            paper: FetchedPaper object

        Returns:
            PaperEvaluation object
        """
        # Build evaluation prompt
        prompt = self.EVALUATION_PROMPT_TEMPLATE.format(
            title=paper.title,
            journal=paper.journal,
            year=paper.year,
            authors=", ".join(paper.authors[:10]),
            doi=paper.doi or "N/A",
            pmid=paper.pmid or "N/A",
            citations=paper.citation_count,
            abstract=paper.abstract or "초록 없음",
            full_text=paper.full_text[:15000] if paper.full_text else "전문 없음 (초록만 분석)"
        )

        # Call Gemini API
        response = await self.gemini.generate(
            prompt=prompt,
            system_instruction=self.SYSTEM_INSTRUCTION
        )

        # Parse JSON response
        eval_data = self._parse_json_response(response)

        # Create evaluation object
        evaluation = PaperEvaluation(
            title=paper.title,
            authors=paper.authors,
            journal=paper.journal,
            year=paper.year,
            doi=paper.doi,
            pmid=paper.pmid,
            url=paper.url,
            citation_count=paper.citation_count,
            model_used=self.model,
            **eval_data
        )

        return evaluation

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("No JSON found in response")
                return {}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            # Try to fix common issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                return json.loads(json_str)
            except:
                return {}


async def main():
    parser = argparse.ArgumentParser(description="논문 심층 평가 시스템")
    parser.add_argument("--doi", type=str, help="논문 DOI")
    parser.add_argument("--pmid", type=str, help="PubMed ID")
    parser.add_argument("--url", type=str, help="논문 URL")
    parser.add_argument("--search", type=str, help="PubMed 검색어")
    parser.add_argument("--output", type=str, help="출력 파일 경로 (.md 또는 .json)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini 모델명")
    parser.add_argument("--list-models", action="store_true", help="사용 가능한 모델 목록")

    args = parser.parse_args()

    if args.list_models:
        print("\n📋 사용 가능한 Gemini 모델:")
        print("  - gemini-2.0-flash (기본, 가장 빠르고 저렴)")
        print("  - gemini-2.0-flash-lite (더 빠르고 저렴)")
        print("  - gemini-1.5-pro (고품질, 더 비쌈)")
        print("  - gemini-1.5-flash (균형)")
        return

    if not any([args.doi, args.pmid, args.url, args.search]):
        parser.print_help()
        print("\n예시:")
        print("  python scripts/paper_evaluator.py --doi '10.1038/s41586-025-09896-x'")
        print("  python scripts/paper_evaluator.py --pmid '41501451'")
        print("  python scripts/paper_evaluator.py --search 'mitochondrial transfer neuropathy 2026'")
        return

    print("🔬 논문 평가 시스템 시작...")
    print(f"   모델: {args.model}\n")

    try:
        evaluator = PaperEvaluator(model=args.model)
    except ValueError as e:
        print(f"❌ 오류: {e}")
        print("   GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경 변수를 설정하세요.")
        return

    # Fetch paper
    paper = None

    if args.search:
        print(f"🔍 PubMed 검색 중: '{args.search}'")
        papers = await evaluator.search_and_get_latest(args.search, max_results=5)

        if not papers:
            print("❌ 검색 결과 없음")
            return

        print(f"\n📚 검색 결과 ({len(papers)}편):")
        for i, p in enumerate(papers, 1):
            print(f"  {i}. [{p.year}] {p.title[:60]}...")

        print("\n1번 논문을 평가합니다...")
        paper = papers[0]
    else:
        print("📥 논문 정보 수집 중...")
        paper = await evaluator.fetch_paper(
            doi=args.doi,
            pmid=args.pmid,
            url=args.url
        )

    if not paper:
        print("❌ 논문을 찾을 수 없습니다")
        return

    print(f"✓ 논문: {paper.title[:60]}...")
    print(f"  저널: {paper.journal} ({paper.year})")
    print(f"  인용: {paper.citation_count}")

    if paper.full_text:
        print(f"  전문: {len(paper.full_text):,} 글자")
    else:
        print("  전문: 없음 (초록만 분석)")

    # Evaluate
    print("\n🤖 Gemini API로 분석 중...")
    evaluation = await evaluator.evaluate(paper)

    # Output
    if args.output:
        output_path = Path(args.output)

        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation.to_dict(), f, ensure_ascii=False, indent=2)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(evaluation.to_markdown())

        print(f"\n✅ 평가 결과 저장: {output_path}")
    else:
        # Print to console
        print("\n" + "=" * 80)
        print(evaluation.to_markdown())

    print("\n✅ 평가 완료!")


if __name__ == "__main__":
    asyncio.run(main())
