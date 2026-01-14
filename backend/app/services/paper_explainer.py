"""
논문 추천 설명 서비스 (Paper Recommendation Explainer).

검색 결과에서 논문이 왜 추천되었는지, 논문의 특성은 무엇인지
Gemini API를 사용하여 설명을 생성합니다.

Features:
- 검색어와 논문 내용 기반 추천 이유 생성
- 논문 특성 분석 (연구 유형, 방법론, 핵심 발견 등)
- 장점/한계점 요약
- 적합성 평가
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re

try:
    import aiohttp
except ImportError:
    aiohttp = None

# 새로운 google.genai SDK 사용 (deprecated google.generativeai 대체)
try:
    from google import genai as new_genai
    from google.genai import types as genai_types
    NEW_GENAI_AVAILABLE = True
except ImportError:
    new_genai = None
    genai_types = None
    NEW_GENAI_AVAILABLE = False

# 레거시 google.generativeai (fallback)
try:
    import google.generativeai as genai
    from google.oauth2 import service_account
    VERTEXAI_AVAILABLE = True
except ImportError:
    genai = None
    VERTEXAI_AVAILABLE = False

from backend.app.core.config import setup_logging

logger = setup_logging(__name__)

# Gemini API 설정
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = os.getenv("PAPER_EXPLAINER_MODEL", "gemini-3-pro-preview")
USE_VERTEX = os.getenv("PAPER_EXPLAINER_USE_VERTEX", "false").lower() == "true"


@dataclass
class PaperCharacteristics:
    """논문 특성 분석 결과."""
    # 연구 유형
    study_type: str = ""  # "Clinical Trial", "Review", "Meta-analysis", etc.
    study_design: str = ""  # "RCT", "Cohort", "Case-control", etc.

    # 핵심 내용
    main_finding: str = ""  # 한 문장 핵심 발견
    methodology: str = ""  # 방법론 요약
    sample_info: str = ""  # 샘플 정보 (n=, population)

    # 평가
    evidence_level: str = ""  # "High", "Medium", "Low"
    clinical_relevance: str = ""  # "High", "Medium", "Low"

    # 장점/한계
    strengths: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    # 키워드/태그
    key_genes: List[str] = field(default_factory=list)
    key_pathways: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)


@dataclass
class RecommendationExplanation:
    """논문 추천 설명."""
    # 왜 추천되었는가
    why_recommended: str = ""
    relevance_factors: List[str] = field(default_factory=list)
    query_match_explanation: str = ""

    # 논문 특성
    characteristics: Optional[PaperCharacteristics] = None

    # 적합성 점수 (1-5)
    relevance_score: int = 0
    novelty_score: int = 0
    quality_score: int = 0

    # 메타
    generated_at: str = ""
    model_used: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        result = asdict(self)
        return result


class VertexAIClient:
    """Vertex AI를 사용한 Gemini 클라이언트 (새로운 google.genai SDK)."""

    def __init__(self, model: str = DEFAULT_MODEL):
        if not NEW_GENAI_AVAILABLE:
            raise ImportError("google-genai 패키지가 필요합니다: pip install google-genai")

        self.model_name = model

        # 서비스 계정 인증을 위한 Vertex AI 클라이언트 생성
        credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("VERTEX_CREDENTIALS_FILE")
        project_id = os.getenv("VERTEX_PROJECT_ID", "gen-lang-client-0509379035")
        location = os.getenv("VERTEX_LOCATION", "us-central1")

        if credentials_file and os.path.exists(credentials_file):
            # Vertex AI 모드로 클라이언트 생성
            self.client = new_genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            logger.info(f"Vertex AI 서비스 계정 인증 완료: {credentials_file}")
        else:
            # API 키 fallback (AI Studio)
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                self.client = new_genai.Client(api_key=api_key)
                logger.info("API 키로 Gemini 인증 (AI Studio)")
            else:
                raise ValueError("Vertex AI 인증 정보가 없습니다")

    async def generate(self, prompt: str, system_instruction: str = None) -> str:
        """Generate content using new google.genai SDK."""
        try:
            # system_instruction이 있으면 프롬프트에 포함
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"

            # 비동기 생성
            config = genai_types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=2048,
            )

            # 동기 호출을 비동기로 래핑
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=config
                )
            )

            if response.text:
                return response.text
            return ""

        except Exception as e:
            logger.error(f"Vertex AI 생성 오류: {e}")
            raise


class GeminiClient:
    """Gemini API 클라이언트 (REST API 직접 호출)."""

    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경 변수가 필요합니다")

        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=30) if aiohttp else None

    async def generate(self, prompt: str, system_instruction: str = None) -> str:
        """Generate content using Gemini API."""
        if not aiohttp:
            raise ImportError("aiohttp 패키지가 필요합니다")

        url = f"{GEMINI_API_URL}/{self.model}:generateContent?key={self.api_key}"

        contents = [{"parts": [{"text": prompt}]}]
        body = {"contents": contents}

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        body["generationConfig"] = {
            "temperature": 0.3,  # 더 일관된 출력
            "topP": 0.9,
            "maxOutputTokens": 2048,
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(url, json=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error: {response.status} - {error_text}")

                data = await response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return ""


class PaperExplainer:
    """논문 추천 설명 생성기."""

    SYSTEM_INSTRUCTION = """당신은 바이오메디컬 연구 논문을 분석하는 전문가입니다.
주어진 검색어와 논문 정보를 바탕으로:
1. 왜 이 논문이 검색어와 관련되는지 설명
2. 논문의 주요 특성 분석
3. 장점과 한계점 요약

모든 응답은 한국어로 작성하세요.
응답은 반드시 유효한 JSON 형식이어야 합니다."""

    EXPLANATION_PROMPT = """검색어: "{query}"

논문 제목: {title}
섹션: {section}
매칭된 내용:
{content}

{extra_info}

다음 JSON 형식으로 응답해주세요:

```json
{{
    "why_recommended": "이 논문이 검색어와 관련된 이유를 2-3문장으로 설명",
    "relevance_factors": ["관련성 요인1", "관련성 요인2", "관련성 요인3"],
    "query_match_explanation": "검색어의 어떤 부분이 논문과 매칭되는지 설명",
    "characteristics": {{
        "study_type": "연구 유형 (Review, Clinical Trial, Basic Research 등)",
        "study_design": "연구 설계 (Meta-analysis, RCT, Cohort, Case Study 등)",
        "main_finding": "핵심 발견 한 문장",
        "methodology": "사용된 방법론 요약",
        "sample_info": "샘플/대상 정보",
        "evidence_level": "High/Medium/Low",
        "clinical_relevance": "High/Medium/Low",
        "strengths": ["장점1", "장점2"],
        "limitations": ["한계점1", "한계점2"],
        "key_genes": ["유전자1", "유전자2"],
        "key_pathways": ["경로1"],
        "techniques": ["기법1", "기법2"]
    }},
    "relevance_score": 4,
    "novelty_score": 3,
    "quality_score": 4
}}
```

참고: 점수는 1-5 사이 정수 (5=최고)"""

    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL, use_vertex: bool = None):
        self.model = model
        self._client = None
        self._api_key = api_key
        # use_vertex가 None이면 환경변수에서 결정
        self._use_vertex = use_vertex if use_vertex is not None else USE_VERTEX

    @property
    def client(self):
        """Lazy initialization of Gemini/Vertex AI client."""
        if self._client is None:
            if self._use_vertex and NEW_GENAI_AVAILABLE:
                logger.info(f"Vertex AI 클라이언트 사용 (새 SDK): {self.model}")
                self._client = VertexAIClient(model=self.model)
            else:
                logger.info(f"Gemini REST API 클라이언트 사용: {self.model}")
                self._client = GeminiClient(api_key=self._api_key, model=self.model)
        return self._client

    @property
    def gemini(self):
        """Backward compatibility alias."""
        return self.client

    async def explain(
        self,
        query: str,
        title: str,
        content: str,
        section: str = "",
        pmid: str = None,
        year: str = None,
        matched_terms: List[str] = None,
    ) -> RecommendationExplanation:
        """
        논문이 왜 추천되었는지 설명 생성.

        Args:
            query: 검색어
            title: 논문 제목
            content: 매칭된 내용
            section: 섹션명
            pmid: PubMed ID
            year: 출판연도
            matched_terms: 매칭된 검색어들

        Returns:
            RecommendationExplanation 객체
        """
        # Extra info 구성
        extra_parts = []
        if pmid:
            extra_parts.append(f"PMID: {pmid}")
        if year:
            extra_parts.append(f"출판연도: {year}")
        if matched_terms:
            extra_parts.append(f"매칭된 용어: {', '.join(matched_terms)}")

        extra_info = "\n".join(extra_parts) if extra_parts else ""

        prompt = self.EXPLANATION_PROMPT.format(
            query=query,
            title=title,
            section=section,
            content=content[:2000],  # 토큰 제한
            extra_info=extra_info
        )

        try:
            response = await self.gemini.generate(
                prompt=prompt,
                system_instruction=self.SYSTEM_INSTRUCTION
            )

            data = self._parse_json_response(response)

            # 특성 객체 생성
            char_data = data.get("characteristics", {})
            characteristics = PaperCharacteristics(
                study_type=char_data.get("study_type", ""),
                study_design=char_data.get("study_design", ""),
                main_finding=char_data.get("main_finding", ""),
                methodology=char_data.get("methodology", ""),
                sample_info=char_data.get("sample_info", ""),
                evidence_level=char_data.get("evidence_level", ""),
                clinical_relevance=char_data.get("clinical_relevance", ""),
                strengths=char_data.get("strengths", []),
                limitations=char_data.get("limitations", []),
                key_genes=char_data.get("key_genes", []),
                key_pathways=char_data.get("key_pathways", []),
                techniques=char_data.get("techniques", []),
            )

            return RecommendationExplanation(
                why_recommended=data.get("why_recommended", ""),
                relevance_factors=data.get("relevance_factors", []),
                query_match_explanation=data.get("query_match_explanation", ""),
                characteristics=characteristics,
                relevance_score=data.get("relevance_score", 0),
                novelty_score=data.get("novelty_score", 0),
                quality_score=data.get("quality_score", 0),
                model_used=self.model
            )

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            # 기본 설명 반환
            return RecommendationExplanation(
                why_recommended=f"'{query}' 검색어와 관련된 내용이 논문에서 발견되었습니다.",
                relevance_factors=matched_terms or [],
                model_used=self.model
            )

    async def explain_batch(
        self,
        query: str,
        results: List[Dict],
        max_concurrent: int = 3
    ) -> List[RecommendationExplanation]:
        """
        여러 검색 결과에 대해 설명 일괄 생성.

        Args:
            query: 검색어
            results: 검색 결과 리스트
            max_concurrent: 최대 동시 요청 수

        Returns:
            설명 리스트
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def explain_with_limit(result):
            async with semaphore:
                return await self.explain(
                    query=query,
                    title=result.get("paper_title", ""),
                    content=result.get("content", ""),
                    section=result.get("section", ""),
                    pmid=result.get("pmid"),
                    year=result.get("year"),
                    matched_terms=result.get("matched_terms", [])
                )

        tasks = [explain_with_limit(r) for r in results]
        return await asyncio.gather(*tasks)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response."""
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                return json.loads(json_str)
            except:
                return {}


# Rule-based fallback (Gemini API 없을 때 사용)
class RuleBasedExplainer:
    """규칙 기반 설명 생성기 (API 없이 동작)."""

    # 연구 유형 키워드
    STUDY_TYPE_KEYWORDS = {
        "Review": ["review", "overview", "systematic review", "literature review"],
        "Meta-analysis": ["meta-analysis", "meta analysis", "pooled analysis"],
        "Clinical Trial": ["clinical trial", "randomized", "RCT", "phase I", "phase II", "phase III"],
        "Cohort Study": ["cohort", "prospective", "retrospective", "follow-up"],
        "Case Study": ["case report", "case study", "case series"],
        "Basic Research": ["in vitro", "in vivo", "cell line", "mouse model", "animal model"],
        "Bioinformatics": ["RNA-seq", "transcriptome", "TCGA", "GEO", "bioinformatics", "computational"],
    }

    # 기법 키워드
    TECHNIQUE_KEYWORDS = {
        "RNA-seq": ["RNA-seq", "RNA sequencing", "transcriptome"],
        "qPCR": ["qPCR", "RT-PCR", "real-time PCR"],
        "Western Blot": ["western blot", "immunoblot"],
        "Immunohistochemistry": ["immunohistochemistry", "IHC"],
        "CRISPR": ["CRISPR", "Cas9", "gene editing"],
        "Flow Cytometry": ["flow cytometry", "FACS"],
        "Machine Learning": ["machine learning", "deep learning", "neural network", "random forest"],
    }

    def explain(
        self,
        query: str,
        title: str,
        content: str,
        section: str = "",
        matched_terms: List[str] = None,
    ) -> RecommendationExplanation:
        """규칙 기반으로 설명 생성."""
        text = f"{title} {content}".lower()
        matched_terms = matched_terms or []

        # 연구 유형 감지
        study_type = "연구 논문"
        for stype, keywords in self.STUDY_TYPE_KEYWORDS.items():
            if any(kw.lower() in text for kw in keywords):
                study_type = stype
                break

        # 기법 감지
        techniques = []
        for tech, keywords in self.TECHNIQUE_KEYWORDS.items():
            if any(kw.lower() in text for kw in keywords):
                techniques.append(tech)

        # 유전자 감지 (대문자 2-6글자)
        gene_pattern = r'\b([A-Z][A-Z0-9]{1,5})\b'
        potential_genes = list(set(re.findall(gene_pattern, f"{title} {content}")))
        # 일반적인 약어 제외
        exclude = {"RNA", "DNA", "PCR", "qPCR", "USA", "FDA", "WHO", "THE", "AND", "FOR"}
        key_genes = [g for g in potential_genes if g not in exclude][:5]

        # 관련성 요인 생성
        relevance_factors = []
        query_terms = query.lower().split()
        for term in query_terms:
            if term in text:
                relevance_factors.append(f"'{term}' 키워드 매칭")

        if matched_terms:
            relevance_factors.extend([f"'{t}' 용어 포함" for t in matched_terms[:3]])

        characteristics = PaperCharacteristics(
            study_type=study_type,
            key_genes=key_genes,
            techniques=techniques,
        )

        why_recommended = f"검색어 '{query}'와 관련된 {study_type} 논문입니다."
        if techniques:
            why_recommended += f" {', '.join(techniques[:2])} 기법을 사용합니다."
        if key_genes:
            why_recommended += f" 주요 유전자: {', '.join(key_genes[:3])}."

        return RecommendationExplanation(
            why_recommended=why_recommended,
            relevance_factors=relevance_factors,
            query_match_explanation=f"검색어의 핵심 단어가 논문의 {section or '본문'}에서 발견되었습니다.",
            characteristics=characteristics,
            relevance_score=3,
            novelty_score=3,
            quality_score=3,
            model_used="rule-based"
        )


def get_paper_explainer(use_llm: bool = True) -> Any:
    """
    적절한 설명 생성기 반환.

    Args:
        use_llm: LLM 사용 여부 (False면 규칙 기반)

    Returns:
        PaperExplainer 또는 RuleBasedExplainer
    """
    if use_llm:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            return PaperExplainer()
        else:
            logger.warning("Gemini API key not found, using rule-based explainer")
            return RuleBasedExplainer()
    return RuleBasedExplainer()
