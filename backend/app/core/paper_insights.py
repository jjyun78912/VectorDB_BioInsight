"""
Paper Insights Module - Quick paper evaluation metrics.

Provides:
1. Bottom Line: One-sentence clinical/research takeaway
2. Study Quality Score: Methodological rigor assessment
3. Key Outcomes: Extracted effect sizes and primary endpoints
4. Population Quick View: Patient demographics and inclusion criteria
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

# Using unified LLM helper for Vertex AI / API key auto-selection


class StudyDesign(Enum):
    """Study design hierarchy for evidence strength."""
    META_ANALYSIS = ("Meta-Analysis", 10)
    SYSTEMATIC_REVIEW = ("Systematic Review", 9)
    RCT = ("Randomized Controlled Trial", 8)
    COHORT = ("Cohort Study", 6)
    CASE_CONTROL = ("Case-Control", 5)
    CROSS_SECTIONAL = ("Cross-Sectional", 4)
    CASE_SERIES = ("Case Series", 3)
    CASE_REPORT = ("Case Report", 2)
    EXPERT_OPINION = ("Expert Opinion/Review", 1)
    IN_VITRO = ("In Vitro/Animal", 0)
    UNKNOWN = ("Unknown", 0)

    def __init__(self, label: str, score: int):
        self.label = label
        self.evidence_score = score


@dataclass
class BottomLine:
    """One-sentence takeaway from a paper."""
    summary: str  # The bottom line itself
    clinical_relevance: str  # "High", "Medium", "Low"
    action_type: str  # "Treatment", "Diagnosis", "Mechanism", "Epidemiology"
    confidence: float  # 0-1


@dataclass
class StudyQuality:
    """Quality assessment of a research paper."""
    design: StudyDesign
    sample_size: Optional[int]
    quality_score: float  # 0-10
    strengths: List[str]
    limitations: List[str]
    bias_risk: str  # "Low", "Medium", "High", "Unclear"

    @property
    def quality_label(self) -> str:
        if self.quality_score >= 8:
            return "High"
        elif self.quality_score >= 5:
            return "Medium"
        else:
            return "Low"


@dataclass
class KeyOutcome:
    """Extracted primary outcome with effect size."""
    outcome_name: str  # e.g., "Overall Survival"
    metric_type: str  # "HR", "OR", "RR", "MD", "SMD", "Response Rate"
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[float] = None
    interpretation: str = ""  # e.g., "36% reduced risk of death"


@dataclass
class PopulationInfo:
    """Quick view of study population."""
    total_n: Optional[int]
    condition: str
    age_range: str  # e.g., "45-75" or "median 62"
    female_percent: Optional[float]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    setting: str  # e.g., "Multicenter", "Single-center"


@dataclass
class PaperInsights:
    """Complete insights for a paper."""
    bottom_line: Optional[BottomLine] = None
    quality: Optional[StudyQuality] = None
    key_outcomes: List[KeyOutcome] = field(default_factory=list)
    population: Optional[PopulationInfo] = None


class PaperInsightsExtractor:
    """
    Extracts quick evaluation insights from papers.

    Designed for speed - researchers should understand paper value in <10 seconds.
    """

    # Study design detection patterns
    DESIGN_PATTERNS = {
        StudyDesign.META_ANALYSIS: [
            r"meta-analysis", r"meta analysis", r"pooled analysis"
        ],
        StudyDesign.SYSTEMATIC_REVIEW: [
            r"systematic review", r"systematic literature review"
        ],
        StudyDesign.RCT: [
            r"randomized", r"randomised", r"RCT", r"phase [IViv123]+",
            r"double-blind", r"placebo-controlled", r"clinical trial"
        ],
        StudyDesign.COHORT: [
            r"cohort study", r"prospective study", r"retrospective study",
            r"longitudinal", r"follow-up study"
        ],
        StudyDesign.CASE_CONTROL: [
            r"case-control", r"case control"
        ],
        StudyDesign.CROSS_SECTIONAL: [
            r"cross-sectional", r"cross sectional", r"prevalence study"
        ],
        StudyDesign.CASE_SERIES: [
            r"case series", r"consecutive patients", r"patient series"
        ],
        StudyDesign.CASE_REPORT: [
            r"case report", r"case presentation"
        ],
        StudyDesign.IN_VITRO: [
            r"in vitro", r"cell line", r"mouse model", r"animal model",
            r"xenograft", r"preclinical"
        ]
    }

    # Sample size extraction patterns
    SAMPLE_PATTERNS = [
        r"n\s*=\s*(\d+)",
        r"(\d+)\s*patients",
        r"(\d+)\s*participants",
        r"(\d+)\s*subjects",
        r"sample size of\s*(\d+)",
        r"enrolled\s*(\d+)",
        r"included\s*(\d+)",
        r"(\d+)\s*individuals",
    ]

    # Effect size patterns
    EFFECT_PATTERNS = {
        "HR": r"(?:hazard ratio|HR)\s*[=:]?\s*([\d.]+)\s*(?:\(|,|;)?\s*(?:95%?\s*CI)?[:\s]*([\d.]+)\s*[-–to]+\s*([\d.]+)",
        "OR": r"(?:odds ratio|OR)\s*[=:]?\s*([\d.]+)\s*(?:\(|,|;)?\s*(?:95%?\s*CI)?[:\s]*([\d.]+)\s*[-–to]+\s*([\d.]+)",
        "RR": r"(?:relative risk|risk ratio|RR)\s*[=:]?\s*([\d.]+)\s*(?:\(|,|;)?\s*(?:95%?\s*CI)?[:\s]*([\d.]+)\s*[-–to]+\s*([\d.]+)",
    }

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        """Lazy load LLM using unified helper (Vertex AI or API key)."""
        if self._llm is None:
            from .llm_helper import get_langchain_llm
            self._llm = get_langchain_llm(temperature=0.1)  # Low temperature for factual extraction
        return self._llm

    def extract_all(self, title: str, abstract: str, full_text: str = None) -> PaperInsights:
        """Extract all insights from a paper."""
        text = full_text or abstract

        return PaperInsights(
            bottom_line=self.extract_bottom_line(title, abstract),
            quality=self.extract_quality(title, abstract, text),
            key_outcomes=self.extract_outcomes(text),
            population=self.extract_population(text)
        )

    def extract_bottom_line(self, title: str, abstract: str) -> Optional[BottomLine]:
        """
        Extract the "bottom line" - a one-sentence clinical/research takeaway.

        This is the most important insight: "Why should I care about this paper?"
        """
        if not abstract:
            return None

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", """논문에서 핵심 메시지를 한 문장으로 요약하는 BOTTOM LINE을 추출합니다.

작업: 가장 중요한 발견이나 결론을 담은 한 문장을 추출하세요.

가이드라인:
1. 반드시 한국어로 작성 (30단어 이내)
2. 주요 발견, 결론에 집중
3. 구체적인 숫자/퍼센트가 있으면 포함 (예: "90%의 환자에서", "2배 증가")
4. 리뷰 논문: 핵심 통찰이나 종합 요약
5. 메커니즘 연구: 발견된 내용 설명
6. 임상 연구: 주요 결과 기술
7. 모호한 초록이라도 반드시 bottom_line 제공

연구 유형별 예시:
- 임상: "펨브롤리주맙+화학요법이 진행성 위암에서 전체 생존율을 개선함 (HR 0.65)"
- 메커니즘: "KRAS 돌연변이가 RAF-MEK-ERK 신호전달 활성화를 통해 췌장암 진행을 유도함"
- 리뷰: "현재 근거는 HER2+ 유방암의 1차 치료로 병용요법을 지지함"
- 바이오마커: "높은 PD-L1 발현(>50%)이 NSCLC에서 면역치료 반응을 예측함"
- 역학: "50세 미만 성인에서 대장암 발생률이 지난 10년간 50% 증가함"

JSON만 출력 (마크다운 없이):
{{
  "bottom_line": "한국어로 작성된 핵심 요약 문장",
  "clinical_relevance": "High|Medium|Low",
  "action_type": "Treatment|Diagnosis|Mechanism|Prevention|Prognosis|Epidemiology|Review|Biomarker"
}}"""),
            ("human", """제목: {title}

초록: {abstract}

Bottom line을 추출하세요 (JSON만 응답):""")
        ])

        try:
            chain = prompt | self.llm
            result = chain.invoke({"title": title, "abstract": abstract})

            import json
            content = result.content.strip()

            # Clean up markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # Try to find JSON object in content
            content = content.strip()
            if not content.startswith("{"):
                # Try to find JSON object
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    content = content[start:end]

            data = json.loads(content)

            bottom_line_text = data.get("bottom_line", "")

            # Validate that we got something useful
            if not bottom_line_text or len(bottom_line_text) < 10:
                # Fallback: extract from title if bottom_line failed
                bottom_line_text = title if len(title) < 100 else title[:100] + "..."

            return BottomLine(
                summary=bottom_line_text,
                clinical_relevance=data.get("clinical_relevance", "Medium"),
                action_type=data.get("action_type", "Unknown"),
                confidence=0.8 if data.get("bottom_line") else 0.5
            )
        except Exception as e:
            print(f"Bottom line extraction error: {e}")
            # Fallback: return title-based bottom line
            return BottomLine(
                summary=title[:100] if len(title) > 100 else title,
                clinical_relevance="Medium",
                action_type="Unknown",
                confidence=0.3
            )

    def extract_quality(self, title: str, abstract: str, text: str = None) -> StudyQuality:
        """
        Extract study quality indicators.

        Returns a quality score (0-10) based on:
        - Study design (0-10 base)
        - Sample size adequacy
        - Methodology indicators
        """
        full_text = f"{title} {abstract} {text or ''}"
        text_lower = full_text.lower()

        # Detect study design
        design = self._detect_study_design(text_lower)

        # Extract sample size
        sample_size = self._extract_sample_size(text_lower)

        # Calculate quality score
        quality_score = design.evidence_score

        # Adjust for sample size
        if sample_size:
            if sample_size >= 1000:
                quality_score = min(10, quality_score + 1)
            elif sample_size >= 100:
                quality_score = min(10, quality_score + 0.5)
            elif sample_size < 30:
                quality_score = max(0, quality_score - 1)

        # Detect strengths
        strengths = []
        if "multicenter" in text_lower or "multi-center" in text_lower:
            strengths.append("Multicenter study")
            quality_score = min(10, quality_score + 0.5)
        if "double-blind" in text_lower:
            strengths.append("Double-blind design")
            quality_score = min(10, quality_score + 0.5)
        if "intention-to-treat" in text_lower or "itt" in text_lower:
            strengths.append("Intention-to-treat analysis")
        if "pre-registered" in text_lower or "registered" in text_lower:
            strengths.append("Pre-registered")

        # Detect limitations
        limitations = []
        if "single-center" in text_lower or "single center" in text_lower:
            limitations.append("Single-center")
        if "retrospective" in text_lower and design != StudyDesign.COHORT:
            limitations.append("Retrospective design")
        if "small sample" in text_lower or (sample_size and sample_size < 50):
            limitations.append("Small sample size")
        if "selection bias" in text_lower:
            limitations.append("Potential selection bias noted")

        # Assess bias risk
        bias_risk = "Medium"
        if design in [StudyDesign.RCT, StudyDesign.META_ANALYSIS] and "double-blind" in text_lower:
            bias_risk = "Low"
        elif design in [StudyDesign.CASE_REPORT, StudyDesign.CASE_SERIES]:
            bias_risk = "High"
        elif "bias" in text_lower or "confound" in text_lower:
            bias_risk = "Medium" if "adjusted" in text_lower else "High"

        return StudyQuality(
            design=design,
            sample_size=sample_size,
            quality_score=round(quality_score, 1),
            strengths=strengths,
            limitations=limitations,
            bias_risk=bias_risk
        )

    def extract_outcomes(self, text: str) -> List[KeyOutcome]:
        """Extract key outcomes with effect sizes."""
        if not text:
            return []

        outcomes = []
        text_lower = text.lower()

        # Common outcome names to look for
        outcome_keywords = {
            "overall survival": "OS",
            "progression-free survival": "PFS",
            "disease-free survival": "DFS",
            "response rate": "ORR",
            "complete response": "CR",
            "mortality": "Mortality",
            "recurrence": "Recurrence"
        }

        # Try to extract HR, OR, RR with confidence intervals
        for metric_type, pattern in self.EFFECT_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    ci_lower = float(match.group(2)) if match.group(2) else None
                    ci_upper = float(match.group(3)) if match.group(3) else None

                    # Find associated outcome name
                    context_start = max(0, match.start() - 100)
                    context = text[context_start:match.start()].lower()

                    outcome_name = "Primary Outcome"
                    for keyword, abbrev in outcome_keywords.items():
                        if keyword in context:
                            outcome_name = abbrev
                            break

                    # Generate interpretation
                    interpretation = self._interpret_effect(metric_type, value, ci_lower, ci_upper)

                    outcomes.append(KeyOutcome(
                        outcome_name=outcome_name,
                        metric_type=metric_type,
                        value=value,
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                        interpretation=interpretation
                    ))
                except (ValueError, IndexError):
                    continue

        return outcomes[:3]  # Return top 3 outcomes

    def extract_population(self, text: str) -> Optional[PopulationInfo]:
        """Extract study population information."""
        if not text:
            return None

        text_lower = text.lower()

        # Extract sample size
        total_n = self._extract_sample_size(text_lower)

        # Extract age info
        age_range = ""
        age_match = re.search(r"(?:median age|mean age|age)[:\s]*(\d+)(?:\s*[-–to]\s*(\d+))?(?:\s*years)?", text_lower)
        if age_match:
            if age_match.group(2):
                age_range = f"{age_match.group(1)}-{age_match.group(2)}"
            else:
                age_range = f"median {age_match.group(1)}"

        # Extract gender info
        female_percent = None
        gender_match = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:female|women)", text_lower)
        if gender_match:
            female_percent = float(gender_match.group(1))

        # Try to find condition
        condition = ""
        # Look for common disease patterns
        disease_patterns = [
            r"patients with ([\w\s]+(?:cancer|carcinoma|disease|syndrome))",
            r"([\w\s]+(?:cancer|carcinoma)) patients",
            r"diagnosed with ([\w\s]+)"
        ]
        for pattern in disease_patterns:
            match = re.search(pattern, text_lower)
            if match:
                condition = match.group(1).strip().title()
                break

        # Setting detection
        setting = "Not specified"
        if "multicenter" in text_lower or "multi-center" in text_lower:
            setting = "Multicenter"
        elif "single-center" in text_lower or "single center" in text_lower:
            setting = "Single-center"

        return PopulationInfo(
            total_n=total_n,
            condition=condition,
            age_range=age_range,
            female_percent=female_percent,
            inclusion_criteria=[],  # Would need more complex extraction
            exclusion_criteria=[],
            setting=setting
        )

    def _detect_study_design(self, text: str) -> StudyDesign:
        """Detect study design from text."""
        for design, patterns in self.DESIGN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return design
        return StudyDesign.UNKNOWN

    def _extract_sample_size(self, text: str) -> Optional[int]:
        """Extract sample size from text."""
        for pattern in self.SAMPLE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    n = int(match.group(1))
                    if 1 <= n <= 1000000:  # Reasonable range
                        return n
                except ValueError:
                    continue
        return None

    def _interpret_effect(self, metric_type: str, value: float,
                          ci_lower: float = None, ci_upper: float = None) -> str:
        """Generate human-readable interpretation of effect size."""
        if metric_type in ["HR", "OR", "RR"]:
            if value < 1:
                reduction = (1 - value) * 100
                return f"{reduction:.0f}% reduced risk"
            else:
                increase = (value - 1) * 100
                return f"{increase:.0f}% increased risk"
        return ""


# Singleton instance
_insights_extractor: Optional[PaperInsightsExtractor] = None


def get_insights_extractor() -> PaperInsightsExtractor:
    """Get or create the insights extractor."""
    global _insights_extractor
    if _insights_extractor is None:
        _insights_extractor = PaperInsightsExtractor()
    return _insights_extractor


def extract_paper_insights(title: str, abstract: str, full_text: str = None) -> Dict[str, Any]:
    """
    Main entry point for extracting paper insights.

    Returns a dictionary suitable for API response.
    """
    extractor = get_insights_extractor()
    insights = extractor.extract_all(title, abstract, full_text)

    return {
        "bottom_line": {
            "summary": insights.bottom_line.summary if insights.bottom_line else None,
            "clinical_relevance": insights.bottom_line.clinical_relevance if insights.bottom_line else None,
            "action_type": insights.bottom_line.action_type if insights.bottom_line else None,
        } if insights.bottom_line else None,
        "quality": {
            "design": insights.quality.design.label if insights.quality else None,
            "design_score": insights.quality.design.evidence_score if insights.quality else None,
            "sample_size": insights.quality.sample_size if insights.quality else None,
            "quality_score": insights.quality.quality_score if insights.quality else None,
            "quality_label": insights.quality.quality_label if insights.quality else None,
            "bias_risk": insights.quality.bias_risk if insights.quality else None,
            "strengths": insights.quality.strengths if insights.quality else [],
            "limitations": insights.quality.limitations if insights.quality else [],
        } if insights.quality else None,
        "key_outcomes": [
            {
                "outcome": o.outcome_name,
                "metric": o.metric_type,
                "value": o.value,
                "ci": f"{o.ci_lower}-{o.ci_upper}" if o.ci_lower and o.ci_upper else None,
                "interpretation": o.interpretation
            }
            for o in insights.key_outcomes
        ],
        "population": {
            "n": insights.population.total_n if insights.population else None,
            "condition": insights.population.condition if insights.population else None,
            "age": insights.population.age_range if insights.population else None,
            "female_percent": insights.population.female_percent if insights.population else None,
            "setting": insights.population.setting if insights.population else None,
        } if insights.population else None,
    }
