"""
뉴스레터 HTML 생성기 v3 - 신문 스타일
멀티소스 + PDF 다운로드 지원

Features:
- Deep Plum 컬러 (#4C1D95 → #5B21B6)
- Noto Serif KR + Noto Sans KR 폰트
- 신문 스타일 레이아웃 (2단 컬럼, 카드 그리드)
- PDF 다운로드 (html2pdf.js)
"""

from jinja2 import Environment, FileSystemLoader
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import os
import json


class NewsletterGenerator:
    """신문 스타일 뉴스레터 HTML 생성기"""

    def __init__(self, template_dir: Optional[str] = None):
        """
        Args:
            template_dir: 템플릿 디렉토리 경로 (기본: templates/)
        """
        if template_dir is None:
            # 현재 파일 기준 상대 경로
            base_dir = Path(__file__).parent.parent
            template_dir = base_dir / "templates"

        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False  # HTML을 그대로 렌더링하기 위해
        )
        self.template = self.env.get_template("newsletter_template.html")

    def generate(self, data: dict, issue_number: int) -> str:
        """뉴스레터 HTML 생성

        Args:
            data: 뉴스레터 데이터 딕셔너리
            issue_number: 발행 호수

        Returns:
            렌더링된 HTML 문자열
        """
        now = datetime.now()

        context = {
            # 메타 정보
            "issue_number": issue_number,
            "date_en": now.strftime("%A, %B %d, %Y"),
            "date_kr": now.strftime("%Y년 %m월 %d일"),
            "date_file": now.strftime("%Y-%m-%d"),
            "total_papers": data.get("total_papers", 0),

            # 헤드라인
            "headline_title": data.get("headline", {}).get("title", "오늘의 주요 바이오 뉴스"),
            "headline_summary": data.get("headline", {}).get("summary", ""),
            "headline_why": data.get("headline", {}).get("why_important", ""),

            # 규제 뉴스
            "regulatory_news": self._format_regulatory(data.get("regulatory", [])),

            # 임상시험
            "clinical_trials": self._format_clinical(data.get("clinical_trials", [])),

            # 연구 논문
            "research_papers": self._format_research(data.get("research", [])),

            # 핫토픽
            "hot_topics": self._format_topics(data.get("hot_topics", [])),

            # 에디터 코멘트
            "editor_quote": data.get("editor", {}).get("quote", ""),
            "editor_note": data.get("editor", {}).get("note", "")
        }

        return self.template.render(**context)

    def _format_regulatory(self, items: list) -> list:
        """규제 뉴스 포맷팅"""
        result = []
        badge_map = {
            "approved": ("approved", "✓ 승인"),
            "pending": ("pending", "⏳ 심사중"),
            "warning": ("hot", "⚠️ 경고"),
            "rejected": ("hot", "✗ 거절"),
            "fast_track": ("new", "🚀 패스트트랙"),
            "breakthrough": ("phase3", "💎 혁신신약"),
            "safety": ("hot", "⚠️ 안전성")
        }

        for item in items[:3]:  # 최대 3개
            status = item.get("status", "pending")
            badge_type, badge_text = badge_map.get(status, ("pending", "📋 기타"))

            result.append({
                "badge_type": badge_type,
                "badge_text": badge_text,
                "title": item.get("title", ""),
                "description": item.get("description", "")
            })
        return result

    def _format_clinical(self, items: list) -> list:
        """임상시험 포맷팅"""
        result = []
        badge_map = {
            "phase3_positive": ("approved", "Phase 3 ✓"),
            "phase3_negative": ("hot", "Phase 3 ✗"),
            "phase3_completed": ("phase3", "Phase 3 완료"),
            "new_trial": ("new", "신규 임상"),
            "stopped": ("hot", "중단"),
            "phase2": ("pending", "Phase 2"),
            "phase1": ("pending", "Phase 1")
        }

        for item in items[:3]:  # 최대 3개
            trial_type = item.get("type", "phase3_completed")
            badge_type, badge_text = badge_map.get(trial_type, ("phase3", "Phase 3"))

            result.append({
                "badge_type": badge_type,
                "badge_text": badge_text,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "patients": item.get("patients"),  # 환자 수 (예: "1,200명")
                "disease": item.get("disease")  # 질환 (예: "흑색종")
            })
        return result

    def _format_research(self, items: list) -> list:
        """연구 논문 포맷팅"""
        result = []
        journal_map = {
            "nature": "nature",
            "science": "nature",
            "cell": "cell",
            "nejm": "nejm",
            "new england journal of medicine": "nejm",
            "lancet": "nejm",
            "the lancet": "nejm",
            "jama": "nejm",
            "biorxiv": "biorxiv",
            "medrxiv": "biorxiv",
            "nature medicine": "nature",
            "nature genetics": "nature",
            "nature biotechnology": "nature",
            "cell metabolism": "cell",
            "cell stem cell": "cell",
            "cancer cell": "cell"
        }

        for item in items[:4]:  # 최대 4개 (2x2 그리드)
            journal = item.get("journal", "")
            journal_lower = journal.lower()
            journal_class = journal_map.get(journal_lower, "nature")

            # 저널명 표시 형식
            display_journal = journal.upper()
            if len(display_journal) > 15:
                display_journal = display_journal[:15] + "..."

            result.append({
                "journal": display_journal,
                "journal_class": journal_class,
                "title": item.get("title", ""),
                "insight": item.get("insight", item.get("summary", ""))
            })
        return result

    def _format_topics(self, items: list) -> list:
        """핫토픽 포맷팅"""
        result = []
        event_type_map = {
            "approval": "approved",
            "first_approval": "hot",
            "mna": "pending",
            "phase3": "phase3",
            "breakthrough": "new",
            "warning": "hot"
        }

        for idx, item in enumerate(items[:5], start=1):  # 최대 5개
            change = item.get("change", 0)

            # 변동 타입 결정
            if change > 20:
                change_type = "up"
                change_text = f"🔥 +{change}%"
            elif change > 0:
                change_type = "up"
                change_text = f"↑ +{change}%"
            elif change < 0:
                change_type = "down"
                change_text = f"↓ {change}%"
            else:
                change_type = "same"
                change_text = "→ 유지"

            # 순위 변동 정보
            prev_rank = item.get("prev_rank")  # 전주 순위 (없으면 None = 신규)
            if prev_rank is None:
                rank_change_type = "new"
                rank_display = f"#{idx} 🆕"
            elif prev_rank > idx:
                rank_diff = prev_rank - idx
                rank_change_type = "up"
                rank_display = f"#{idx} ▲{rank_diff}"
            elif prev_rank < idx:
                rank_diff = idx - prev_rank
                rank_change_type = "down"
                rank_display = f"#{idx} ▼{rank_diff}"
            else:
                rank_change_type = "same"
                rank_display = f"#{idx} —"

            result.append({
                "name": item.get("name", ""),
                "count": item.get("count", 0),
                "change_type": change_type,
                "change_text": change_text,
                "event": item.get("event"),
                "event_type": event_type_map.get(item.get("event_type"), "approved"),
                "rank_change_type": rank_change_type,
                "rank_display": rank_display
            })
        return result

    def save(self, html: str, issue_number: int, output_dir: Optional[str] = None) -> str:
        """HTML 파일 저장

        Args:
            html: 렌더링된 HTML 문자열
            issue_number: 발행 호수
            output_dir: 출력 디렉토리 (기본: output/)

        Returns:
            저장된 파일 경로
        """
        if output_dir is None:
            base_dir = Path(__file__).parent.parent
            output_dir = base_dir / "output"

        output_dir = Path(output_dir)
        html_dir = output_dir / "html"
        html_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"bio_daily_briefing_{issue_number}_{date_str}.html"
        filepath = html_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        return str(filepath)

    def generate_and_save(self, data: dict, issue_number: int, output_dir: Optional[str] = None) -> str:
        """HTML 생성 및 저장 (편의 메서드)

        Args:
            data: 뉴스레터 데이터
            issue_number: 발행 호수
            output_dir: 출력 디렉토리

        Returns:
            저장된 파일 경로
        """
        html = self.generate(data, issue_number)
        return self.save(html, issue_number, output_dir)


def create_sample_data() -> dict:
    """테스트용 샘플 데이터 생성"""
    return {
        "total_papers": 205,

        "headline": {
            "title": "FDA, 버텍스 CRISPR 겸상적혈구 치료제 최종 승인",
            "summary": "2012년 CRISPR 발견 이후 11년 만에 실제 환자 치료제로 FDA 승인을 획득했습니다. 버텍스와 CRISPR Therapeutics가 공동 개발한 이 치료제는 유전자 편집 기술을 활용해 겸상적혈구병 환자의 조혈모세포를 교정합니다. 이번 승인은 유전자 치료의 새로운 시대를 여는 이정표로 평가받고 있습니다.",
            "why_important": "CRISPR 기술의 첫 FDA 승인 - 유전자 치료 시대의 본격적인 시작"
        },

        "regulatory": [
            {
                "status": "approved",
                "title": "노보 노디스크, 경구용 세마글루타이드 비만 적응증 승인",
                "description": "위고비의 경구 버전이 FDA 승인. 주사 없이 GLP-1 수용체 작용제 복용 가능"
            },
            {
                "status": "pending",
                "title": "릴리, 도나네맙 알츠하이머 치료제 심사 중",
                "description": "FDA 자문위원회 6월 검토 예정. 레카네맙과의 경쟁 주목"
            },
            {
                "status": "fast_track",
                "title": "모더나 개인맞춤 암백신, 패스트트랙 지정",
                "description": "흑색종 대상 mRNA 네오항원 백신. 키트루다 병용 임상 진행 중"
            }
        ],

        "clinical_trials": [
            {
                "type": "phase3_positive",
                "title": "BioNTech 개인맞춤 암백신, 흑색종 재발률 44% 감소",
                "description": "mRNA 네오항원 백신 + 키트루다 병용. 무재발 생존기간 유의미하게 연장",
                "disease": "흑색종",
                "patients": "1,089명"
            },
            {
                "type": "new_trial",
                "title": "알닐람, RNAi 기반 심부전 치료제 Phase 3 시작",
                "description": "ATTR 심근병증 대상. 기존 약물 대비 6개월마다 1회 투여",
                "disease": "ATTR 심근병증",
                "patients": "약 800명 모집 예정"
            },
            {
                "type": "stopped",
                "title": "바이오젠, SMA 유전자치료제 임상 조기 중단",
                "description": "안전성 우려로 환자 등록 중단. 간독성 이슈 조사 중",
                "disease": "척수성 근위축증(SMA)"
            }
        ],

        "research": [
            {
                "journal": "Nature",
                "title": "종양미세환경 리프로그래밍 새 기전 발견",
                "insight": "CAR-T 효능 높이는 병용전략 제시. 면역억제 극복 방안 도출"
            },
            {
                "journal": "NEJM",
                "title": "GLP-1 작용제, 심혈관 사망률 20% 감소",
                "insight": "SELECT 임상 최종 결과. 비만 치료 넘어 심혈관 보호 효과 입증"
            },
            {
                "journal": "Cell",
                "title": "장내 미생물-뇌 축 새로운 신호전달 경로",
                "insight": "파킨슨병 조기 진단 바이오마커 가능성. 미생물 대사체 프로파일링"
            },
            {
                "journal": "bioRxiv",
                "title": "AlphaFold3, 항체-항원 복합체 예측 정확도 향상",
                "insight": "프리프린트 (피어리뷰 전). 신약 개발 가속화 기대"
            }
        ],

        "hot_topics": [
            {
                "name": "GLP-1",
                "count": 45,
                "change": 23,
                "prev_rank": 2,  # 전주 2위 → 이번주 1위 (▲1)
                "event": "경구제 승인",
                "event_type": "approval"
            },
            {
                "name": "CRISPR",
                "count": 38,
                "change": 156,
                "prev_rank": None,  # 신규 진입 (🆕)
                "event": "첫 승인",
                "event_type": "first_approval"
            },
            {
                "name": "CAR-T",
                "count": 32,
                "change": 12,
                "prev_rank": 1,  # 전주 1위 → 이번주 3위 (▼2)
                "event": None,
                "event_type": None
            },
            {
                "name": "mRNA 백신",
                "count": 28,
                "change": 8,
                "prev_rank": 4,  # 전주 4위 → 이번주 4위 (—)
                "event": "암백신 Phase 3",
                "event_type": "phase3"
            },
            {
                "name": "ADC",
                "count": 25,
                "change": -5,
                "prev_rank": 3,  # 전주 3위 → 이번주 5위 (▼2)
                "event": None,
                "event_type": None
            }
        ],

        "editor": {
            "quote": "오늘 최대 이슈는 <strong>CRISPR 치료제 FDA 승인</strong>입니다. 2012년 Jennifer Doudna와 Emmanuelle Charpentier가 CRISPR-Cas9를 발표한 지 11년 만에, 이 기술이 실제 환자 치료에 사용되는 역사적인 순간입니다.",
            "note": "버텍스 주가는 장전 거래에서 8% 상승했으며, CRISPR Therapeutics는 12% 급등했습니다. 유전자 편집 치료제 시대의 서막이 열렸습니다."
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIO 데일리 브리핑 뉴스레터 생성기 (신문 스타일)")
    parser.add_argument("--test", action="store_true", help="샘플 데이터로 테스트 생성")
    parser.add_argument("--issue", type=int, default=1, help="발행 호수")
    parser.add_argument("--output", type=str, help="출력 디렉토리")
    args = parser.parse_args()

    if args.test:
        print("🗞️ BIO 데일리 브리핑 테스트 생성 중...")

        generator = NewsletterGenerator()
        sample_data = create_sample_data()

        filepath = generator.generate_and_save(
            data=sample_data,
            issue_number=args.issue,
            output_dir=args.output
        )

        print(f"✅ 뉴스레터 생성 완료: {filepath}")
        print(f"📂 브라우저에서 열어보세요: file://{os.path.abspath(filepath)}")
    else:
        print("사용법: python -m src.newsletter_generator --test")
        print("옵션:")
        print("  --test     샘플 데이터로 테스트 생성")
        print("  --issue N  발행 호수 지정 (기본: 1)")
        print("  --output   출력 디렉토리 지정")
