# BIO 데일리 브리핑 v3.0

AI 기반 멀티소스 바이오/의학 뉴스레터 시스템 (평일 오전 6시 자동 생성)

## 주요 특징

- **멀티소스 수집**: FDA, ClinicalTrials.gov, bioRxiv/medRxiv, PubMed 통합
- **자동 트렌드 추출**: 논문에서 핫 키워드 자동 분석 (🔥⬆️⬇️➡️)
- **AI 뉴스 변환**: Claude API로 논문을 읽기 쉬운 뉴스로 변환
- **신문 스타일**: PDF 다운로드 지원, 반응형 HTML
- **자동 스케줄링**: launchd (macOS) / cron 지원

## 데이터 소스

| 소스 | 내용 | 수집 주기 |
|------|------|-----------|
| **FDA** | 신약 승인, 안전성 경고, 리콜 | 72시간 |
| **ClinicalTrials.gov** | Phase 3 결과, 신규 임상, 중단 | 30일 |
| **bioRxiv/medRxiv** | 프리프린트 | 3일 |
| **PubMed** | Peer-reviewed 논문, High-impact journals | 2일 |

## 프로젝트 구조

```
bio-daily-briefing/
├── src/
│   ├── __init__.py
│   ├── scheduler.py           # 스케줄러 (⚠️ 데이터 형식 변환 포함)
│   ├── aggregator.py          # 멀티소스 통합 (dict 형식 반환)
│   ├── newsletter_generator.py # HTML 생성 (list 형식 필요!)
│   ├── pubmed_fetcher.py      # PubMed 논문 수집
│   ├── trend_analyzer.py      # 트렌드/키워드 분석
│   ├── ai_summarizer.py       # Claude 요약
│   └── sources/
│       ├── fda_fetcher.py     # FDA 뉴스 수집
│       ├── clinicaltrials_fetcher.py  # 임상시험 수집
│       └── biorxiv_fetcher.py # 프리프린트 수집
├── templates/
│   └── newsletter_template.html  # 신문 스타일 템플릿
├── config/
│   ├── .env.example           # 환경변수 템플릿
│   └── subscribers.json       # 구독자 목록
├── output/
│   ├── html/                  # 생성된 HTML
│   ├── *.json                 # API용 JSON
│   └── history/               # 트렌드 히스토리
├── requirements.txt
└── README.md
```

## ⚠️ 개발자 주의사항

**데이터 형식 불일치 문제 (반드시 숙지!)**

`aggregator.py`는 dict 형식을 반환하지만, `newsletter_generator.py`는 list 형식을 기대합니다.
**`scheduler.py`에서 반드시 변환해야 합니다!**

```python
# ❌ Wrong - newsletter_generator에서 KeyError 발생
newsletter_data = {
    "clinical_trials": agg_dict.get("clinical_trials", {}),  # dict 형식
    "research": agg_dict.get("research", {}),  # dict 형식
}

# ✅ Correct - list 형식으로 변환
clinical_list = []
for item in ct_dict.get("phase3_results", [])[:3]:
    clinical_list.append({
        "type": "phase3_completed",
        "title": item.get("title", ""),
        "description": item.get("summary", "")
    })

newsletter_data = {
    "clinical_trials": clinical_list,  # list 형식!
}
```

검증 로그가 빈 섹션을 경고합니다:
```
⚠️ WARNING: clinical_list is empty! Check ClinicalTrials fetcher.
[Data Validation] regulatory: 5, clinical: 5, research: 6
```

## 설치

### 1. 환경 설정

```bash
cd bio-daily-briefing

# Python 가상환경
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp config/.env.example config/.env
# .env 파일 편집
```

### 2. 환경변수 (.env)

```env
# 필수: AI API 키 (하나 이상)
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=your-google-api-key

# 필수: 이메일 설정
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# 선택: 스케줄 설정
GENERATE_TIME=06:00
SEND_TIME=08:00
```

## 사용법

### 설정 확인

```bash
python -m src.scheduler --check-config
```

### 즉시 실행 (테스트)

```bash
python -m src.scheduler --run-now
```

### 뉴스레터 생성만

```bash
python -m src.scheduler --generate-only
```

### 데몬 모드 (자동 스케줄)

```bash
python -m src.scheduler --daemon
```

### 구독자 관리

```bash
# 추가
python -m src.scheduler --add-subscriber user@example.com

# 삭제
python -m src.scheduler --remove-subscriber user@example.com

# 목록
python -m src.scheduler --list-subscribers
```

### 테스트 이메일

```bash
python -m src.scheduler --test-email your@email.com
```

## Docker 배포

```bash
cd deploy

# 데몬 모드
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 1회 실행
docker-compose --profile run-once run bio-briefing-run
```

## 트렌드 분석 방식

### 키워드 추출

1. **MeSH Terms**: PubMed 공식 의학 주제어 (높은 가중치)
2. **Author Keywords**: 저자 지정 키워드
3. **Known Phrases**: 바이오 분야 주요 용어 (CAR-T, CRISPR, PD-1 등)
4. **Title/Abstract**: TF-IDF 기반 중요 단어

### 트렌드 비교

```
오늘 키워드 카운트 vs 어제 vs 일주일 전

🔥 Hot: +50% 이상
⬆️ Rising: +10% 이상
➡️ Stable: -10% ~ +10%
⬇️ Declining: -10% 이하
```

### 히스토리 저장

```
output/history/
├── trends_20250105.json
├── trends_20250104.json
└── ...
```

## 뉴스레터 구성

```
📰 BIO 데일리 브리핑 #127

🔥 오늘의 핫 키워드
   - 🔥 CAR-T (15건, +45%)
   - ⬆️ CRISPR (12건, +12%)
   - ➡️ PD-1 (8건, +5%)

📌 CAR-T
   💬 [Hook 질문]
   [뉴스 제목]
   [본문 - 배경 → 발견 → 의미]
   💡 인사이트
   📄 출처

📌 CRISPR
   ...

💬 에디터 코멘트
   [오늘의 핵심 정리]

📅 내일 예고
```

## API 참조

### PubMedFetcher

```python
from src import PubMedFetcher

fetcher = PubMedFetcher()
papers = await fetcher.fetch_recent_papers(
    max_results=100,
    days=7
)
```

### TrendAnalyzer

```python
from src import TrendAnalyzer

analyzer = TrendAnalyzer()
trends = analyzer.get_hot_topics(papers, top_n=5)

for trend in trends:
    print(f"{trend.trend_indicator} {trend.keyword}: {trend.count}")
```

### AISummarizer

```python
from src import AISummarizer

summarizer = AISummarizer(language="ko")
article = summarizer.summarize_paper(paper)
```

### NewsletterGenerator

```python
from src import NewsletterGenerator

generator = NewsletterGenerator()
html = generator.generate_html(trends, articles_by_trend, editor_comment)
generator.save_html(html)
```

## 문제 해결

### "No LLM API key configured"

`.env` 파일에 `ANTHROPIC_API_KEY` 또는 `GOOGLE_API_KEY` 설정

### "Email send error"

1. SMTP 설정 확인
2. Gmail의 경우 [앱 비밀번호](https://support.google.com/accounts/answer/185833) 사용
3. 2단계 인증 활성화 필요

### "No papers found"

1. 검색 기간 늘리기: `LOOKBACK_DAYS=14`
2. PubMed API 상태 확인
3. 네트워크 연결 확인

## 라이선스

MIT License
