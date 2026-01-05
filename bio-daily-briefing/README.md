# BIO 데일리 브리핑 v2.0

AI 기반 자동 트렌드 분석 바이오/의학 뉴스레터 시스템

## 주요 특징

- **자동 트렌드 추출**: 고정 카테고리 없이 논문에서 핫 키워드 자동 분석
- **트렌드 비교**: 전일/전주 대비 증감율 자동 계산 (🔥⬆️⬇️➡️)
- **AI 뉴스 변환**: Claude/Gemini로 논문을 읽기 쉬운 뉴스로 변환
- **HTML 뉴스레터**: 이메일 클라이언트 호환 반응형 디자인
- **자동 발송**: 매일 지정 시간에 구독자에게 발송

## 프로젝트 구조

```
bio-daily-briefing/
├── src/
│   ├── __init__.py
│   ├── pubmed_fetcher.py      # PubMed 논문 수집
│   ├── trend_analyzer.py      # 트렌드/키워드 분석 (핵심)
│   ├── ai_summarizer.py       # Claude/Gemini 요약
│   ├── newsletter_generator.py # HTML 이메일 생성
│   └── scheduler.py           # 자동화 스케줄러
├── config/
│   ├── .env.example           # 환경변수 템플릿
│   └── subscribers.json       # 구독자 목록
├── output/                    # 생성된 뉴스레터
│   └── history/               # 트렌드 히스토리
├── deploy/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
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
