# BioInsight AI - 배포 가이드

## 📋 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [필수 API 키](#필수-api-키)
3. [로컬 실행](#로컬-실행)
4. [GCP Cloud Run 배포](#gcp-cloud-run-배포)
5. [Docker 배포](#docker-배포)

---

## 시스템 요구사항

### 필수 소프트웨어

| 소프트웨어 | 버전 | 용도 |
|-----------|------|------|
| Python | 3.11+ | 백엔드 |
| Node.js | 20+ | 프론트엔드 |
| R | 4.3+ | DESeq2 (RNA-seq) |

### 서버 사양

| 항목 | 최소 | 권장 |
|------|------|------|
| RAM | 4GB | 8GB |
| 디스크 | 5GB | 10GB |
| CPU | 2 core | 4 core |

---

## 필수 API 키

### 발급 필요 (배포 전 준비)

| API | 발급처 | 용도 | 필수 |
|-----|--------|------|------|
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | 기본 LLM | ✅ |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/account/keys) | RAG 해석 | ✅ |
| Google AI | [aistudio.google.com](https://aistudio.google.com/app/apikey) | Paper Explainer | ⚠️ 권장 |
| NCBI | [ncbi.nlm.nih.gov](https://www.ncbi.nlm.nih.gov/account/settings/) | PubMed 검색 | ⚠️ 권장 |

### 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# 필수 키 입력
OPENAI_API_KEY=sk-proj-xxx...
ANTHROPIC_API_KEY=sk-ant-xxx...
GOOGLE_API_KEY=AIzaSy...
NCBI_API_KEY=xxx...
```

---

## 로컬 실행 (macOS/Linux)

### 1. 저장소 클론

```bash
# 홈 디렉토리에서 시작
cd ~

git clone https://github.com/jjyun78912/VectorDB_BioInsight.git
cd VectorDB_BioInsight
```

### 2. Python 환경

```bash
# ⚠️ 반드시 프로젝트 루트 디렉토리에서 실행
# 현재 위치 확인: pwd 결과가 ~/VectorDB_BioInsight 여야 함
pwd
# /Users/yourname/VectorDB_BioInsight

# python3 명령어 사용 (시스템에 따라 python일 수도 있음)
python3 --version  # Python 3.11 이상인지 확인

# 가상환경 생성 (프로젝트 루트에 .venv 폴더 생성됨)
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# 활성화 확인 - 프롬프트 앞에 (.venv) 표시됨
# (.venv) user@hostname:~/VectorDB_BioInsight$

# pip 업그레이드 후 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
# 260126 07:34 여까지 작업함
pip install -r requirements-rnaseq.txt  # RNA-seq 분석용
```

### 3. Frontend 환경

```bash
# ⚠️ 프로젝트 루트에서 frontend/react_app으로 이동
cd ~/VectorDB_BioInsight/frontend/react_app

npm install
```

### 4. 환경 변수

```bash
# ⚠️ 프로젝트 루트로 이동
cd ~/VectorDB_BioInsight

cp .env.example .env

# .env 파일 편집하여 API 키 입력
vim .env  # 또는 nano .env
```

### 5. 서버 실행

```bash
# 터미널 1: Backend (포트 8000)
cd ~/VectorDB_BioInsight
source .venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000

# 터미널 2: Frontend (포트 5173)
cd ~/VectorDB_BioInsight/frontend/react_app
npm run dev
```

### 접속

- Frontend: http://localhost:5173
- API 문서: http://localhost:8000/docs

---

## GCP Compute Engine 배포 (VM)

Docker 없이 직접 VM에 배포하는 방법입니다.

### 1. VM 인스턴스 생성

- OS: Debian 12 (Bookworm)
- 머신 유형: e2-medium (2 vCPU, 4GB) 이상 권장
- 디스크: 20GB 이상
- 방화벽: HTTP(80), HTTPS(443) 허용

### 2. 서버 환경 설정

```bash
# SSH 접속 후 스크립트 다운로드
cd ~
curl -O https://raw.githubusercontent.com/jjyun78912/VectorDB_BioInsight/main/docs/deploy/setup-server.sh
chmod +x setup-server.sh

# 실행 (Python, Node.js, R, Nginx 설치)
./setup-server.sh

# ⚠️ 설치 완료 후 반드시 실행 (pyenv 로드)
source ~/.bashrc

# 설치 확인
python --version   # Python 3.11.9
node --version     # v20.x.x
R --version        # R version 4.x.x
```

### 3. 앱 배포

```bash
# 배포 스크립트 다운로드 및 실행
curl -O https://raw.githubusercontent.com/jjyun78912/VectorDB_BioInsight/main/docs/deploy/deploy-app.sh
chmod +x deploy-app.sh
./deploy-app.sh
```

### 4. API 키 설정

```bash
vim /opt/bioinsight/VectorDB_BioInsight/.env
# OPENAI_API_KEY, ANTHROPIC_API_KEY 등 입력
```

### 5. 데이터 업로드 (로컬에서)

```bash
# 로컬 PC에서 실행
scp -r chroma_db/ username@서버IP:/opt/bioinsight/VectorDB_BioInsight/
scp -r models/ username@서버IP:/opt/bioinsight/VectorDB_BioInsight/
```

### 6. 서비스 시작

```bash
sudo systemctl start bioinsight-backend
sudo systemctl status bioinsight-backend

# 로그 확인
sudo journalctl -u bioinsight-backend -f
```

### 7. 접속 테스트

```bash
# 서버에서
curl http://localhost/health

# 브라우저에서
http://서버외부IP
```

---

## GCP Cloud Run 배포

### 사전 준비

```bash
# gcloud CLI 설치 및 로그인
brew install google-cloud-sdk
gcloud auth login

# 프로젝트 설정
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="asia-northeast3"  # 서울
```

### 배포 명령어

```bash
cd deploy/gcp

# 1. GCP 설정 (최초 1회)
./deploy.sh setup

# 2. 데이터 업로드 (chroma_db, models)
./deploy.sh upload-data

# 3. 빌드 및 배포
./deploy.sh all

# 4. 상태 확인
./deploy.sh status
```

### 예상 비용

| 서비스 | 비용 |
|--------|------|
| Cloud Run (Backend) | ~$20-30/월 |
| Cloud Run (Frontend) | ~$5-10/월 |
| Cloud Storage | ~$1/월 |
| **총계** | **~$25-45/월** |

---

## Docker 배포

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# R 설치 (DESeq2용)
RUN apt-get update && apt-get install -y r-base

# Python 패키지
COPY requirements.txt .
RUN pip install -r requirements.txt

# 데이터 복사
COPY data/chroma_db/ /app/data/chroma_db/
COPY models/rnaseq/ /app/models/rnaseq/

# 소스 코드
COPY . .

# 환경 변수
ENV CHROMA_PERSIST_DIR=/app/data/chroma_db

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 빌드 및 실행

```bash
# 빌드
docker build -t bioinsight-backend .

# 실행
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  bioinsight-backend
```

---

## 배포 체크리스트

### 배포 전

- [ ] API 키 발급 완료 (OpenAI, Anthropic)
- [ ] `.env` 파일 설정
- [ ] `data/chroma_db/` 데이터 준비 (2.2GB)
- [ ] `models/rnaseq/` 모델 파일 준비 (~200MB)

### 배포 후

- [ ] 백엔드 헬스체크: `curl http://server:8000/health`
- [ ] 프론트엔드 접속 확인
- [ ] 검색 기능 테스트
- [ ] RNA-seq 업로드 테스트

---

## 문제 해결

### 백엔드 시작 실패

```bash
# 로그 확인
docker logs bioinsight-backend

# 포트 충돌 확인
lsof -i :8000
```

### ChromaDB 로드 실패

```bash
# 데이터 폴더 확인
ls -la data/chroma_db/

# 권한 확인
chmod -R 755 data/chroma_db/
```

### API 키 오류

```bash
# 환경 변수 확인
echo $OPENAI_API_KEY

# .env 로드 확인
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY')[:10])"
```

---

*Last Updated: 2026-01-26*
