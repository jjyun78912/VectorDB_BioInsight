# 🚀 BioInsight AI - GCP Cloud Run 배포 가이드

## 📋 목차

1. [사전 요구사항](#사전-요구사항)
2. [빠른 시작](#빠른-시작)
3. [단계별 가이드](#단계별-가이드)
4. [비용 최적화](#비용-최적화)
5. [문제 해결](#문제-해결)

---

## 사전 요구사항

### 1. GCP 계정 및 프로젝트

1. [Google Cloud Console](https://console.cloud.google.com) 접속
2. 새 프로젝트 생성 또는 기존 프로젝트 선택
3. 결제 계정 연결 (무료 크레딧 $300 사용 가능)

### 2. 로컬 환경 설정

```bash
# gcloud CLI 설치 (macOS)
brew install google-cloud-sdk

# 또는 공식 설치 스크립트
curl https://sdk.cloud.google.com | bash

# Docker 설치 확인
docker --version

# gcloud 로그인
gcloud auth login
gcloud auth configure-docker asia-northeast3-docker.pkg.dev
```

### 3. 환경 변수 설정

```bash
# 프로젝트 ID 설정 (GCP Console에서 확인)
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="asia-northeast3"  # 서울 리전
export GCS_BUCKET_NAME="bioinsight-data"
```

---

## 빠른 시작

```bash
# 1. 배포 디렉토리로 이동
cd deploy/gcp

# 2. 최초 설정 (1회만)
./deploy.sh setup

# 3. 데이터 업로드 (모델, ChromaDB)
./deploy.sh upload-data

# 4. 빌드 및 배포
./deploy.sh all

# 5. 상태 확인
./deploy.sh status
```

배포 완료 후 출력되는 URL로 접속하세요.

---

## 단계별 가이드

### Step 1: GCP 프로젝트 설정

```bash
./deploy.sh setup
```

이 명령어가 수행하는 작업:
- ✅ 필요한 GCP API 활성화 (Cloud Run, Artifact Registry, Secret Manager 등)
- ✅ Docker 이미지 저장소 생성
- ✅ Cloud Storage 버킷 생성
- ✅ `.env` 파일의 API 키를 Secret Manager에 등록

### Step 2: 데이터 업로드

```bash
./deploy.sh upload-data
```

Cloud Storage에 업로드되는 데이터:
- `models/rnaseq/pancancer/` - ML 모델 파일 (~200MB)
- `chroma_db/` - 논문 벡터 DB (~500MB)

### Step 3: Docker 이미지 빌드

```bash
./deploy.sh build
```

⏱️ **예상 시간**: 10-15분 (최초 빌드)

빌드되는 이미지:
- `bioinsight-backend` - FastAPI + R + Python (~3GB)
- `bioinsight-frontend` - React + Nginx (~100MB)

### Step 4: Cloud Run 배포

```bash
./deploy.sh deploy
```

배포 설정:

| 서비스 | CPU | 메모리 | 타임아웃 | 인스턴스 |
|--------|-----|--------|----------|----------|
| Backend | 4 vCPU | 8GB | 60분 | 0-5 |
| Frontend | 1 vCPU | 512MB | 60초 | 0-10 |

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      사용자 요청                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Cloud Run (Frontend)                        │
│                 React + Nginx                                │
│                 https://bioinsight-frontend-xxx.run.app     │
└─────────────────────────┬───────────────────────────────────┘
                          │ API 호출
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Cloud Run (Backend)                         │
│                 FastAPI + R + Python                        │
│                 https://bioinsight-backend-xxx.run.app      │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ RNA-seq     │  │ Paper RAG   │  │ ML Predict  │        │
│  │ Pipeline    │  │ Search      │  │ (CatBoost)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Secret     │  │   Cloud      │  │   외부 API   │
│   Manager    │  │   Storage    │  │              │
│              │  │              │  │  • OpenAI    │
│  • API Keys  │  │  • 모델      │  │  • Anthropic │
│              │  │  • ChromaDB  │  │  • PubMed    │
│              │  │  • 결과      │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 비용 최적화

### 예상 월간 비용

| 서비스 | 사용량 | 비용 |
|--------|--------|------|
| Cloud Run (Backend) | 하루 2시간 × 30일 | ~$20-30 |
| Cloud Run (Frontend) | 상시 최소 | ~$5-10 |
| Cloud Storage | 1GB | ~$0.5 |
| Secret Manager | 5개 시크릿 | ~$0.5 |
| **총계** | | **~$25-45/월** |

### 비용 절감 팁

#### 1. 최소 인스턴스 0으로 설정 (기본값)
```bash
--min-instances=0  # 사용 안 할 때 비용 $0
```

#### 2. 리전 선택
- `asia-northeast3` (서울) - 한국 사용자에게 빠름
- `us-central1` - 가장 저렴

#### 3. 메모리 최적화
RNA-seq 파이프라인 사용 안 하면:
```bash
--memory=4Gi  # 8GB → 4GB로 줄이기
```

#### 4. 예산 알림 설정
```bash
# GCP Console > 결제 > 예산 및 알림
# 월 $50 초과 시 알림 설정 권장
```

---

## 운영 명령어

### 로그 확인
```bash
# 최근 로그
./deploy.sh logs

# 실시간 로그
gcloud run services logs tail bioinsight-backend --region=asia-northeast3
```

### 서비스 상태
```bash
./deploy.sh status
```

### 서비스 재시작
```bash
# 새 리비전 배포로 재시작
gcloud run services update bioinsight-backend --region=asia-northeast3
```

### 서비스 삭제
```bash
./deploy.sh destroy
```

---

## 문제 해결

### 1. 빌드 실패: 메모리 부족

```bash
# Docker Desktop 메모리 늘리기 (8GB 권장)
# Docker Desktop > Settings > Resources > Memory
```

### 2. 배포 실패: 권한 오류

```bash
# 서비스 계정 권한 확인
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:$(gcloud config get-value account)" \
    --role="roles/run.admin"
```

### 3. Cloud Run 시작 실패: 시크릿 접근 오류

```bash
# Cloud Run 서비스 계정에 Secret Manager 권한 부여
PROJECT_NUMBER=$(gcloud projects describe $GCP_PROJECT_ID --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding OPENAI_API_KEY \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 4. RNA-seq 파이프라인 타임아웃

Cloud Run 최대 타임아웃은 60분입니다. 더 긴 작업이 필요하면:
1. Cloud Run Jobs 사용
2. Compute Engine VM으로 전환

### 5. Cold Start 느림

```bash
# 최소 인스턴스 1로 설정 (비용 증가)
gcloud run services update bioinsight-backend \
    --min-instances=1 \
    --region=asia-northeast3
```

---

## 파일 구조

```
deploy/gcp/
├── README.md              # 이 문서
├── deploy.sh              # 배포 스크립트
├── Dockerfile.backend     # Backend 이미지
├── Dockerfile.frontend    # Frontend 이미지
├── nginx.conf             # Frontend Nginx 설정
└── storage_sync.py        # Cloud Storage 동기화 유틸리티
```

---

## 커스텀 도메인 설정 (선택)

```bash
# 1. 도메인 매핑
gcloud run domain-mappings create \
    --service=bioinsight-frontend \
    --domain=bioinsight.yourdomain.com \
    --region=asia-northeast3

# 2. DNS 설정 (도메인 제공업체에서)
# CNAME: bioinsight.yourdomain.com → ghs.googlehosted.com
```

---

## 지원

문제가 발생하면:
1. `./deploy.sh logs`로 로그 확인
2. [GCP Console](https://console.cloud.google.com/run)에서 서비스 상태 확인
3. GitHub Issues에 문의

---

*Last Updated: 2026-01-26*
