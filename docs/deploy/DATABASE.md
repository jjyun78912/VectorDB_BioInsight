# BioInsight AI - 데이터베이스 구성 가이드

## 📋 목차

1. [개요](#개요)
2. [ChromaDB 동작 방식](#chromadb-동작-방식)
3. [데이터 디렉토리 구조](#데이터-디렉토리-구조)
4. [배포 시 데이터 준비](#배포-시-데이터-준비)
5. [백업 및 복원](#백업-및-복원)
6. [트러블슈팅](#트러블슈팅)

---

## 개요

### 데이터베이스 구성

| DB 종류 | 사용 여부 | 설명 |
|---------|----------|------|
| **ChromaDB** | ✅ 사용 | 벡터 DB (논문 임베딩) |
| PostgreSQL | ❌ 미사용 | - |
| MongoDB | ❌ 미사용 | - |
| Redis | ❌ 미사용 | - |

### ChromaDB 기본 정보

| 항목 | 값 |
|------|-----|
| 저장 방식 | **로컬 파일 기반** (Embedded Mode) |
| 저장 경로 | `./data/chroma_db/` |
| 총 용량 | ~2.2GB |
| 문서 수 | 53,000+ chunks |
| 임베딩 모델 | PubMedBERT (768차원) |

---

## ChromaDB 동작 방식

### 핵심: 별도 설치/실행 불필요

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ChromaDB = Embedded Database (SQLite와 유사)                  │
│                                                                 │
│   ✅ 별도 DB 서버 실행 불필요                                   │
│   ✅ 별도 설치 스크립트 불필요                                  │
│   ✅ pip install chromadb 만 하면 됨                            │
│   ✅ 백엔드 실행 시 자동으로 로드됨                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 서버 시작 시 자동 초기화 흐름

```
백엔드 서버 실행
      │
      ▼
uvicorn backend.app.main:app
      │
      ▼
vector_store.py 로드
      │
      ▼
chromadb.PersistentClient(path="./data/chroma_db/")
      │
      ├── 폴더 없으면 → 자동 생성 (빈 DB)
      │
      └── 폴더 있으면 → 기존 데이터 자동 로드
      │
      ▼
서비스 준비 완료
```

### 실제 코드 (backend/app/core/vector_store.py)

```python
import chromadb
from chromadb.config import Settings

# 백엔드 시작 시 자동 실행됨
client = chromadb.PersistentClient(
    path="./data/chroma_db/",      # 이 경로의 데이터 자동 로드
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# 컬렉션 자동 로드/생성
collection = client.get_or_create_collection(name="bio_papers")
```

### 시나리오별 동작

| 상황 | ChromaDB 동작 |
|------|---------------|
| 최초 실행 (빈 폴더) | 새 DB 생성, 빈 컬렉션 생성 |
| 기존 데이터 있음 | **기존 데이터 자동 로드** |
| 데이터 폴더 없음 | 폴더 자동 생성 후 빈 DB |

---

## 데이터 디렉토리 구조

```
VectorDB_BioInsight/
├── data/
│   └── chroma_db/              # ⭐ 벡터 DB (2.2GB)
│       ├── chroma.sqlite3      # 메타데이터
│       └── [collection-uuid]/  # 임베딩 데이터 (90+ 폴더)
│
└── models/
    └── rnaseq/                 # ⭐ ML 모델 (~200MB)
        ├── pancancer/
        │   ├── pancancer_model.cbm
        │   ├── preprocessor.joblib
        │   └── shap_explainer.joblib
        └── breast/
            └── breast_cancer_model.cbm
```

### 용량 요약

| 디렉토리 | 용량 | 배포 시 필요 |
|----------|------|-------------|
| `data/chroma_db/` | 2.2GB | ✅ 필수 |
| `models/rnaseq/` | ~200MB | ✅ 필수 |
| **총계** | ~2.4GB | |

---

## 배포 시 데이터 준비

### 방법 1: 로컬에서 데이터 복사

```bash
# 1. 데이터 압축
tar -czvf bioinsight_data.tar.gz data/chroma_db/ models/rnaseq/

# 2. 서버로 전송
scp bioinsight_data.tar.gz user@server:/app/

# 3. 서버에서 압축 해제
ssh user@server "cd /app && tar -xzvf bioinsight_data.tar.gz"

# 4. 백엔드 실행 → ChromaDB 자동 로드
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### 방법 2: GCP Cloud Storage 사용

```bash
# 1. 로컬에서 업로드 (최초 1회)
gsutil -m cp -r data/chroma_db/ gs://your-bucket/chroma_db/
gsutil -m cp -r models/rnaseq/ gs://your-bucket/models/

# 2. 서버 시작 스크립트 (entrypoint.sh)
#!/bin/bash
gsutil -m rsync -r gs://your-bucket/chroma_db/ /app/data/chroma_db/
gsutil -m rsync -r gs://your-bucket/models/ /app/models/rnaseq/
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### 방법 3: Docker 이미지에 포함

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# 데이터 복사 (이미지 크기 증가)
COPY data/chroma_db/ /app/data/chroma_db/
COPY models/rnaseq/ /app/models/rnaseq/

COPY . .
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0"]
```

---

## 백업 및 복원

### 백업

```bash
# 전체 데이터 백업
tar -czvf backup_$(date +%Y%m%d).tar.gz \
    data/chroma_db/ \
    models/rnaseq/

# Cloud Storage 백업
gsutil -m rsync -r data/chroma_db/ gs://your-bucket/backup/chroma_db/
```

### 복원

```bash
# 로컬 복원
tar -xzvf backup_20260126.tar.gz

# Cloud Storage에서 복원
gsutil -m rsync -r gs://your-bucket/backup/chroma_db/ data/chroma_db/
```

---

## 트러블슈팅

### 1. "Collection not found" 오류

```bash
# 원인: 데이터 폴더가 비어있거나 없음
# 해결: 데이터 폴더 확인
ls -la data/chroma_db/

# 데이터가 없으면 백업에서 복원 또는 재인덱싱
python scripts/collect_rnaseq_papers.py --all --count 50
```

### 2. "Database locked" 오류

```bash
# 원인: 다른 프로세스가 DB 사용 중
# 해결: 기존 프로세스 종료
pkill -f "uvicorn"
lsof data/chroma_db/chroma.sqlite3  # 사용 중인 프로세스 확인
```

### 3. 메모리 부족

```bash
# 원인: 임베딩 모델 로드 시 메모리 부족
# 해결: 서버 메모리 확인 (최소 4GB 권장)
free -h

# 또는 스왑 메모리 추가
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 4. 검색 결과 없음

```python
# 확인 스크립트
import chromadb
client = chromadb.PersistentClient(path="./data/chroma_db/")

# 컬렉션 목록 확인
for col in client.list_collections():
    print(f"{col.name}: {col.count()} docs")
```

---

## 환경 변수

```bash
# .env
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=pritamdeka/S-PubMedBert-MS-MARCO
```

---

*Last Updated: 2026-01-26*
