# BioInsight 임베딩 & RAG 아키텍처 기술 분석

> 작성일: 2025-12-31
> 버전: 1.0
> 분석 대상: `backend/app/core/embeddings.py`, `vector_store.py`, `config.py`

---

## 목차

1. [현재 아키텍처 개요](#1-현재-아키텍처-개요)
2. [임베딩 모델 분석](#2-임베딩-모델-분석)
3. [하이브리드 검색 아키텍처](#3-하이브리드-검색-아키텍처)
4. [청킹 전략](#4-청킹-전략)
5. [RAG 파이프라인](#5-rag-파이프라인)
6. [성능 최적화 로드맵](#6-성능-최적화-로드맵)
7. [대안 모델 비교](#7-대안-모델-비교)

---

## 1. 현재 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BioInsight RAG Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌─────────────────────────────────────────┐                │
│  │  Query   │───▶│           Query Processing              │                │
│  └──────────┘    │  - Korean → English Translation         │                │
│                  │  - Disease/MeSH Detection               │                │
│                  └─────────────────────────────────────────┘                │
│                                    │                                         │
│                                    ▼                                         │
│         ┌──────────────────────────┴──────────────────────────┐             │
│         │                                                      │             │
│         ▼                                                      ▼             │
│  ┌─────────────────┐                                  ┌─────────────────┐   │
│  │  Dense Search   │                                  │  Sparse Search  │   │
│  │  (PubMedBERT)   │                                  │     (BM25)      │   │
│  │                 │                                  │                 │   │
│  │  768-dim vector │                                  │  TF-IDF based   │   │
│  │  Semantic match │                                  │  Keyword match  │   │
│  └────────┬────────┘                                  └────────┬────────┘   │
│           │                                                     │            │
│           │              ┌─────────────────────┐               │            │
│           └─────────────▶│   RRF Fusion        │◀──────────────┘            │
│                          │   (60:40 weight)    │                            │
│                          └─────────┬───────────┘                            │
│                                    │                                         │
│                                    ▼                                         │
│                          ┌─────────────────────┐                            │
│                          │    Top-K Results    │                            │
│                          │    (k=5 default)    │                            │
│                          └─────────┬───────────┘                            │
│                                    │                                         │
│                                    ▼                                         │
│                          ┌─────────────────────┐                            │
│                          │   LLM Generation    │                            │
│                          │   (Gemini 2.0)      │                            │
│                          └─────────────────────┘                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 핵심 설정 (`config.py`)

```python
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIM = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4
```

---

## 2. 임베딩 모델 분석

### 2.1 현재 모델: S-PubMedBert-MS-MARCO

| 항목 | 값 |
|------|-----|
| **모델명** | `pritamdeka/S-PubMedBert-MS-MARCO` |
| **차원** | 768 |
| **기반** | PubMedBERT + MS-MARCO fine-tuning |
| **최대 토큰** | 512 |
| **모델 크기** | ~420MB |

### 2.2 장점

| 장점 | 설명 |
|------|------|
| **도메인 특화** | PubMed 21M+ 논문으로 사전학습되어 생의학 용어(유전자명, 단백질, 질병명, 약물명)에 최적화 |
| **검색 최적화** | MS-MARCO 데이터셋으로 fine-tuning되어 passage retrieval 성능 우수 |
| **Sentence-level** | 문장/단락 단위 의미 유사도 계산에 적합 |
| **검증된 성능** | BEIR 벤치마크에서 biomedical 도메인 상위권 |

### 2.3 단점

| 단점 | 설명 | 영향도 |
|------|------|--------|
| **단일 벡터 표현** | 문서 전체를 하나의 768차원 벡터로 압축 → 세부 정보 손실 | 중간 |
| **영어 전용** | 한국어 논문/쿼리 직접 처리 불가 (번역 필요) | 높음 |
| **토큰 제한** | 512 토큰 초과 시 truncation 발생 | 중간 |
| **Late Interaction 없음** | ColBERT처럼 토큰 단위 매칭 불가 | 중간 |
| **초기 로딩 시간** | 모델 로딩에 수 초 소요 | 낮음 |

### 2.4 구현 코드 분석

```python
# embeddings.py:45-80
class PubMedBertEmbedder:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dimension = settings.EMBEDDING_DIM

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """텍스트를 768차원 벡터로 변환"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 정규화로 코사인 유사도 = 내적
        )
        return embeddings
```

**핵심 포인트:**
- `normalize_embeddings=True`: 벡터 정규화로 코사인 유사도 계산 시 단순 내적으로 처리 가능
- 배치 처리 지원으로 대량 문서 인덱싱 시 효율적

---

## 3. 하이브리드 검색 아키텍처

### 3.1 구조

```
Query
  │
  ├──────────────────────────────────────────┐
  │                                          │
  ▼                                          ▼
┌─────────────────────┐            ┌─────────────────────┐
│    Dense Search     │            │    Sparse Search    │
│                     │            │                     │
│  PubMedBERT 임베딩  │            │      BM25 Index     │
│         ↓           │            │         ↓           │
│   ChromaDB 검색     │            │  rank_bm25 라이브러리│
│         ↓           │            │         ↓           │
│  Top-100 candidates │            │  Top-100 candidates │
└──────────┬──────────┘            └──────────┬──────────┘
           │                                   │
           └───────────────┬───────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Reciprocal Rank       │
              │   Fusion (RRF)          │
              │                         │
              │   score = Σ 1/(k+rank)  │
              │   k = 60 (smoothing)    │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Final Ranked Results  │
              └─────────────────────────┘
```

### 3.2 RRF (Reciprocal Rank Fusion) 구현

```python
# embeddings.py:158-180
class HybridSearcher:
    def __init__(self, dense_weight: float = 0.6, sparse_weight: float = 0.4):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def reciprocal_rank_fusion(
        self,
        dense_results: List[str],
        sparse_results: List[str],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        RRF 알고리즘: 각 결과의 순위 역수를 가중 합산

        Score(doc) = Σ weight_i / (k + rank_i)

        k=60은 상위 랭크와 하위 랭크 간 점수 차이를 완화
        """
        scores = defaultdict(float)

        for rank, doc_id in enumerate(dense_results):
            scores[doc_id] += self.dense_weight * (1 / (k + rank + 1))

        for rank, doc_id in enumerate(sparse_results):
            scores[doc_id] += self.sparse_weight * (1 / (k + rank + 1))

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 3.3 장점

| 장점 | 설명 |
|------|------|
| **상호 보완성** | Dense(의미적 유사도) + Sparse(키워드 매칭) 결합으로 검색 품질 향상 |
| **RRF 안정성** | 점수 스케일 차이에 강건하며, 하이퍼파라미터 튜닝 최소화 |
| **Recall 향상** | 둘 중 하나라도 찾으면 결과에 포함 |
| **전문용어 처리** | Sparse가 정확한 유전자명/약물명 매칭 담당 |

### 3.4 단점

| 단점 | 설명 | 개선 방안 |
|------|------|----------|
| **이중 인덱싱 비용** | Dense 벡터 + BM25 인덱스 별도 관리 | BGE-M3 통합 모델 |
| **가중치 고정** | 0.6:0.4 비율이 모든 쿼리에 적용 | 쿼리 유형별 동적 조정 |
| **Re-ranking 부재** | 검색 후 정제 단계 없음 | Cross-Encoder 추가 |

### 3.5 개선 제안: 동적 가중치

```python
def get_dynamic_weights(query: str) -> Tuple[float, float]:
    """
    쿼리 유형에 따라 Dense/Sparse 가중치 동적 조정
    """
    # 유전자명, 약물명 등 전문용어 패턴
    biomedical_entities = ['BRCA1', 'TP53', 'EGFR', 'pembrolizumab', ...]

    if any(entity.lower() in query.lower() for entity in biomedical_entities):
        # 정확한 용어 매칭 중요 → Sparse 가중치 증가
        return (0.4, 0.6)

    if query.endswith('?'):
        # 자연어 질문 → Dense 가중치 유지
        return (0.7, 0.3)

    # 기본값
    return (0.6, 0.4)
```

---

## 4. 청킹 전략

### 4.1 현재 설정

```python
CHUNK_SIZE = 1000      # 문자 수
CHUNK_OVERLAP = 200    # 오버랩 문자 수
```

### 4.2 장점

| 장점 | 설명 |
|------|------|
| **적절한 크기** | 1000자 ≈ 150-200 토큰, PubMedBERT 512 토큰 한계 내 |
| **오버랩** | 200자 중복으로 문장 단절 방지, 문맥 유지 |
| **구현 단순성** | 고정 크기로 예측 가능한 청크 수 |

### 4.3 단점

| 단점 | 설명 | 영향도 |
|------|------|--------|
| **의미 단위 무시** | 섹션(Abstract, Methods, Results) 경계 무시 | 높음 |
| **메타데이터 손실** | 청크에서 원본 섹션 정보 추적 어려움 | 중간 |
| **테이블/수식 분할** | 표나 수식이 중간에 잘릴 수 있음 | 중간 |

### 4.4 개선 제안: 섹션 기반 청킹

```python
def semantic_chunking(text: str, metadata: dict) -> List[dict]:
    """
    논문 구조를 고려한 의미 단위 청킹
    """
    sections = ['Abstract', 'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion']
    chunks = []

    for section in sections:
        section_text = extract_section(text, section)
        if not section_text:
            continue

        # 섹션 내에서 단락 단위로 분할
        paragraphs = section_text.split('\n\n')

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= 1000:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'section': section,
                        'metadata': metadata
                    })
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'section': section,
                'metadata': metadata
            })

    return chunks
```

---

## 5. RAG 파이프라인

### 5.1 현재 플로우

```
사용자 질문
     │
     ▼
┌─────────────────────────────────────────┐
│  1. Query Processing                    │
│     - 한국어 감지 → 영어 번역           │
│     - MeSH 용어 추출                    │
│     - 질병 도메인 감지                  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  2. Hybrid Retrieval                    │
│     - Dense: ChromaDB 벡터 검색         │
│     - Sparse: BM25 키워드 검색          │
│     - RRF Fusion                        │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  3. Context Selection                   │
│     - Top-5 청크 선택                   │
│     - 총 ~5000자 컨텍스트               │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  4. LLM Generation                      │
│     - Gemini 2.0 Flash                  │
│     - Citation 포함 답변 생성           │
└─────────────────────────────────────────┘
```

### 5.2 장점

| 장점 | 설명 |
|------|------|
| **ChromaDB 경량성** | 로컬 실행 가능, 영속성 지원 |
| **t-SNE 시각화** | 유사 논문 클러스터링 제공 |
| **메타데이터 필터링** | 연도, 저자, 저널 기반 필터 가능 |
| **한국어 지원** | 자동 번역 파이프라인 내장 |

### 5.3 단점 및 개선점

| 이슈 | 현재 상태 | 영향 | 개선 방안 |
|------|----------|------|----------|
| **고정 Top-K** | k=5 고정 | 복잡한 질문에 불충분 | 쿼리 복잡도에 따라 k 조정 |
| **Re-ranking 없음** | RRF 후 바로 반환 | 정밀도 손실 | Cross-Encoder 추가 |
| **Citation 추적** | 청크 단위 | 원문 위치 불명확 | 섹션+문단 번호 추가 |
| **쿼리 확장** | 없음 | 동의어/약어 누락 | MeSH/UMLS 확장 |
| **캐싱** | 없음 | 반복 쿼리 비효율 | Redis 캐싱 |

---

## 6. 성능 최적화 로드맵

### 6.1 단기 (1-2주)

| 항목 | 작업 | 예상 효과 |
|------|------|----------|
| **쿼리 캐싱** | 자주 검색되는 키워드 Redis 캐싱 | 응답 속도 50% 향상 |
| **배치 임베딩** | 논문 인덱싱 시 GPU 배치 처리 | 인덱싱 속도 3x |
| **인덱스 최적화** | ChromaDB HNSW 파라미터 조정 | 검색 속도 20% 향상 |

```python
# Redis 캐싱 예시
import redis
import hashlib
import json

class QueryCache:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.ttl = 3600  # 1시간

    def get_cached(self, query: str) -> Optional[dict]:
        key = f"rag:{hashlib.md5(query.encode()).hexdigest()}"
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None

    def set_cache(self, query: str, result: dict):
        key = f"rag:{hashlib.md5(query.encode()).hexdigest()}"
        self.redis.setex(key, self.ttl, json.dumps(result))
```

### 6.2 중기 (1-2개월)

| 항목 | 작업 | 예상 효과 |
|------|------|----------|
| **Cross-Encoder Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-12-v2` | 정밀도 15-20% 향상 |
| **쿼리 확장** | UMLS/MeSH 기반 동의어 확장 | Recall 10% 향상 |
| **섹션별 가중치** | Abstract > Results > Methods | 관련성 향상 |

```python
# Cross-Encoder Re-ranker 예시
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 5
    ) -> List[dict]:
        """초기 후보를 Cross-Encoder로 재정렬"""
        pairs = [(query, c['content']) for c in candidates]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = float(score)

        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
```

### 6.3 장기 (3-6개월)

| 항목 | 작업 | 예상 효과 |
|------|------|----------|
| **BGE-M3 마이그레이션** | Dense + Sparse + Multi-vector 통합 | 단일 모델로 하이브리드 |
| **ColBERT Late Interaction** | 토큰 레벨 유사도 계산 | 정밀도 대폭 향상 |
| **HyDE** | LLM으로 가상 답변 생성 후 검색 | 복잡한 질문 처리 |

---

## 7. 대안 모델 비교

### 7.1 임베딩 모델 비교

| 모델 | 차원 | 장점 | 단점 | 적합성 |
|------|------|------|------|--------|
| **현재: S-PubMedBert-MS-MARCO** | 768 | 생의학 특화, 검색 최적화 | 영어 전용, 단일 벡터 | ★★★★☆ |
| `BAAI/bge-m3` | 1024 | 다국어, Dense+Sparse+ColBERT 통합 | 모델 크기 2GB+ | ★★★★★ |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 8192 토큰 지원, 경량 | 생의학 특화 X | ★★★☆☆ |
| `colbert-ir/colbertv2.0` | 128x토큰수 | Late interaction 정밀도 | 인덱싱 비용 높음 | ★★★★☆ |
| `intfloat/e5-large-v2` | 1024 | 범용 고성능 | 생의학 특화 X | ★★★☆☆ |

### 7.2 벡터 DB 비교

| DB | 장점 | 단점 | 현재 사용 |
|----|------|------|----------|
| **ChromaDB** | 경량, 로컬, 영속성 | 대규모 확장 어려움 | ✅ |
| Pinecone | 관리형, 확장성 | 비용, 종속성 | ❌ |
| Weaviate | 하이브리드 검색 내장 | 설정 복잡 | ❌ |
| Milvus | 대규모 지원, 고성능 | 운영 복잡 | ❌ |
| Qdrant | Rust 기반 고성능 | 생태계 작음 | ❌ |

### 7.3 권장 마이그레이션 경로

```
현재 (S-PubMedBert + BM25)
           │
           ▼ (단기)
Cross-Encoder Re-ranker 추가
           │
           ▼ (중기)
BGE-M3으로 임베딩 모델 교체
(Dense + Sparse 통합)
           │
           ▼ (장기)
ColBERT Late Interaction 추가
또는 Milvus로 벡터 DB 마이그레이션
```

---

## 8. 결론 및 권장사항

### 8.1 현재 시스템 평가

| 영역 | 점수 | 평가 |
|------|------|------|
| 임베딩 모델 | ★★★★☆ | 도메인 특화 잘 되어 있음 |
| 하이브리드 검색 | ★★★☆☆ | Re-ranking 추가 필요 |
| 청킹 전략 | ★★★☆☆ | 섹션 기반 개선 필요 |
| RAG 파이프라인 | ★★★☆☆ | 쿼리 확장, 캐싱 필요 |
| **종합** | **★★★☆☆** | 기본 아키텍처 양호, 최적화 여지 있음 |

### 8.2 우선순위 권장사항

1. **즉시 적용 (높은 ROI)**
   - Redis 쿼리 캐싱
   - Cross-Encoder Re-ranker 추가

2. **중기 개선 (중간 ROI)**
   - MeSH/UMLS 쿼리 확장
   - 섹션 기반 청킹

3. **장기 투자 (전략적)**
   - BGE-M3 마이그레이션
   - ColBERT 도입 검토

---

*이 문서는 BioInsight 프로젝트의 기술 분석을 위해 작성되었습니다.*
