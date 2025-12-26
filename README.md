# VectorDB BioInsight

Bio 논문 전용 Vector Database - **Pheochromocytoma (갈색세포종)** 도메인 특화

## 아키텍처

```
PDF 논문
    ↓
[PDF Parser] PyMuPDF - 섹션 구조 파싱
    ↓
[Text Splitter] 섹션 단위 split → RecursiveCharacterTextSplitter
    ↓
[Embedding] PubMedBERT (Sentence-Transformers)
    ↓
[Vector DB] ChromaDB + Metadata
    ↓
[유사도 검색] 질문 → 벡터화 → 검색
```

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
```

## 사용법

### 1. 논문 인덱싱

```bash
# 단일 PDF 인덱싱
python main.py index -d pheochromocytoma -f ./data/papers/paper.pdf

# 디렉토리 전체 인덱싱
python main.py index -d pheochromocytoma -p ./data/papers/
```

### 2. 유사도 검색

```bash
# 기본 검색
python main.py search -d pheochromocytoma -q "RET mutation genetic analysis"

# 섹션 필터 검색
python main.py search -d pheochromocytoma -q "RNA sequencing" -s Methods

# 결과 개수 지정
python main.py search -d pheochromocytoma -q "catecholamine" -k 10
```

### 3. 통계 확인

```bash
python main.py stats -d pheochromocytoma
```

### 4. 인덱싱된 논문 목록

```bash
python main.py list -d pheochromocytoma
```

## Python API

```python
from src.indexer import create_indexer
from src.search import create_searcher

# 인덱싱
indexer = create_indexer(disease_domain="pheochromocytoma")
indexer.index_pdf("./paper.pdf")
indexer.index_directory("./papers/")

# 검색
searcher = create_searcher(disease_domain="pheochromocytoma")
results = searcher.search("RET mutation", top_k=5)

# 섹션별 검색
results = searcher.search("RNA extraction", section_filter="Methods")

# 결과 출력
print(searcher.format_results(results))
```

## 메타데이터 구조

각 chunk에 저장되는 메타데이터:

| 필드 | 설명 |
|------|------|
| `paper_title` | 논문 제목 |
| `doi` | DOI |
| `year` | 출판 연도 |
| `keywords` | 키워드 |
| `section` | 섹션명 (Abstract, Introduction, Methods, Results, Discussion, Conclusion) |
| `parent_section` | 상위 섹션 (Methods의 subsection인 경우) |
| `disease_domain` | 질병 도메인 |
| `chunk_index` | 전체 chunk 인덱스 |
| `source_file` | 원본 PDF 경로 |

## 프로젝트 구조

```
VectorDB_BioInsight/
├── main.py                 # CLI 진입점
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py          # 설정
│   ├── pdf_parser.py      # PDF 파싱
│   ├── text_splitter.py   # 텍스트 분할
│   ├── embeddings.py      # PubMedBERT 임베딩
│   ├── vector_store.py    # ChromaDB 연동
│   ├── indexer.py         # 인덱싱 파이프라인
│   └── search.py          # 유사도 검색
├── data/
│   ├── papers/            # PDF 논문 저장
│   └── chroma_db/         # ChromaDB 영구 저장소
└── examples/
    └── example_usage.py   # 사용 예제
```

## 지원 섹션

자동 인식되는 Bio 논문 섹션:
- Abstract
- Introduction
- Materials and Methods / Methods
- Results
- Discussion
- Conclusion

Methods 하위 섹션:
- RNA extraction
- DNA extraction
- Library preparation
- RNA-seq processing
- Differential expression analysis
- Statistical analysis
- 등

## 임베딩 모델

PubMedBERT 기반 모델 사용:
- `pritamdeka/S-PubMedBert-MS-MARCO` (기본값, retrieval 최적화)
- `NeuML/pubmedbert-base-embeddings`
- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`

## 다른 질병 도메인 추가

```python
# 새 도메인 생성
indexer = create_indexer(disease_domain="lung_cancer")
indexer.index_directory("./data/papers/lung_cancer/")

# 검색
searcher = create_searcher(disease_domain="lung_cancer")
```
