#!/usr/bin/env python3
"""
RNA-seq RAG Agent 테스트 스크립트 - 전체 파이프라인 구현

파이프라인 단계:
  [1] LOAD    → Count matrix + Metadata 로드
  [2] ANALYZE → scipy DEG 분석 (PyDESeq2 대체)
  [3] CHUNK   → 유전자별 텍스트 청크 생성
  [4] EMBED   → PubMedBERT 벡터화 ✅
  [5] STORE   → ChromaDB 저장 ✅
  [6] RETRIEVE/GENERATE → RAG 검색 + 답변 생성 ✅
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("  RNA-seq RAG Pipeline Test (Full Implementation)")
print("=" * 70)

# ============================================================================
# 1. DEGResult 데이터 클래스
# ============================================================================

@dataclass
class DEGResult:
    """DEG 분석 결과"""
    gene_symbol: str
    base_mean: float
    log2_fold_change: float
    p_value: float
    adjusted_p_value: float

    @property
    def is_significant(self) -> bool:
        return self.adjusted_p_value < 0.05 and abs(self.log2_fold_change) > 1.0

    @property
    def direction(self) -> str:
        if self.log2_fold_change > 1.0 and self.adjusted_p_value < 0.05:
            return "up"
        elif self.log2_fold_change < -1.0 and self.adjusted_p_value < 0.05:
            return "down"
        return "unchanged"

    @property
    def regulation(self) -> str:
        return {"up": "Upregulated", "down": "Downregulated"}.get(self.direction, "Not significant")


# ============================================================================
# 2. 합성 데이터 생성
# ============================================================================

print("\n[1] 합성 Count Matrix 생성 (10 genes × 4 samples)")
print("-" * 70)

np.random.seed(42)

genes = ["TP53", "KRAS", "EGFR", "BRCA1", "MYC", "GAPDH", "ACTB", "CDK4", "RB1", "VEGFA"]
samples = ["Case_1", "Case_2", "Control_1", "Control_2"]

# 기본 counts
base_counts = np.random.poisson(lam=500, size=(10, 4))

# 차등 발현 적용 (Case vs Control)
multipliers = {
    "TP53": [0.3, 0.4, 1.0, 1.0],   # 하향 (tumor suppressor)
    "KRAS": [3.5, 4.0, 1.0, 1.0],   # 상향 (oncogene)
    "EGFR": [2.5, 2.8, 1.0, 1.0],   # 상향
    "BRCA1": [0.4, 0.5, 1.0, 1.0],  # 하향
    "MYC": [3.0, 3.2, 1.0, 1.0],    # 상향
    "GAPDH": [1.0, 1.0, 1.0, 1.0],  # 변화없음 (housekeeping)
    "ACTB": [1.0, 1.0, 1.0, 1.0],   # 변화없음
    "CDK4": [2.0, 2.2, 1.0, 1.0],   # 상향
    "RB1": [0.5, 0.4, 1.0, 1.0],    # 하향
    "VEGFA": [2.8, 3.0, 1.0, 1.0],  # 상향
}

for i, gene in enumerate(genes):
    base_counts[i] = (base_counts[i] * multipliers[gene]).astype(int) + np.random.poisson(20, 4)

count_matrix = pd.DataFrame(base_counts, index=genes, columns=samples)
print(count_matrix)

metadata = pd.DataFrame({
    "condition": ["case", "case", "control", "control"]
}, index=samples)

print("\n메타데이터:")
print(metadata)


# ============================================================================
# 3. DEG 분석 (scipy t-test 폴백)
# ============================================================================

print("\n[2] DEG 분석 (scipy Welch's t-test + BH correction)")
print("-" * 70)

def run_deg_scipy(counts: pd.DataFrame, meta: pd.DataFrame) -> List[DEGResult]:
    """scipy 기반 DEG 분석"""
    case_samples = meta[meta["condition"] == "case"].index.tolist()
    ctrl_samples = meta[meta["condition"] == "control"].index.tolist()

    results = []
    pvals = []

    for gene in counts.index:
        case_vals = counts.loc[gene, case_samples].values + 1  # pseudocount
        ctrl_vals = counts.loc[gene, ctrl_samples].values + 1

        log2fc = np.log2(np.mean(case_vals) / np.mean(ctrl_vals))
        _, pval = stats.ttest_ind(np.log2(case_vals), np.log2(ctrl_vals), equal_var=False)
        base_mean = np.mean(counts.loc[gene].values)

        pvals.append(pval if not np.isnan(pval) else 1.0)
        results.append({"gene": gene, "log2fc": log2fc, "pval": pvals[-1], "base_mean": base_mean})

    # BH correction
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    padj = np.ones(n)
    for rank, idx in enumerate(sorted_idx, 1):
        padj[idx] = min(1.0, pvals[idx] * n / rank)

    return [
        DEGResult(
            gene_symbol=r["gene"],
            base_mean=r["base_mean"],
            log2_fold_change=r["log2fc"],
            p_value=r["pval"],
            adjusted_p_value=padj[i]
        )
        for i, r in enumerate(results)
    ]

deg_results = run_deg_scipy(count_matrix, metadata)
deg_results.sort(key=lambda x: x.adjusted_p_value)

print(f"{'Gene':<10} {'log2FC':>10} {'padj':>12} {'Direction':>12} {'Significant':>12}")
print("-" * 60)
for d in deg_results:
    print(f"{d.gene_symbol:<10} {d.log2_fold_change:>10.2f} {d.adjusted_p_value:>12.2e} {d.direction:>12} {'✓' if d.is_significant else '':>12}")

sig_up = sum(1 for d in deg_results if d.is_significant and d.direction == "up")
sig_down = sum(1 for d in deg_results if d.is_significant and d.direction == "down")
print(f"\n유의미한 DEG: {sig_up} 상향, {sig_down} 하향")


# ============================================================================
# 4. GeneCard 생성
# ============================================================================

print("\n[3] GeneCard 생성 (유의미한 DEG)")
print("-" * 70)

@dataclass
class GeneCard:
    gene_symbol: str
    regulation: str
    log2_fold_change: float
    p_value: float
    fold_change: float
    diseases: List[Dict] = field(default_factory=list)
    top_disease: Optional[str] = None
    therapeutics: List[str] = field(default_factory=list)

# 샘플 질병 연관성 데이터
DISEASE_DB = {
    "KRAS": [{"name": "Pancreatic Cancer", "score": 0.92}, {"name": "Lung Cancer", "score": 0.88}],
    "TP53": [{"name": "Li-Fraumeni Syndrome", "score": 0.95}, {"name": "Breast Cancer", "score": 0.85}],
    "EGFR": [{"name": "Non-Small Cell Lung Cancer", "score": 0.90}],
    "MYC": [{"name": "Burkitt Lymphoma", "score": 0.88}],
    "BRCA1": [{"name": "Hereditary Breast Cancer", "score": 0.93}],
}

gene_cards = []
for d in deg_results:
    if d.is_significant:
        diseases = DISEASE_DB.get(d.gene_symbol, [])
        card = GeneCard(
            gene_symbol=d.gene_symbol,
            regulation=d.regulation,
            log2_fold_change=d.log2_fold_change,
            p_value=d.adjusted_p_value,
            fold_change=2 ** abs(d.log2_fold_change),
            diseases=diseases,
            top_disease=diseases[0]["name"] if diseases else None
        )
        gene_cards.append(card)
        print(f"  {card.gene_symbol}: {card.regulation} ({card.fold_change:.1f}x) → {card.top_disease or 'N/A'}")


# ============================================================================
# 5. RAG 청크 생성
# ============================================================================

print("\n[4] RAG 청크 생성")
print("-" * 70)

def create_rag_chunks(gene_cards: List[GeneCard], deg_results: List[DEGResult]) -> List[Dict[str, Any]]:
    """GeneCard를 RAG 청크로 변환"""
    chunks = []

    for card in gene_cards:
        chunk_text = f"""
Gene: {card.gene_symbol}
Expression: {card.regulation} ({card.fold_change:.1f}x fold change)
Log2 Fold Change: {card.log2_fold_change:+.2f}
Adjusted p-value: {card.p_value:.2e}
Top Disease Association: {card.top_disease or 'Unknown'}
Related Diseases: {', '.join([d['name'] for d in card.diseases]) if card.diseases else 'None identified'}
Clinical Significance: This gene shows {'significant upregulation' if card.log2_fold_change > 0 else 'significant downregulation'} in the case samples compared to controls.
        """.strip()

        metadata = {
            "gene_symbol": card.gene_symbol,
            "direction": "up" if "Up" in card.regulation else "down",
            "log2fc": round(card.log2_fold_change, 3),
            "padj": card.p_value,
            "fold_change": round(card.fold_change, 2),
            "disease": card.top_disease or "Unknown",
            "chunk_type": "deg_analysis"
        }

        chunks.append({
            "text": chunk_text,
            "metadata": metadata
        })

    return chunks

rag_chunks = create_rag_chunks(gene_cards, deg_results)
print(f"생성된 RAG 청크: {len(rag_chunks)}개")

for i, chunk in enumerate(rag_chunks[:3]):  # 처음 3개만 출력
    print(f"\n  [Chunk {i+1}] {chunk['metadata']['gene_symbol']}")
    print(f"  Direction: {chunk['metadata']['direction']}, log2FC: {chunk['metadata']['log2fc']}")


# ============================================================================
# 6. EMBED - PubMedBERT 벡터화
# ============================================================================

print("\n[5] EMBED - PubMedBERT 벡터화")
print("-" * 70)

try:
    from backend.app.core.embeddings import PubMedBertEmbedder, get_embedder

    # PubMedBERT 임베더 초기화
    embedder = get_embedder()
    print(f"모델: {embedder.model_name}")
    print(f"디바이스: {embedder.device}")
    print(f"임베딩 차원: {embedder.embedding_dimension}")

    # 청크 텍스트 임베딩 생성
    chunk_texts = [chunk["text"] for chunk in rag_chunks]
    embeddings = embedder.embed_texts(chunk_texts, show_progress=False)

    print(f"\n생성된 임베딩: {len(embeddings)}개")
    print(f"임베딩 벡터 크기: {len(embeddings[0])}차원")

    # 첫 번째 임베딩 샘플 출력
    print(f"\n첫 번째 임베딩 샘플 (처음 10개 값):")
    print(f"  {embeddings[0][:10]}")

    EMBED_SUCCESS = True
except ImportError as e:
    print(f"⚠️  임베딩 모듈 로드 실패: {e}")
    print("   sentence-transformers 설치 필요: pip install sentence-transformers")
    EMBED_SUCCESS = False
    embeddings = []
except Exception as e:
    print(f"⚠️  임베딩 생성 실패: {e}")
    EMBED_SUCCESS = False
    embeddings = []


# ============================================================================
# 7. STORE - ChromaDB 저장
# ============================================================================

print("\n[6] STORE - ChromaDB 저장")
print("-" * 70)

STORE_SUCCESS = False
vector_store = None

try:
    import chromadb
    from chromadb.config import Settings

    # ChromaDB 클라이언트 생성 (임시 저장소)
    chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))

    # RNA-seq 전용 컬렉션 생성
    collection_name = "rnaseq_deg_analysis"

    # 기존 컬렉션 삭제 후 재생성
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={
            "description": "RNA-seq DEG Analysis Results",
            "analysis_type": "differential_expression",
            "embedding_model": "PubMedBERT"
        }
    )

    print(f"컬렉션 생성: {collection_name}")

    if EMBED_SUCCESS and embeddings:
        # ChromaDB에 청크 저장
        ids = [f"deg_{chunk['metadata']['gene_symbol']}" for chunk in rag_chunks]
        documents = [chunk["text"] for chunk in rag_chunks]
        metadatas = [chunk["metadata"] for chunk in rag_chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"저장된 문서: {collection.count()}개")
        STORE_SUCCESS = True
        vector_store = collection
    else:
        print("⚠️  임베딩이 없어 저장을 건너뜁니다")

except ImportError as e:
    print(f"⚠️  ChromaDB 로드 실패: {e}")
    print("   chromadb 설치 필요: pip install chromadb")
except Exception as e:
    print(f"⚠️  ChromaDB 저장 실패: {e}")


# ============================================================================
# 8. RETRIEVE - RAG 검색
# ============================================================================

print("\n[7] RETRIEVE - RAG 검색")
print("-" * 70)

RETRIEVE_SUCCESS = False
retrieved_context = ""

if STORE_SUCCESS and vector_store:
    try:
        # 테스트 쿼리
        test_queries = [
            "Which genes are upregulated in cancer?",
            "What oncogenes show differential expression?",
            "Which tumor suppressor genes are downregulated?"
        ]

        for query in test_queries:
            print(f"\n쿼리: \"{query}\"")

            # 쿼리 임베딩 생성
            query_embedding = embedder.embed_query(query)

            # ChromaDB 검색
            results = vector_store.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )

            print(f"  검색 결과 {len(results['ids'][0])}개:")
            distances = results['distances'][0]
            min_dist = min(distances) if distances else 0
            max_dist = max(distances) if distances else 1

            for i, (doc_id, distance) in enumerate(zip(results['ids'][0], distances)):
                gene = results['metadatas'][0][i]['gene_symbol']
                direction = results['metadatas'][0][i]['direction']
                # 상대적 relevance: 최소 거리가 100%, 최대 거리가 낮은 점수
                if max_dist > min_dist:
                    relevance = ((max_dist - distance) / (max_dist - min_dist)) * 50 + 50
                else:
                    relevance = 75.0
                print(f"    {i+1}. {gene} ({direction}) - distance: {distance:.2f}, relevance: {relevance:.1f}%")

        # 마지막 쿼리 결과를 컨텍스트로 사용
        retrieved_context = "\n\n".join(results['documents'][0])
        RETRIEVE_SUCCESS = True

    except Exception as e:
        print(f"⚠️  검색 실패: {e}")
else:
    print("⚠️  벡터 스토어가 없어 검색을 건너뜁니다")


# ============================================================================
# 9. GENERATE - LLM 답변 생성
# ============================================================================

print("\n[8] GENERATE - 답변 생성")
print("-" * 70)

GENERATE_SUCCESS = False

if RETRIEVE_SUCCESS and retrieved_context:
    # LLM 없이도 동작하는 템플릿 기반 답변 생성
    print("템플릿 기반 답변 생성 (LLM 없음):")

    # 검색된 컨텍스트 기반 요약 생성
    upregulated_genes = [c['metadata']['gene_symbol'] for c in rag_chunks if c['metadata']['direction'] == 'up']
    downregulated_genes = [c['metadata']['gene_symbol'] for c in rag_chunks if c['metadata']['direction'] == 'down']

    template_response = f"""
## RNA-seq DEG 분석 결과 요약

### 상향 조절된 유전자 (Upregulated)
{', '.join(upregulated_genes)}

### 하향 조절된 유전자 (Downregulated)
{', '.join(downregulated_genes)}

### 주요 발견
- 총 {len(gene_cards)}개의 유의미한 DEG 검출
- 상향 조절: {len(upregulated_genes)}개, 하향 조절: {len(downregulated_genes)}개
- 주요 암 관련 유전자 ({gene_cards[0].gene_symbol if gene_cards else 'N/A'}) 발현 변화 확인

### 질병 연관성
"""

    for card in gene_cards[:3]:
        if card.top_disease:
            template_response += f"- **{card.gene_symbol}**: {card.top_disease}\n"

    print(template_response)
    GENERATE_SUCCESS = True

    # Gemini API가 있으면 LLM 답변도 시도
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")

        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')

            prompt = f"""Based on the following RNA-seq differential expression analysis results,
provide a brief summary of the key findings:

{retrieved_context}

Focus on:
1. Which genes are significantly upregulated/downregulated
2. Potential disease associations
3. Clinical implications
"""

            print("\n" + "-" * 40)
            print("Gemini LLM 답변:")
            print("-" * 40)

            response = model.generate_content(prompt)
            print(response.text[:500] + "..." if len(response.text) > 500 else response.text)

    except Exception as e:
        print(f"\n(LLM 답변 생성 건너뜀: {e})")

else:
    print("⚠️  검색된 컨텍스트가 없어 답변 생성을 건너뜁니다")


# ============================================================================
# 완료
# ============================================================================

print("\n" + "=" * 70)
print("  ✅ RNA-seq RAG Pipeline 테스트 완료!")
print("=" * 70)

# 파이프라인 상태 요약
pipeline_status = {
    "LOAD": "✅",
    "ANALYZE": "✅",
    "CHUNK": "✅",
    "EMBED": "✅" if EMBED_SUCCESS else "❌",
    "STORE": "✅" if STORE_SUCCESS else "❌",
    "RETRIEVE": "✅" if RETRIEVE_SUCCESS else "❌",
    "GENERATE": "✅" if GENERATE_SUCCESS else "❌"
}

print(f"""
파이프라인 단계:
  [1] LOAD     {pipeline_status['LOAD']} Count matrix + Metadata 로드
  [2] ANALYZE  {pipeline_status['ANALYZE']} scipy DEG 분석
  [3] CHUNK    {pipeline_status['CHUNK']} 유전자별 텍스트 청크 생성
  [4] EMBED    {pipeline_status['EMBED']} PubMedBERT 벡터화
  [5] STORE    {pipeline_status['STORE']} ChromaDB 저장
  [6] RETRIEVE {pipeline_status['RETRIEVE']} RAG 검색
  [7] GENERATE {pipeline_status['GENERATE']} 답변 생성

결과 요약:
  - 분석 유전자: {len(deg_results)}개
  - 유의미 DEG: {len(gene_cards)}개 (↑{sig_up}, ↓{sig_down})
  - GeneCard 생성: {len(gene_cards)}개
  - RAG 청크: {len(rag_chunks)}개
  - 임베딩 차원: {len(embeddings[0]) if embeddings else 'N/A'}
""")
