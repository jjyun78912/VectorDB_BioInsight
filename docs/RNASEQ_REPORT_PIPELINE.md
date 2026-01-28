# RNA-seq Report Pipeline: Complete Technical Documentation

## Overview

이 문서는 BioInsight AI의 RNA-seq 분석 파이프라인이 최종 리포트를 생성하기까지의 전체 과정을 기술합니다. 각 단계에서 사용하는 모델, 검증 방법, 그리고 바이오 도메인 지식이 어떻게 적용되는지 상세히 설명합니다.

---

## Pipeline Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          RNA-seq Report Generation Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   INPUT                                                                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                                 │
│   │count_matrix  │  │ metadata.csv │  │ config.json  │                                 │
│   │   .csv       │  │              │  │              │                                 │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                                 │
│          └─────────────────┴─────────────────┘                                          │
│                             │                                                           │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                       AGENT 1: DIFFERENTIAL EXPRESSION                                  │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │ DESeq2 (R) + apeglm LFC Shrinkage                                               │   │
│   │ ────────────────────────────────────────                                        │   │
│   │ • Negative Binomial GLM (count data 특성 반영)                                  │   │
│   │ • Size Factor Normalization (sequencing depth 보정)                             │   │
│   │ • Wald Test for differential expression                                         │   │
│   │ • apeglm: Adaptive shrinkage (효과 크기 과대추정 방지)                          │   │
│   │ • Benjamini-Hochberg FDR correction                                             │   │
│   │                                                                                 │   │
│   │ Filters: padj < 0.05, |log2FC| > 1.0                                            │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                             │                                                           │
│              ┌──────────────┼──────────────┐                                            │
│              ▼              ▼              ▼                                            │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                                 │
│   │deg_signif.csv│  │norm_counts   │  │counts_pca    │                                 │
│   │(DEG list)    │  │.csv          │  │.csv (VST)    │                                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                                 │
│                                                                                         │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                       AGENT 2: CO-EXPRESSION NETWORK                                    │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Spearman Correlation + NetworkX                                                 │   │
│   │ ───────────────────────────────────                                             │   │
│   │ • Spearman rank correlation (비선형 관계 포착)                                   │   │
│   │ • Threshold: |ρ| > 0.7 (강한 상관관계만)                                        │   │
│   │ • NetworkX graph construction                                                   │   │
│   │                                                                                 │   │
│   │ Hub Gene Centrality (Adaptive):                                                 │   │
│   │ ┌─────────────────────────────────────────────────────────────────────────────┐ │   │
│   │ │ edges > 1M   → Weighted Degree (17초)                                       │ │   │
│   │ │ nodes > 1K   → k-sampling Betweenness (k=100)                               │ │   │
│   │ │ else         → Exact Betweenness Centrality                                 │ │   │
│   │ └─────────────────────────────────────────────────────────────────────────────┘ │   │
│   │                                                                                 │   │
│   │ Hub Gene Score = |log2FC| × -log10(padj)                                        │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                             │                                                           │
│              ┌──────────────┼──────────────┐                                            │
│              ▼              ▼              ▼                                            │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                                 │
│   │hub_genes.csv │  │network_edges │  │network_stats │                                 │
│   │(Top 20)      │  │.csv          │  │.json         │                                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                                 │
│                                                                                         │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                       AGENT 3: PATHWAY ENRICHMENT                                       │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Enrichr API (gseapy)                                                            │   │
│   │ ─────────────────────                                                           │   │
│   │ Gene Sets:                                                                      │   │
│   │ • GO_Biological_Process_2023: 생물학적 기능                                     │   │
│   │ • GO_Molecular_Function_2023: 분자 기능                                         │   │
│   │ • GO_Cellular_Component_2023: 세포 내 위치                                      │   │
│   │ • KEGG_2021_Human: 대사/신호 경로                                              │   │
│   │ • Reactome_2022: 상세 신호 전달 경로                                           │   │
│   │ • MSigDB_Hallmark_2020: 암 hallmark 경로                                       │   │
│   │                                                                                 │   │
│   │ Combined P-value: Fisher's method (multiple gene sets)                          │   │
│   │ Filter: adjusted_p < 0.05                                                       │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                             │                                                           │
│              ┌──────────────┼──────────────┐                                            │
│              ▼              ▼              ▼                                            │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                                 │
│   │pathway_sum   │  │gene_to_      │  │enrichr_raw   │                                 │
│   │mary.csv      │  │pathway.csv   │  │.json         │                                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                                 │
│                                                                                         │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                       AGENT 4: DATABASE VALIDATION                                      │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Cancer Gene Database Validation                                                 │   │
│   │ ─────────────────────────────────                                               │   │
│   │                                                                                 │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ COSMIC (Catalogue of Somatic Mutations in Cancer)                          │  │   │
│   │ │ • Census genes: 알려진 암 driver genes                                      │  │   │
│   │ │ • Tier 1: 강력한 문헌 근거                                                  │  │   │
│   │ │ • Tier 2: 추가 연구 필요                                                    │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ OncoKB (Precision Oncology Knowledge Base)                                 │  │   │
│   │ │ • Actionability levels: 1-4 (FDA 승인 → 연구 단계)                         │  │   │
│   │ │ • Oncogene/TSG annotation                                                   │  │   │
│   │ │ • Drug associations                                                         │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Score Calculation:                                                              │   │
│   │ • COSMIC Tier 1: +30점                                                          │   │
│   │ • COSMIC Tier 2: +15점                                                          │   │
│   │ • OncoKB Level 1-2: +25점                                                       │   │
│   │ • OncoKB Level 3-4: +15점                                                       │   │
│   │ • log2FC weight: |log2FC| × 5                                                   │   │
│   │ • padj weight: -log10(padj) × 2                                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │ RAG Gene Interpretation (ChromaDB + Claude)                                     │   │
│   │ ────────────────────────────────────────────                                    │   │
│   │                                                                                 │   │
│   │ Step 1: Embedding Query                                                         │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ Query: "{gene} {cancer_type} {direction}"                                  │  │   │
│   │ │ Model: S-PubMedBert-MS-MARCO (768-dim)                                     │  │   │
│   │ │ Collection: rnaseq_{cancer_type} (~50 papers, ~3000 chunks)                │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Step 2: Hybrid Search                                                           │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ Dense: Cosine similarity (60%)                                             │  │   │
│   │ │ Sparse: BM25 keyword matching (40%)                                        │  │   │
│   │ │ Fusion: Reciprocal Rank Fusion (k=60)                                      │  │   │
│   │ │ Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2                            │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Step 3: LLM Interpretation                                                      │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ Model: Claude Sonnet 4 (claude-sonnet-4-20250514)                          │  │   │
│   │ │ Context: Top-5 relevant paper chunks + PMID                                │  │   │
│   │ │ Output: 한국어 해석 + PMID 인용                                              │  │   │
│   │ │ Fallback: Gemini 2.5 Pro (Google Vertex AI)                                │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Example Output:                                                                 │   │
│   │ "BRCA1의 하향 발현은 DNA 손상 복구 능력 저하와 관련되어 유방암의                │   │
│   │  genomic instability를 유발할 수 있습니다 [PMID: 28952959]"                     │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                             │                                                           │
│              ┌──────────────┼──────────────┐                                            │
│              ▼              ▼              ▼                                            │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                                 │
│   │integrated_   │  │cosmic_hits   │  │rag_interp    │                                 │
│   │gene_table.csv│  │.csv          │  │.json         │                                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                                 │
│                                                                                         │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                       AGENT 5: VISUALIZATION                                            │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Publication-Quality Figures (Matplotlib + Plotly)                               │   │
│   │ ─────────────────────────────────────────────────                               │   │
│   │                                                                                 │   │
│   │ Static Plots (Matplotlib):                                                      │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ • Volcano Plot: log2FC vs -log10(padj), hub genes 강조                     │  │   │
│   │ │ • Heatmap: Top 50 DEGs, hierarchical clustering (ward)                     │  │   │
│   │ │ • PCA Plot: PC1 vs PC2, condition별 색상                                   │  │   │
│   │ │ • Pathway Barplot: Top 15 enriched pathways                                │  │   │
│   │ │ • MA Plot: baseMean vs log2FC                                              │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Interactive Plots (Plotly):                                                     │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ • 3D Network: force-directed layout, hub genes 하이라이트                  │  │   │
│   │ │ • Interactive Volcano: hover로 gene 정보 표시                               │  │   │
│   │ │ • Sankey Diagram: Gene → Pathway → Function                                │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Color Scheme:                                                                   │   │
│   │ • Upregulated: #E74C3C (빨강)                                                   │   │
│   │ • Downregulated: #3498DB (파랑)                                                 │   │
│   │ • Not significant: #95A5A6 (회색)                                               │   │
│   │ • Hub genes: #F39C12 (주황)                                                     │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                             │                                                           │
│              ┌──────────────┼──────────────┐                                            │
│              ▼              ▼              ▼                                            │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                                 │
│   │figures/      │  │network_3d_   │  │plot_config   │                                 │
│   │*.png (16+)   │  │interactive   │  │.json         │                                 │
│   └──────────────┘  │.html         │  └──────────────┘                                 │
│                     └──────────────┘                                                    │
│                                                                                         │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                       ML PREDICTION (Optional)                                          │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │ CatBoost Pan-Cancer Classifier                                                  │   │
│   │ ────────────────────────────────                                                │   │
│   │                                                                                 │   │
│   │ Model Info:                                                                     │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ Algorithm: CatBoost (Gradient Boosting on Decision Trees)                  │  │   │
│   │ │ Training Data: TCGA Pan-Cancer (17 cancer types, 11,000+ samples)          │  │   │
│   │ │ Features: 5,000 HVGs (Highly Variable Genes)                               │  │   │
│   │ │ Hyperparameters: Optuna optimization (500 trials)                          │  │   │
│   │ │                                                                            │  │   │
│   │ │ ⚠️ WARNING: Data Leakage 가능성 있음 (TROUBLESHOOTING_ML_OVERFITTING.md)     │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Preprocessing:                                                                  │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ 1. Gene ID mapping (Entrez → Ensembl/Symbol)                               │  │   │
│   │ │ 2. Log2(TPM + 1) transformation                                            │  │   │
│   │ │ 3. Feature alignment (train HVGs만 사용)                                    │  │   │
│   │ │ 4. StandardScaler normalization (train mean/std 적용)                       │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Output:                                                                         │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ • Primary prediction (top cancer type)                                     │  │   │
│   │ │ • Probability distribution (17 classes)                                    │  │   │
│   │ │ • Confidence gap (top1 - top2)                                             │  │   │
│   │ │ • Confusable pair warning                                                  │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ SHAP Explanation:                                                               │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ Model: TreeExplainer (CatBoost 특화)                                        │  │   │
│   │ │ Output: Top 10 genes driving prediction                                    │  │   │
│   │ │ Visualization: Waterfall plot, Summary plot                                │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                             │                                                           │
│              ┌──────────────┼──────────────┐                                            │
│              ▼              ▼              ▼                                            │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                                 │
│   │cancer_pred   │  │shap_values   │  │shap_waterfall│                                 │
│   │iction.json   │  │.csv          │  │.png          │                                 │
│   └──────────────┘  └──────────────┘  └──────────────┘                                 │
│                                                                                         │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                       AGENT 6: REPORT GENERATION                                        │
│   ═══════════════════════════════════════════════════════════════════════════════════   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Interactive HTML Report                                                         │   │
│   │ ──────────────────────────                                                      │   │
│   │                                                                                 │   │
│   │ Extended Abstract Generation (LLM):                                             │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ Model: Claude Opus 4 (claude-opus-4-20250514) - Premium storytelling       │  │   │
│   │ │ Input: DEG stats, Pathway summary, Hub genes, Cancer type                  │  │   │
│   │ │ Output: 4-6 paragraph Korean narrative (1500-2000자)                       │  │   │
│   │ │ RAG Context: 암종 특화 논문 (rnaseq_{cancer_type} collection)               │  │   │
│   │ │                                                                            │  │   │
│   │ │ Structure:                                                                 │  │   │
│   │ │ 1. 배경 및 목적 (왜 이 분석을 하는가)                                       │  │   │
│   │ │ 2. 주요 발견 (DEG 패턴, Hub genes)                                         │  │   │
│   │ │ 3. 생물학적 의미 (Pathway, 암 메커니즘)                                     │  │   │
│   │ │ 4. 임상적 시사점 (치료 표적, 바이오마커)                                    │  │   │
│   │ │ 5. 한계 및 향후 연구 방향                                                   │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Research Recommendations (LLM):                                                 │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ Model: Claude Sonnet 4                                                     │  │   │
│   │ │                                                                            │  │   │
│   │ │ Sections Generated:                                                        │  │   │
│   │ │ • therapeutic_targets: 고/중 우선순위 치료 표적                            │  │   │
│   │ │ • drug_repurposing: 약물 재목적화 후보 (DGIdb 연동)                        │  │   │
│   │ │ • experimental_validation: qPCR, Western, IHC 프로토콜                     │  │   │
│   │ │ • biomarker_development: 진단/예후 마커 후보                               │  │   │
│   │ │ • future_research: 단기/중기/장기 연구 방향                                │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ DGIdb Integration (Drug-Gene Interactions):                                     │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │ API: DGIdb v5 GraphQL                                                      │  │   │
│   │ │ Query: Hub genes + Top DEGs                                                │  │   │
│   │ │                                                                            │  │   │
│   │ │ Retrieved Information:                                                     │  │   │
│   │ │ • Drug-gene interactions (interaction_type, score)                         │  │   │
│   │ │ • Gene categories (KINASE, RECEPTOR, DRUGGABLE)                            │  │   │
│   │ │ • FDA approval status                                                      │  │   │
│   │ │ • Literature evidence (PMIDs)                                              │  │   │
│   │ │ • Data sources (DrugBank, PharmGKB, ChEMBL, OncoKB)                        │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │ Report Sections:                                                                │   │
│   │ ┌────────────────────────────────────────────────────────────────────────────┐  │   │
│   │ │  # │ Section              │ Data Source                                    │  │   │
│   │ │ ───┼──────────────────────┼──────────────────────────────────────────────│  │   │
│   │ │  0 │ Cover                │ config.json, timestamp                         │  │   │
│   │ │  1 │ Summary              │ Aggregated stats                               │  │   │
│   │ │  2 │ Extended Abstract    │ LLM (Opus) + RAG                               │  │   │
│   │ │  3 │ QC                   │ PCA plot, sample clustering                    │  │   │
│   │ │  4 │ DEG Analysis         │ Agent 1 outputs                                │  │   │
│   │ │  5 │ Pathway              │ Agent 3 outputs                                │  │   │
│   │ │  6 │ Driver Genes         │ Agent 4 (COSMIC/OncoKB)                        │  │   │
│   │ │  7 │ Network              │ Agent 2 outputs                                │  │   │
│   │ │  8 │ Clinical             │ LLM + DGIdb                                    │  │   │
│   │ │  9 │ Follow-up            │ LLM recommendations                            │  │   │
│   │ │ 10 │ Research             │ research_recommendations.json                  │  │   │
│   │ │ 11 │ Methods              │ config.json, parameters                        │  │   │
│   │ │ 12 │ References           │ RAG PMIDs                                      │  │   │
│   │ │ 13 │ Appendix             │ Full DEG table (sortable)                      │  │   │
│   │ └────────────────────────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                           FINAL OUTPUT                                          │   │
│   │                                                                                 │   │
│   │   report.html (10-15MB, Interactive)                                            │   │
│   │   ├── Embedded figures (base64)                                                 │   │
│   │   ├── Interactive tables (sortable, searchable)                                 │   │
│   │   ├── 3D network visualization (Plotly)                                         │   │
│   │   ├── Collapsible sections                                                      │   │
│   │   └── Print-optimized CSS                                                       │   │
│   │                                                                                 │   │
│   │   Supporting Files:                                                             │   │
│   │   ├── report_data.json (structured data)                                        │   │
│   │   ├── research_recommendations.json                                             │   │
│   │   └── figures/ (standalone PNGs)                                                │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Bio Domain Knowledge Integration

### 1. DESeq2 + apeglm: 왜 이 조합인가?

**RNA-seq count data의 특성**:
- Count data는 음수가 없고, 정수이며, overdispersion이 있음
- Poisson 분포보다 Negative Binomial이 적합
- Low-count genes에서 variance가 과대추정됨

**DESeq2의 해결책**:
```
Count ~ Negative Binomial(μ, α)
where μ = size_factor × 2^(β × condition)
```

- `size_factor`: Sequencing depth 차이 보정
- `α` (dispersion): Gene별 variance 모델링
- Empirical Bayes shrinkage로 dispersion 안정화

**apeglm Shrinkage**:
- 기존 DESeq2의 log2FC는 low-count genes에서 과대추정
- apeglm: Adaptive shrinkage로 효과 크기를 "정직하게" 추정
- 결과: Hub gene selection에서 false positive 감소

### 2. Spearman vs Pearson Correlation

**Pearson** (선형 관계):
```
r = Σ(x-x̄)(y-ȳ) / √[Σ(x-x̄)² × Σ(y-ȳ)²]
```

**Spearman** (단조 관계):
```
ρ = 1 - 6Σd² / n(n²-1)  where d = rank(x) - rank(y)
```

**RNA-seq에서 Spearman을 선택한 이유**:
1. Gene expression은 종종 비선형 관계 (saturation, threshold effects)
2. Outlier에 robust
3. Log-transformed data에서도 안정적

### 3. Hub Gene Score 공식

```
Hub Score = |log2FC| × -log10(padj)
```

**생물학적 해석**:
- `|log2FC|`: 발현 변화의 "크기" (biological effect size)
- `-log10(padj)`: 통계적 신뢰도 (statistical confidence)
- 곱셈: 둘 다 높아야 높은 점수

**왜 이 점수인가?**:
- 발현이 많이 바뀌어도 p-value가 높으면 신뢰 불가
- p-value가 낮아도 발현 변화가 작으면 생물학적 의미 제한
- 기존 연구(Langfelder & Horvath, WGCNA)의 "gene significance" 개념 차용

### 4. Pathway Enrichment 해석

**Over-Representation Analysis (ORA)**:
```
P-value = Hypergeometric test
         = C(K,k) × C(N-K, n-k) / C(N,n)

where:
  N = total genes in background
  K = genes in pathway (background)
  n = DEGs
  k = DEGs in pathway
```

**Fisher's Exact Test 사용 이유**:
- Small sample sizes (DEG list)에서 정확
- Chi-square보다 conservative
- Multiple testing correction (BH FDR) 필수

### 5. COSMIC/OncoKB Tier 시스템

**COSMIC Cancer Gene Census**:

| Tier | 정의 | 근거 수준 |
|------|------|-----------|
| Tier 1 | Documented activity in cancer | 강력한 문헌 + 기능 연구 |
| Tier 2 | Strong indication | 상관관계 있으나 기능 검증 부족 |

**OncoKB Actionability Levels**:

| Level | 정의 | 예시 |
|-------|------|------|
| 1 | FDA-approved therapy | BRAF V600E → Vemurafenib |
| 2 | Standard care | EGFR exon19del → Erlotinib |
| 3 | Clinical evidence | KRAS G12C → Sotorasib (trials) |
| 4 | Biological evidence | PIK3CA → 전임상 단계 |

### 6. RAG 시스템의 특성과 한계

**RAG의 핵심 목적**:
- LLM hallucination 방지: 검색된 문헌에 기반하여 응답 생성
- 출처 추적 가능: 모든 해석에 PMID 인용 포함
- 최신 정보 반영: LLM 학습 cutoff 이후 논문도 활용 가능

**강점**:
- 최신 문헌 반영 (PubMed 2020-2024)
- PMID 인용으로 검증 가능 (사용자가 원문 확인 가능)
- 암종 특화 context (17종 각 ~50편, ~3000 chunks)
- Hallucination 대폭 감소 (순수 LLM 대비)

**실제 한계** (hallucination이 아닌):
- **Retrieval 품질**: 관련 문서를 찾지 못하면 해석 불가능
- **Corpus 커버리지**: 컬렉션에 없는 유전자/암종 조합은 약함
- **Embedding 한계**: Semantic similarity가 항상 relevance를 보장하지 않음
- **Context 길이**: Top-k 문서만 사용하므로 중요 정보 누락 가능

**Guardrail 적용**:
```python
# Agent 4에서 적용되는 검증
if len(retrieved_chunks) == 0:
    interpretation = "해당 유전자에 대한 문헌 정보가 부족합니다."
elif relevance_score < 0.5:
    interpretation += "\n⚠️ 관련 문헌이 제한적이며, 추가 검증이 권장됩니다."
```

### 7. Hub Gene vs Driver Gene: 핵심 개념 차이

파이프라인에서 식별하는 **Hub Gene**과 암 **Driver Gene**은 근본적으로 다른 개념입니다:

| 구분 | Hub Gene (발현 기반) | Driver Gene (변이 기반) |
|------|----------------------|------------------------|
| **정의** | DEG 기반 발현 변화 큰 유전자 | 암 발생/진행 유발 유전자 |
| **데이터** | RNA-seq (mRNA 발현) | DNA 변이 (WGS/WES) |
| **산출 방식** | \|log2FC\| × -log10(p) | 체세포 변이 빈도 + 기능적 증거 |
| **의미** | 발현 수준 변화 반영 | 단백질 기능 변화 유발 |
| **신뢰도** | PREDICTION (DB 매칭) | IDENTIFICATION (변이 검증) |

**Hub Gene ≠ Driver Gene인 이유**:
- Driver gene은 **변이(mutation)**를 통해 작용하며, mRNA 발현 변화가 없을 수 있음
- 예: TP53 변이는 단백질 기능 상실을 유발하지만 mRNA 수준은 정상일 수 있음
- **Hub gene은 암의 "결과"**, **Driver gene은 암의 "원인"**을 반영

### 8. 암종별 Known Driver Genes 예시

COSMIC/OncoKB 검증 시 참조되는 암종별 주요 드라이버:

| 암종 (TCGA) | Tier 1 Driver Genes | 주요 변이 |
|-------------|---------------------|-----------|
| **PAAD** (췌장암) | KRAS, TP53, CDKN2A, SMAD4 | KRAS G12D/V/R (>90%) |
| **LUAD** (폐선암) | EGFR, KRAS, ALK, ROS1 | EGFR exon19del, L858R |
| **BRCA** (유방암) | BRCA1, BRCA2, TP53, PIK3CA | PIK3CA H1047R |
| **COAD** (대장암) | APC, KRAS, TP53, BRAF | BRAF V600E |
| **SKCM** (흑색종) | BRAF, NRAS, NF1 | BRAF V600E (50%) |
| **GBM** (교모세포종) | IDH1, EGFR, TP53, PTEN | IDH1 R132H |
| **LIHC** (간암) | TP53, CTNNB1, AXIN1 | CTNNB1 hotspots |

**리포트에서의 활용**:
- Agent 4가 Hub genes를 위 목록과 대조
- 매칭되면: "KRAS는 PAAD의 알려진 driver gene입니다 (COSMIC Tier 1)"
- 매칭 안되면: "발현 변화가 있으나 알려진 driver는 아닙니다"

### 9. Multi-omic Pipeline (WGS/WES 통합)

**RNA-seq만으로는 Driver Gene "예측"만 가능**, 실제 "식별"을 위해서는 WGS/WES 데이터 필요:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Multi-omic Driver Identification                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   RNA-seq Only (현재 파이프라인):                                        │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Hub Genes → COSMIC/OncoKB 매칭 → "PREDICTION"                  │   │
│   │  ⚠️ 신뢰도: 낮음 (발현 변화 ≠ driver mutation)                   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   RNA-seq + WGS/WES (Multi-omic):                                       │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Step 1: Bulk RNA-seq 6-Agent Pipeline                          │   │
│   │          → DEGs, Hub genes, Pathways                            │   │
│   │                                                                 │   │
│   │  Step 2: Variant Analysis                                       │   │
│   │          VCF/MAF → Hotspot detection → VAF filtering            │   │
│   │          → KRAS G12C, BRAF V600E, TP53 R175H 등                 │   │
│   │                                                                 │   │
│   │  Step 3: Integrated Driver Analysis                             │   │
│   │          Mutation + Expression evidence 통합                    │   │
│   │          → "IDENTIFICATION" (confirmed driver)                  │   │
│   │          ✅ 신뢰도: 높음                                         │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Integrated Driver 분류 체계**:

| 분류 | 점수 | 근거 |
|------|------|------|
| **confirmed_driver** | ≥80 | Mutation + Expression 모두 존재 |
| **high_confidence** | ≥60 | 강한 증거 (hotspot 또는 높은 발현 변화) |
| **candidate** | ≥40 | 일부 증거 존재 |
| **mutation_only** | <40 | 변이만 있고 발현 변화 없음 |
| **expression_only** | <40 | 발현 변화만 있고 변이 없음 |

**VCF/MAF 파일 자동 감지**:
- 입력 폴더에 `.vcf`, `.vcf.gz`, `.maf` 파일이 있으면 자동으로 Multi-omic 파이프라인 실행
- 없으면 RNA-seq only (발현 기반 예측)

---

## Model Summary Table

| 단계 | 모델/도구 | 용도 | 비고 |
|------|----------|------|------|
| **DEG** | DESeq2 (R) | 차등발현 통계 | Negative Binomial GLM |
| | apeglm | LFC shrinkage | Adaptive prior |
| **Network** | NetworkX | 그래프 분석 | Python |
| | Spearman | 상관관계 | Non-parametric |
| **Pathway** | Enrichr API | ORA | Fisher's exact |
| | gseapy | Python wrapper | |
| **Validation** | COSMIC | 암 유전자 DB | Local cache |
| | OncoKB | Actionability | API |
| **Embedding** | S-PubMedBert | 문서 임베딩 | 768-dim |
| **Re-rank** | ms-marco-MiniLM | Cross-encoder | 정밀도 향상 |
| **LLM** | Claude Sonnet 4 | 일반 해석 | Primary |
| | Claude Opus 4 | Extended Abstract | Premium |
| | Gemini 2.5 Pro | Fallback | Google |
| **ML** | CatBoost | 암종 분류 | Gradient Boosting |
| | SHAP | 설명 가능성 | TreeExplainer |
| **Viz** | Matplotlib | Static plots | Publication quality |
| | Plotly | Interactive | 3D network |

---

## Output Files Reference

| 파일명 | 생성 Agent | 설명 |
|--------|-----------|------|
| `deg_all_results.csv` | Agent 1 | 전체 DESeq2 결과 |
| `deg_significant.csv` | Agent 1 | 필터링된 DEGs |
| `normalized_counts.csv` | Agent 1 | 정규화된 발현량 |
| `counts_for_pca.csv` | Agent 1 | VST 변환 (PCA용) |
| `hub_genes.csv` | Agent 2 | Top 20 hub genes |
| `network_edges.csv` | Agent 2 | 모든 edge 정보 |
| `pathway_summary.csv` | Agent 3 | Enrichment 결과 |
| `gene_to_pathway.csv` | Agent 3 | Gene-pathway 매핑 |
| `integrated_gene_table.csv` | Agent 4 | 통합 유전자 테이블 |
| `cosmic_hits.csv` | Agent 4 | COSMIC 매칭 결과 |
| `rag_interpretations.json` | Agent 4 | RAG 해석 |
| `figures/*.png` | Agent 5 | 시각화 (16+ files) |
| `network_3d_interactive.html` | Agent 5 | 3D 네트워크 |
| `cancer_prediction.json` | ML Module | 암종 예측 결과 |
| `shap_values.csv` | ML Module | SHAP 값 |
| `research_recommendations.json` | Agent 6 | 연구 추천 |
| `report_data.json` | Agent 6 | 리포트 데이터 |
| `report.html` | Agent 6 | 최종 리포트 |

---

## Quality Control Checkpoints

### 1. Input Validation (Agent 1)

```python
# 샘플 수 검증
if n_samples < 6:
    warning("Insufficient samples for DESeq2")
    fallback_to_synthetic()

# Condition 검증
if len(unique_conditions) != 2:
    error("Binary comparison required")
```

### 2. DEG Quality Check

```python
# 과도한 DEG 경고
if n_degs > 10000:
    warning("Unusually high DEG count - check batch effects")

# DEG 부족 경고
if n_degs < 50:
    warning("Few DEGs detected - consider relaxing thresholds")
```

### 3. Pathway Validation

```python
# 유의한 pathway 없음
if n_significant_pathways == 0:
    info("No pathways enriched - biological interpretation limited")
```

### 4. ML Confidence Check

```python
# 불확실한 예측
if confidence_gap < 0.15:
    warning("Low confidence - multiple cancer types possible")

# 혼동 가능 쌍
if is_confusable_pair(top1, top2):
    warning(f"{top1} and {top2} are biologically similar")
```

---

## Known Limitations

1. **ML 과적합 가능성**: Data leakage 이슈 (별도 문서 참조)
2. **RAG 커버리지**: 17종 암종만 특화 컬렉션 보유
3. **Single-cell 지원**: Bulk 대비 제한적
4. **외부 DB 의존**: COSMIC/OncoKB API 가용성
5. **LLM 비용**: Opus 사용 시 ~$0.30/report

---

*Last Updated: 2026-01-27*
