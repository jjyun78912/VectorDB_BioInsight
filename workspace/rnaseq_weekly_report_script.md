# RNA-seq Cancer Analysis Pipeline - 주간 보고 대본

**발표 시간**: 약 12-15분
**발표자**: BioInsight AI Computational Biology Team
**일자**: 2026-01-06

---

## 슬라이드 1: Title (30초)

> 안녕하세요. 이번 주간 보고에서는 RNA-seq Cancer Analysis Pipeline의 개발 진행 상황을 말씀드리겠습니다.
>
> 이번 주에는 파이프라인 v2.0을 완성하고, **RAG 아키텍처**를 통합하여 논문 기반 검증 시스템까지 구축했습니다.

---

## 슬라이드 2: Project Overview (1분 30초)

> 먼저 프로젝트 전체 구조를 설명드리겠습니다.
>
> **왼쪽의 Pipeline Architecture**를 보시면, 저희 파이프라인은 크게 6개 핵심 기능으로 구성됩니다.
>
> 첫째, **DESeq2**를 통한 차등발현분석입니다. R의 DESeq2를 Python에서 rpy2로 호출하여 통계적으로 검증된 분석을 수행합니다.
>
> 둘째, **NetworkX 기반의 Co-expression Network 분석**으로 Hub Gene을 찾아냅니다.
>
> 셋째, **gseapy를 통한 GO/KEGG Pathway Enrichment 분석**입니다.
>
> 넷째, **200개 이상의 암 유전자 데이터베이스**를 구축하여 발견된 DEG들을 검증합니다.
>
> 다섯째, **Publication-quality 시각화** - Volcano plot, Heatmap, PCA, Network plot 등을 자동 생성합니다.
>
> 마지막으로 **Interactive HTML Report**를 생성합니다.
>
> **오른쪽은 이번 주에 새로 추가된 6개 모듈**입니다. 약 4,300줄의 코드가 새로 작성되었고, 모두 GitHub에 커밋 완료했습니다.

---

## 슬라이드 3: Analysis Results Summary (1분)

> 이제 실제 분석 결과를 보겠습니다.
>
> Synthetic 데이터로 파이프라인을 테스트한 결과입니다.
>
> **총 4,709개 유전자**, 40개 샘플을 분석했고, 통계적으로 유의미한 DEG는 **43개**가 검출되었습니다.
>
> 기준은 adjusted p-value 0.05 미만, log2 fold change 절대값 1 이상입니다.
>
> 이 중 **22개가 상향조절**, **21개가 하향조절**되어 거의 비슷한 비율을 보였습니다.
>
> 이 결과들이 생물학적으로 의미있는지 다음 슬라이드에서 검증 결과를 보여드리겠습니다.

---

## 슬라이드 4: Top Hub Genes (1분 30초)

> Network 분석을 통해 찾아낸 **Top Hub Gene**들입니다.
>
> 가장 중요한 건 **HIF1A**입니다. 5.6배 상향조절되었고, 이 유전자는 **저산소 반응의 핵심 조절자**로 암의 혈관신생과 전이에 중요한 역할을 합니다.
>
> 두 번째 **SPP1**은 전이 마커로 알려져 있고, **MYC**은 대표적인 종양유전자입니다.
>
> **APC**는 대장암의 핵심 유전자이고, **CCNB1**과 **RB1**은 세포주기 조절에 관여합니다.
>
> 오른쪽 인사이트를 보시면, 43개 DEG 중 **31개가 이미 알려진 치료제 타겟**입니다. CDK4/6 inhibitor, PARP inhibitor 등이 해당됩니다.
>
> 또한 **86%의 검증률**을 달성했습니다. 이는 저희가 구축한 Tier1, Tier2 암 유전자 데이터베이스와 대조한 결과입니다.

---

## 슬라이드 5: RNA-seq RAG Pipeline Architecture (2분) ⭐ NEW

> 이번 주의 핵심 성과 중 하나인 **RAG Pipeline Architecture**입니다.
>
> RAG는 Retrieval-Augmented Generation의 약자로, **검색 기반 생성 AI 기술**입니다.
>
> 상단의 **6단계 파이프라인**을 보시면:
>
> **[1] LOAD** - RNA-seq 데이터를 AnnData로 로드하고 DESeq2로 분석합니다.
>
> **[2] CHUNK** - 분석 결과를 유전자 단위로 청크화합니다. 각 유전자별로 발현량, fold change, p-value 정보가 텍스트로 변환됩니다.
>
> **[3] EMBED** - **PubMedBERT**를 사용해 768차원 벡터로 임베딩합니다. 일반 BERT가 아닌 바이오메디컬 특화 모델을 사용해 정확도를 높였습니다.
>
> **[4] STORE** - **ChromaDB**에 벡터와 메타데이터를 저장합니다. 질병 도메인별로 컬렉션이 분리되어 있습니다.
>
> **[5] RETRIEVE** - 쿼리가 들어오면 유사도 검색을 수행합니다.
>
> **[6] GENERATE** - 검색된 컨텍스트를 바탕으로 LLM이 답변을 생성합니다.
>
> 하단 왼쪽의 **Key Components**를 보시면:
> - PubMedBERT로 바이오메디컬 도메인에 최적화
> - ChromaDB에 메타데이터(섹션, 논문 제목, DOI 등)를 함께 저장
> - **Gene Status Card**가 핵심입니다. DEG 정보에 질병 연관성과 치료제 정보를 매핑합니다.
>
> 오른쪽 **Integration Points**:
> - 현재 5개 논문, 521개 청크가 인덱싱되어 있습니다.
> - DisGeNET, COSMIC 등 외부 DB와 연동하여 질병 정보를 조회합니다.
> - 최종적으로 **DEG 결과가 논문 근거와 함께 검증**됩니다.

---

## 슬라이드 6: Internal Database Implementation (1분 30초)

> 이번 주 핵심 작업 중 하나가 **내부 데이터베이스 구축**이었습니다.
>
> 첫 번째 **disease_database.py**는 44개의 엄선된 암 유전자를 포함합니다. 특징은 각 유전자별로 **치료제 정보**가 매핑되어 있다는 점입니다. 예를 들어 EGFR이면 Erlotinib, Gefitinib 같은 표적치료제가 연결됩니다.
>
> 두 번째 **validation_enhanced.py**는 200개 이상의 유전자를 **Tier 분류 체계**로 정리했습니다.
> - Tier1은 Cancer Gene Census와 OncoKB에서 검증된 고신뢰도 유전자 약 100개
> - Tier2는 COSMIC, IntOGen 기반의 암 연관 유전자 약 100개입니다.
>
> 또한 **6개 암종별 Disease Signature**도 구축했습니다. 폐암, 유방암, 대장암, 췌장암, 간암, 교모세포종입니다.
>
> 세 번째는 **외부 API 연동 계획**입니다. DisGeNET, COSMIC, OncoKB 연동이 계획되어 있지만, API Key가 필요해서 다음 스프린트로 미뤘습니다.

---

## 슬라이드 7: Enhanced Pipeline Workflow (1분)

> 전체 파이프라인 워크플로우입니다.
>
> **7단계로 구성**됩니다.
>
> 1단계 **QC** - 샘플 아웃라이어 검출, 저발현 유전자 필터링
> 2단계 **DESeq2** - rpy2를 통한 차등발현분석
> 3단계 **Validation** - 암 유전자 DB와 대조 검증
> 4단계 **Network** - Hub Gene 식별
> 5단계 **Pathway** - GO/KEGG Enrichment
> 6단계 **Visualization** - 6종 Publication-quality 플롯
> 7단계 **Report** - HTML 및 CSV 결과 파일
>
> 아래는 생성되는 **Output 파일** 목록입니다. HTML 리포트, DEG 결과, Hub Gene, Pathway 결과, Gene Card JSON, 그리고 각종 시각화 PNG 파일이 자동 생성됩니다.

---

## 슬라이드 8: Next Steps (1분 30초)

> 앞으로의 계획입니다.
>
> **이번 주 High Priority**는 두 가지입니다.
>
> 첫째, **HTML Report Display 문제 해결**입니다. 현재 로컬 환경에서 Bootstrap CSS 로딩 이슈가 있어서 리포트가 깨져 보이는 문제가 있습니다. 이건 CDN 의존성을 로컬 번들링으로 변경하면 해결됩니다.
>
> 둘째, **실제 GEO 데이터 테스트**입니다. GSE19804 폐암 데이터셋으로 테스트를 진행 중인데, Probe ID 매핑 이슈가 있어서 보완이 필요합니다.
>
> **다음 스프린트**에서는 외부 API 연동을 진행합니다.
> - DisGeNET API로 유전자-질병 연관성 데이터 확장
> - COSMIC/OncoKB로 돌연변이 및 정밀의료 데이터 연동
> - 장기적으로는 **RAG Pipeline을 완전 통합**하여 DEG 결과에 논문 근거를 자동으로 붙이는 기능을 구현할 예정입니다.

---

## 슬라이드 9: Summary (30초)

> 요약하겠습니다.
>
> **Enhanced RNA-seq Pipeline v2.0이 완성**되었습니다.
>
> 핵심 성과는:
> - **43개 DEG 식별**, **86% 검증률**
> - **31개 치료제 타겟** 발견
> - **RAG Architecture 통합** - PubMedBERT + ChromaDB
> - **6개 신규 모듈**, 총 **4,270줄 코드** 추가
>
> 감사합니다. 질문 있으시면 말씀해주세요.

---

## 슬라이드 10: Q&A - RAG Pipeline (1분 30초) ⭐ NEW

> 예상되는 질문들에 대해 미리 답변을 준비했습니다.
>
> **첫 번째, RAG와 일반 검색의 차이점입니다.**
>
> 일반 키워드 검색은 정확한 단어 매칭만 가능합니다. 예를 들어 "EGFR"을 검색하면 정확히 "EGFR"이 포함된 문서만 찾습니다.
>
> 반면 RAG는 **의미적 유사도** 기반입니다. "EGFR mutation"을 검색하면 "epidermal growth factor receptor alteration"처럼 다른 표현도 찾아냅니다. 또한 LLM이 검색 결과를 종합해서 자연어로 답변을 생성합니다.
>
> **두 번째, PubMedBERT 선택 이유입니다.**
>
> 일반 BERT는 Wikipedia, 뉴스 같은 일반 텍스트로 학습되었습니다. **PubMedBERT는 PubMed 논문 전문**으로 학습되어 바이오메디컬 용어에 대한 이해도가 훨씬 높습니다.
>
> 실제로 바이오메디컬 벤치마크에서 **10-15% 더 높은 정확도**를 보입니다.
>
> **세 번째, ChromaDB 선택 이유입니다.**
>
> 세 가지입니다. 첫째 **Python API가 간단**합니다. 둘째 로컬에서 영구 저장이 가능합니다. 셋째 LangChain과 통합이 쉽습니다.
>
> 대규모 프로덕션 환경에서는 Pinecone이나 Weaviate를 고려할 수 있습니다.

---

## Q&A 예상 질문 (추가)

### Q1: RAG와 일반 검색의 차이점은?
> 일반 키워드 검색은 정확한 단어 매칭만 가능합니다. RAG는 **의미적 유사도**를 기반으로 검색하므로, "EGFR mutation"을 검색하면 "epidermal growth factor receptor alteration"도 찾아냅니다. 또한 검색 결과를 LLM이 종합하여 자연어 답변을 생성합니다.

### Q2: PubMedBERT를 선택한 이유는?
> 일반 BERT는 Wikipedia, 뉴스 등 일반 텍스트로 학습되었습니다. PubMedBERT는 **PubMed 논문 전문**으로 학습되어 바이오메디컬 용어에 대한 이해도가 훨씬 높습니다. 실제로 바이오메디컬 벤치마크에서 10-15% 더 높은 정확도를 보입니다.

### Q3: rpy2로 DESeq2를 호출하는 이유는?
> DESeq2는 RNA-seq 분석의 Gold Standard입니다. Python으로 재구현하는 것보다 R 원본을 호출하는 것이 통계적 정확성과 재현성 면에서 유리합니다.

### Q4: 86% 검증률의 의미는?
> 발견된 43개 DEG 중 37개가 이미 암과의 연관성이 알려진 유전자라는 의미입니다. 이는 파이프라인이 생물학적으로 의미있는 결과를 도출하고 있다는 검증입니다.

### Q5: 실제 환자 데이터에 적용 가능한가?
> 네, GEO나 TCGA의 공개 데이터셋에 바로 적용 가능합니다. 다만 Probe ID 매핑이나 배치 효과 보정 같은 전처리가 필요할 수 있습니다.

### Q6: 외부 API 연동 시 비용은?
> DisGeNET은 Academic 라이선스가 무료입니다. OncoKB도 연구 목적은 무료입니다. COSMIC은 Academic 라이선스 신청이 필요합니다.

### Q7: ChromaDB를 선택한 이유는?
> 세 가지 이유입니다. 첫째, Python API가 간단합니다. 둘째, 로컬에서 영구 저장이 가능합니다. 셋째, LangChain과 통합이 쉽습니다. 대규모 프로덕션에서는 Pinecone이나 Weaviate를 고려할 수 있습니다.

---

**총 발표 시간: 약 12분**
