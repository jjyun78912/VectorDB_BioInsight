/**
 * i18n Translation files for BioInsight
 */

export type Language = 'en' | 'ko';

export interface Translations {
  // Header
  appName: string;
  tagline: string;

  // Hero Section
  heroBadge: string;
  heroTitle1: string;
  heroTitle2: string;
  heroSubtitle: string;
  heroHint: string;

  // Search
  searchPlaceholder: string;
  searchPlaceholderPubmed: string;
  searchPlaceholderDoi: string;
  searchButton: string;
  searching: string;

  // Search modes
  localDB: string;
  pubmedLive: string;
  doiUrl: string;

  // Results
  results: string;
  noResults: string;
  relevanceScore: string;
  section: string;
  year: string;

  // Paper details
  abstract: string;
  authors: string;
  journal: string;
  keywords: string;
  viewPaper: string;
  similarPapers: string;

  // Chat
  askQuestion: string;
  typeQuestion: string;
  send: string;
  sources: string;
  confidence: string;

  // Upload
  uploadPdf: string;
  uploadPdfShort: string;
  uploadRnaseq: string;
  uploadRnaseqShort: string;
  dragDrop: string;
  processing: string;
  uploadSuccess: string;
  uploadError: string;

  // Trending
  trendingPapers: string;
  categories: string;
  trendAnalysis: string;
  hotTopics: string;
  researchTrends: string;

  // Language
  language: string;

  // Disease domains
  pancreaticCancer: string;
  bloodCancer: string;
  glioblastoma: string;
  alzheimer: string;
  pcos: string;
  pheochromocytoma: string;
  adhd: string;

  // Common
  loading: string;
  error: string;
  retry: string;
  close: string;
  viewMore: string;
  showLess: string;

  // Diagnostics
  diagnostics: string;
  searchStrategy: string;
  detectedDisease: string;
  meshTerm: string;
  totalCandidates: string;
  filteredResults: string;
  matchField: string;

  // Match fields
  titleMatch: string;
  abstractMatch: string;
  fullTextMatch: string;

  // Footer
  poweredBy: string;

  // Tools
  bioResearchDaily: string;
  bioResearchDailyDesc: string;
  openTool: string;

  // Research Tools Section
  researchTools: string;
  researchToolsSubtitle: string;
  powerfulTools: string;

  // Tool Names
  literatureReview: string;
  literatureReviewDesc: string;
  chatWithPdf: string;
  chatWithPdfDesc: string;
  researchLibrary: string;
  researchLibraryDesc: string;
  knowledgeGraph: string;
  knowledgeGraphDesc: string;
  dailyBriefing: string;
  dailyBriefingDesc: string;
  geneNetwork: string;
  geneNetworkDesc: string;

  // Feature Suite
  featureSuiteTitle: string;
  featureSuiteSubtitle: string;
  featureRag: string;
  featureRagDesc: string;
  featureRealtime: string;
  featureRealtimeDesc: string;
  featureKnowledge: string;
  featureKnowledgeDesc: string;
  featureBriefing: string;
  featureBriefingDesc: string;
  featureRnaseq: string;
  featureRnaseqDesc: string;
  featureMl: string;
  featureMlDesc: string;

  // Knowledge Graph Section
  knowledgeUniverse: string;
  exploreKnowledge: string;
  knowledgeGraphSubtitle: string;
  launchKnowledgeGraph: string;
  geneNetworks: string;
  geneNetworksDesc: string;
  diseaseLinks: string;
  diseaseLinksDesc: string;
  paperClusters: string;
  paperClustersDesc: string;

  // CTA Section
  ctaTitle: string;
  ctaSubtitle: string;
  getStarted: string;
  learnMore: string;

  // Footer
  footerTagline: string;
  footerCopyright: string;
  quickLinks: string;
  resources: string;
  contact: string;

  // Trending Papers
  trendingNow: string;
  latestResearch: string;
  viewAll: string;
  papers: string;

  // Daily Briefing Modal
  todaysBriefing: string;
  aiGeneratedSummary: string;
  keyHighlights: string;
  readFullArticle: string;

  // Literature Review Modal
  addToReview: string;
  removeFromReview: string;
  generateSummary: string;
  exportReview: string;
  paperCount: string;

  // Chat With PDF
  uploadYourPdf: string;
  askAnything: string;
  citationSource: string;

  // Gene Network
  selectAnalysis: string;
  hubGenes: string;
  upregulated: string;
  downregulated: string;
  edges: string;
  legend: string;
  dbValidated: string;
  correlation: string;
  searchGene: string;

  // FeatureSuite Tab Labels
  featureTabPaper: string;
  featureTabRnaseq: string;
  featureTabMl: string;
  featureTabAssistant: string;

  // FeatureSuite - Paper Analysis
  paperAnalysisTitle: string;
  paperAnalysisDesc: string;
  paperBenefit1: string;
  paperBenefit2: string;
  paperBenefit3: string;
  paperBenefit4: string;
  learnMoreAbout: string;

  // FeatureSuite - RNA-seq
  rnaseqTitle: string;
  rnaseqDesc: string;
  rnaseqBenefit1: string;
  rnaseqBenefit2: string;
  rnaseqBenefit3: string;
  rnaseqBenefit4: string;

  // FeatureSuite - ML
  mlTitle: string;
  mlDesc: string;
  mlBenefit1: string;
  mlBenefit2: string;
  mlBenefit3: string;
  mlBenefit4: string;

  // FeatureSuite - AI Assistant
  assistantTitle: string;
  assistantDesc: string;
  assistantBenefit1: string;
  assistantBenefit2: string;
  assistantBenefit3: string;
  assistantBenefit4: string;

  // FeatureSuite Header
  allInOnePlatform: string;
  yourBioInsight: string;
  aiSuite: string;
  platformSubtitle: string;

  // TrendingPapers
  realTimeFromPubmed: string;
  trendingResearchPapers: string;
  trendingPapersSubtitle: string;
  multipleResearchAreas: string;
  cached: string;
  refresh: string;
  fetchingTrending: string;
  noTrendingPapers: string;
  tryRefreshing: string;
  trend: string;
  velocity: string;
  surge: string;
  citations: string;
  dataSourcedFrom: string;
  updatedRealTime: string;

  // Category Names
  catOncology: string;
  catImmunotherapy: string;
  catGeneTherapy: string;
  catNeurology: string;
  catCardiology: string;
  catInfectiousDisease: string;
  catMetabolic: string;
  catRareDisease: string;

  // Navbar
  navProduct: string;
  navEnterprise: string;
  navPricing: string;
  navDocs: string;
  logIn: string;
}

export const translations: Record<Language, Translations> = {
  en: {
    // Header
    appName: 'BioInsight',
    tagline: 'AI-Powered Biomedical Research Assistant',

    // Hero Section
    heroBadge: 'AI-Powered Research Platform',
    heroTitle1: 'Biological Insight,',
    heroTitle2: 'Starts with a Question.',
    heroSubtitle: 'Search literature, analyze data, and interpret results — in one unified platform.',
    heroHint: 'Try: "BRCA1 mutations" or "What causes drug resistance in cancer?"',

    // Search
    searchPlaceholder: 'Search genes, diseases, or ask a question...',
    searchPlaceholderPubmed: 'Search PubMed for papers...',
    searchPlaceholderDoi: 'Enter DOI (e.g., 10.1038/s41586-023-...) or URL',
    searchButton: 'Search',
    searching: 'Searching...',

    // Search modes
    localDB: 'Local DB',
    pubmedLive: 'PubMed Live',
    doiUrl: 'DOI/URL',

    // Results
    results: 'Results',
    noResults: 'No results found',
    relevanceScore: 'Relevance',
    section: 'Section',
    year: 'Year',

    // Paper details
    abstract: 'Abstract',
    authors: 'Authors',
    journal: 'Journal',
    keywords: 'Keywords',
    viewPaper: 'View Paper',
    similarPapers: 'Similar Papers',

    // Chat
    askQuestion: 'Ask a question about this paper',
    typeQuestion: 'Type your question...',
    send: 'Send',
    sources: 'Sources',
    confidence: 'Confidence',

    // Upload
    uploadPdf: 'Upload PDF',
    uploadPdfShort: 'PDF',
    uploadRnaseq: 'Upload RNA-seq Data',
    uploadRnaseqShort: 'RNA-seq',
    dragDrop: 'Drag & drop a PDF or click to browse',
    processing: 'Processing...',
    uploadSuccess: 'Paper uploaded successfully',
    uploadError: 'Upload failed',

    // Trending
    trendingPapers: 'Trending Papers',
    categories: 'Categories',
    trendAnalysis: 'Trend Analysis',
    hotTopics: 'Hot Topics',
    researchTrends: 'Research Trends',

    // Language
    language: 'Language',

    // Disease domains
    pancreaticCancer: 'Pancreatic Cancer',
    bloodCancer: 'Blood Cancer',
    glioblastoma: 'Glioblastoma',
    alzheimer: "Alzheimer's Disease",
    pcos: 'PCOS',
    pheochromocytoma: 'Pheochromocytoma',
    adhd: 'ADHD',

    // Common
    loading: 'Loading...',
    error: 'Error',
    retry: 'Retry',
    close: 'Close',
    viewMore: 'View More',
    showLess: 'Show Less',

    // Diagnostics
    diagnostics: 'Search Diagnostics',
    searchStrategy: 'Strategy',
    detectedDisease: 'Detected Disease',
    meshTerm: 'MeSH Term',
    totalCandidates: 'Total Candidates',
    filteredResults: 'Filtered Results',
    matchField: 'Match Field',

    // Match fields
    titleMatch: 'Title',
    abstractMatch: 'Abstract',
    fullTextMatch: 'Full Text',

    // Footer
    poweredBy: 'Powered by PubMedBERT & Gemini AI',

    // Tools
    bioResearchDaily: 'BIO Research Daily',
    bioResearchDailyDesc: 'Daily updates on life science research trends and analysis',
    openTool: 'Open Tool',

    // Research Tools Section
    researchTools: 'Research Tools',
    researchToolsSubtitle: 'Access AI-powered research tools designed to accelerate your scientific discoveries',
    powerfulTools: 'Powerful Tools at Your Fingertips',

    // Tool Names
    literatureReview: 'Literature Review',
    literatureReviewDesc: 'Search and analyze papers with AI-generated summaries and table view',
    chatWithPdf: 'Chat with PDF',
    chatWithPdfDesc: 'Upload papers and ask questions to get instant answers with citations',
    researchLibrary: 'Research Library',
    researchLibraryDesc: 'Organize your papers into collections with tags and annotations',
    knowledgeGraph: 'Knowledge Graph',
    knowledgeGraphDesc: 'Visualize connections between genes, diseases, and research in 3D',
    dailyBriefing: 'Daily Briefing',
    dailyBriefingDesc: 'AI-curated daily bio/healthcare research trends and news',
    geneNetwork: 'Gene Network',
    geneNetworkDesc: '3D visualization of gene co-expression networks from RNA-seq',

    // Feature Suite
    featureSuiteTitle: 'Everything You Need for Research',
    featureSuiteSubtitle: 'From literature search to data analysis, all in one platform',
    featureRag: 'Paper RAG',
    featureRagDesc: 'AI-powered paper analysis with semantic search and Q&A',
    featureRealtime: 'Real-time Search',
    featureRealtimeDesc: 'Search PubMed, bioRxiv, and more in real-time',
    featureKnowledge: 'Knowledge Graph',
    featureKnowledgeDesc: '3D visualization of research connections',
    featureBriefing: 'Daily Briefing',
    featureBriefingDesc: 'AI-curated research news and trends',
    featureRnaseq: 'RNA-seq Analysis',
    featureRnaseqDesc: '6-agent pipeline for differential expression analysis',
    featureMl: 'ML Prediction',
    featureMlDesc: 'Machine learning models for biomarker discovery',

    // Knowledge Graph Section
    knowledgeUniverse: 'Knowledge Universe',
    exploreKnowledge: 'Explore the',
    knowledgeGraphSubtitle: 'Visualize connections between genes, diseases, pathways, and research papers in an interactive 3D space. Discover hidden relationships in your research.',
    launchKnowledgeGraph: 'Launch Knowledge Graph',
    geneNetworks: 'Gene Networks',
    geneNetworksDesc: 'Explore gene-gene interactions and pathways',
    diseaseLinks: 'Disease Links',
    diseaseLinksDesc: 'Connect diseases to genetic markers',
    paperClusters: 'Paper Clusters',
    paperClustersDesc: 'See research paper relationships',

    // CTA Section
    ctaTitle: 'Ready to Accelerate Your Research?',
    ctaSubtitle: 'Join thousands of researchers using BioInsight to make discoveries faster.',
    getStarted: 'Get Started',
    learnMore: 'Learn More',

    // Footer
    footerTagline: 'AI-Powered Biomedical Research Platform',
    footerCopyright: '© 2024 BioInsight. All rights reserved.',
    quickLinks: 'Quick Links',
    resources: 'Resources',
    contact: 'Contact',

    // Trending Papers
    trendingNow: 'Trending Now',
    latestResearch: 'Latest Research',
    viewAll: 'View All',
    papers: 'papers',

    // Daily Briefing Modal
    todaysBriefing: "Today's Briefing",
    aiGeneratedSummary: 'AI-Generated Summary',
    keyHighlights: 'Key Highlights',
    readFullArticle: 'Read Full Article',

    // Literature Review Modal
    addToReview: 'Add to Review',
    removeFromReview: 'Remove',
    generateSummary: 'Generate Summary',
    exportReview: 'Export',
    paperCount: 'papers',

    // Chat With PDF
    uploadYourPdf: 'Upload your PDF',
    askAnything: 'Ask anything about the paper',
    citationSource: 'Source',

    // Gene Network
    selectAnalysis: 'Select Analysis',
    hubGenes: 'Hub Genes',
    upregulated: 'Upregulated',
    downregulated: 'Downregulated',
    edges: 'edges',
    legend: 'Legend',
    dbValidated: 'DB Validated',
    correlation: 'Correlation',
    searchGene: 'Search gene...',

    // FeatureSuite Tab Labels
    featureTabPaper: 'Paper Analysis',
    featureTabRnaseq: 'RNA-seq',
    featureTabMl: 'ML Prediction',
    featureTabAssistant: 'AI Assistant',

    // FeatureSuite - Paper Analysis
    paperAnalysisTitle: 'AI-Powered Literature Analysis',
    paperAnalysisDesc: 'Upload research papers and get instant AI summaries, key findings extraction, and semantic search across your entire library.',
    paperBenefit1: 'Automatic PDF parsing & structure recognition',
    paperBenefit2: 'PubMedBERT-powered semantic embeddings',
    paperBenefit3: 'Key findings & methodology extraction',
    paperBenefit4: 'Similar paper recommendations',
    learnMoreAbout: 'Learn more about',

    // FeatureSuite - RNA-seq
    rnaseqTitle: 'Automated RNA-seq Analysis',
    rnaseqDesc: 'Upload count matrices and get publication-ready DESeq2 analysis, visualizations, and pathway enrichment in minutes—no coding required.',
    rnaseqBenefit1: 'DESeq2 differential expression analysis',
    rnaseqBenefit2: 'Auto-generated Volcano, PCA & Heatmaps',
    rnaseqBenefit3: 'KEGG & GO pathway enrichment',
    rnaseqBenefit4: 'Batch correction (ComBat, Limma)',

    // FeatureSuite - ML
    mlTitle: 'Automated ML Model Training',
    mlDesc: 'Build predictive models from your gene signatures automatically. Train, validate, and interpret with SHAP—all without writing code.',
    mlBenefit1: 'XGBoost, Random Forest, SVM models',
    mlBenefit2: 'Automatic cross-validation',
    mlBenefit3: 'ROC-AUC, Precision, Recall metrics',
    mlBenefit4: 'SHAP feature importance analysis',

    // FeatureSuite - AI Assistant
    assistantTitle: 'Integrated Research Assistant',
    assistantDesc: 'Ask complex questions that span your papers, DEG results, and pathway data. Get context-aware answers with source citations.',
    assistantBenefit1: 'Cross-data Q&A (Papers + DEGs + Pathways)',
    assistantBenefit2: 'Experiment design suggestions',
    assistantBenefit3: 'Source citations for every answer',
    assistantBenefit4: 'Follow-up question support',

    // FeatureSuite Header
    allInOnePlatform: 'All-in-One Platform',
    yourBioInsight: 'Your BioInsight',
    aiSuite: 'AI Suite',
    platformSubtitle: 'Data analysis + Literature knowledge + ML experiments — unified in one platform for the first time.',

    // TrendingPapers
    realTimeFromPubmed: 'Real-time from PubMed',
    trendingResearchPapers: 'Trending Research Papers',
    trendingPapersSubtitle: 'Latest high-impact papers across',
    multipleResearchAreas: 'multiple research areas',
    cached: 'Cached',
    refresh: 'Refresh',
    fetchingTrending: 'Fetching trending papers from PubMed...',
    noTrendingPapers: 'No trending papers found for',
    tryRefreshing: 'Try refreshing.',
    trend: 'Trend',
    velocity: 'Velocity',
    surge: 'Surge',
    citations: 'citations',
    dataSourcedFrom: 'Data sourced from PubMed E-utilities',
    updatedRealTime: 'Updated in real-time',

    // Category Names
    catOncology: 'Oncology',
    catImmunotherapy: 'Immunotherapy',
    catGeneTherapy: 'Gene Therapy',
    catNeurology: 'Neurology',
    catCardiology: 'Cardiology',
    catInfectiousDisease: 'Infectious Disease',
    catMetabolic: 'Metabolic',
    catRareDisease: 'Rare Disease',

    // Navbar
    navProduct: 'Product',
    navEnterprise: 'Enterprise',
    navPricing: 'Pricing',
    navDocs: 'Docs',
    logIn: 'Log in',
  },

  ko: {
    // Header
    appName: 'BioInsight',
    tagline: 'AI 기반 생의학 연구 어시스턴트',

    // Hero Section
    heroBadge: 'AI 기반 연구 플랫폼',
    heroTitle1: '생물학적 인사이트,',
    heroTitle2: '질문에서 시작됩니다.',
    heroSubtitle: '문헌 검색, 데이터 분석, 결과 해석 — 하나의 통합 플랫폼에서.',
    heroHint: '예시: "BRCA1 돌연변이" 또는 "암에서 약물 내성의 원인은?"',

    // Search
    searchPlaceholder: '유전자, 질병 검색 또는 질문하기...',
    searchPlaceholderPubmed: 'PubMed에서 논문 검색...',
    searchPlaceholderDoi: 'DOI 입력 (예: 10.1038/s41586-023-...) 또는 URL',
    searchButton: '검색',
    searching: '검색 중...',

    // Search modes
    localDB: '로컬 DB',
    pubmedLive: 'PubMed 실시간',
    doiUrl: 'DOI/URL',

    // Results
    results: '결과',
    noResults: '검색 결과가 없습니다',
    relevanceScore: '관련도',
    section: '섹션',
    year: '연도',

    // Paper details
    abstract: '초록',
    authors: '저자',
    journal: '저널',
    keywords: '키워드',
    viewPaper: '논문 보기',
    similarPapers: '유사 논문',

    // Chat
    askQuestion: '이 논문에 대해 질문하세요',
    typeQuestion: '질문을 입력하세요...',
    send: '전송',
    sources: '출처',
    confidence: '신뢰도',

    // Upload
    uploadPdf: 'PDF 업로드',
    uploadPdfShort: 'PDF',
    uploadRnaseq: 'RNA-seq 데이터 업로드',
    uploadRnaseqShort: 'RNA-seq',
    dragDrop: 'PDF를 드래그하거나 클릭하여 선택',
    processing: '처리 중...',
    uploadSuccess: '논문 업로드 완료',
    uploadError: '업로드 실패',

    // Trending
    trendingPapers: '인기 논문',
    categories: '카테고리',
    trendAnalysis: '트렌드 분석',
    hotTopics: '급상승 주제',
    researchTrends: '연구 트렌드',

    // Language
    language: '언어',

    // Disease domains
    pancreaticCancer: '췌장암',
    bloodCancer: '혈액암',
    glioblastoma: '교모세포종',
    alzheimer: '알츠하이머',
    pcos: '다낭성난소증후군',
    pheochromocytoma: '갈색세포종',
    adhd: 'ADHD',

    // Common
    loading: '로딩 중...',
    error: '오류',
    retry: '다시 시도',
    close: '닫기',
    viewMore: '더 보기',
    showLess: '접기',

    // Diagnostics
    diagnostics: '검색 진단',
    searchStrategy: '검색 전략',
    detectedDisease: '감지된 질병',
    meshTerm: 'MeSH 용어',
    totalCandidates: '전체 후보',
    filteredResults: '필터링 결과',
    matchField: '매칭 필드',

    // Match fields
    titleMatch: '제목',
    abstractMatch: '초록',
    fullTextMatch: '본문',

    // Footer
    poweredBy: 'PubMedBERT & Gemini AI 기반',

    // Tools
    bioResearchDaily: 'BIO 연구 데일리',
    bioResearchDailyDesc: '매일 업데이트되는 생명과학 연구 동향과 트렌드 분석',
    openTool: '열기',

    // Research Tools Section
    researchTools: '연구 도구',
    researchToolsSubtitle: '과학적 발견을 가속화하는 AI 기반 연구 도구에 접근하세요',
    powerfulTools: '강력한 도구를 손쉽게',

    // Tool Names
    literatureReview: '문헌 리뷰',
    literatureReviewDesc: 'AI 생성 요약과 테이블 뷰로 논문 검색 및 분석',
    chatWithPdf: 'PDF와 대화',
    chatWithPdfDesc: '논문을 업로드하고 인용과 함께 즉각적인 답변 받기',
    researchLibrary: '연구 라이브러리',
    researchLibraryDesc: '태그와 주석으로 논문을 컬렉션으로 정리',
    knowledgeGraph: '지식 그래프',
    knowledgeGraphDesc: '유전자, 질병, 연구 간의 연결을 3D로 시각화',
    dailyBriefing: '데일리 브리핑',
    dailyBriefingDesc: 'AI가 선별한 바이오/헬스케어 연구 트렌드와 뉴스',
    geneNetwork: '유전자 네트워크',
    geneNetworkDesc: 'RNA-seq 유전자 공발현 네트워크 3D 시각화',

    // Feature Suite
    featureSuiteTitle: '연구에 필요한 모든 것',
    featureSuiteSubtitle: '문헌 검색부터 데이터 분석까지, 하나의 플랫폼에서',
    featureRag: 'Paper RAG',
    featureRagDesc: '시맨틱 검색과 Q&A를 통한 AI 논문 분석',
    featureRealtime: '실시간 검색',
    featureRealtimeDesc: 'PubMed, bioRxiv 등 실시간 검색',
    featureKnowledge: '지식 그래프',
    featureKnowledgeDesc: '연구 연결 관계 3D 시각화',
    featureBriefing: '데일리 브리핑',
    featureBriefingDesc: 'AI가 선별한 연구 뉴스와 트렌드',
    featureRnaseq: 'RNA-seq 분석',
    featureRnaseqDesc: '차등 발현 분석을 위한 6-에이전트 파이프라인',
    featureMl: 'ML 예측',
    featureMlDesc: '바이오마커 발견을 위한 머신러닝 모델',

    // Knowledge Graph Section
    knowledgeUniverse: '지식 우주',
    exploreKnowledge: '탐험하세요',
    knowledgeGraphSubtitle: '유전자, 질병, 경로, 연구 논문 간의 연결을 인터랙티브 3D 공간에서 시각화하세요. 연구에서 숨겨진 관계를 발견하세요.',
    launchKnowledgeGraph: '지식 그래프 시작',
    geneNetworks: '유전자 네트워크',
    geneNetworksDesc: '유전자 간 상호작용과 경로 탐색',
    diseaseLinks: '질병 연결',
    diseaseLinksDesc: '질병과 유전자 마커 연결',
    paperClusters: '논문 클러스터',
    paperClustersDesc: '연구 논문 관계 확인',

    // CTA Section
    ctaTitle: '연구를 가속화할 준비가 되셨나요?',
    ctaSubtitle: 'BioInsight를 사용하여 더 빠르게 발견하는 수천 명의 연구자들과 함께하세요.',
    getStarted: '시작하기',
    learnMore: '더 알아보기',

    // Footer
    footerTagline: 'AI 기반 생의학 연구 플랫폼',
    footerCopyright: '© 2024 BioInsight. All rights reserved.',
    quickLinks: '빠른 링크',
    resources: '리소스',
    contact: '연락처',

    // Trending Papers
    trendingNow: '지금 인기',
    latestResearch: '최신 연구',
    viewAll: '전체 보기',
    papers: '논문',

    // Daily Briefing Modal
    todaysBriefing: '오늘의 브리핑',
    aiGeneratedSummary: 'AI 생성 요약',
    keyHighlights: '주요 하이라이트',
    readFullArticle: '전체 기사 읽기',

    // Literature Review Modal
    addToReview: '리뷰에 추가',
    removeFromReview: '제거',
    generateSummary: '요약 생성',
    exportReview: '내보내기',
    paperCount: '논문',

    // Chat With PDF
    uploadYourPdf: 'PDF 업로드',
    askAnything: '논문에 대해 무엇이든 물어보세요',
    citationSource: '출처',

    // Gene Network
    selectAnalysis: '분석 선택',
    hubGenes: '허브 유전자',
    upregulated: '상향 조절',
    downregulated: '하향 조절',
    edges: '연결',
    legend: '범례',
    dbValidated: 'DB 검증됨',
    correlation: '상관관계',
    searchGene: '유전자 검색...',

    // FeatureSuite Tab Labels
    featureTabPaper: '논문 분석',
    featureTabRnaseq: 'RNA-seq',
    featureTabMl: 'ML 예측',
    featureTabAssistant: 'AI 어시스턴트',

    // FeatureSuite - Paper Analysis
    paperAnalysisTitle: 'AI 기반 문헌 분석',
    paperAnalysisDesc: '연구 논문을 업로드하고 AI 요약, 핵심 결과 추출, 전체 라이브러리 시맨틱 검색을 즉시 받아보세요.',
    paperBenefit1: '자동 PDF 파싱 및 구조 인식',
    paperBenefit2: 'PubMedBERT 기반 시맨틱 임베딩',
    paperBenefit3: '핵심 결과 및 방법론 추출',
    paperBenefit4: '유사 논문 추천',
    learnMoreAbout: '자세히 알아보기',

    // FeatureSuite - RNA-seq
    rnaseqTitle: '자동화된 RNA-seq 분석',
    rnaseqDesc: 'Count 매트릭스를 업로드하면 출판 수준의 DESeq2 분석, 시각화, 경로 농축 분석을 코딩 없이 몇 분 만에 받아보세요.',
    rnaseqBenefit1: 'DESeq2 차등 발현 분석',
    rnaseqBenefit2: 'Volcano, PCA, Heatmap 자동 생성',
    rnaseqBenefit3: 'KEGG & GO 경로 농축 분석',
    rnaseqBenefit4: '배치 보정 (ComBat, Limma)',

    // FeatureSuite - ML
    mlTitle: '자동화된 ML 모델 학습',
    mlDesc: '유전자 시그니처로 예측 모델을 자동 구축하세요. 코딩 없이 학습, 검증, SHAP 해석까지.',
    mlBenefit1: 'XGBoost, Random Forest, SVM 모델',
    mlBenefit2: '자동 교차 검증',
    mlBenefit3: 'ROC-AUC, Precision, Recall 메트릭',
    mlBenefit4: 'SHAP 특성 중요도 분석',

    // FeatureSuite - AI Assistant
    assistantTitle: '통합 연구 어시스턴트',
    assistantDesc: '논문, DEG 결과, 경로 데이터를 아우르는 복잡한 질문을 하세요. 출처 인용과 함께 맥락 인식 답변을 받으세요.',
    assistantBenefit1: '교차 데이터 Q&A (논문 + DEGs + 경로)',
    assistantBenefit2: '실험 설계 제안',
    assistantBenefit3: '모든 답변에 출처 인용',
    assistantBenefit4: '후속 질문 지원',

    // FeatureSuite Header
    allInOnePlatform: '올인원 플랫폼',
    yourBioInsight: 'BioInsight',
    aiSuite: 'AI 스위트',
    platformSubtitle: '데이터 분석 + 문헌 지식 + ML 실험 — 최초로 하나의 플랫폼에서 통합.',

    // TrendingPapers
    realTimeFromPubmed: 'PubMed 실시간',
    trendingResearchPapers: '인기 연구 논문',
    trendingPapersSubtitle: '다양한 연구 분야의 최신 고영향력 논문',
    multipleResearchAreas: '다양한 연구 분야',
    cached: '캐시됨',
    refresh: '새로고침',
    fetchingTrending: 'PubMed에서 인기 논문 가져오는 중...',
    noTrendingPapers: '인기 논문이 없습니다:',
    tryRefreshing: '새로고침을 시도해보세요.',
    trend: '트렌드',
    velocity: '속도',
    surge: '급증',
    citations: '인용',
    dataSourcedFrom: 'PubMed E-utilities 데이터 기반',
    updatedRealTime: '실시간 업데이트',

    // Category Names
    catOncology: '종양학',
    catImmunotherapy: '면역치료',
    catGeneTherapy: '유전자치료',
    catNeurology: '신경학',
    catCardiology: '심장학',
    catInfectiousDisease: '감염병',
    catMetabolic: '대사질환',
    catRareDisease: '희귀질환',

    // Navbar
    navProduct: '제품',
    navEnterprise: '엔터프라이즈',
    navPricing: '가격',
    navDocs: '문서',
    logIn: '로그인',
  },
};

export default translations;
