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
  },
};

export default translations;
