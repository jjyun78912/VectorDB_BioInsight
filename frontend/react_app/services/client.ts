/**
 * BioInsight API Client
 */

// Use proxy in development, direct URL in production
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export interface SearchResult {
  content: string;
  relevance_score: number;
  paper_title: string;
  section: string;
  parent_section?: string;
  pmid?: string;
  year?: string;
  doi?: string;
}

export interface SearchResponse {
  query: string;
  domain: string;
  results: SearchResult[];
  total: number;
}

// Precision Search types
export interface PrecisionSearchResult {
  rank: number;
  content: string;
  relevance_score: number;
  disease_relevance: number;
  paper_title: string;
  section: string;
  pmid?: string;
  year?: string;
  doi?: string;
  match_field: 'mesh' | 'title' | 'abstract' | 'full_text' | 'none';
  matched_terms: string[];
  explanation: string;
}

export interface SearchDiagnostics {
  query: string;
  detected_disease?: string;
  mesh_term?: string;
  search_terms: string[];
  modifiers: string[];
  total_candidates: number;
  filtered_results: number;
  strategy_used: string;
  explanation: string;
}

export interface PrecisionSearchResponse {
  query: string;
  domain: string;
  results: PrecisionSearchResult[];
  diagnostics: SearchDiagnostics;
  total: number;
}

export interface SupportedDisease {
  key: string;
  mesh_term: string;
  mesh_id: string;
  synonyms: string[];
  abbreviations: string[];
}

export interface Source {
  citation_index: number;  // 1-based index for [1], [2], etc.
  paper_title: string;
  section: string;
  relevance_score: number;
  excerpt: string;
  full_content?: string;  // Full content for expanded view
}

export interface ChatResponse {
  question: string;
  answer: string;
  sources: Source[];
  confidence: number;
}

export interface Paper {
  title: string;
  doi?: string;
  year?: number;
  authors?: string[];
  chunk_count: number;
}

export interface PaperListResponse {
  papers: Paper[];
  total: number;
}

export interface StatsResponse {
  collection_name: string;
  disease_domain: string;
  total_papers: number;
  total_chunks: number;
  embedding_model: string;
  chunks_by_section: Record<string, number>;
}

// ============== Web Crawler Types ==============

export interface CrawlerPaper {
  id: string;
  source: string;
  title: string;
  title_ko?: string;  // Korean translation
  authors: string[];
  abstract: string;
  abstract_ko?: string;  // Korean translation
  journal: string;
  year: number;
  doi: string;
  pmid: string;
  pmcid: string;
  url: string;
  keywords: string[];
  citation_count: number;
  trend_score: number;
  recency_score: number;
  fetched_at: string;
  // NEW: Advanced trend metrics
  citation_velocity?: number;  // Citation growth rate
  publication_surge?: number;  // Topic publication growth rate
  influential_citation_count?: number;  // Citations from influential papers
}

export interface TrendingResponse {
  category: string;
  papers: CrawlerPaper[];
  cached: boolean;
}

export interface DailyPapersResponse {
  category: string;
  date: string;  // YYYY-MM-DD
  papers: CrawlerPaper[];
  total: number;
  cached: boolean;
}

// ============== Enhanced Trending Types ==============

export interface TrendMatch {
  id: string;
  name: string;
  emoji: string;
  color: string;
  why_trending: string;
  clinical_relevance: string;
  matched_terms: string[];
}

export interface EnhancedCrawlerPaper extends CrawlerPaper {
  trend_match?: TrendMatch;
}

export interface TrendGroup {
  trend_id: string;
  trend_name: string;
  emoji: string;
  color: string;
  why_trending: string;
  papers: EnhancedCrawlerPaper[];
  paper_count: number;
}

export interface CategoryGroup {
  category_id: string;
  category_name: string;
  emoji: string;
  trends: Record<string, TrendGroup>;
  total_papers: number;
}

export interface TrendDefinition {
  id: string;
  name: string;
  category: string;
  keywords: string[];
  why_trending: string;
  clinical_relevance: string;
  emoji: string;
  color: string;
}

export interface EnhancedTrendingResponse {
  domain: string;
  total_papers: number;
  categories: Record<string, CategoryGroup>;
  all_trends: TrendDefinition[];
  cached: boolean;
}

export interface CrawlerSearchResponse {
  query: string;
  query_translated?: string;  // English translation of Korean query
  was_translated?: boolean;  // Whether query was translated
  total_results: number;
  papers: CrawlerPaper[];
}

// ============== Trend Analysis Types ==============

export interface YearlyCount {
  year: number;
  count: number;
  growth_rate: number | null;
}

export interface KeywordTrend {
  keyword: string;
  total_count: number;
  yearly_counts: YearlyCount[];
  trend_direction: 'rising' | 'stable' | 'declining';
  growth_5yr: number | null;
  peak_year: number | null;
}

export interface TrendAnalysisResponse {
  query: string;
  years: number[];
  trends: KeywordTrend[];
  comparison_keywords: string[];
  analysis_date: string;
}

export interface HotTopic {
  keyword: string;
  recent_count: number;
  previous_count: number;
  growth_rate: number;
  sample_titles: string[];
}

export interface HotTopicsResponse {
  domain: string;
  hot_topics: HotTopic[];
  analysis_period: string;
}

export interface TrendDomain {
  key: string;
  name: string;
  keyword_count: number;
}

export interface EmergingTopic {
  topic: string;
  modifier: string;
  current_year_count: number;
  two_years_ago_count: number;
  growth_rate: number;
}

export interface EmergingTopicsResponse {
  base_keyword: string;
  emerging_topics: EmergingTopic[];
  analysis_period: string;
}

// ============== Validated Trend Types ==============

export interface ValidatedTrend {
  keyword: string;

  // Scores (0-100)
  publication_score: number;
  diversity_score: number;
  review_score: number;
  clinical_score: number;
  gap_score: number;
  total_score: number;

  // Confidence
  confidence_level: 'high' | 'medium' | 'emerging' | 'uncertain';
  confidence_emoji: string;

  // Evidence
  summary: string;
  evidence_summary: string[];

  // Raw metrics
  total_papers_5yr: number;
  growth_rate_5yr: number;
  growth_rate_yoy: number;
  unique_journals: number;
  high_if_journals: number;
  systematic_reviews: number;
  meta_analyses: number;
  active_clinical_trials: number;
  future_research_mentions: number;

  // Metadata
  validated_at: string;
  data_period: string;
}

export interface ValidatedTrendsResponse {
  trends: ValidatedTrend[];
  total_validated: number;
  methodology: string;
  last_updated: string;
}

// ============== Enhanced Hot Topics Types ==============

export interface MultiDimensionalScore {
  rising_score: number;      // YoY growth rate (0-100)
  interest_score: number;    // Search/citation interest (0-100)
  activity_score: number;    // Active research volume (0-100)
  future_score: number;      // Future research potential (0-100)
  total_score: number;       // Weighted composite score
}

export interface EnhancedHotTopic {
  keyword: string;
  scores: MultiDimensionalScore;

  // Raw metrics
  current_year_papers: number;
  previous_year_papers: number;
  growth_rate: number;
  clinical_trials: number;
  future_mentions: number;

  // Insights
  trend_label: string;       // "🔥 Explosive", "📈 Rising", etc.
  research_stage: string;    // "Early Stage", "Growth Phase", etc.
  recommendation: string;    // Brief insight
}

export interface EnhancedHotTopicsResponse {
  domain: string;
  hot_topics: EnhancedHotTopic[];
  analysis_period: string;
  methodology: string;
  last_updated: string;
}

// ============== Paper Insights Types ==============

export interface BottomLineResponse {
  summary: string | null;
  clinical_relevance: string | null;  // "High", "Medium", "Low"
  action_type: string | null;  // "Treatment", "Diagnosis", "Mechanism", etc.
}

export interface QualityResponse {
  design: string | null;  // Study design name
  design_score: number | null;  // 0-10 evidence score
  sample_size: number | null;
  quality_score: number | null;  // 0-10 overall quality
  quality_label: string | null;  // "High", "Medium", "Low"
  bias_risk: string | null;  // "Low", "Medium", "High", "Unclear"
  strengths: string[];
  limitations: string[];
}

export interface OutcomeResponse {
  outcome: string;  // e.g., "Overall Survival"
  metric: string;  // "HR", "OR", "RR"
  value: number;
  ci: string | null;  // e.g., "0.64-0.82"
  interpretation: string | null;  // e.g., "36% reduced risk"
}

export interface PopulationResponse {
  n: number | null;  // Sample size
  condition: string | null;  // e.g., "Pancreatic Cancer"
  age: string | null;  // e.g., "median 62" or "45-75"
  female_percent: number | null;
  setting: string | null;  // "Multicenter", "Single-center"
}

export interface PaperInsightsResponse {
  bottom_line: BottomLineResponse | null;
  quality: QualityResponse | null;
  key_outcomes: OutcomeResponse[];
  population: PopulationResponse | null;
}

class BioInsightAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Search the vector database
   */
  async search(
    query: string,
    domain: string = 'pheochromocytoma',
    options?: { section?: string; topK?: number }
  ): Promise<SearchResponse> {
    const params = new URLSearchParams({
      query,
      domain,
      ...(options?.section && { section: options.section }),
      ...(options?.topK && { top_k: options.topK.toString() }),
    });

    const response = await fetch(`${this.baseUrl}/search/?${params}`);
    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Precision search with MeSH vocabulary and field-aware ranking
   * domain='auto' will auto-detect disease from query
   */
  async precisionSearch(
    query: string,
    domain: string = 'auto',
    options?: { section?: string; topK?: number; requireTitleMatch?: boolean }
  ): Promise<PrecisionSearchResponse> {
    const params = new URLSearchParams({
      query,
      domain,
      ...(options?.section && { section: options.section }),
      ...(options?.topK && { top_k: options.topK.toString() }),
      require_title_match: (options?.requireTitleMatch !== false).toString(),
    });

    const response = await fetch(`${this.baseUrl}/search/precision?${params}`);
    if (!response.ok) {
      throw new Error(`Precision search failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get list of supported diseases with MeSH terms
   */
  async getSupportedDiseases(): Promise<{ diseases: SupportedDisease[]; total: number }> {
    const response = await fetch(`${this.baseUrl}/search/supported-diseases`);
    if (!response.ok) {
      throw new Error(`Get diseases failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Ask a question using RAG
   */
  async ask(
    question: string,
    domain: string = 'pheochromocytoma',
    options?: { section?: string; topK?: number }
  ): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/chat/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        domain,
        section: options?.section,
        top_k: options?.topK || 5,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ask failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * List all papers
   */
  async listPapers(domain: string = 'pheochromocytoma'): Promise<PaperListResponse> {
    const response = await fetch(`${this.baseUrl}/papers?domain=${domain}`);
    if (!response.ok) {
      throw new Error(`List papers failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get collection stats
   */
  async getStats(domain: string = 'pheochromocytoma'): Promise<StatsResponse> {
    const response = await fetch(`${this.baseUrl}/papers/stats?domain=${domain}`);
    if (!response.ok) {
      throw new Error(`Get stats failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Upload and index a PDF
   */
  async uploadPaper(file: File, domain: string = 'pheochromocytoma'): Promise<{
    success: boolean;
    message: string;
    paper_title?: string;
    chunks_created: number;
  }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('domain', domain);

    const response = await fetch(`${this.baseUrl}/papers/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Summarize a paper
   */
  async summarize(title: string, domain: string = 'pheochromocytoma', brief: boolean = false): Promise<{
    title: string;
    summary: string;
    key_findings: string[];
    methodology?: string;
  }> {
    const response = await fetch(`${this.baseUrl}/chat/summarize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, domain, brief }),
    });

    if (!response.ok) {
      throw new Error(`Summarize failed: ${response.statusText}`);
    }
    return response.json();
  }

  // ============== Paper Agent API ==============

  /**
   * Upload a PDF to create a dedicated chat agent
   */
  async uploadToAgent(file: File): Promise<{
    success: boolean;
    session_id: string;
    paper_title: string;
    chunks_indexed: number;
    message: string;
  }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/chat/agent/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Ask a question to a paper-specific agent
   */
  async askAgent(sessionId: string, question: string, topK: number = 5): Promise<{
    question: string;
    answer: string;
    sources: Source[];
    confidence: number;
    is_answerable: boolean;
  }> {
    const response = await fetch(`${this.baseUrl}/chat/agent/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        question,
        top_k: topK,
      }),
    });

    if (!response.ok) {
      // Include status code in error message for better error handling
      const errorDetail = await response.text().catch(() => response.statusText);
      throw new Error(`Ask failed (${response.status}): ${errorDetail}`);
    }
    return response.json();
  }

  /**
   * Get agent session info
   */
  async getAgentSession(sessionId: string): Promise<{
    session_id: string;
    paper_title: string;
    chunks_count: number;
  }> {
    const response = await fetch(`${this.baseUrl}/chat/agent/session/${sessionId}`);
    if (!response.ok) {
      throw new Error(`Get session failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Delete agent session
   */
  async deleteAgentSession(sessionId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/chat/agent/session/${sessionId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(`Delete session failed: ${response.statusText}`);
    }
  }

  // ============== Web Crawler API ==============

  /**
   * Get trending papers from PubMed
   * @param category - Paper category (oncology, immunotherapy, etc.)
   * @param limit - Number of papers to fetch
   * @param lang - Language for paper metadata: 'en' or 'ko'
   */
  async getTrendingPapers(
    category: string = 'oncology',
    limit: number = 10,
    lang: string = 'en'
  ): Promise<TrendingResponse> {
    const response = await fetch(
      `${this.baseUrl}/crawler/trending/${category}?limit=${limit}&lang=${lang}`
    );
    if (!response.ok) {
      throw new Error(`Get trending failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get enhanced trending papers grouped by defined trends
   * Returns papers with "why this is trending" context
   */
  async getEnhancedTrendingPapers(
    domain: string = 'oncology',
    limit: number = 20
  ): Promise<EnhancedTrendingResponse> {
    const response = await fetch(
      `${this.baseUrl}/crawler/trending-enhanced/${domain}?limit=${limit}`
    );
    if (!response.ok) {
      throw new Error(`Get enhanced trending failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get list of trending categories
   */
  async getTrendingCategories(): Promise<{ categories: string[] }> {
    const response = await fetch(`${this.baseUrl}/crawler/categories`);
    if (!response.ok) {
      throw new Error(`Get categories failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get today's papers (within last 3 days, sorted by recency)
   * Cache refreshes at KST 07:00 daily
   */
  async getDailyPapers(
    category: string = 'oncology',
    limit: number = 10
  ): Promise<DailyPapersResponse> {
    const response = await fetch(
      `${this.baseUrl}/crawler/daily/${category}?limit=${limit}`
    );
    if (!response.ok) {
      throw new Error(`Get daily papers failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Search PubMed in real-time (uses hybrid search by default)
   */
  async searchPubMed(
    query: string,
    options?: {
      limit?: number;
      hybrid?: boolean;
      minYear?: number;
      sort?: 'relevance' | 'pub_date';
    }
  ): Promise<CrawlerSearchResponse> {
    const params = new URLSearchParams({
      q: query,
      limit: (options?.limit || 20).toString(),  // Default to 20 results
      hybrid: (options?.hybrid !== false).toString(), // default true
      sort: options?.sort || 'relevance',
    });

    if (options?.minYear) {
      params.set('min_year', options.minYear.toString());
    }

    const response = await fetch(`${this.baseUrl}/crawler/search?${params}`);
    if (!response.ok) {
      throw new Error(`PubMed search failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Fetch paper by DOI
   */
  async fetchByDOI(doi: string): Promise<CrawlerPaper> {
    const response = await fetch(`${this.baseUrl}/crawler/fetch/doi`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doi }),
    });
    if (!response.ok) {
      throw new Error(`DOI fetch failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Fetch paper by URL (DOI, PubMed, or PMC URL)
   */
  async fetchByURL(url: string): Promise<CrawlerPaper> {
    const response = await fetch(`${this.baseUrl}/crawler/fetch/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    if (!response.ok) {
      throw new Error(`URL fetch failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get full text of a paper and create a chat session
   * Tries PMC (free) first, then Playwright for publisher pages
   */
  async getFullText(params: {
    title: string;
    pmid?: string;
    pmcid?: string;
    doi?: string;
    url?: string;
    language?: string;  // "ko" for Korean, "en" for English (default)
  }): Promise<{
    success: boolean;
    session_id?: string;
    paper_title: string;
    full_text_preview: string;
    full_text_length: number;
    chunks_created: number;
    source: string;
    ai_summary?: {
      summary: string;
      key_findings: string[];
      methodology: string;
    };
    error?: string;
  }> {
    const response = await fetch(`${this.baseUrl}/crawler/full-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!response.ok) {
      throw new Error(`Full text fetch failed: ${response.statusText}`);
    }
    return response.json();
  }

  // ============== Abstract-based AI API ==============

  /**
   * Summarize a paper from its abstract (for PubMed papers not in local DB)
   * @param language - "ko" for Korean output, "en" for English (default)
   */
  async summarizeAbstract(title: string, abstract: string, language: string = 'en'): Promise<{
    title: string;
    summary: string;
    key_points: string[];
  }> {
    const response = await fetch(`${this.baseUrl}/chat/summarize-abstract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, abstract, language }),
    });

    if (!response.ok) {
      throw new Error(`Summarize failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Ask a question about a paper based on its abstract
   */
  async askAbstract(title: string, abstract: string, question: string): Promise<{
    question: string;
    answer: string;
  }> {
    const response = await fetch(`${this.baseUrl}/chat/ask-abstract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, abstract, question }),
    });

    if (!response.ok) {
      throw new Error(`Ask failed: ${response.statusText}`);
    }
    return response.json();
  }

  // ============== Translation API ==============

  /**
   * Translate Korean search query to English for PubMed search
   */
  async translateQuery(text: string): Promise<{
    original: string;
    translated: string;
    detected_lang: string;
    is_biomedical: boolean;
  }> {
    const response = await fetch(`${this.baseUrl}/chat/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`Translation failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Check if text contains Korean characters
   */
  containsKorean(text: string): boolean {
    return /[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]/.test(text);
  }

  // ============== Paper Insights API ==============

  /**
   * Get comprehensive paper insights (Bottom Line, Quality, Outcomes, Population)
   */
  async getPaperInsights(
    title: string,
    abstract: string,
    fullText?: string
  ): Promise<PaperInsightsResponse> {
    const response = await fetch(`${this.baseUrl}/chat/insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, abstract, full_text: fullText }),
    });

    if (!response.ok) {
      throw new Error(`Get insights failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get just the bottom line summary (fastest)
   */
  async getBottomLine(
    title: string,
    abstract: string
  ): Promise<BottomLineResponse> {
    const response = await fetch(`${this.baseUrl}/chat/insights/bottom-line`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, abstract }),
    });

    if (!response.ok) {
      throw new Error(`Get bottom line failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get just the quality assessment (fast, rule-based)
   */
  async getQualityScore(
    title: string,
    abstract: string,
    fullText?: string
  ): Promise<QualityResponse> {
    const response = await fetch(`${this.baseUrl}/chat/insights/quality`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, abstract, full_text: fullText }),
    });

    if (!response.ok) {
      throw new Error(`Get quality failed: ${response.statusText}`);
    }
    return response.json();
  }

  // ============== Trend Analysis API ==============

  /**
   * Analyze keyword trends over 5 years
   */
  async analyzeTrends(
    keywords: string[],
    options?: { startYear?: number; endYear?: number }
  ): Promise<TrendAnalysisResponse> {
    const params = new URLSearchParams({
      keywords: keywords.join(','),
      ...(options?.startYear && { start_year: options.startYear.toString() }),
      ...(options?.endYear && { end_year: options.endYear.toString() }),
    });

    const response = await fetch(`${this.baseUrl}/trends/keyword?${params}`);
    if (!response.ok) {
      throw new Error(`Trend analysis failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Compare keyword trends
   */
  async compareTrends(
    keyword1: string,
    keyword2: string,
    keyword3?: string
  ): Promise<TrendAnalysisResponse> {
    const params = new URLSearchParams({
      keyword1,
      keyword2,
      ...(keyword3 && { keyword3 }),
    });

    const response = await fetch(`${this.baseUrl}/trends/compare?${params}`);
    if (!response.ok) {
      throw new Error(`Trend comparison failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get hot topics in a domain (legacy)
   */
  async getHotTopics(domain: string, limit: number = 10): Promise<HotTopicsResponse> {
    const response = await fetch(
      `${this.baseUrl}/trends/hot-topics/${domain}?limit=${limit}`
    );
    if (!response.ok) {
      throw new Error(`Hot topics failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get enhanced hot topics with multi-dimensional analysis
   */
  async getEnhancedHotTopics(domain: string, limit: number = 10): Promise<EnhancedHotTopicsResponse> {
    const response = await fetch(
      `${this.baseUrl}/trends/hot-topics-enhanced/${domain}?limit=${limit}`
    );
    if (!response.ok) {
      throw new Error(`Enhanced hot topics failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get available domains for hot topic analysis
   */
  async getTrendDomains(): Promise<{ domains: TrendDomain[] }> {
    const response = await fetch(`${this.baseUrl}/trends/domains`);
    if (!response.ok) {
      throw new Error(`Get domains failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Find emerging topics in a research area
   */
  async findEmergingTopics(
    baseKeyword: string,
    limit: number = 10
  ): Promise<EmergingTopicsResponse> {
    const params = new URLSearchParams({
      base_keyword: baseKeyword,
      limit: limit.toString(),
    });

    const response = await fetch(`${this.baseUrl}/trends/emerging?${params}`);
    if (!response.ok) {
      throw new Error(`Emerging topics failed: ${response.statusText}`);
    }
    return response.json();
  }

  // ============== Validated Trend API ==============

  /**
   * Validate a single keyword as a research trend
   * Returns comprehensive validation with multi-dimensional scores
   */
  async validateKeyword(keyword: string): Promise<ValidatedTrend> {
    const response = await fetch(
      `${this.baseUrl}/trends/validate/${encodeURIComponent(keyword)}`
    );
    if (!response.ok) {
      throw new Error(`Validation failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get validated default keywords for TrendAnalysis
   * Returns top 5 pre-validated hot topics
   */
  async getValidatedDefaults(): Promise<ValidatedTrendsResponse> {
    const response = await fetch(`${this.baseUrl}/trends/validated-defaults`);
    if (!response.ok) {
      throw new Error(`Get validated defaults failed: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Get validated hot topics for a domain
   * Each topic includes validation scores and evidence
   */
  async getValidatedDomainTrends(
    domain: string,
    options?: { limit?: number; minScore?: number }
  ): Promise<ValidatedTrendsResponse> {
    const params = new URLSearchParams({
      ...(options?.limit && { limit: options.limit.toString() }),
      ...(options?.minScore && { min_score: options.minScore.toString() }),
    });

    const response = await fetch(
      `${this.baseUrl}/trends/validated-domain/${domain}?${params}`
    );
    if (!response.ok) {
      throw new Error(`Validated domain trends failed: ${response.statusText}`);
    }
    return response.json();
  }
}

export const api = new BioInsightAPI();
export default api;
