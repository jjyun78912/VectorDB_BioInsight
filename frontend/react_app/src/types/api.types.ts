/**
 * API Type Definitions
 * Centralized types for all API responses and requests
 */

// ============== Search Types ==============

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

// ============== Chat Types ==============

export interface Source {
  citation_index: number;
  paper_title: string;
  section: string;
  relevance_score: number;
  excerpt: string;
  full_content?: string;
}

export interface ChatResponse {
  question: string;
  answer: string;
  sources: Source[];
  confidence: number;
}

export interface AgentChatResponse extends ChatResponse {
  is_answerable: boolean;
}

export interface AgentSession {
  session_id: string;
  paper_title: string;
  chunks_count: number;
}

export interface AgentUploadResponse {
  success: boolean;
  session_id: string;
  paper_title: string;
  chunks_indexed: number;
  message: string;
}

// ============== Paper Types ==============

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

export interface PaperUploadResponse {
  success: boolean;
  message: string;
  paper_title?: string;
  chunks_created: number;
}

export interface PaperSummaryResponse {
  title: string;
  summary: string;
  key_findings: string[];
  methodology?: string;
}

// ============== Crawler Types ==============

export interface CrawlerPaper {
  id: string;
  source: string;
  title: string;
  title_ko?: string;
  authors: string[];
  abstract: string;
  abstract_ko?: string;
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
  citation_velocity?: number;
  publication_surge?: number;
  influential_citation_count?: number;
}

export interface TrendingResponse {
  category: string;
  papers: CrawlerPaper[];
  cached: boolean;
}

export interface DailyPapersResponse {
  category: string;
  date: string;
  papers: CrawlerPaper[];
  total: number;
  cached: boolean;
}

export interface CrawlerSearchResponse {
  query: string;
  query_translated?: string;
  was_translated?: boolean;
  total_results: number;
  papers: CrawlerPaper[];
}

export interface FullTextResponse {
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
  publication_score: number;
  diversity_score: number;
  review_score: number;
  clinical_score: number;
  gap_score: number;
  total_score: number;
  confidence_level: 'high' | 'medium' | 'emerging' | 'uncertain';
  confidence_emoji: string;
  summary: string;
  evidence_summary: string[];
  total_papers_5yr: number;
  growth_rate_5yr: number;
  growth_rate_yoy: number;
  unique_journals: number;
  high_if_journals: number;
  systematic_reviews: number;
  meta_analyses: number;
  active_clinical_trials: number;
  future_research_mentions: number;
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
  rising_score: number;
  interest_score: number;
  activity_score: number;
  future_score: number;
  total_score: number;
}

export interface EnhancedHotTopic {
  keyword: string;
  scores: MultiDimensionalScore;
  current_year_papers: number;
  previous_year_papers: number;
  growth_rate: number;
  clinical_trials: number;
  future_mentions: number;
  trend_label: string;
  research_stage: string;
  recommendation: string;
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
  clinical_relevance: string | null;
  action_type: string | null;
}

export interface QualityResponse {
  design: string | null;
  design_score: number | null;
  sample_size: number | null;
  quality_score: number | null;
  quality_label: string | null;
  bias_risk: string | null;
  strengths: string[];
  limitations: string[];
}

export interface OutcomeResponse {
  outcome: string;
  metric: string;
  value: number;
  ci: string | null;
  interpretation: string | null;
}

export interface PopulationResponse {
  n: number | null;
  condition: string | null;
  age: string | null;
  female_percent: number | null;
  setting: string | null;
}

export interface PaperInsightsResponse {
  bottom_line: BottomLineResponse | null;
  quality: QualityResponse | null;
  key_outcomes: OutcomeResponse[];
  population: PopulationResponse | null;
}

// ============== Paper Explanation Types ==============

export interface PaperCharacteristics {
  study_type: string;
  study_design: string;
  main_finding: string;
  methodology: string;
  sample_info: string;
  evidence_level: string;
  clinical_relevance: string;
  strengths: string[];
  limitations: string[];
  key_genes: string[];
  key_pathways: string[];
  techniques: string[];
}

export interface PaperExplanation {
  why_recommended: string;
  relevance_factors: string[];
  query_match_explanation: string;
  characteristics: PaperCharacteristics | null;
  relevance_score: number;
  novelty_score: number;
  quality_score: number;
  model_used: string;
}

// ============== Translation Types ==============

export interface TranslationResponse {
  original: string;
  translated: string;
  detected_lang: string;
  is_biomedical: boolean;
}

export interface AbstractSummaryResponse {
  title: string;
  summary: string;
  key_points: string[];
}

export interface AbstractAskResponse {
  question: string;
  answer: string;
}
