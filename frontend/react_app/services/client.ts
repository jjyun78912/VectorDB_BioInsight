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
  authors: string[];
  abstract: string;
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
}

export interface TrendingResponse {
  category: string;
  papers: CrawlerPaper[];
  cached: boolean;
}

export interface CrawlerSearchResponse {
  query: string;
  total_results: number;
  papers: CrawlerPaper[];
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
      throw new Error(`Ask failed: ${response.statusText}`);
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
   */
  async getTrendingPapers(
    category: string = 'oncology',
    limit: number = 10
  ): Promise<TrendingResponse> {
    const response = await fetch(
      `${this.baseUrl}/crawler/trending/${category}?limit=${limit}`
    );
    if (!response.ok) {
      throw new Error(`Get trending failed: ${response.statusText}`);
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
   * Search PubMed in real-time (uses hybrid search by default)
   */
  async searchPubMed(
    query: string,
    options?: { limit?: number; hybrid?: boolean; minYear?: number }
  ): Promise<CrawlerSearchResponse> {
    const params = new URLSearchParams({
      q: query,
      limit: (options?.limit || 10).toString(),
      hybrid: (options?.hybrid !== false).toString(), // default true
      ...(options?.minYear && { min_year: options.minYear.toString() }),
    });

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

  // ============== Abstract-based AI API ==============

  /**
   * Summarize a paper from its abstract (for PubMed papers not in local DB)
   */
  async summarizeAbstract(title: string, abstract: string): Promise<{
    title: string;
    summary: string;
    key_points: string[];
  }> {
    const response = await fetch(`${this.baseUrl}/chat/summarize-abstract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, abstract }),
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
}

export const api = new BioInsightAPI();
export default api;
