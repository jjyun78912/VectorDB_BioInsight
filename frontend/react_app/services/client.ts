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
}

export interface SearchResponse {
  query: string;
  domain: string;
  results: SearchResult[];
  total: number;
}

export interface Source {
  paper_title: string;
  section: string;
  relevance_score: number;
  excerpt: string;
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
}

export const api = new BioInsightAPI();
export default api;
