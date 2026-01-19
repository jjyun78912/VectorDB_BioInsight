/**
 * Search API
 * Local vector DB search and precision search
 */

import { httpClient } from './http-client';
import type {
  SearchResponse,
  PrecisionSearchResponse,
  SupportedDisease,
  PaperExplanation,
} from '@/types/api.types';

export interface SearchOptions {
  section?: string;
  topK?: number;
}

export interface PrecisionSearchOptions extends SearchOptions {
  requireTitleMatch?: boolean;
}

export interface ExplainOptions {
  section?: string;
  pmid?: string;
  year?: string;
  matchedTerms?: string[];
}

export const searchApi = {
  /**
   * Search the vector database
   */
  search: (
    query: string,
    domain: string = 'pheochromocytoma',
    options?: SearchOptions
  ): Promise<SearchResponse> => {
    return httpClient.get('/search/', {
      params: {
        query,
        domain,
        section: options?.section,
        top_k: options?.topK,
      },
    });
  },

  /**
   * Precision search with MeSH vocabulary and field-aware ranking
   * domain='auto' will auto-detect disease from query
   */
  precisionSearch: (
    query: string,
    domain: string = 'auto',
    options?: PrecisionSearchOptions
  ): Promise<PrecisionSearchResponse> => {
    return httpClient.get('/search/precision', {
      params: {
        query,
        domain,
        section: options?.section,
        top_k: options?.topK,
        require_title_match: options?.requireTitleMatch !== false,
      },
    });
  },

  /**
   * Get list of supported diseases with MeSH terms
   */
  getSupportedDiseases: (): Promise<{ diseases: SupportedDisease[]; total: number }> => {
    return httpClient.get('/search/supported-diseases');
  },

  /**
   * Get detailed explanation for why a paper was recommended
   */
  explainPaper: (
    query: string,
    title: string,
    content: string,
    options?: ExplainOptions
  ): Promise<PaperExplanation> => {
    return httpClient.post('/search/explain', {
      query,
      title,
      content,
      section: options?.section || '',
      pmid: options?.pmid,
      year: options?.year,
      matched_terms: options?.matchedTerms || [],
    });
  },

  /**
   * Quick explanation (rule-based, faster)
   */
  explainPaperQuick: (
    query: string,
    title: string,
    content: string,
    section?: string
  ): Promise<PaperExplanation> => {
    return httpClient.get('/search/explain-quick', {
      params: {
        query,
        title,
        content: content || '',
        section: section || '',
      },
    });
  },
};

export default searchApi;
