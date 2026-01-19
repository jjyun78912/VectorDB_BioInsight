/**
 * Crawler API
 * PubMed search, trending papers, and full-text fetching
 */

import { httpClient } from './http-client';
import type {
  TrendingResponse,
  EnhancedTrendingResponse,
  DailyPapersResponse,
  CrawlerSearchResponse,
  CrawlerPaper,
  FullTextResponse,
} from '@/types/api.types';

export interface PubMedSearchOptions {
  limit?: number;
  hybrid?: boolean;
  minYear?: number;
  sort?: 'relevance' | 'pub_date';
}

export interface FullTextParams {
  title: string;
  pmid?: string;
  pmcid?: string;
  doi?: string;
  url?: string;
  language?: string;
}

export const crawlerApi = {
  /**
   * Get trending papers from PubMed
   */
  getTrendingPapers: (
    category: string = 'oncology',
    limit: number = 10,
    lang: string = 'en'
  ): Promise<TrendingResponse> => {
    return httpClient.get(`/crawler/trending/${category}`, {
      params: { limit, lang },
    });
  },

  /**
   * Get enhanced trending papers grouped by defined trends
   */
  getEnhancedTrendingPapers: (
    domain: string = 'oncology',
    limit: number = 20
  ): Promise<EnhancedTrendingResponse> => {
    return httpClient.get(`/crawler/trending-enhanced/${domain}`, {
      params: { limit },
    });
  },

  /**
   * Get list of trending categories
   */
  getTrendingCategories: (): Promise<{ categories: string[] }> => {
    return httpClient.get('/crawler/categories');
  },

  /**
   * Get today's papers (within last 3 days, sorted by recency)
   */
  getDailyPapers: (
    category: string = 'oncology',
    limit: number = 10
  ): Promise<DailyPapersResponse> => {
    return httpClient.get(`/crawler/daily/${category}`, {
      params: { limit },
    });
  },

  /**
   * Search PubMed in real-time (uses hybrid search by default)
   */
  searchPubMed: (
    query: string,
    options?: PubMedSearchOptions
  ): Promise<CrawlerSearchResponse> => {
    return httpClient.get('/crawler/search', {
      params: {
        q: query,
        limit: options?.limit || 20,
        hybrid: options?.hybrid !== false,
        sort: options?.sort || 'relevance',
        min_year: options?.minYear,
      },
    });
  },

  /**
   * Fetch paper by DOI
   */
  fetchByDOI: (doi: string): Promise<CrawlerPaper> => {
    return httpClient.post('/crawler/fetch/doi', { doi });
  },

  /**
   * Fetch paper by URL (DOI, PubMed, or PMC URL)
   */
  fetchByURL: (url: string): Promise<CrawlerPaper> => {
    return httpClient.post('/crawler/fetch/url', { url });
  },

  /**
   * Get full text of a paper and create a chat session
   */
  getFullText: (params: FullTextParams): Promise<FullTextResponse> => {
    return httpClient.post('/crawler/full-text', params);
  },
};

export default crawlerApi;
