/**
 * Trends API
 * Keyword trends, hot topics, and trend validation
 */

import { httpClient } from './http-client';
import type {
  TrendAnalysisResponse,
  HotTopicsResponse,
  EnhancedHotTopicsResponse,
  TrendDomain,
  EmergingTopicsResponse,
  ValidatedTrend,
  ValidatedTrendsResponse,
} from '@/types/api.types';

export interface TrendAnalysisOptions {
  startYear?: number;
  endYear?: number;
}

export interface ValidatedDomainOptions {
  limit?: number;
  minScore?: number;
}

export const trendsApi = {
  /**
   * Analyze keyword trends over 5 years
   */
  analyzeTrends: (
    keywords: string[],
    options?: TrendAnalysisOptions
  ): Promise<TrendAnalysisResponse> => {
    return httpClient.get('/trends/keyword', {
      params: {
        keywords: keywords.join(','),
        start_year: options?.startYear,
        end_year: options?.endYear,
      },
    });
  },

  /**
   * Compare keyword trends
   */
  compareTrends: (
    keyword1: string,
    keyword2: string,
    keyword3?: string
  ): Promise<TrendAnalysisResponse> => {
    return httpClient.get('/trends/compare', {
      params: { keyword1, keyword2, keyword3 },
    });
  },

  /**
   * Get hot topics in a domain (legacy)
   */
  getHotTopics: (domain: string, limit: number = 10): Promise<HotTopicsResponse> => {
    return httpClient.get(`/trends/hot-topics/${domain}`, {
      params: { limit },
    });
  },

  /**
   * Get enhanced hot topics with multi-dimensional analysis
   */
  getEnhancedHotTopics: (
    domain: string,
    limit: number = 10
  ): Promise<EnhancedHotTopicsResponse> => {
    return httpClient.get(`/trends/hot-topics-enhanced/${domain}`, {
      params: { limit },
    });
  },

  /**
   * Get available domains for hot topic analysis
   */
  getTrendDomains: (): Promise<{ domains: TrendDomain[] }> => {
    return httpClient.get('/trends/domains');
  },

  /**
   * Find emerging topics in a research area
   */
  findEmergingTopics: (
    baseKeyword: string,
    limit: number = 10
  ): Promise<EmergingTopicsResponse> => {
    return httpClient.get('/trends/emerging', {
      params: {
        base_keyword: baseKeyword,
        limit,
      },
    });
  },

  // ============== Validated Trends ==============

  /**
   * Validate a single keyword as a research trend
   */
  validateKeyword: (keyword: string): Promise<ValidatedTrend> => {
    return httpClient.get(`/trends/validate/${encodeURIComponent(keyword)}`);
  },

  /**
   * Get validated default keywords (top 5 pre-validated hot topics)
   */
  getValidatedDefaults: (): Promise<ValidatedTrendsResponse> => {
    return httpClient.get('/trends/validated-defaults');
  },

  /**
   * Get validated hot topics for a domain
   */
  getValidatedDomainTrends: (
    domain: string,
    options?: ValidatedDomainOptions
  ): Promise<ValidatedTrendsResponse> => {
    return httpClient.get(`/trends/validated-domain/${domain}`, {
      params: {
        limit: options?.limit,
        min_score: options?.minScore,
      },
    });
  },
};

export default trendsApi;
