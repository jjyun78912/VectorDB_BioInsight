/**
 * API Module Exports
 * Centralized API exports for clean imports
 */

// HTTP Client
export { httpClient, ApiError } from './http-client';

// Domain APIs
export { searchApi } from './search.api';
export { chatApi } from './chat.api';
export { papersApi } from './papers.api';
export { crawlerApi } from './crawler.api';
export { trendsApi } from './trends.api';

// Re-export types
export type * from '@/types/api.types';

// Legacy compatibility - unified API object
import { searchApi } from './search.api';
import { chatApi } from './chat.api';
import { papersApi } from './papers.api';
import { crawlerApi } from './crawler.api';
import { trendsApi } from './trends.api';

/**
 * Unified API object for backward compatibility
 * Prefer using individual API modules for new code
 */
export const api = {
  // Search
  search: searchApi.search,
  precisionSearch: searchApi.precisionSearch,
  getSupportedDiseases: searchApi.getSupportedDiseases,
  explainPaper: searchApi.explainPaper,
  explainPaperQuick: searchApi.explainPaperQuick,

  // Chat
  ask: chatApi.ask,
  summarize: chatApi.summarize,
  summarizeAbstract: chatApi.summarizeAbstract,
  askAbstract: chatApi.askAbstract,
  translateQuery: chatApi.translateQuery,
  translateText: chatApi.translateText,
  containsKorean: chatApi.containsKorean,
  getPaperInsights: chatApi.getPaperInsights,
  getBottomLine: chatApi.getBottomLine,
  getQualityScore: chatApi.getQualityScore,
  uploadToAgent: chatApi.uploadToAgent,
  askAgent: chatApi.askAgent,
  getAgentSession: chatApi.getAgentSession,
  deleteAgentSession: chatApi.deleteAgentSession,

  // Papers
  listPapers: papersApi.listPapers,
  getStats: papersApi.getStats,
  uploadPaper: papersApi.uploadPaper,

  // Crawler
  getTrendingPapers: crawlerApi.getTrendingPapers,
  getEnhancedTrendingPapers: crawlerApi.getEnhancedTrendingPapers,
  getTrendingCategories: crawlerApi.getTrendingCategories,
  getDailyPapers: crawlerApi.getDailyPapers,
  searchPubMed: crawlerApi.searchPubMed,
  fetchByDOI: crawlerApi.fetchByDOI,
  fetchByURL: crawlerApi.fetchByURL,
  getFullText: crawlerApi.getFullText,

  // Trends
  analyzeTrends: trendsApi.analyzeTrends,
  compareTrends: trendsApi.compareTrends,
  getHotTopics: trendsApi.getHotTopics,
  getEnhancedHotTopics: trendsApi.getEnhancedHotTopics,
  getTrendDomains: trendsApi.getTrendDomains,
  findEmergingTopics: trendsApi.findEmergingTopics,
  validateKeyword: trendsApi.validateKeyword,
  getValidatedDefaults: trendsApi.getValidatedDefaults,
  getValidatedDomainTrends: trendsApi.getValidatedDomainTrends,
};

export default api;
