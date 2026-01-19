/**
 * Trending Hooks
 * React Query hooks for trending papers and topics
 */

import { useQuery } from '@tanstack/react-query';
import { crawlerApi, trendsApi } from '@/services/api';

// Query keys
export const trendingKeys = {
  all: ['trending'] as const,
  papers: (category: string, limit: number) => [...trendingKeys.all, 'papers', category, limit] as const,
  enhanced: (domain: string, limit: number) => [...trendingKeys.all, 'enhanced', domain, limit] as const,
  daily: (category: string, limit: number) => [...trendingKeys.all, 'daily', category, limit] as const,
  categories: () => [...trendingKeys.all, 'categories'] as const,
  hotTopics: (domain: string, limit: number) => [...trendingKeys.all, 'hotTopics', domain, limit] as const,
  enhancedHotTopics: (domain: string, limit: number) => [...trendingKeys.all, 'enhancedHotTopics', domain, limit] as const,
  domains: () => [...trendingKeys.all, 'domains'] as const,
  validated: (domain: string) => [...trendingKeys.all, 'validated', domain] as const,
};

/**
 * Hook for trending papers
 */
export function useTrendingPapers(category = 'oncology', limit = 10, lang = 'en') {
  return useQuery({
    queryKey: trendingKeys.papers(category, limit),
    queryFn: () => crawlerApi.getTrendingPapers(category, limit, lang),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Hook for enhanced trending papers with trend context
 */
export function useEnhancedTrendingPapers(domain = 'oncology', limit = 20) {
  return useQuery({
    queryKey: trendingKeys.enhanced(domain, limit),
    queryFn: () => crawlerApi.getEnhancedTrendingPapers(domain, limit),
    staleTime: 10 * 60 * 1000,
  });
}

/**
 * Hook for daily papers
 */
export function useDailyPapers(category = 'oncology', limit = 10) {
  return useQuery({
    queryKey: trendingKeys.daily(category, limit),
    queryFn: () => crawlerApi.getDailyPapers(category, limit),
    staleTime: 30 * 60 * 1000, // 30 minutes
  });
}

/**
 * Hook for trending categories
 */
export function useTrendingCategories() {
  return useQuery({
    queryKey: trendingKeys.categories(),
    queryFn: crawlerApi.getTrendingCategories,
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

/**
 * Hook for hot topics
 */
export function useHotTopics(domain: string, limit = 10) {
  return useQuery({
    queryKey: trendingKeys.hotTopics(domain, limit),
    queryFn: () => trendsApi.getHotTopics(domain, limit),
    staleTime: 15 * 60 * 1000, // 15 minutes
  });
}

/**
 * Hook for enhanced hot topics
 */
export function useEnhancedHotTopics(domain: string, limit = 10) {
  return useQuery({
    queryKey: trendingKeys.enhancedHotTopics(domain, limit),
    queryFn: () => trendsApi.getEnhancedHotTopics(domain, limit),
    staleTime: 15 * 60 * 1000,
  });
}

/**
 * Hook for trend domains
 */
export function useTrendDomains() {
  return useQuery({
    queryKey: trendingKeys.domains(),
    queryFn: trendsApi.getTrendDomains,
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

/**
 * Hook for validated domain trends
 */
export function useValidatedDomainTrends(domain: string, options?: { limit?: number; minScore?: number }) {
  return useQuery({
    queryKey: trendingKeys.validated(domain),
    queryFn: () => trendsApi.getValidatedDomainTrends(domain, options),
    staleTime: 30 * 60 * 1000, // 30 minutes
  });
}
