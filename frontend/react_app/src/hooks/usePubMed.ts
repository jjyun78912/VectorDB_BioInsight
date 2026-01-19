/**
 * PubMed Search Hooks
 * React Query hooks for PubMed search functionality
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { crawlerApi, chatApi } from '@/services/api';
import type { CrawlerPaper, FullTextResponse } from '@/types/api.types';

// Query keys
export const pubmedKeys = {
  all: ['pubmed'] as const,
  search: (query: string, options?: object) => [...pubmedKeys.all, 'search', query, options] as const,
  doi: (doi: string) => [...pubmedKeys.all, 'doi', doi] as const,
  fullText: (params: object) => [...pubmedKeys.all, 'fullText', params] as const,
};

interface SearchOptions {
  limit?: number;
  hybrid?: boolean;
  minYear?: number;
  sort?: 'relevance' | 'pub_date';
}

/**
 * Hook for PubMed search
 */
export function usePubMedSearch(query: string, options?: SearchOptions, enabled = true) {
  return useQuery({
    queryKey: pubmedKeys.search(query, options),
    queryFn: () => crawlerApi.searchPubMed(query, options),
    enabled: enabled && query.length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Mutation hook for PubMed search (imperative)
 */
export function usePubMedSearchMutation() {
  return useMutation({
    mutationFn: ({ query, options }: { query: string; options?: SearchOptions }) =>
      crawlerApi.searchPubMed(query, options),
  });
}

/**
 * Hook for fetching paper by DOI
 */
export function usePaperByDOI(doi: string, enabled = true) {
  return useQuery({
    queryKey: pubmedKeys.doi(doi),
    queryFn: () => crawlerApi.fetchByDOI(doi),
    enabled: enabled && doi.length > 0,
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

/**
 * Mutation hook for fetching paper by DOI (imperative)
 */
export function useFetchByDOI() {
  return useMutation({
    mutationFn: (doi: string) => crawlerApi.fetchByDOI(doi),
  });
}

/**
 * Mutation hook for fetching paper by URL
 */
export function useFetchByURL() {
  return useMutation({
    mutationFn: (url: string) => crawlerApi.fetchByURL(url),
  });
}

/**
 * Mutation hook for getting full text
 */
export function useFullText() {
  return useMutation({
    mutationFn: (params: {
      title: string;
      pmid?: string;
      pmcid?: string;
      doi?: string;
      url?: string;
      language?: string;
    }) => crawlerApi.getFullText(params),
  });
}

/**
 * Hook for translating Korean query to English
 */
export function useTranslateQuery(text: string, enabled = true) {
  const shouldTranslate = enabled && chatApi.containsKorean(text);

  return useQuery({
    queryKey: ['translate', text],
    queryFn: () => chatApi.translateQuery(text),
    enabled: shouldTranslate,
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

/**
 * Combined hook for PubMed search with auto-translation
 */
export function usePubMedSearchWithTranslation(query: string, options?: SearchOptions, enabled = true) {
  const needsTranslation = chatApi.containsKorean(query);

  // Translate if Korean
  const translation = useTranslateQuery(query, enabled && needsTranslation);

  // Use translated query if available, otherwise original
  const searchQuery = needsTranslation && translation.data ? translation.data.translated : query;

  // Search with the appropriate query
  const search = usePubMedSearch(searchQuery, options, enabled && (!needsTranslation || translation.isSuccess));

  return {
    ...search,
    isTranslating: translation.isLoading,
    translatedQuery: translation.data?.translated,
    wasTranslated: needsTranslation && translation.isSuccess,
  };
}
