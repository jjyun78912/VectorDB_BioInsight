/**
 * Search Hooks
 * React Query hooks for search functionality
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { searchApi, chatApi } from '@/services/api';
import { useSearchStore } from '@/stores';
import type {
  SearchResponse,
  PrecisionSearchResponse,
  SupportedDisease,
  PaperExplanation,
} from '@/types/api.types';

// Query keys
export const searchKeys = {
  all: ['search'] as const,
  standard: (query: string, domain: string) => [...searchKeys.all, 'standard', query, domain] as const,
  precision: (query: string, domain: string) => [...searchKeys.all, 'precision', query, domain] as const,
  diseases: () => [...searchKeys.all, 'diseases'] as const,
  explain: (query: string, title: string) => [...searchKeys.all, 'explain', query, title] as const,
};

/**
 * Hook for standard vector search
 */
export function useStandardSearch(query: string, domain: string, enabled = true) {
  const { setLoading, setResults, setError, addRecentSearch } = useSearchStore();

  return useQuery({
    queryKey: searchKeys.standard(query, domain),
    queryFn: async () => {
      setLoading(true);
      try {
        const result = await searchApi.search(query, domain);
        setResults(result.results, result.total);
        addRecentSearch(query);
        return result;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Search failed';
        setError(message);
        throw error;
      } finally {
        setLoading(false);
      }
    },
    enabled: enabled && query.length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook for precision search with MeSH vocabulary
 */
export function usePrecisionSearch(query: string, domain: string = 'auto', enabled = true) {
  const { setLoading, setPrecisionResults, setError, addRecentSearch } = useSearchStore();

  return useQuery({
    queryKey: searchKeys.precision(query, domain),
    queryFn: async () => {
      setLoading(true);
      try {
        const result = await searchApi.precisionSearch(query, domain);
        setPrecisionResults(result.results, result.diagnostics, result.total);
        addRecentSearch(query);
        return result;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Precision search failed';
        setError(message);
        throw error;
      } finally {
        setLoading(false);
      }
    },
    enabled: enabled && query.length > 0,
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Hook to get supported diseases
 */
export function useSupportedDiseases() {
  return useQuery({
    queryKey: searchKeys.diseases(),
    queryFn: searchApi.getSupportedDiseases,
    staleTime: 30 * 60 * 1000, // 30 minutes
  });
}

/**
 * Mutation hook for paper explanation
 */
export function useExplainPaper() {
  return useMutation({
    mutationFn: ({
      query,
      title,
      content,
      options,
    }: {
      query: string;
      title: string;
      content: string;
      options?: { section?: string; pmid?: string; year?: string; matchedTerms?: string[] };
    }) => searchApi.explainPaper(query, title, content, options),
  });
}

/**
 * Hook for quick paper explanation (rule-based)
 */
export function useExplainPaperQuick(
  query: string,
  title: string,
  content: string,
  section?: string,
  enabled = true
) {
  return useQuery({
    queryKey: searchKeys.explain(query, title),
    queryFn: () => searchApi.explainPaperQuick(query, title, content, section),
    enabled: enabled && query.length > 0 && title.length > 0,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Combined search hook with auto mode selection
 */
export function useSearch() {
  const { query, domain, searchMode, setQuery, setDomain, setSearchMode, clearResults } = useSearchStore();
  const queryClient = useQueryClient();

  const standardSearch = useStandardSearch(query, domain, searchMode === 'standard' && query.length > 0);
  const precisionSearch = usePrecisionSearch(query, domain, searchMode === 'precision' && query.length > 0);

  const activeSearch = searchMode === 'precision' ? precisionSearch : standardSearch;

  const search = async (newQuery: string, newDomain?: string) => {
    setQuery(newQuery);
    if (newDomain) setDomain(newDomain);
    // Query will auto-execute due to enabled condition
  };

  const reset = () => {
    clearResults();
    setQuery('');
  };

  return {
    query,
    domain,
    searchMode,
    setQuery,
    setDomain,
    setSearchMode,
    search,
    reset,
    isLoading: activeSearch.isLoading,
    error: activeSearch.error,
    data: activeSearch.data,
    refetch: activeSearch.refetch,
  };
}
