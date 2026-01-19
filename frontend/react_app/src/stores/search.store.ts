/**
 * Search Store
 * Global state for search functionality
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { SearchResult, PrecisionSearchResult, SearchDiagnostics } from '@/types/api.types';

interface SearchState {
  // Search query and results
  query: string;
  domain: string;
  results: SearchResult[];
  precisionResults: PrecisionSearchResult[];
  diagnostics: SearchDiagnostics | null;
  total: number;

  // UI state
  isLoading: boolean;
  error: string | null;
  searchMode: 'standard' | 'precision';

  // Search history
  recentSearches: string[];

  // Actions
  setQuery: (query: string) => void;
  setDomain: (domain: string) => void;
  setResults: (results: SearchResult[], total: number) => void;
  setPrecisionResults: (results: PrecisionSearchResult[], diagnostics: SearchDiagnostics, total: number) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSearchMode: (mode: 'standard' | 'precision') => void;
  addRecentSearch: (query: string) => void;
  clearResults: () => void;
  reset: () => void;
}

const initialState = {
  query: '',
  domain: 'auto',
  results: [],
  precisionResults: [],
  diagnostics: null,
  total: 0,
  isLoading: false,
  error: null,
  searchMode: 'precision' as const,
  recentSearches: [],
};

export const useSearchStore = create<SearchState>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        setQuery: (query) => set({ query }),

        setDomain: (domain) => set({ domain }),

        setResults: (results, total) =>
          set({
            results,
            total,
            precisionResults: [],
            diagnostics: null,
            error: null,
          }),

        setPrecisionResults: (results, diagnostics, total) =>
          set({
            precisionResults: results,
            diagnostics,
            total,
            results: [],
            error: null,
          }),

        setLoading: (isLoading) => set({ isLoading }),

        setError: (error) => set({ error, isLoading: false }),

        setSearchMode: (searchMode) => set({ searchMode }),

        addRecentSearch: (query) => {
          const { recentSearches } = get();
          const filtered = recentSearches.filter((s) => s !== query);
          const updated = [query, ...filtered].slice(0, 10);
          set({ recentSearches: updated });
        },

        clearResults: () =>
          set({
            results: [],
            precisionResults: [],
            diagnostics: null,
            total: 0,
            error: null,
          }),

        reset: () => set(initialState),
      }),
      {
        name: 'bioinsight-search',
        partialize: (state) => ({
          recentSearches: state.recentSearches,
          searchMode: state.searchMode,
          domain: state.domain,
        }),
      }
    ),
    { name: 'SearchStore' }
  )
);
