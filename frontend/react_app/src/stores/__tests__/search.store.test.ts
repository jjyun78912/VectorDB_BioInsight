import { describe, it, expect, beforeEach } from 'vitest';
import { useSearchStore } from '../search.store';

describe('SearchStore', () => {
  beforeEach(() => {
    // Reset store before each test
    useSearchStore.getState().reset();
  });

  it('has correct initial state', () => {
    const state = useSearchStore.getState();
    expect(state.query).toBe('');
    expect(state.domain).toBe('auto');
    expect(state.results).toEqual([]);
    expect(state.precisionResults).toEqual([]);
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
    expect(state.searchMode).toBe('precision');
  });

  it('setQuery updates query', () => {
    useSearchStore.getState().setQuery('test query');
    expect(useSearchStore.getState().query).toBe('test query');
  });

  it('setDomain updates domain', () => {
    useSearchStore.getState().setDomain('breast_cancer');
    expect(useSearchStore.getState().domain).toBe('breast_cancer');
  });

  it('setResults updates results and clears precision results', () => {
    const mockResults = [
      { content: 'test', relevance_score: 0.9, paper_title: 'Test Paper', section: 'abstract' }
    ];
    useSearchStore.getState().setResults(mockResults, 1);

    const state = useSearchStore.getState();
    expect(state.results).toEqual(mockResults);
    expect(state.total).toBe(1);
    expect(state.precisionResults).toEqual([]);
    expect(state.diagnostics).toBeNull();
  });

  it('setPrecisionResults updates precision results and clears standard results', () => {
    const mockResults = [
      {
        rank: 1,
        content: 'test',
        relevance_score: 0.9,
        disease_relevance: 0.8,
        paper_title: 'Test Paper',
        section: 'abstract',
        match_field: 'title' as const,
        matched_terms: ['test'],
        explanation: 'Test explanation'
      }
    ];
    const mockDiagnostics = {
      query: 'test',
      search_terms: ['test'],
      modifiers: [],
      total_candidates: 10,
      filtered_results: 1,
      strategy_used: 'precision',
      explanation: 'Test'
    };

    useSearchStore.getState().setPrecisionResults(mockResults, mockDiagnostics, 1);

    const state = useSearchStore.getState();
    expect(state.precisionResults).toEqual(mockResults);
    expect(state.diagnostics).toEqual(mockDiagnostics);
    expect(state.total).toBe(1);
    expect(state.results).toEqual([]);
  });

  it('setLoading updates loading state', () => {
    useSearchStore.getState().setLoading(true);
    expect(useSearchStore.getState().isLoading).toBe(true);

    useSearchStore.getState().setLoading(false);
    expect(useSearchStore.getState().isLoading).toBe(false);
  });

  it('setError updates error and stops loading', () => {
    useSearchStore.getState().setLoading(true);
    useSearchStore.getState().setError('Test error');

    const state = useSearchStore.getState();
    expect(state.error).toBe('Test error');
    expect(state.isLoading).toBe(false);
  });

  it('setSearchMode updates search mode', () => {
    useSearchStore.getState().setSearchMode('standard');
    expect(useSearchStore.getState().searchMode).toBe('standard');

    useSearchStore.getState().setSearchMode('precision');
    expect(useSearchStore.getState().searchMode).toBe('precision');
  });

  it('addRecentSearch adds search to history and limits to 10', () => {
    for (let i = 0; i < 15; i++) {
      useSearchStore.getState().addRecentSearch(`query ${i}`);
    }

    const state = useSearchStore.getState();
    expect(state.recentSearches.length).toBe(10);
    expect(state.recentSearches[0]).toBe('query 14'); // Most recent first
  });

  it('addRecentSearch moves existing query to front', () => {
    useSearchStore.getState().addRecentSearch('query 1');
    useSearchStore.getState().addRecentSearch('query 2');
    useSearchStore.getState().addRecentSearch('query 1'); // Add again

    const state = useSearchStore.getState();
    expect(state.recentSearches[0]).toBe('query 1');
    expect(state.recentSearches.filter(s => s === 'query 1').length).toBe(1);
  });

  it('clearResults clears all results', () => {
    useSearchStore.getState().setResults([{ content: 'test', relevance_score: 0.9, paper_title: 'Test', section: 'abstract' }], 1);
    useSearchStore.getState().clearResults();

    const state = useSearchStore.getState();
    expect(state.results).toEqual([]);
    expect(state.precisionResults).toEqual([]);
    expect(state.diagnostics).toBeNull();
    expect(state.total).toBe(0);
  });

  it('reset returns to initial state', () => {
    useSearchStore.getState().setQuery('test');
    useSearchStore.getState().setDomain('breast_cancer');
    useSearchStore.getState().setLoading(true);
    useSearchStore.getState().reset();

    const state = useSearchStore.getState();
    expect(state.query).toBe('');
    expect(state.domain).toBe('auto');
    expect(state.isLoading).toBe(false);
  });
});
