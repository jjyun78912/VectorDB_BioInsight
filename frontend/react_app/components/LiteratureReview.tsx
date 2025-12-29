import React, { useState, useEffect } from 'react';
import {
  Search, Filter, Download, Plus, Trash2, BookOpen, FileText,
  ChevronDown, ChevronRight, Table, Grid3X3, MessageSquare,
  Sparkles, Loader2, X, ExternalLink, Copy, Check, Star,
  ArrowUpDown, Calendar, Users, Building2
} from 'lucide-react';
import api, { SearchResult } from '../services/client';

interface Paper {
  id: string;
  title: string;
  authors: string[];
  year: number;
  journal?: string;
  abstract?: string;
  citations?: number;
  relevance: number;
  doi?: string;
  pmid?: string;
  isStarred?: boolean;
  content?: string;
  section?: string;
}

interface LiteratureReviewProps {
  isOpen: boolean;
  onClose: () => void;
  initialQuery?: string;
}

type ViewMode = 'table' | 'grid';
type SortField = 'relevance' | 'year' | 'citations' | 'title';

export const LiteratureReview: React.FC<LiteratureReviewProps> = ({ isOpen, onClose, initialQuery = '' }) => {
  const [query, setQuery] = useState(initialQuery);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [sortField, setSortField] = useState<SortField>('relevance');
  const [sortAsc, setSortAsc] = useState(false);
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set());
  const [expandedPaper, setExpandedPaper] = useState<string | null>(null);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Search papers
  const handleSearch = async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    setPapers([]);
    setAiSummary(null);

    try {
      // Search across all domains
      const domains = ['pancreatic_cancer', 'blood_cancer', 'glioblastoma', 'alzheimer', 'pcos'];
      const allResults: Paper[] = [];

      for (const domain of domains) {
        try {
          const response = await api.search(query, domain, { topK: 10 });
          const domainPapers = response.results.map((r, idx) => ({
            id: `${domain}-${idx}-${Date.now()}`,
            title: r.paper_title,
            authors: [],
            year: 2024,
            journal: domain.replace('_', ' ').toUpperCase(),
            abstract: r.content,
            relevance: r.relevance_score,
            section: r.section,
            content: r.content,
          }));
          allResults.push(...domainPapers);
        } catch (e) {
          console.log(`No results from ${domain}`);
        }
      }

      // Deduplicate by title and sort by relevance
      const uniquePapers = allResults.reduce((acc, paper) => {
        const existing = acc.find(p => p.title === paper.title);
        if (!existing) {
          acc.push(paper);
        } else if (paper.relevance > existing.relevance) {
          const idx = acc.indexOf(existing);
          acc[idx] = paper;
        }
        return acc;
      }, [] as Paper[]);

      uniquePapers.sort((a, b) => b.relevance - a.relevance);
      setPapers(uniquePapers.slice(0, 30));
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Generate AI summary
  const generateSummary = async () => {
    if (papers.length === 0) return;

    setIsGeneratingSummary(true);
    try {
      const response = await api.ask(
        `Based on the search results, provide a comprehensive literature review summary about: ${query}`,
        'pancreatic_cancer'
      );
      setAiSummary(response.answer);
    } catch (err) {
      console.error('Summary generation failed:', err);
    } finally {
      setIsGeneratingSummary(false);
    }
  };

  // Sort papers
  const sortedPapers = [...papers].sort((a, b) => {
    let comparison = 0;
    switch (sortField) {
      case 'relevance':
        comparison = b.relevance - a.relevance;
        break;
      case 'year':
        comparison = b.year - a.year;
        break;
      case 'citations':
        comparison = (b.citations || 0) - (a.citations || 0);
        break;
      case 'title':
        comparison = a.title.localeCompare(b.title);
        break;
    }
    return sortAsc ? -comparison : comparison;
  });

  // Toggle paper selection
  const toggleSelect = (id: string) => {
    const newSelected = new Set(selectedPapers);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedPapers(newSelected);
  };

  // Toggle star
  const toggleStar = (id: string) => {
    setPapers(papers.map(p =>
      p.id === id ? { ...p, isStarred: !p.isStarred } : p
    ));
  };

  // Copy citation
  const copyCitation = (paper: Paper) => {
    const citation = `${paper.authors.join(', ')} (${paper.year}). ${paper.title}. ${paper.journal || 'Journal'}.`;
    navigator.clipboard.writeText(citation);
    setCopiedId(paper.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Export selected
  const exportSelected = () => {
    const selected = papers.filter(p => selectedPapers.has(p.id));
    const csv = [
      ['Title', 'Authors', 'Year', 'Journal', 'Relevance', 'Abstract'].join(','),
      ...selected.map(p => [
        `"${p.title}"`,
        `"${p.authors.join('; ')}"`,
        p.year,
        `"${p.journal || ''}"`,
        p.relevance.toFixed(1),
        `"${(p.abstract || '').substring(0, 200)}..."`
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `literature_review_${query.replace(/\s+/g, '_')}.csv`;
    a.click();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] bg-gradient-to-br from-violet-50 via-purple-50 to-indigo-100">
      {/* Header */}
      <header className="glass-4 border-b border-purple-100/50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onClose}
              className="p-2 hover:bg-purple-100/50 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
                <BookOpen className="w-4 h-4 text-white" />
              </div>
              <span className="font-bold text-gray-900">Literature Review</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {selectedPapers.size > 0 && (
              <button
                onClick={exportSelected}
                className="px-4 py-2 glass-3 border border-purple-200/50 rounded-lg text-sm font-medium text-gray-700 hover:bg-purple-50/50 transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Export ({selectedPapers.size})
              </button>
            )}
            <button
              onClick={generateSummary}
              disabled={papers.length === 0 || isGeneratingSummary}
              className="px-4 py-2 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-lg text-sm font-semibold hover:from-violet-700 hover:to-purple-700 transition-all flex items-center gap-2 disabled:opacity-50 btn-glow"
            >
              {isGeneratingSummary ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              Generate Summary
            </button>
          </div>
        </div>
      </header>

      {/* Search Bar */}
      <div className="glass-3 border-b border-purple-100/50 px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={(e) => { e.preventDefault(); handleSearch(); }} className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-purple-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your research question or topic..."
              className="w-full pl-12 pr-32 py-4 glass-4 border border-purple-200/50 rounded-xl text-gray-900 placeholder:text-gray-400 focus:ring-2 focus:ring-purple-400/50 focus:border-purple-300 text-lg"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-6 py-2.5 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-lg font-semibold hover:from-violet-700 hover:to-purple-700 transition-all disabled:opacity-50"
            >
              {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Search'}
            </button>
          </form>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex">
        {/* Results Panel */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* AI Summary */}
          {aiSummary && (
            <div className="max-w-7xl mx-auto mb-6 glass-3 rounded-2xl border border-purple-200/50 p-6 animate-appear">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <Sparkles className="w-4 h-4 text-white" />
                </div>
                <h3 className="font-bold text-gray-900">AI Literature Summary</h3>
              </div>
              <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{aiSummary}</p>
            </div>
          )}

          {/* Results Header */}
          {papers.length > 0 && (
            <div className="max-w-7xl mx-auto mb-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <span className="text-sm font-medium text-gray-600">
                  {papers.length} papers found
                </span>
                <div className="flex items-center gap-1 glass-2 rounded-lg p-1">
                  <button
                    onClick={() => setViewMode('table')}
                    className={`p-2 rounded-md transition-colors ${viewMode === 'table' ? 'bg-white shadow-sm text-purple-600' : 'text-gray-500 hover:text-gray-700'}`}
                  >
                    <Table className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`p-2 rounded-md transition-colors ${viewMode === 'grid' ? 'bg-white shadow-sm text-purple-600' : 'text-gray-500 hover:text-gray-700'}`}
                  >
                    <Grid3X3 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">Sort by:</span>
                <select
                  value={sortField}
                  onChange={(e) => setSortField(e.target.value as SortField)}
                  className="glass-3 border border-purple-200/50 rounded-lg px-3 py-1.5 text-sm text-gray-700 focus:ring-2 focus:ring-purple-400/50"
                >
                  <option value="relevance">Relevance</option>
                  <option value="year">Year</option>
                  <option value="citations">Citations</option>
                  <option value="title">Title</option>
                </select>
                <button
                  onClick={() => setSortAsc(!sortAsc)}
                  className="p-2 glass-3 border border-purple-200/50 rounded-lg hover:bg-purple-50/50 transition-colors"
                >
                  <ArrowUpDown className="w-4 h-4 text-gray-600" />
                </button>
              </div>
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="max-w-7xl mx-auto flex flex-col items-center justify-center py-20 gap-4">
              <Loader2 className="w-10 h-10 text-purple-500 animate-spin" />
              <p className="text-gray-600">Searching across literature databases...</p>
            </div>
          )}

          {/* Empty State */}
          {!isLoading && papers.length === 0 && (
            <div className="max-w-7xl mx-auto flex flex-col items-center justify-center py-20 gap-4">
              <div className="w-20 h-20 glass-3 rounded-2xl flex items-center justify-center">
                <Search className="w-10 h-10 text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-800">Start your literature review</h3>
              <p className="text-gray-500 text-center max-w-md">
                Enter a research question or topic above to search across our database of biomedical papers
              </p>
            </div>
          )}

          {/* Table View */}
          {!isLoading && papers.length > 0 && viewMode === 'table' && (
            <div className="max-w-7xl mx-auto glass-3 rounded-2xl border border-purple-100/50 overflow-hidden">
              <table className="w-full">
                <thead className="glass-4 border-b border-purple-100/50">
                  <tr>
                    <th className="w-10 px-4 py-3">
                      <input
                        type="checkbox"
                        checked={selectedPapers.size === papers.length}
                        onChange={() => {
                          if (selectedPapers.size === papers.length) {
                            setSelectedPapers(new Set());
                          } else {
                            setSelectedPapers(new Set(papers.map(p => p.id)));
                          }
                        }}
                        className="rounded border-purple-300 text-purple-600 focus:ring-purple-500"
                      />
                    </th>
                    <th className="w-10 px-2 py-3"></th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider">Title</th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider w-24">Year</th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider w-32">Domain</th>
                    <th className="text-left px-4 py-3 text-xs font-semibold text-gray-600 uppercase tracking-wider w-28">Relevance</th>
                    <th className="w-20 px-4 py-3"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-purple-50">
                  {sortedPapers.map((paper) => (
                    <React.Fragment key={paper.id}>
                      <tr
                        className={`hover:bg-purple-50/50 transition-colors cursor-pointer ${expandedPaper === paper.id ? 'bg-purple-50/50' : ''}`}
                        onClick={() => setExpandedPaper(expandedPaper === paper.id ? null : paper.id)}
                      >
                        <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked={selectedPapers.has(paper.id)}
                            onChange={() => toggleSelect(paper.id)}
                            className="rounded border-purple-300 text-purple-600 focus:ring-purple-500"
                          />
                        </td>
                        <td className="px-2 py-3" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => toggleStar(paper.id)}
                            className={`p-1 rounded transition-colors ${paper.isStarred ? 'text-yellow-500' : 'text-gray-300 hover:text-yellow-500'}`}
                          >
                            <Star className="w-4 h-4" fill={paper.isStarred ? 'currentColor' : 'none'} />
                          </button>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            {expandedPaper === paper.id ? (
                              <ChevronDown className="w-4 h-4 text-gray-400 flex-shrink-0" />
                            ) : (
                              <ChevronRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
                            )}
                            <span className="text-sm font-medium text-gray-900 line-clamp-2">{paper.title}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm text-gray-600">{paper.year}</span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs font-medium text-purple-600 bg-purple-100/80 px-2 py-1 rounded-full">
                            {paper.journal}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-16 h-2 bg-purple-100 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-gradient-to-r from-violet-500 to-purple-500 rounded-full"
                                style={{ width: `${paper.relevance}%` }}
                              />
                            </div>
                            <span className="text-xs font-medium text-gray-600">{paper.relevance.toFixed(0)}%</span>
                          </div>
                        </td>
                        <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => copyCitation(paper)}
                            className="p-2 text-gray-400 hover:text-purple-600 hover:bg-purple-100/50 rounded-lg transition-colors"
                            title="Copy citation"
                          >
                            {copiedId === paper.id ? (
                              <Check className="w-4 h-4 text-emerald-500" />
                            ) : (
                              <Copy className="w-4 h-4" />
                            )}
                          </button>
                        </td>
                      </tr>
                      {expandedPaper === paper.id && (
                        <tr className="bg-purple-50/30">
                          <td colSpan={7} className="px-6 py-4">
                            <div className="pl-10 space-y-3 animate-appear">
                              <div>
                                <span className="text-xs font-semibold text-gray-500 uppercase">Section: </span>
                                <span className="text-xs text-purple-600">{paper.section}</span>
                              </div>
                              <p className="text-sm text-gray-700 leading-relaxed">{paper.abstract}</p>
                              <div className="flex items-center gap-3 pt-2">
                                <button className="px-3 py-1.5 glass-3 border border-purple-200/50 rounded-lg text-xs font-medium text-gray-700 hover:bg-purple-50/50 transition-colors flex items-center gap-1">
                                  <MessageSquare className="w-3.5 h-3.5" />
                                  Chat with Paper
                                </button>
                                <button className="px-3 py-1.5 glass-3 border border-purple-200/50 rounded-lg text-xs font-medium text-gray-700 hover:bg-purple-50/50 transition-colors flex items-center gap-1">
                                  <ExternalLink className="w-3.5 h-3.5" />
                                  View Source
                                </button>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Grid View */}
          {!isLoading && papers.length > 0 && viewMode === 'grid' && (
            <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {sortedPapers.map((paper) => (
                <div
                  key={paper.id}
                  className="glass-3 rounded-2xl border border-purple-100/50 p-5 card-hover animate-appear"
                >
                  <div className="flex items-start justify-between mb-3">
                    <span className="text-xs font-medium text-purple-600 bg-purple-100/80 px-2 py-1 rounded-full">
                      {paper.journal}
                    </span>
                    <button
                      onClick={() => toggleStar(paper.id)}
                      className={`p-1 rounded transition-colors ${paper.isStarred ? 'text-yellow-500' : 'text-gray-300 hover:text-yellow-500'}`}
                    >
                      <Star className="w-4 h-4" fill={paper.isStarred ? 'currentColor' : 'none'} />
                    </button>
                  </div>
                  <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2">{paper.title}</h3>
                  <p className="text-sm text-gray-600 mb-4 line-clamp-3">{paper.abstract}</p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      <span className="text-sm text-gray-600">{paper.year}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-12 h-2 bg-purple-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-violet-500 to-purple-500 rounded-full"
                          style={{ width: `${paper.relevance}%` }}
                        />
                      </div>
                      <span className="text-xs font-medium text-gray-600">{paper.relevance.toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
