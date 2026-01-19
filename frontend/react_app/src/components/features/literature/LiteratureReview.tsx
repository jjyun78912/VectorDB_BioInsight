import React, { useState } from 'react';
import {
  Download, Trash2, BookOpen, FileText,
  ChevronDown, ChevronRight, Table, Grid3X3, MessageSquare,
  Sparkles, Loader2, X, ExternalLink, Copy, Check, Star,
  ArrowUpDown, Calendar, Users, Building2, Columns, FileDown
} from 'lucide-react';
import api from '../services/client';
import type { ReviewPaper } from '../App';

interface LiteratureReviewProps {
  isOpen: boolean;
  onClose: () => void;
  papers: ReviewPaper[];
  onRemovePaper: (id: string) => void;
  onClearAll: () => void;
}

type ViewMode = 'table' | 'grid' | 'compare';
type SortField = 'relevance' | 'year' | 'title';

export const LiteratureReview: React.FC<LiteratureReviewProps> = ({
  isOpen,
  onClose,
  papers,
  onRemovePaper,
  onClearAll
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [sortField, setSortField] = useState<SortField>('relevance');
  const [sortAsc, setSortAsc] = useState(false);
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set());
  const [expandedPaper, setExpandedPaper] = useState<string | null>(null);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [comparePapers, setComparePapers] = useState<string[]>([]);

  // Chat with paper state
  const [chatPaper, setChatPaper] = useState<ReviewPaper | null>(null);
  const [chatQuestion, setChatQuestion] = useState('');
  const [chatAnswer, setChatAnswer] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);

  // Generate AI summary from papers
  const generateSummary = async () => {
    if (papers.length === 0) return;

    setIsGeneratingSummary(true);
    try {
      const topPapers = papers.slice(0, 10);
      const combinedAbstract = topPapers
        .map((p, i) => `[${i + 1}] "${p.title}" (${p.year})\n${p.abstract || 'No abstract available.'}`)
        .join('\n\n');

      const response = await api.askAbstract(
        `Literature Review`,
        combinedAbstract,
        `Based on these ${topPapers.length} research papers, provide a comprehensive literature review summary. Include key findings, common themes, and research trends. Write in a scholarly tone.`
      );
      setAiSummary(response.answer);
    } catch (err) {
      console.error('Summary generation failed:', err);
      setAiSummary('Failed to generate summary. Please try again.');
    } finally {
      setIsGeneratingSummary(false);
    }
  };

  // Sort papers
  const sortedPapers = [...papers].sort((a, b) => {
    let comparison = 0;
    switch (sortField) {
      case 'relevance':
        comparison = (b.relevance || 0) - (a.relevance || 0);
        break;
      case 'year':
        comparison = b.year - a.year;
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

  // Toggle compare
  const toggleCompare = (id: string) => {
    setComparePapers(prev => {
      if (prev.includes(id)) {
        return prev.filter(p => p !== id);
      }
      if (prev.length >= 3) {
        return [...prev.slice(1), id]; // Max 3 papers
      }
      return [...prev, id];
    });
  };

  // Copy citation
  const copyCitation = (paper: ReviewPaper) => {
    const citation = `${paper.authors?.join(', ') || 'Unknown'} (${paper.year}). ${paper.title}. ${paper.journal || 'Journal'}.`;
    navigator.clipboard.writeText(citation);
    setCopiedId(paper.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Open paper source
  const openPaperSource = (paper: ReviewPaper) => {
    if (paper.doi) {
      window.open(`https://doi.org/${paper.doi}`, '_blank');
    } else if (paper.pmid) {
      window.open(`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`, '_blank');
    } else {
      const searchUrl = `https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(paper.title)}`;
      window.open(searchUrl, '_blank');
    }
  };

  // Chat with paper
  const askAboutPaper = async (paper: ReviewPaper, question: string) => {
    if (!question.trim() || !paper.abstract) return;

    setIsChatLoading(true);
    try {
      const response = await api.askAbstract(paper.title, paper.abstract, question);
      setChatAnswer(response.answer);
    } catch (err) {
      setChatAnswer('Failed to get answer. Please try again.');
    } finally {
      setIsChatLoading(false);
    }
  };

  // Export to CSV
  const exportCSV = () => {
    const papersToExport = selectedPapers.size > 0
      ? papers.filter(p => selectedPapers.has(p.id))
      : papers;

    const csv = [
      ['Title', 'Authors', 'Year', 'Journal', 'DOI', 'PMID', 'Abstract'].join(','),
      ...papersToExport.map(p => [
        `"${p.title.replace(/"/g, '""')}"`,
        `"${p.authors?.join('; ') || ''}"`,
        p.year,
        `"${p.journal || ''}"`,
        p.doi || '',
        p.pmid || '',
        `"${(p.abstract || '').replace(/"/g, '""').substring(0, 500)}"`
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `literature_review_${Date.now()}.csv`;
    a.click();
  };

  // Export to BibTeX
  const exportBibTeX = () => {
    const papersToExport = selectedPapers.size > 0
      ? papers.filter(p => selectedPapers.has(p.id))
      : papers;

    const bibtex = papersToExport.map((p, i) => {
      const key = `paper${i + 1}_${p.year}`;
      return `@article{${key},
  title = {${p.title}},
  author = {${p.authors?.join(' and ') || 'Unknown'}},
  year = {${p.year}},
  journal = {${p.journal || 'Unknown'}},
  doi = {${p.doi || ''}},
  pmid = {${p.pmid || ''}}
}`;
    }).join('\n\n');

    const blob = new Blob([bibtex], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `literature_review_${Date.now()}.bib`;
    a.click();
  };

  // Get papers for comparison
  const papersToCompare = papers.filter(p => comparePapers.includes(p.id));

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] bg-gradient-to-br from-violet-50 via-purple-50 to-indigo-100 flex flex-col">
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
              <span className="text-sm text-gray-500">({papers.length} papers)</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {papers.length > 0 && (
              <>
                <div className="flex items-center gap-1 glass-2 rounded-lg p-1">
                  <button
                    onClick={exportCSV}
                    className="px-3 py-1.5 rounded-md text-xs font-medium text-gray-600 hover:bg-white/50 transition-colors flex items-center gap-1"
                    title="Export to CSV"
                  >
                    <Download className="w-3.5 h-3.5" />
                    CSV
                  </button>
                  <button
                    onClick={exportBibTeX}
                    className="px-3 py-1.5 rounded-md text-xs font-medium text-gray-600 hover:bg-white/50 transition-colors flex items-center gap-1"
                    title="Export to BibTeX"
                  >
                    <FileDown className="w-3.5 h-3.5" />
                    BibTeX
                  </button>
                </div>
                <button
                  onClick={generateSummary}
                  disabled={isGeneratingSummary}
                  className="px-4 py-2 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-lg text-sm font-semibold hover:from-violet-700 hover:to-purple-700 transition-all flex items-center gap-2 disabled:opacity-50 btn-glow"
                >
                  {isGeneratingSummary ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                  Generate Summary
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Empty State */}
      {papers.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-20 h-20 bg-purple-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <BookOpen className="w-10 h-10 text-purple-400" />
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">No papers yet</h3>
            <p className="text-gray-500 mb-6 max-w-md">
              Search for papers in the main search bar and click "Add to Review" to start building your literature review.
            </p>
            <button
              onClick={onClose}
              className="px-6 py-3 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-xl font-semibold hover:from-violet-700 hover:to-purple-700 transition-all"
            >
              Go to Search
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* Toolbar */}
          <div className="glass-3 border-b border-purple-100/50 px-6 py-3">
            <div className="max-w-7xl mx-auto flex items-center justify-between">
              <div className="flex items-center gap-4">
                {/* View Mode Toggle */}
                <div className="flex items-center gap-1 glass-2 rounded-lg p-1">
                  <button
                    onClick={() => setViewMode('table')}
                    className={`p-2 rounded-md transition-colors ${viewMode === 'table' ? 'bg-white shadow-sm text-purple-600' : 'text-gray-400'}`}
                  >
                    <Table className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`p-2 rounded-md transition-colors ${viewMode === 'grid' ? 'bg-white shadow-sm text-purple-600' : 'text-gray-400'}`}
                  >
                    <Grid3X3 className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode('compare')}
                    className={`p-2 rounded-md transition-colors ${viewMode === 'compare' ? 'bg-white shadow-sm text-purple-600' : 'text-gray-400'}`}
                    title="Compare papers"
                  >
                    <Columns className="w-4 h-4" />
                  </button>
                </div>

                {viewMode === 'compare' && (
                  <span className="text-sm text-gray-500">
                    Select up to 3 papers to compare ({comparePapers.length}/3)
                  </span>
                )}

                {selectedPapers.size > 0 && (
                  <span className="text-sm text-purple-600 font-medium">
                    {selectedPapers.size} selected
                  </span>
                )}
              </div>

              <div className="flex items-center gap-3">
                {/* Sort */}
                <select
                  value={sortField}
                  onChange={(e) => setSortField(e.target.value as SortField)}
                  className="glass-3 border border-purple-200/50 rounded-lg px-3 py-1.5 text-sm text-gray-700"
                >
                  <option value="relevance">Sort by Relevance</option>
                  <option value="year">Sort by Year</option>
                  <option value="title">Sort by Title</option>
                </select>

                <button
                  onClick={() => setSortAsc(!sortAsc)}
                  className="p-2 glass-3 border border-purple-200/50 rounded-lg"
                >
                  <ArrowUpDown className={`w-4 h-4 ${sortAsc ? 'text-purple-600' : 'text-gray-400'}`} />
                </button>

                {/* Clear All */}
                <button
                  onClick={onClearAll}
                  className="px-3 py-1.5 text-red-500 hover:bg-red-50 rounded-lg text-sm font-medium transition-colors"
                >
                  Clear All
                </button>
              </div>
            </div>
          </div>

          {/* Main Content */}
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

            {/* Compare View */}
            {viewMode === 'compare' && (
              <div className="max-w-7xl mx-auto">
                {papersToCompare.length === 0 ? (
                  <div className="text-center py-12 text-gray-500">
                    Click on papers below to add them to comparison
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                    {papersToCompare.map((paper) => (
                      <div key={paper.id} className="glass-3 rounded-xl border border-purple-200/50 p-4">
                        <div className="flex items-start justify-between mb-3">
                          <span className="text-xs font-medium text-purple-600 bg-purple-100 px-2 py-0.5 rounded-full">
                            {paper.year}
                          </span>
                          <button
                            onClick={() => toggleCompare(paper.id)}
                            className="p-1 hover:bg-red-100 rounded text-red-500"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                        <h4 className="font-semibold text-gray-900 text-sm mb-2 line-clamp-2">{paper.title}</h4>
                        <p className="text-xs text-gray-500 mb-3">
                          {paper.authors?.slice(0, 2).join(', ')}{paper.authors && paper.authors.length > 2 && ' et al.'}
                        </p>
                        <div className="bg-gray-50 rounded-lg p-3 mb-3 max-h-48 overflow-y-auto">
                          <p className="text-xs text-gray-600 leading-relaxed">{paper.abstract || 'No abstract available.'}</p>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => openPaperSource(paper)}
                            className="text-xs text-purple-600 hover:underline flex items-center gap-1"
                          >
                            <ExternalLink className="w-3 h-3" />
                            View Source
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Paper selection list for comparison */}
                <div className="glass-3 rounded-xl border border-purple-200/50 p-4">
                  <h4 className="font-semibold text-gray-900 mb-3">Select papers to compare:</h4>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {sortedPapers.map((paper) => (
                      <div
                        key={paper.id}
                        onClick={() => toggleCompare(paper.id)}
                        className={`p-3 rounded-lg cursor-pointer transition-colors ${
                          comparePapers.includes(paper.id)
                            ? 'bg-purple-100 border-2 border-purple-400'
                            : 'hover:bg-purple-50 border-2 border-transparent'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-800 line-clamp-1">{paper.title}</span>
                          <span className="text-xs text-gray-500">{paper.year}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Table View */}
            {viewMode === 'table' && (
              <div className="max-w-7xl mx-auto glass-3 rounded-2xl border border-purple-200/50 overflow-hidden">
                <table className="w-full">
                  <thead className="bg-purple-50/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase">Title</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase w-20">Year</th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase w-32">Journal</th>
                      <th className="px-4 py-3 text-right text-xs font-semibold text-gray-500 uppercase w-24">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-purple-100/50">
                    {sortedPapers.map((paper) => (
                      <React.Fragment key={paper.id}>
                        <tr className="hover:bg-purple-50/30 transition-colors">
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-3">
                              <button
                                onClick={() => setExpandedPaper(expandedPaper === paper.id ? null : paper.id)}
                                className="p-1 hover:bg-purple-100 rounded"
                              >
                                {expandedPaper === paper.id ? (
                                  <ChevronDown className="w-4 h-4 text-gray-400" />
                                ) : (
                                  <ChevronRight className="w-4 h-4 text-gray-400" />
                                )}
                              </button>
                              <div className="min-w-0">
                                <p className="font-medium text-gray-900 text-sm line-clamp-2">{paper.title}</p>
                                <p className="text-xs text-gray-500 mt-1">
                                  {paper.authors?.slice(0, 2).join(', ')}{paper.authors && paper.authors.length > 2 && ' et al.'}
                                </p>
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-600">{paper.year}</td>
                          <td className="px-4 py-3 text-xs text-gray-500 line-clamp-1">{paper.journal || '-'}</td>
                          <td className="px-4 py-3">
                            <div className="flex items-center justify-end gap-1">
                              <button
                                onClick={() => openPaperSource(paper)}
                                className="p-1.5 hover:bg-purple-100 rounded text-purple-600"
                                title="View source"
                              >
                                <ExternalLink className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => copyCitation(paper)}
                                className="p-1.5 hover:bg-purple-100 rounded text-gray-500"
                                title="Copy citation"
                              >
                                {copiedId === paper.id ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                              </button>
                              <button
                                onClick={() => onRemovePaper(paper.id)}
                                className="p-1.5 hover:bg-red-100 rounded text-red-500"
                                title="Remove"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                          </td>
                        </tr>
                        {expandedPaper === paper.id && (
                          <tr className="bg-purple-50/30">
                            <td colSpan={4} className="px-6 py-4">
                              <div className="pl-10 space-y-3">
                                <p className="text-sm text-gray-700 leading-relaxed">{paper.abstract || 'No abstract available.'}</p>
                                <div className="flex items-center gap-3 pt-2">
                                  <button
                                    onClick={() => {
                                      setChatPaper(paper);
                                      setChatQuestion('');
                                      setChatAnswer('');
                                    }}
                                    className="px-3 py-1.5 glass-3 border border-purple-200/50 rounded-lg text-xs font-medium text-gray-700 hover:bg-purple-50/50 transition-colors flex items-center gap-1"
                                  >
                                    <MessageSquare className="w-3.5 h-3.5" />
                                    Chat with Paper
                                  </button>
                                  <button
                                    onClick={() => openPaperSource(paper)}
                                    className="px-3 py-1.5 glass-3 border border-purple-200/50 rounded-lg text-xs font-medium text-gray-700 hover:bg-purple-50/50 transition-colors flex items-center gap-1"
                                  >
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
            {viewMode === 'grid' && (
              <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {sortedPapers.map((paper) => (
                  <div key={paper.id} className="glass-3 rounded-xl border border-purple-200/50 p-4 group hover:border-purple-300 transition-colors">
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-xs font-medium text-purple-600 bg-purple-100 px-2 py-0.5 rounded-full">
                        {paper.year}
                      </span>
                      <button
                        onClick={() => onRemovePaper(paper.id)}
                        className="p-1 hover:bg-red-100 rounded text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                    <h4 className="font-semibold text-gray-900 text-sm mb-2 line-clamp-2">{paper.title}</h4>
                    <p className="text-xs text-gray-500 mb-3">
                      {paper.authors?.slice(0, 2).join(', ')}{paper.authors && paper.authors.length > 2 && ' et al.'}
                    </p>
                    {paper.journal && (
                      <p className="text-xs text-purple-600 mb-3">{paper.journal}</p>
                    )}
                    <div className="flex items-center justify-between pt-2 border-t border-purple-100/50">
                      <button
                        onClick={() => openPaperSource(paper)}
                        className="text-xs text-purple-600 hover:underline flex items-center gap-1"
                      >
                        <ExternalLink className="w-3 h-3" />
                        View
                      </button>
                      <button
                        onClick={() => copyCitation(paper)}
                        className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1"
                      >
                        {copiedId === paper.id ? <Check className="w-3 h-3 text-green-500" /> : <Copy className="w-3 h-3" />}
                        Cite
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {/* Chat with Paper Modal */}
      {chatPaper && (
        <div className="fixed inset-0 z-[110] bg-black/50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] flex flex-col">
            <div className="p-4 border-b border-gray-100 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center">
                  <MessageSquare className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">Chat with Paper</h3>
                  <p className="text-xs text-gray-500 line-clamp-1">{chatPaper.title}</p>
                </div>
              </div>
              <button onClick={() => setChatPaper(null)} className="p-2 hover:bg-gray-100 rounded-lg">
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div className="bg-purple-50 rounded-xl p-4">
                <p className="text-xs font-semibold text-purple-600 uppercase mb-2">Abstract</p>
                <p className="text-sm text-gray-700 leading-relaxed">{chatPaper.abstract || 'No abstract available.'}</p>
              </div>

              {!chatAnswer && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-gray-500">Quick Questions:</p>
                  <div className="flex flex-wrap gap-2">
                    {['What are the main findings?', 'What methodology was used?', 'What are the limitations?'].map((q) => (
                      <button
                        key={q}
                        onClick={() => {
                          setChatQuestion(q);
                          askAboutPaper(chatPaper, q);
                        }}
                        className="px-3 py-1.5 bg-gray-100 hover:bg-purple-100 rounded-full text-xs text-gray-700 transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {chatAnswer && (
                <div className="bg-gradient-to-br from-violet-50 to-purple-50 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="w-4 h-4 text-purple-600" />
                    <p className="text-xs font-semibold text-purple-600">AI Answer</p>
                  </div>
                  <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">{chatAnswer}</p>
                </div>
              )}

              {isChatLoading && (
                <div className="flex items-center gap-2 text-purple-600">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Analyzing paper...</span>
                </div>
              )}
            </div>

            <div className="p-4 border-t border-gray-100">
              <form onSubmit={(e) => { e.preventDefault(); askAboutPaper(chatPaper, chatQuestion); }} className="flex gap-2">
                <input
                  type="text"
                  value={chatQuestion}
                  onChange={(e) => setChatQuestion(e.target.value)}
                  placeholder="Ask a question about this paper..."
                  className="flex-1 px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-purple-400/50 focus:border-purple-300"
                />
                <button
                  type="submit"
                  disabled={!chatQuestion.trim() || isChatLoading}
                  className="px-4 py-2 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-lg font-medium hover:from-violet-700 hover:to-purple-700 disabled:opacity-50 transition-all"
                >
                  Ask
                </button>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
