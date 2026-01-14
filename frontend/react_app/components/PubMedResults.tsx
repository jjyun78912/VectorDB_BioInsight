import React, { useState } from 'react';
import { Globe, X, Loader2, FileText, Sparkles, Plus, FolderPlus, ArrowUpDown, Calendar, Hash, MessageSquare, Telescope, ExternalLink, Lightbulb, Award, Target, FlaskConical, Users, BookOpen, CheckCircle, AlertTriangle, Dna, Activity } from 'lucide-react';
import api, { CrawlerPaper, PaperExplanation } from '../services/client';
import { PaperInsightsCard } from './PaperInsightsCard';
import { KnowledgeGraph } from './KnowledgeGraph';
import { ChatPanel } from './ChatPanel';
import { useChat } from '../hooks/useChat';
import type { ReviewPaper } from '../App';

// Search filter options
interface SearchFilters {
  sort: 'relevance' | 'pub_date';
  yearRange: 'all' | '1' | '3' | '5';
  limit: number;
}

interface PubMedResultsProps {
  papers: CrawlerPaper[];
  onClose: () => void;
  isLoading?: boolean;
  onAddToReview?: (paper: ReviewPaper) => void;
  onAddAllToReview?: (papers: ReviewPaper[]) => void;
  useKorean?: boolean;
  query?: string;
  onSearch?: (filters: SearchFilters) => void;
  filters?: SearchFilters;
}

export const PubMedResults: React.FC<PubMedResultsProps> = ({
  papers,
  onClose,
  isLoading,
  onAddToReview,
  onAddAllToReview,
  useKorean = false,
  query,
  onSearch,
  filters
}) => {
  const [selectedPaper, setSelectedPaper] = useState<CrawlerPaper | null>(null);
  const [showGalaxy, setShowGalaxy] = useState(false);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);
  const [chatMode, setChatMode] = useState(false);
  const [keyPoints, setKeyPoints] = useState<string[]>([]);

  // Paper explanation state (why recommended + characteristics)
  const [paperExplanation, setPaperExplanation] = useState<PaperExplanation | null>(null);
  const [isLoadingExplanation, setIsLoadingExplanation] = useState(false);

  // Full text state
  const [fullTextSessionId, setFullTextSessionId] = useState<string | null>(null);
  const [isLoadingFullText, setIsLoadingFullText] = useState(false);
  const [fullTextError, setFullTextError] = useState<string | null>(null);
  const [fullTextMode, setFullTextMode] = useState(false);
  const [fullTextSummary, setFullTextSummary] = useState<{
    summary: string;
    key_findings: string[];
    methodology: string;
  } | null>(null);

  const { chatHistory, question, setQuestion, isLoading: isAskingQuestion, sendMessage, clearHistory, chatEndRef } = useChat();

  // Generate AI summary when paper is selected
  const handleSelectPaper = async (paper: CrawlerPaper) => {
    setSelectedPaper(paper);
    setAiSummary(null);
    setKeyPoints([]);
    clearHistory();
    setChatMode(false);
    setFullTextSessionId(null);
    setFullTextError(null);
    setFullTextMode(false);
    setFullTextSummary(null);
    setIsLoadingSummary(true);
    setPaperExplanation(null);
    setIsLoadingExplanation(true);

    // Fetch paper explanation (why recommended + characteristics) in parallel
    if (query && paper.abstract) {
      api.explainPaper(query, paper.title, paper.abstract, {
        pmid: paper.pmid,
        year: paper.year?.toString()
      })
        .then((explanation) => {
          setPaperExplanation(explanation);
        })
        .catch((err) => {
          console.error('Failed to get paper explanation:', err);
        })
        .finally(() => {
          setIsLoadingExplanation(false);
        });
    } else {
      setIsLoadingExplanation(false);
    }

    // Try to get full text first
    try {
      const fullTextResult = await api.getFullText({
        title: paper.title,
        pmid: paper.pmid,
        pmcid: paper.pmcid,
        doi: paper.doi,
        url: paper.url,
        language: useKorean ? 'ko' : 'en'
      });

      if (fullTextResult.success && fullTextResult.session_id) {
        setFullTextSessionId(fullTextResult.session_id);
        setFullTextMode(true);

        if (fullTextResult.ai_summary) {
          setFullTextSummary(fullTextResult.ai_summary);
          setAiSummary(fullTextResult.ai_summary.summary);
          setKeyPoints(fullTextResult.ai_summary.key_findings);
        }
        setIsLoadingSummary(false);
        return;
      }
    } catch (err) {
      console.log('Full text not available, falling back to abstract');
    }

    // Fallback: use abstract-based summary
    if (paper.abstract) {
      try {
        const response = await api.summarizeAbstract(paper.title, paper.abstract, useKorean ? 'ko' : 'en');
        setAiSummary(response.summary);
        setKeyPoints(response.key_points || []);
      } catch (err) {
        setAiSummary('Unable to generate summary.');
      }
    }
    setIsLoadingSummary(false);
  };

  // Handle AI Q&A
  const handleAskQuestion = async () => {
    if (!selectedPaper) return;

    await sendMessage(async (currentQuestion) => {
      if (fullTextMode && fullTextSessionId) {
        const response = await api.askAgent(fullTextSessionId, currentQuestion);
        return response.answer;
      } else {
        const response = await api.askAbstract(
          selectedPaper.title,
          selectedPaper.abstract || '',
          currentQuestion
        );
        return response.answer;
      }
    });
  };

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Backdrop */}
        <div className="absolute inset-0 bg-black/30 backdrop-blur-sm" onClick={onClose} />

        {/* Modal with Split View */}
        <div
          className="relative z-[51] w-full max-w-6xl bg-white/95 backdrop-blur-xl rounded-2xl shadow-2xl border border-purple-200/50 flex animate-appear"
          style={{ height: 'calc(100vh - 8rem)', maxHeight: '750px', minHeight: '500px' }}
        >
          {/* Left: Paper List */}
          <div className={`${selectedPaper ? 'w-2/5' : 'w-full'} flex flex-col border-r border-purple-100/50 transition-all min-h-0`}>
            {/* Header */}
            <div className="flex-shrink-0 bg-white/80 backdrop-blur border-b border-purple-100/50 px-5 py-3 rounded-tl-2xl">
              <div className="flex justify-between items-center mb-3">
                <div className="flex items-center gap-2">
                  <Globe className="w-4 h-4 text-emerald-500" />
                  <span className="text-sm font-semibold text-gray-700">
                    {isLoading ? 'Searching...' : `${papers.length} papers`}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {onAddAllToReview && papers.length > 0 && !isLoading && (
                    <button
                      onClick={() => {
                        const reviewPapers: ReviewPaper[] = papers.map(p => ({
                          id: p.pmid || p.id,
                          title: p.title,
                          authors: p.authors,
                          year: p.year,
                          journal: p.journal,
                          abstract: p.abstract,
                          doi: p.doi,
                          pmid: p.pmid,
                          relevance: p.trend_score
                        }));
                        onAddAllToReview(reviewPapers);
                      }}
                      className="px-3 py-1.5 bg-gradient-to-r from-violet-600 to-purple-600 text-white text-xs font-medium rounded-lg hover:from-violet-700 hover:to-purple-700 transition-all flex items-center gap-1.5"
                    >
                      <FolderPlus className="w-3.5 h-3.5" />
                      Add All
                    </button>
                  )}
                  <button onClick={onClose} className="p-1.5 hover:bg-purple-100/50 rounded-full transition-colors">
                    <X className="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </div>

              {/* Filter Bar */}
              {onSearch && filters && (
                <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-purple-100/30">
                  <div className="flex items-center gap-1">
                    <ArrowUpDown className="w-3.5 h-3.5 text-gray-400" />
                    <select
                      value={filters.sort}
                      onChange={(e) => onSearch({ ...filters, sort: e.target.value as 'relevance' | 'pub_date' })}
                      disabled={isLoading}
                      className="text-xs bg-white border border-gray-200 rounded-lg px-2 py-1.5 focus:ring-1 focus:ring-purple-400 focus:border-purple-400 disabled:opacity-50"
                    >
                      <option value="relevance">Relevance</option>
                      <option value="pub_date">Newest First</option>
                    </select>
                  </div>

                  <div className="flex items-center gap-1">
                    <Calendar className="w-3.5 h-3.5 text-gray-400" />
                    <select
                      value={filters.yearRange}
                      onChange={(e) => onSearch({ ...filters, yearRange: e.target.value as 'all' | '1' | '3' | '5' })}
                      disabled={isLoading}
                      className="text-xs bg-white border border-gray-200 rounded-lg px-2 py-1.5 focus:ring-1 focus:ring-purple-400 focus:border-purple-400 disabled:opacity-50"
                    >
                      <option value="all">All Years</option>
                      <option value="1">Last 1 Year</option>
                      <option value="3">Last 3 Years</option>
                      <option value="5">Last 5 Years</option>
                    </select>
                  </div>

                  <div className="flex items-center gap-1">
                    <Hash className="w-3.5 h-3.5 text-gray-400" />
                    <select
                      value={filters.limit}
                      onChange={(e) => onSearch({ ...filters, limit: parseInt(e.target.value) })}
                      disabled={isLoading}
                      className="text-xs bg-white border border-gray-200 rounded-lg px-2 py-1.5 focus:ring-1 focus:ring-purple-400 focus:border-purple-400 disabled:opacity-50"
                    >
                      <option value="10">10 papers</option>
                      <option value="20">20 papers</option>
                      <option value="30">30 papers</option>
                      <option value="50">50 papers</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            {/* Paper List */}
            <div className="flex-1 overflow-y-auto min-h-0">
              {isLoading ? (
                <div className="flex flex-col items-center justify-center py-12 gap-3">
                  <Loader2 className="w-8 h-8 text-emerald-500 animate-spin" />
                  <p className="text-sm text-gray-500">Fetching from PubMed...</p>
                </div>
              ) : papers.length === 0 ? (
                <div className="p-8 text-center text-gray-500">
                  <FileText className="w-10 h-10 mx-auto mb-3 text-gray-400" />
                  <p>No papers found.</p>
                </div>
              ) : (
                <div className="divide-y divide-purple-50/50">
                  {papers.map((paper) => (
                    <div
                      key={paper.id}
                      className={`group p-4 cursor-pointer transition-all ${
                        selectedPaper?.id === paper.id
                          ? 'bg-purple-100/50 border-l-4 border-purple-500'
                          : 'hover:bg-purple-50/50'
                      }`}
                      onClick={() => handleSelectPaper(paper)}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        {paper.year > 0 && (
                          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                            paper.year >= 2024 ? 'text-orange-600 bg-orange-50' : 'text-gray-500 bg-gray-100'
                          }`}>
                            {paper.year}
                          </span>
                        )}
                        {paper.citation_count > 0 && (
                          <span className="text-xs text-blue-600">{paper.citation_count} citations</span>
                        )}
                      </div>
                      <h4 className="text-sm font-medium text-gray-800 line-clamp-2">
                        {paper.title_ko || paper.title}
                      </h4>
                      {paper.title_ko && (
                        <p className="text-xs text-gray-400 line-clamp-1 mt-0.5">{paper.title}</p>
                      )}
                      <div className="flex items-center justify-between mt-1">
                        {paper.authors.length > 0 && (
                          <p className="text-xs text-gray-500 line-clamp-1 flex-1">
                            {paper.authors.slice(0, 2).join(', ')}{paper.authors.length > 2 && ' et al.'}
                          </p>
                        )}
                        {onAddToReview && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onAddToReview({
                                id: paper.pmid || paper.id,
                                title: paper.title,
                                authors: paper.authors,
                                year: paper.year,
                                journal: paper.journal,
                                abstract: paper.abstract,
                                doi: paper.doi,
                                pmid: paper.pmid,
                                relevance: paper.trend_score
                              });
                            }}
                            className="ml-2 p-1.5 text-purple-500 hover:bg-purple-100 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                            title="Add to Literature Review"
                          >
                            <Plus className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right: Preview Panel */}
          {selectedPaper && (
            <div className="w-3/5 flex flex-col min-h-0">
              {/* Preview Header */}
              <div className="shrink-0 bg-gradient-to-r from-purple-50 to-indigo-50 border-b border-purple-100/50 px-6 py-4 rounded-tr-2xl">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                  <span className="text-xs font-semibold text-emerald-600 bg-emerald-100 px-2.5 py-1 rounded-full">
                    {selectedPaper.source === 'pubmed' ? 'PubMed' : selectedPaper.source}
                  </span>
                  {selectedPaper.pmid && (
                    <span className="text-xs text-gray-500">PMID: {selectedPaper.pmid}</span>
                  )}
                  {selectedPaper.pmcid ? (
                    <span className="text-xs font-semibold text-green-700 bg-green-100 px-2 py-0.5 rounded-full flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                      Open Access
                    </span>
                  ) : (
                    <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">
                      Abstract Only
                    </span>
                  )}
                </div>
                <h3 className="text-lg font-bold text-gray-900 leading-snug">
                  {selectedPaper.title_ko || selectedPaper.title}
                </h3>
                {selectedPaper.title_ko && (
                  <p className="text-sm text-gray-500 mt-1">{selectedPaper.title}</p>
                )}
                {selectedPaper.authors.length > 0 && (
                  <p className="text-sm text-gray-600 mt-1">
                    {selectedPaper.authors.slice(0, 5).join(', ')}{selectedPaper.authors.length > 5 && ' et al.'}
                  </p>
                )}
              </div>

              {/* Preview Content - Scrollable */}
              <div className="flex-1 min-h-0 overflow-y-auto p-6 space-y-5">
                {/* AI Summary */}
                <div className="glass-2 rounded-xl p-4 border border-purple-100/50">
                  <h4 className="text-xs font-bold text-purple-600 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <Sparkles className="w-3.5 h-3.5" />
                    AI Summary
                    {fullTextMode && (
                      <span className="ml-auto px-2 py-0.5 text-[10px] font-semibold bg-emerald-100 text-emerald-700 rounded-full">
                        Full Text
                      </span>
                    )}
                  </h4>
                  {isLoadingSummary ? (
                    <div className="flex items-center gap-2 text-gray-500">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Fetching full text & generating AI summary...</span>
                    </div>
                  ) : aiSummary ? (
                    <div className="space-y-3">
                      <p className="text-sm text-gray-700 leading-relaxed">{aiSummary}</p>
                      {fullTextSummary?.methodology && (
                        <div className="pt-2 border-t border-purple-100/50">
                          <h5 className="text-xs font-semibold text-blue-500 mb-1">Methodology</h5>
                          <p className="text-xs text-gray-600">{fullTextSummary.methodology}</p>
                        </div>
                      )}
                      {keyPoints.length > 0 && (
                        <div className="pt-2 border-t border-purple-100/50">
                          <h5 className="text-xs font-semibold text-purple-500 mb-2">Key Points</h5>
                          <ul className="space-y-1.5">
                            {keyPoints.map((point, i) => (
                              <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                                <span className="w-5 h-5 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center text-xs font-semibold flex-shrink-0 mt-0.5">
                                  {i + 1}
                                </span>
                                <span>{point}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-400">No abstract available for summary.</p>
                  )}
                </div>

                {/* Why Recommended Section */}
                {(isLoadingExplanation || paperExplanation) && (
                  <div className="glass-2 rounded-xl p-4 border border-emerald-100/50 bg-gradient-to-br from-emerald-50/50 to-teal-50/50">
                    <h4 className="text-xs font-bold text-emerald-600 uppercase tracking-wider mb-3 flex items-center gap-2">
                      <Lightbulb className="w-3.5 h-3.5" />
                      Why This Paper Is Recommended
                    </h4>
                    {isLoadingExplanation ? (
                      <div className="flex items-center gap-2 text-gray-500">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span className="text-sm">Analyzing relevance to your query...</span>
                      </div>
                    ) : paperExplanation ? (
                      <div className="space-y-3">
                        <p className="text-sm text-gray-700 leading-relaxed">{paperExplanation.why_recommended}</p>

                        {/* Relevance Factors */}
                        {paperExplanation.relevance_factors && paperExplanation.relevance_factors.length > 0 && (
                          <div className="pt-2 border-t border-emerald-100/50">
                            <h5 className="text-xs font-semibold text-teal-600 mb-2 flex items-center gap-1">
                              <Target className="w-3 h-3" />
                              Relevance Factors
                            </h5>
                            <ul className="flex flex-wrap gap-2">
                              {paperExplanation.relevance_factors.map((factor, i) => (
                                <li key={i} className="text-xs px-2.5 py-1 bg-teal-100 text-teal-700 rounded-full">
                                  {factor}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Scores */}
                        <div className="flex items-center gap-3 pt-2 border-t border-emerald-100/50">
                          <div className="flex items-center gap-1.5">
                            <Target className="w-3.5 h-3.5 text-emerald-500" />
                            <span className="text-xs text-gray-600">Relevance:</span>
                            <span className={`text-xs font-bold ${
                              paperExplanation.relevance_score >= 8 ? 'text-emerald-600' :
                              paperExplanation.relevance_score >= 6 ? 'text-blue-600' : 'text-gray-600'
                            }`}>
                              {paperExplanation.relevance_score}/10
                            </span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <Sparkles className="w-3.5 h-3.5 text-violet-500" />
                            <span className="text-xs text-gray-600">Novelty:</span>
                            <span className="text-xs font-bold text-violet-600">{paperExplanation.novelty_score}/10</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <Award className="w-3.5 h-3.5 text-amber-500" />
                            <span className="text-xs text-gray-600">Quality:</span>
                            <span className="text-xs font-bold text-amber-600">{paperExplanation.quality_score}/10</span>
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                )}

                {/* Paper Characteristics Section */}
                {paperExplanation?.characteristics && (
                  <div className="glass-2 rounded-xl p-4 border border-indigo-100/50 bg-gradient-to-br from-indigo-50/50 to-purple-50/50">
                    <h4 className="text-xs font-bold text-indigo-600 uppercase tracking-wider mb-3 flex items-center gap-2">
                      <BookOpen className="w-3.5 h-3.5" />
                      Paper Characteristics
                    </h4>
                    <div className="space-y-3">
                      {/* Study Type & Evidence Level */}
                      <div className="flex flex-wrap gap-2">
                        {paperExplanation.characteristics.study_type && (
                          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-indigo-100 text-indigo-700 text-xs font-medium rounded-full">
                            <FlaskConical className="w-3 h-3" />
                            {paperExplanation.characteristics.study_type}
                          </span>
                        )}
                        {paperExplanation.characteristics.evidence_level && (
                          <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full ${
                            paperExplanation.characteristics.evidence_level.toLowerCase().includes('high')
                              ? 'bg-emerald-100 text-emerald-700'
                              : paperExplanation.characteristics.evidence_level.toLowerCase().includes('moderate')
                              ? 'bg-blue-100 text-blue-700'
                              : 'bg-gray-100 text-gray-700'
                          }`}>
                            <Award className="w-3 h-3" />
                            {paperExplanation.characteristics.evidence_level}
                          </span>
                        )}
                      </div>

                      {/* Main Finding */}
                      {paperExplanation.characteristics.main_finding && (
                        <div>
                          <h5 className="text-xs font-semibold text-purple-600 mb-1">Main Finding</h5>
                          <p className="text-sm text-gray-700">{paperExplanation.characteristics.main_finding}</p>
                        </div>
                      )}

                      {/* Methodology & Sample */}
                      <div className="grid grid-cols-2 gap-3">
                        {paperExplanation.characteristics.methodology && (
                          <div>
                            <h5 className="text-xs font-semibold text-blue-600 mb-1 flex items-center gap-1">
                              <FlaskConical className="w-3 h-3" />
                              Methodology
                            </h5>
                            <p className="text-xs text-gray-600">{paperExplanation.characteristics.methodology}</p>
                          </div>
                        )}
                        {paperExplanation.characteristics.sample_info && (
                          <div>
                            <h5 className="text-xs font-semibold text-cyan-600 mb-1 flex items-center gap-1">
                              <Users className="w-3 h-3" />
                              Sample Info
                            </h5>
                            <p className="text-xs text-gray-600">{paperExplanation.characteristics.sample_info}</p>
                          </div>
                        )}
                      </div>

                      {/* Clinical Relevance */}
                      {paperExplanation.characteristics.clinical_relevance && (
                        <div className="bg-white/50 rounded-lg p-2.5">
                          <h5 className="text-xs font-semibold text-rose-600 mb-1">Clinical Relevance</h5>
                          <p className="text-xs text-gray-600">{paperExplanation.characteristics.clinical_relevance}</p>
                        </div>
                      )}

                      {/* Strengths & Limitations */}
                      <div className="grid grid-cols-2 gap-3">
                        {paperExplanation.characteristics.strengths && paperExplanation.characteristics.strengths.length > 0 && (
                          <div>
                            <h5 className="text-xs font-semibold text-emerald-600 mb-1.5 flex items-center gap-1">
                              <CheckCircle className="w-3 h-3" />
                              Strengths
                            </h5>
                            <ul className="space-y-1">
                              {paperExplanation.characteristics.strengths.slice(0, 3).map((s, i) => (
                                <li key={i} className="text-xs text-gray-600 flex items-start gap-1">
                                  <span className="text-emerald-500 mt-0.5">+</span>
                                  {s}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {paperExplanation.characteristics.limitations && paperExplanation.characteristics.limitations.length > 0 && (
                          <div>
                            <h5 className="text-xs font-semibold text-amber-600 mb-1.5 flex items-center gap-1">
                              <AlertTriangle className="w-3 h-3" />
                              Limitations
                            </h5>
                            <ul className="space-y-1">
                              {paperExplanation.characteristics.limitations.slice(0, 3).map((l, i) => (
                                <li key={i} className="text-xs text-gray-600 flex items-start gap-1">
                                  <span className="text-amber-500 mt-0.5">-</span>
                                  {l}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>

                      {/* Key Genes, Pathways, Techniques */}
                      <div className="pt-2 border-t border-indigo-100/50 space-y-2">
                        {paperExplanation.characteristics.key_genes && paperExplanation.characteristics.key_genes.length > 0 && (
                          <div className="flex items-start gap-2">
                            <Dna className="w-3.5 h-3.5 text-rose-500 mt-0.5 flex-shrink-0" />
                            <div className="flex flex-wrap gap-1">
                              {paperExplanation.characteristics.key_genes.map((gene, i) => (
                                <span key={i} className="text-xs px-2 py-0.5 bg-rose-100 text-rose-700 rounded-full font-medium">
                                  {gene}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        {paperExplanation.characteristics.key_pathways && paperExplanation.characteristics.key_pathways.length > 0 && (
                          <div className="flex items-start gap-2">
                            <Activity className="w-3.5 h-3.5 text-violet-500 mt-0.5 flex-shrink-0" />
                            <div className="flex flex-wrap gap-1">
                              {paperExplanation.characteristics.key_pathways.map((pathway, i) => (
                                <span key={i} className="text-xs px-2 py-0.5 bg-violet-100 text-violet-700 rounded-full">
                                  {pathway}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        {paperExplanation.characteristics.techniques && paperExplanation.characteristics.techniques.length > 0 && (
                          <div className="flex items-start gap-2">
                            <FlaskConical className="w-3.5 h-3.5 text-blue-500 mt-0.5 flex-shrink-0" />
                            <div className="flex flex-wrap gap-1">
                              {paperExplanation.characteristics.techniques.map((tech, i) => (
                                <span key={i} className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full">
                                  {tech}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Paper Insights */}
                {selectedPaper.abstract && (
                  <PaperInsightsCard
                    title={selectedPaper.title}
                    abstract={selectedPaper.abstract}
                    compact={false}
                  />
                )}

                {/* Q&A Section */}
                {chatMode && (
                  <ChatPanel
                    chatHistory={chatHistory}
                    chatQuestion={question}
                    onQuestionChange={setQuestion}
                    onSubmit={handleAskQuestion}
                    isLoading={isAskingQuestion}
                    chatEndRef={chatEndRef}
                    placeholder="Ask about this paper..."
                  />
                )}

                {/* Keywords */}
                {selectedPaper.keywords.length > 0 && (
                  <div>
                    <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Keywords</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedPaper.keywords.map((kw, i) => (
                        <span key={i} className="px-2.5 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                          {kw}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="shrink-0 bg-white border-t border-purple-100/50 px-6 py-4 rounded-br-2xl">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setChatMode(!chatMode)}
                      className={`px-4 py-2 rounded-full text-sm font-medium transition-all flex items-center gap-2 ${
                        chatMode
                          ? 'bg-blue-100 text-blue-700'
                          : 'bg-gray-100 border border-gray-200 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      <MessageSquare className="w-4 h-4" />
                      {fullTextMode ? 'Full Text Chat' : 'AI Chat'}
                    </button>

                    {fullTextMode ? (
                      <span className="px-3 py-1.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-700 flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5" />
                        Full Text Loaded
                      </span>
                    ) : (
                      <span className="px-3 py-1.5 rounded-full text-xs font-medium bg-gray-100 text-gray-500 flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5" />
                        Abstract Only
                      </span>
                    )}

                    {selectedPaper.pmid && (
                      <button
                        type="button"
                        onClick={() => setShowGalaxy(true)}
                        className="px-4 py-2 rounded-full text-sm font-medium bg-gradient-to-r from-yellow-500 to-orange-500 text-white hover:from-yellow-600 hover:to-orange-600 transition-all flex items-center gap-2"
                      >
                        <Telescope className="w-4 h-4" />
                        Galaxy
                      </button>
                    )}
                  </div>

                  <button
                    type="button"
                    onClick={() => {
                      const url = selectedPaper.url || (selectedPaper.doi ? `https://doi.org/${selectedPaper.doi}` : `https://pubmed.ncbi.nlm.nih.gov/${selectedPaper.pmid}`);
                      if (url) window.open(url, '_blank');
                    }}
                    className="px-5 py-2 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-full text-sm font-semibold hover:from-violet-700 hover:to-purple-700 transition-all flex items-center gap-2 shadow-lg"
                  >
                    View on PubMed
                    <ExternalLink className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Galaxy Visualization */}
      {showGalaxy && selectedPaper?.pmid && (
        <KnowledgeGraph
          isOpen={showGalaxy}
          onClose={() => setShowGalaxy(false)}
          mode="similar"
          sourcePmid={selectedPaper.pmid}
          domain="pancreatic_cancer"
          onPaperClick={(pmid) => {
            console.log('Navigate to paper:', pmid);
          }}
        />
      )}
    </>
  );
};
