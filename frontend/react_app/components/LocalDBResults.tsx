import React, { useState } from 'react';
import { Target, X, FileText, Loader2, Info, Sparkles, BookOpen, MessageSquare, ExternalLink, Lightbulb, FlaskConical, TrendingUp, AlertTriangle, Dna, Microscope } from 'lucide-react';
import api, { PrecisionSearchResult, SearchDiagnostics, PaperExplanation } from '../services/client';
import { ChatPanel } from './ChatPanel';
import { useChat } from '../hooks/useChat';

interface LocalDBResultsProps {
  results: PrecisionSearchResult[];
  diagnostics: SearchDiagnostics | null;
  onClose: () => void;
  onSelectResult: (result: PrecisionSearchResult) => void;
  query?: string;  // Search query for paper explanation
}

// Helper to get match field color
const getMatchFieldStyle = (field: string) => {
  switch (field) {
    case 'title':
      return { bg: 'bg-emerald-100', text: 'text-emerald-700', label: 'Title Match' };
    case 'abstract':
      return { bg: 'bg-blue-100', text: 'text-blue-700', label: 'Abstract Match' };
    case 'mesh':
      return { bg: 'bg-purple-100', text: 'text-purple-700', label: 'MeSH Match' };
    case 'full_text':
      return { bg: 'bg-gray-100', text: 'text-gray-600', label: 'Full Text' };
    default:
      return { bg: 'bg-gray-100', text: 'text-gray-500', label: 'No Match' };
  }
};

// Helper to get search query from URL or context
const getSearchQuery = (): string => {
  const params = new URLSearchParams(window.location.search);
  return params.get('q') || '';
};

export const LocalDBResults: React.FC<LocalDBResultsProps> = ({ results, diagnostics, onClose, onSelectResult, query: initialQuery }) => {
  const [selectedResult, setSelectedResult] = useState<PrecisionSearchResult | null>(null);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [keyPoints, setKeyPoints] = useState<string[]>([]);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);
  const [chatMode, setChatMode] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  // Paper explanation state
  const [paperExplanation, setPaperExplanation] = useState<PaperExplanation | null>(null);
  const [isLoadingExplanation, setIsLoadingExplanation] = useState(false);
  const [searchQuery, setSearchQuery] = useState<string>('');

  const { chatHistory, question, setQuestion, isLoading: isAskingQuestion, sendMessage, clearHistory, chatEndRef } = useChat();

  // Get search query on mount or from props
  React.useEffect(() => {
    const query = initialQuery || getSearchQuery() || diagnostics?.detected_disease || '';
    setSearchQuery(query);
  }, [initialQuery, diagnostics]);

  // Generate AI summary and paper explanation when result is selected
  const handleSelectResult = async (result: PrecisionSearchResult) => {
    setSelectedResult(result);
    setAiSummary(null);
    setKeyPoints([]);
    setPaperExplanation(null);
    clearHistory();
    setChatMode(false);

    if (result.content) {
      // Fetch both summary and explanation in parallel
      setIsLoadingSummary(true);
      setIsLoadingExplanation(true);

      // Get AI Summary
      const summaryPromise = api.summarizeAbstract(result.paper_title, result.content)
        .then(response => {
          setAiSummary(response.summary);
          setKeyPoints(response.key_points || []);
        })
        .catch(() => {
          setAiSummary('Unable to generate summary.');
        })
        .finally(() => {
          setIsLoadingSummary(false);
        });

      // Get Paper Explanation (LLM-based for detailed analysis)
      const query = searchQuery || diagnostics?.detected_disease || '';
      const explanationPromise = api.explainPaper(
        query,
        result.paper_title,
        result.content,
        {
          section: result.section,
          matchedTerms: result.matched_terms
        }
      )
        .then(explanation => {
          setPaperExplanation(explanation);
        })
        .catch(err => {
          console.error('Failed to get paper explanation:', err);
          // Fallback - create a simple explanation
          setPaperExplanation({
            why_recommended: `'${query}' 검색어와 관련된 내용이 발견되었습니다.`,
            relevance_factors: result.matched_terms.slice(0, 3),
            query_match_explanation: `검색어가 논문의 ${result.section || '본문'}에서 매칭되었습니다.`,
            characteristics: null,
            relevance_score: 3,
            novelty_score: 3,
            quality_score: 3,
            generated_at: new Date().toISOString(),
            model_used: 'fallback'
          });
        })
        .finally(() => {
          setIsLoadingExplanation(false);
        });

      await Promise.all([summaryPromise, explanationPromise]);
    }
  };

  // Handle AI Q&A
  const handleAskQuestion = async () => {
    if (!selectedResult) return;

    await sendMessage(async (currentQuestion) => {
      const response = await api.askAbstract(
        selectedResult.paper_title,
        selectedResult.content,
        currentQuestion
      );
      return response.answer;
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/30 backdrop-blur-sm" onClick={onClose} />

      {/* Modal with Split View */}
      <div
        className="relative z-[51] w-full max-w-6xl bg-white/95 backdrop-blur-xl rounded-2xl shadow-2xl border border-purple-200/50 flex animate-appear"
        style={{ height: 'calc(100vh - 8rem)', maxHeight: '750px', minHeight: '500px' }}
      >
        {/* Left: Results List */}
        <div className={`${selectedResult ? 'w-2/5' : 'w-full'} flex flex-col border-r border-purple-100/50 transition-all min-h-0`}>
          {/* Header with Diagnostics Toggle */}
          <div className="flex-shrink-0 bg-white/80 backdrop-blur border-b border-purple-100/50 px-5 py-4 rounded-tl-2xl">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-purple-500" />
                <span className="text-sm font-semibold text-gray-700">
                  {results.length} precision results
                </span>
                {diagnostics?.detected_disease && (
                  <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
                    {diagnostics.detected_disease}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {diagnostics && (
                  <button
                    onClick={() => setShowDiagnostics(!showDiagnostics)}
                    className={`p-1.5 rounded-full transition-colors ${showDiagnostics ? 'bg-purple-100 text-purple-600' : 'hover:bg-purple-100/50 text-gray-400'}`}
                    title="Show search diagnostics"
                  >
                    <Info className="w-4 h-4" />
                  </button>
                )}
                <button onClick={onClose} className="p-1.5 hover:bg-purple-100/50 rounded-full transition-colors">
                  <X className="w-4 h-4 text-gray-500" />
                </button>
              </div>
            </div>

            {/* Diagnostics Panel */}
            {showDiagnostics && diagnostics && (
              <div className="mt-3 p-3 bg-purple-50/50 rounded-xl border border-purple-100/50 text-xs space-y-2 animate-appear">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-purple-700">Search Strategy:</span>
                  <span className="text-gray-600">{diagnostics.strategy_used}</span>
                </div>
                {diagnostics.mesh_term && (
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-purple-700">MeSH Term:</span>
                    <span className="text-gray-600">{diagnostics.mesh_term}</span>
                  </div>
                )}
                {diagnostics.search_terms.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1">
                    <span className="font-semibold text-purple-700">Terms:</span>
                    {diagnostics.search_terms.slice(0, 5).map((term, i) => (
                      <span key={i} className="px-1.5 py-0.5 bg-white rounded text-gray-600 border border-purple-100">
                        {term}
                      </span>
                    ))}
                  </div>
                )}
                <div className="text-gray-500 pt-1 border-t border-purple-100">
                  {diagnostics.total_candidates} candidates → {diagnostics.filtered_results} filtered
                </div>
              </div>
            )}
          </div>

          {/* Results List */}
          <div className="flex-1 overflow-y-auto min-h-0">
            {results.length === 0 ? (
              <div className="p-8 text-center text-gray-500">
                <FileText className="w-10 h-10 mx-auto mb-3 text-gray-400" />
                <p>No results found.</p>
                {diagnostics && (
                  <p className="text-xs mt-2 text-gray-400">
                    {diagnostics.total_candidates} candidates checked, none matched criteria
                  </p>
                )}
              </div>
            ) : (
              <div className="divide-y divide-purple-50/50">
                {results.map((result, idx) => {
                  const matchStyle = getMatchFieldStyle(result.match_field);
                  return (
                    <div
                      key={idx}
                      className={`p-4 cursor-pointer transition-all ${
                        selectedResult?.paper_title === result.paper_title && selectedResult?.section === result.section
                          ? 'bg-purple-100/50 border-l-4 border-purple-500'
                          : 'hover:bg-purple-50/50'
                      }`}
                      onClick={() => handleSelectResult(result)}
                    >
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <span className={`text-xs font-semibold ${matchStyle.bg} ${matchStyle.text} px-2 py-0.5 rounded-full`}>
                          {matchStyle.label}
                        </span>
                        <span className="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
                          {result.section}
                        </span>
                        <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
                          {result.relevance_score.toFixed(1)}%
                        </span>
                      </div>
                      <h4 className="text-sm font-medium text-gray-800 line-clamp-2">{result.paper_title}</h4>
                      {result.matched_terms.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1.5">
                          {result.matched_terms.slice(0, 4).map((term, i) => (
                            <span key={i} className="text-[10px] text-purple-600 bg-purple-50 px-1.5 py-0.5 rounded">
                              {term}
                            </span>
                          ))}
                          {result.matched_terms.length > 4 && (
                            <span className="text-[10px] text-gray-400">+{result.matched_terms.length - 4}</span>
                          )}
                        </div>
                      )}
                      <p className="text-xs text-gray-500 mt-1 line-clamp-2">{result.content}</p>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Right: Preview Panel */}
        {selectedResult && (
          <div className="w-3/5 flex flex-col min-h-0">
            {/* Preview Header */}
            <div className="shrink-0 bg-gradient-to-r from-purple-50 to-indigo-50 border-b border-purple-100/50 px-6 py-4 rounded-tr-2xl">
              <div className="flex items-center gap-2 mb-2 flex-wrap">
                {(() => {
                  const matchStyle = getMatchFieldStyle(selectedResult.match_field);
                  return (
                    <span className={`text-xs font-semibold ${matchStyle.bg} ${matchStyle.text} px-2.5 py-1 rounded-full`}>
                      {matchStyle.label}
                    </span>
                  );
                })()}
                <span className="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
                  {selectedResult.section}
                </span>
                <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
                  {selectedResult.relevance_score.toFixed(1)}% match
                </span>
                <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full">
                  {selectedResult.disease_relevance.toFixed(0)}% disease relevance
                </span>
              </div>
              <h3 className="text-lg font-bold text-gray-900 leading-snug">{selectedResult.paper_title}</h3>
              {selectedResult.explanation && (
                <p className="text-xs text-gray-500 mt-2 italic">{selectedResult.explanation}</p>
              )}
            </div>

            {/* Preview Content - Scrollable */}
            <div className="flex-1 min-h-0 overflow-y-auto p-6 space-y-5">
              {/* Matched Content */}
              <div>
                <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <BookOpen className="w-3.5 h-3.5" />
                  Matched Content
                </h4>
                <p className="text-sm text-gray-700 leading-relaxed">{selectedResult.content}</p>
              </div>

              {/* Why Recommended Section */}
              <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl p-4 border border-amber-200/50">
                <h4 className="text-xs font-bold text-amber-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Lightbulb className="w-3.5 h-3.5" />
                  왜 이 논문이 추천되었나요?
                </h4>
                {isLoadingExplanation ? (
                  <div className="flex items-center gap-2 text-gray-500">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">분석 중...</span>
                  </div>
                ) : paperExplanation ? (
                  <div className="space-y-3">
                    {/* Why Recommended */}
                    <p className="text-sm text-gray-700 leading-relaxed">{paperExplanation.why_recommended}</p>

                    {/* Relevance Factors */}
                    {paperExplanation.relevance_factors.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {paperExplanation.relevance_factors.map((factor, i) => (
                          <span key={i} className="text-xs px-2 py-1 bg-amber-100 text-amber-700 rounded-full">
                            {factor}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Query Match Explanation */}
                    {paperExplanation.query_match_explanation && (
                      <p className="text-xs text-gray-500 italic border-l-2 border-amber-300 pl-2">
                        {paperExplanation.query_match_explanation}
                      </p>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-gray-400">설명을 불러올 수 없습니다.</p>
                )}
              </div>

              {/* Paper Characteristics Section */}
              {paperExplanation?.characteristics && (
                <div className="bg-gradient-to-r from-indigo-50 to-blue-50 rounded-xl p-4 border border-indigo-200/50">
                  <h4 className="text-xs font-bold text-indigo-700 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <FlaskConical className="w-3.5 h-3.5" />
                    논문 특성
                  </h4>
                  <div className="space-y-3">
                    {/* Study Type & Design */}
                    <div className="flex flex-wrap gap-2">
                      {paperExplanation.characteristics.study_type && (
                        <span className="text-xs px-2.5 py-1 bg-indigo-100 text-indigo-700 rounded-full font-medium">
                          {paperExplanation.characteristics.study_type}
                        </span>
                      )}
                      {paperExplanation.characteristics.study_design && (
                        <span className="text-xs px-2.5 py-1 bg-blue-100 text-blue-700 rounded-full font-medium">
                          {paperExplanation.characteristics.study_design}
                        </span>
                      )}
                      {paperExplanation.characteristics.evidence_level && (
                        <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${
                          paperExplanation.characteristics.evidence_level === 'High'
                            ? 'bg-emerald-100 text-emerald-700'
                            : paperExplanation.characteristics.evidence_level === 'Medium'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-gray-100 text-gray-600'
                        }`}>
                          근거수준: {paperExplanation.characteristics.evidence_level}
                        </span>
                      )}
                    </div>

                    {/* Main Finding */}
                    {paperExplanation.characteristics.main_finding && (
                      <div>
                        <h5 className="text-xs font-semibold text-indigo-600 mb-1 flex items-center gap-1">
                          <TrendingUp className="w-3 h-3" />
                          핵심 발견
                        </h5>
                        <p className="text-sm text-gray-700">{paperExplanation.characteristics.main_finding}</p>
                      </div>
                    )}

                    {/* Methodology */}
                    {paperExplanation.characteristics.methodology && (
                      <div>
                        <h5 className="text-xs font-semibold text-indigo-600 mb-1 flex items-center gap-1">
                          <Microscope className="w-3 h-3" />
                          방법론
                        </h5>
                        <p className="text-sm text-gray-600">{paperExplanation.characteristics.methodology}</p>
                      </div>
                    )}

                    {/* Key Genes */}
                    {paperExplanation.characteristics.key_genes && paperExplanation.characteristics.key_genes.length > 0 && (
                      <div>
                        <h5 className="text-xs font-semibold text-indigo-600 mb-1 flex items-center gap-1">
                          <Dna className="w-3 h-3" />
                          주요 유전자
                        </h5>
                        <div className="flex flex-wrap gap-1">
                          {paperExplanation.characteristics.key_genes.map((gene, i) => (
                            <span key={i} className="text-xs px-2 py-0.5 bg-pink-50 text-pink-600 rounded border border-pink-200">
                              {gene}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Techniques */}
                    {paperExplanation.characteristics.techniques && paperExplanation.characteristics.techniques.length > 0 && (
                      <div>
                        <h5 className="text-xs font-semibold text-indigo-600 mb-1">사용 기법</h5>
                        <div className="flex flex-wrap gap-1">
                          {paperExplanation.characteristics.techniques.map((tech, i) => (
                            <span key={i} className="text-xs px-2 py-0.5 bg-cyan-50 text-cyan-600 rounded border border-cyan-200">
                              {tech}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Strengths & Limitations */}
                    {((paperExplanation.characteristics.strengths && paperExplanation.characteristics.strengths.length > 0) ||
                      (paperExplanation.characteristics.limitations && paperExplanation.characteristics.limitations.length > 0)) && (
                      <div className="grid grid-cols-2 gap-3 pt-2 border-t border-indigo-100">
                        {/* Strengths */}
                        {paperExplanation.characteristics.strengths && paperExplanation.characteristics.strengths.length > 0 && (
                          <div>
                            <h5 className="text-xs font-semibold text-emerald-600 mb-1.5 flex items-center gap-1">
                              <TrendingUp className="w-3 h-3" />
                              장점
                            </h5>
                            <ul className="space-y-1">
                              {paperExplanation.characteristics.strengths.map((s, i) => (
                                <li key={i} className="text-xs text-gray-600 flex items-start gap-1">
                                  <span className="text-emerald-500 mt-0.5">•</span>
                                  <span>{s}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {/* Limitations */}
                        {paperExplanation.characteristics.limitations && paperExplanation.characteristics.limitations.length > 0 && (
                          <div>
                            <h5 className="text-xs font-semibold text-orange-600 mb-1.5 flex items-center gap-1">
                              <AlertTriangle className="w-3 h-3" />
                              한계점
                            </h5>
                            <ul className="space-y-1">
                              {paperExplanation.characteristics.limitations.map((l, i) => (
                                <li key={i} className="text-xs text-gray-600 flex items-start gap-1">
                                  <span className="text-orange-500 mt-0.5">•</span>
                                  <span>{l}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* AI Summary */}
              <div className="glass-2 rounded-xl p-4 border border-purple-100/50">
                <h4 className="text-xs font-bold text-purple-600 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <Sparkles className="w-3.5 h-3.5" />
                  AI Summary
                </h4>
                {isLoadingSummary ? (
                  <div className="flex items-center gap-2 text-gray-500">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">Generating summary...</span>
                  </div>
                ) : aiSummary ? (
                  <div className="space-y-3">
                    <p className="text-sm text-gray-700 leading-relaxed">{aiSummary}</p>
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
                  <p className="text-sm text-gray-400">No content available for summary.</p>
                )}
              </div>

              {/* Q&A Section */}
              {chatMode && (
                <ChatPanel
                  chatHistory={chatHistory}
                  chatQuestion={question}
                  onQuestionChange={setQuestion}
                  onSubmit={handleAskQuestion}
                  isLoading={isAskingQuestion}
                  chatEndRef={chatEndRef}
                  placeholder="Ask about this content..."
                />
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
                    AI Chat
                  </button>
                </div>

                <button
                  type="button"
                  onClick={() => onSelectResult(selectedResult)}
                  className="px-5 py-2 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-full text-sm font-semibold hover:from-violet-700 hover:to-purple-700 transition-all flex items-center gap-2 shadow-lg"
                >
                  View Full Details
                  <ExternalLink className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
