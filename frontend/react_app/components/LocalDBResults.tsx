import React, { useState } from 'react';
import { Target, X, FileText, Loader2, Info, Sparkles, BookOpen, MessageSquare, ExternalLink } from 'lucide-react';
import api, { PrecisionSearchResult, SearchDiagnostics } from '../services/client';
import { ChatPanel } from './ChatPanel';
import { useChat } from '../hooks/useChat';

interface LocalDBResultsProps {
  results: PrecisionSearchResult[];
  diagnostics: SearchDiagnostics | null;
  onClose: () => void;
  onSelectResult: (result: PrecisionSearchResult) => void;
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

export const LocalDBResults: React.FC<LocalDBResultsProps> = ({ results, diagnostics, onClose, onSelectResult }) => {
  const [selectedResult, setSelectedResult] = useState<PrecisionSearchResult | null>(null);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [keyPoints, setKeyPoints] = useState<string[]>([]);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);
  const [chatMode, setChatMode] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  const { chatHistory, question, setQuestion, isLoading: isAskingQuestion, sendMessage, clearHistory, chatEndRef } = useChat();

  // Generate AI summary when result is selected
  const handleSelectResult = async (result: PrecisionSearchResult) => {
    setSelectedResult(result);
    setAiSummary(null);
    setKeyPoints([]);
    clearHistory();
    setChatMode(false);

    if (result.content) {
      setIsLoadingSummary(true);
      try {
        const response = await api.summarizeAbstract(result.paper_title, result.content);
        setAiSummary(response.summary);
        setKeyPoints(response.key_points || []);
      } catch (err) {
        setAiSummary('Unable to generate summary.');
      } finally {
        setIsLoadingSummary(false);
      }
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
