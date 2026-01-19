import React, { useState } from 'react';
import { X, Loader2, Sparkles, BookOpen, FileText, MessageSquare, Layers, Send, Telescope, ExternalLink } from 'lucide-react';
import api, { PrecisionSearchResult, SearchResult } from '../services/client';
import { KnowledgeGraph } from './KnowledgeGraph';

interface PaperDetail {
  title: string;
  summary: string;
  key_findings: string[];
  methodology?: string;
}

interface PaperDetailModalProps {
  result: PrecisionSearchResult;
  detail: PaperDetail | null;
  isLoading: boolean;
  onClose: () => void;
}

export const PaperDetailModal: React.FC<PaperDetailModalProps> = ({ result, detail, isLoading, onClose }) => {
  const [chatMode, setChatMode] = useState(false);
  const [chatQuestion, setChatQuestion] = useState('');
  const [chatAnswer, setChatAnswer] = useState<string | null>(null);
  const [isAskingQuestion, setIsAskingQuestion] = useState(false);
  const [similarPapers, setSimilarPapers] = useState<SearchResult[] | null>(null);
  const [isLoadingSimilar, setIsLoadingSimilar] = useState(false);
  const [showGalaxy, setShowGalaxy] = useState(false);

  const handleAskQuestion = async () => {
    if (!chatQuestion.trim()) return;
    setIsAskingQuestion(true);
    setChatAnswer(null);

    try {
      const response = await api.ask(chatQuestion);
      setChatAnswer(response.answer);
    } catch (err) {
      setChatAnswer('Sorry, I could not process your question. Please try again.');
    } finally {
      setIsAskingQuestion(false);
    }
  };

  const handleFindSimilar = async () => {
    setIsLoadingSimilar(true);
    setSimilarPapers(null);

    try {
      const searchQuery = detail?.key_findings?.[0] || result.paper_title;
      const response = await api.search(searchQuery);
      const filtered = response.results.filter(r => r.paper_title !== result.paper_title);
      setSimilarPapers(filtered.slice(0, 5));
    } catch (err) {
      console.error('Failed to find similar papers:', err);
      setSimilarPapers([]);
    } finally {
      setIsLoadingSimilar(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" onClick={onClose}>
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm animate-appear"></div>

      {/* Modal */}
      <div
        className="relative w-full max-w-3xl max-h-[85vh] glass-5 rounded-3xl shadow-2xl border border-purple-200/50 overflow-hidden animate-appear-zoom"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 glass-5 border-b border-purple-100/50 px-6 py-4 flex justify-between items-start z-10">
          <div className="flex-1 pr-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-semibold text-purple-600 bg-purple-100/80 px-2.5 py-1 rounded-full">
                {result.section}
              </span>
              <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
                {result.relevance_score.toFixed(1)}% match
              </span>
            </div>
            <h2 className="text-lg font-bold text-gray-900 leading-snug">{result.paper_title}</h2>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-purple-100/50 rounded-full transition-colors flex-shrink-0">
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(85vh-80px)]">
          {/* Original Search Result */}
          <div className="mb-6 p-4 glass-2 rounded-xl border border-purple-100/50">
            <h3 className="text-xs font-bold text-purple-600 uppercase tracking-wider mb-2 flex items-center gap-2">
              <BookOpen className="w-3.5 h-3.5" />
              Matched Content
            </h3>
            <p className="text-sm text-gray-700 leading-relaxed">{result.content}</p>
          </div>

          {/* AI Summary */}
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-12 gap-4">
              <Loader2 className="w-8 h-8 text-purple-500 animate-spin" />
              <p className="text-sm text-gray-500">Generating AI summary...</p>
            </div>
          ) : detail ? (
            <div className="space-y-6">
              {/* Summary */}
              <div>
                <h3 className="text-xs font-bold text-purple-600 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Sparkles className="w-3.5 h-3.5" />
                  AI Summary
                </h3>
                <p className="text-sm text-gray-700 leading-relaxed glass-2 rounded-xl p-4 border border-purple-100/50">
                  {detail.summary}
                </p>
              </div>

              {/* Key Findings */}
              {detail.key_findings && detail.key_findings.length > 0 && (
                <div>
                  <h3 className="text-xs font-bold text-emerald-600 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-emerald-500 flex items-center justify-center">
                      <span className="w-1.5 h-1.5 bg-white rounded-full"></span>
                    </span>
                    Key Findings
                  </h3>
                  <ul className="space-y-2">
                    {detail.key_findings.map((finding, idx) => (
                      <li key={idx} className="flex items-start gap-3 text-sm text-gray-700 glass-2 rounded-xl p-3 border border-emerald-100/50">
                        <span className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <span className="leading-relaxed">{finding}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Methodology */}
              {detail.methodology && (
                <div>
                  <h3 className="text-xs font-bold text-orange-600 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <FileText className="w-3.5 h-3.5" />
                    Methodology
                  </h3>
                  <p className="text-sm text-gray-700 leading-relaxed glass-2 rounded-xl p-4 border border-orange-100/50">
                    {detail.methodology}
                  </p>
                </div>
              )}

              {/* Q&A Section */}
              {chatMode && (
                <div className="mt-6 space-y-4">
                  <h3 className="text-xs font-bold text-blue-600 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <MessageSquare className="w-3.5 h-3.5" />
                    Ask Questions About This Paper
                  </h3>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={chatQuestion}
                      onChange={(e) => setChatQuestion(e.target.value)}
                      placeholder="Ask a question about this paper..."
                      className="flex-1 px-4 py-3 glass-3 border border-purple-200/50 rounded-xl text-sm focus:ring-2 focus:ring-purple-400/50"
                      onKeyDown={(e) => e.key === 'Enter' && handleAskQuestion()}
                      disabled={isAskingQuestion}
                    />
                    <button
                      onClick={handleAskQuestion}
                      disabled={isAskingQuestion || !chatQuestion.trim()}
                      className="px-4 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-50 flex items-center gap-2"
                    >
                      {isAskingQuestion ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                    </button>
                  </div>
                  {chatAnswer && (
                    <div className="glass-2 rounded-xl p-4 border border-blue-100/50 animate-appear">
                      <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">{chatAnswer}</p>
                    </div>
                  )}
                </div>
              )}

              {/* Similar Papers Section */}
              {similarPapers !== null && (
                <div className="mt-6 space-y-4">
                  <h3 className="text-xs font-bold text-indigo-600 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <Layers className="w-3.5 h-3.5" />
                    Similar Papers
                  </h3>
                  {isLoadingSimilar ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="w-6 h-6 text-purple-500 animate-spin" />
                    </div>
                  ) : similarPapers.length === 0 ? (
                    <p className="text-sm text-gray-500 text-center py-4">No similar papers found</p>
                  ) : (
                    <div className="space-y-2">
                      {similarPapers.map((paper, idx) => (
                        <div key={idx} className="glass-2 rounded-xl p-3 border border-indigo-100/50 hover:bg-indigo-50/30 transition-colors cursor-pointer">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1">
                              <h4 className="text-sm font-medium text-gray-900 line-clamp-1">{paper.paper_title}</h4>
                              <p className="text-xs text-gray-500 mt-1 line-clamp-2">{paper.content}</p>
                            </div>
                            <span className="text-xs font-medium text-indigo-600 bg-indigo-100/80 px-2 py-0.5 rounded-full flex-shrink-0">
                              {paper.relevance_score.toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <p>Unable to load paper details</p>
            </div>
          )}
        </div>

        {/* Footer with Actions */}
        <div className="sticky bottom-0 glass-5 border-t border-purple-100/50 px-6 py-4">
          <div className="flex items-center justify-between gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-purple-50/50 rounded-lg transition-colors"
            >
              Close
            </button>

            <div className="flex items-center gap-2">
              {/* Toggle Q&A Mode */}
              <button
                onClick={() => {
                  setChatMode(!chatMode);
                  setSimilarPapers(null);
                }}
                className={`group px-4 py-2.5 rounded-full text-sm font-semibold transition-all flex items-center gap-2 ${
                  chatMode
                    ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                    : 'glass-3 border border-purple-200/50 text-gray-700 hover:bg-purple-50/50'
                }`}
              >
                <MessageSquare className="w-4 h-4" />
                Q&A
              </button>

              {/* Find Similar Papers */}
              <button
                onClick={() => {
                  handleFindSimilar();
                  setChatMode(false);
                }}
                disabled={isLoadingSimilar}
                className={`group px-4 py-2.5 rounded-full text-sm font-semibold transition-all flex items-center gap-2 ${
                  similarPapers !== null
                    ? 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200'
                    : 'glass-3 border border-purple-200/50 text-gray-700 hover:bg-purple-50/50'
                }`}
              >
                {isLoadingSimilar ? <Loader2 className="w-4 h-4 animate-spin" /> : <Layers className="w-4 h-4" />}
                Similar Papers
              </button>

              {/* Galaxy View */}
              {result.pmid && (
                <button
                  onClick={() => setShowGalaxy(true)}
                  className="group px-4 py-2.5 rounded-full text-sm font-semibold transition-all flex items-center gap-2 bg-gradient-to-r from-yellow-500 to-orange-500 text-white hover:from-yellow-600 hover:to-orange-600 shadow-lg"
                >
                  <Telescope className="w-4 h-4" />
                  Galaxy View
                </button>
              )}

              {/* View Full Paper */}
              <button className="group px-5 py-2.5 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-full text-sm font-semibold hover:from-violet-700 hover:to-purple-700 transition-all flex items-center gap-2 shadow-lg btn-glow">
                View Full Paper
                <ExternalLink className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </button>
            </div>
          </div>
        </div>

        {/* Galaxy Visualization Modal */}
        {showGalaxy && result.pmid && (
          <KnowledgeGraph
            isOpen={showGalaxy}
            onClose={() => setShowGalaxy(false)}
            mode="similar"
            sourcePmid={result.pmid}
            domain="pancreatic_cancer"
            onPaperClick={(pmid) => {
              console.log('Navigate to paper:', pmid);
            }}
          />
        )}
      </div>
    </div>
  );
};
