import React, { useState, useRef, useEffect } from 'react';
import { Search, FileText, Dna, ArrowRight, Loader2, X, Sparkles, BookOpen, ExternalLink, ChevronRight, MessageSquare, Layers, Send, Globe, Link2, Database, Telescope, Target, Info, Tag } from 'lucide-react';
import api, { SearchResult, ChatResponse, CrawlerPaper, PrecisionSearchResult, SearchDiagnostics } from '../services/client';
import { KnowledgeGraph } from './KnowledgeGraph';

interface PaperDetail {
  title: string;
  summary: string;
  key_findings: string[];
  methodology?: string;
}

// Navigation modes after viewing paper detail
type NextAction = 'chat' | 'similar' | null;

// Search modes
type SearchMode = 'local' | 'pubmed' | 'doi';

// PubMed Search Results Component with Preview Panel
interface PubMedResultsProps {
  papers: CrawlerPaper[];
  onClose: () => void;
  isLoading?: boolean;
}

// Chat message interface for history
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const PubMedResults: React.FC<PubMedResultsProps> = ({ papers, onClose, isLoading }) => {
  const [selectedPaper, setSelectedPaper] = useState<CrawlerPaper | null>(null);
  const [showGalaxy, setShowGalaxy] = useState(false);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);
  const [chatMode, setChatMode] = useState(false);
  const [chatQuestion, setChatQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);  // Chat history array
  const [isAskingQuestion, setIsAskingQuestion] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);  // For auto-scroll

  // State for key points
  const [keyPoints, setKeyPoints] = useState<string[]>([]);

  // Generate AI summary when paper is selected
  const handleSelectPaper = async (paper: CrawlerPaper) => {
    setSelectedPaper(paper);
    setAiSummary(null);
    setKeyPoints([]);
    setChatHistory([]);  // Clear chat history when selecting new paper
    setChatMode(false);
    setChatQuestion('');

    if (paper.abstract) {
      setIsLoadingSummary(true);
      try {
        // Use the new summarize-abstract API
        const response = await api.summarizeAbstract(paper.title, paper.abstract);
        setAiSummary(response.summary);
        setKeyPoints(response.key_points || []);
      } catch (err) {
        setAiSummary('Unable to generate summary.');
      } finally {
        setIsLoadingSummary(false);
      }
    }
  };

  // Auto scroll to bottom when chat history updates
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // Handle AI Q&A with chat history
  const handleAskQuestion = async () => {
    if (!chatQuestion.trim() || !selectedPaper) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: chatQuestion,
      timestamp: new Date(),
    };

    setChatHistory(prev => [...prev, userMessage]);
    const currentQuestion = chatQuestion;
    setChatQuestion('');
    setIsAskingQuestion(true);

    try {
      // Use the new ask-abstract API
      const response = await api.askAbstract(
        selectedPaper.title,
        selectedPaper.abstract || '',
        currentQuestion
      );

      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
      };

      setChatHistory(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: 'Sorry, I could not process your question.',
        timestamp: new Date(),
      };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsAskingQuestion(false);
    }
  };

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Backdrop - click to close */}
        <div className="absolute inset-0 bg-black/30 backdrop-blur-sm" onClick={onClose} />

        {/* Modal with Split View */}
        <div
          className="relative z-[51] w-full max-w-6xl bg-white/95 backdrop-blur-xl rounded-2xl shadow-2xl border border-purple-200/50 flex animate-appear"
          style={{ height: 'calc(100vh - 8rem)', maxHeight: '750px', minHeight: '500px' }}
        >
          {/* Left: Paper List */}
          <div className={`${selectedPaper ? 'w-2/5' : 'w-full'} flex flex-col border-r border-purple-100/50 transition-all min-h-0`}>
            {/* Header */}
            <div className="flex-shrink-0 bg-white/80 backdrop-blur border-b border-purple-100/50 px-5 py-4 flex justify-between items-center rounded-tl-2xl">
              <div className="flex items-center gap-2">
                <Globe className="w-4 h-4 text-emerald-500" />
                <span className="text-sm font-semibold text-gray-700">
                  {isLoading ? 'Searching...' : `${papers.length} papers`}
                </span>
              </div>
              <button onClick={onClose} className="p-1.5 hover:bg-purple-100/50 rounded-full transition-colors">
                <X className="w-4 h-4 text-gray-500" />
              </button>
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
                      className={`p-4 cursor-pointer transition-all ${
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
                      <h4 className="text-sm font-medium text-gray-800 line-clamp-2">{paper.title}</h4>
                      {paper.authors.length > 0 && (
                        <p className="text-xs text-gray-500 mt-1 line-clamp-1">
                          {paper.authors.slice(0, 2).join(', ')}{paper.authors.length > 2 && ' et al.'}
                        </p>
                      )}
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
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-emerald-600 bg-emerald-100 px-2.5 py-1 rounded-full">
                    {selectedPaper.source === 'pubmed' ? 'PubMed' : selectedPaper.source}
                  </span>
                  {selectedPaper.pmid && (
                    <span className="text-xs text-gray-500">PMID: {selectedPaper.pmid}</span>
                  )}
                </div>
                <h3 className="text-lg font-bold text-gray-900 leading-snug">{selectedPaper.title}</h3>
                {selectedPaper.authors.length > 0 && (
                  <p className="text-sm text-gray-600 mt-1">
                    {selectedPaper.authors.slice(0, 5).join(', ')}{selectedPaper.authors.length > 5 && ' et al.'}
                  </p>
                )}
              </div>

              {/* Preview Content - Scrollable */}
              <div className="flex-1 min-h-0 overflow-y-auto p-6 space-y-5">
                {/* Abstract */}
                {selectedPaper.abstract && (
                  <div>
                    <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 flex items-center gap-2">
                      <BookOpen className="w-3.5 h-3.5" />
                      Abstract
                    </h4>
                    <p className="text-sm text-gray-700 leading-relaxed">{selectedPaper.abstract}</p>
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
                    <p className="text-sm text-gray-400">No abstract available for summary.</p>
                  )}
                </div>

                {/* Q&A Section with Chat History */}
                {chatMode && (
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold text-blue-600 uppercase tracking-wider flex items-center gap-2">
                      <MessageSquare className="w-3.5 h-3.5" />
                      Ask AI
                    </h4>

                    {/* Chat History */}
                    {chatHistory.length > 0 && (
                      <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
                        {chatHistory.map((msg) => (
                          <div
                            key={msg.id}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                          >
                            <div
                              className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm ${
                                msg.role === 'user'
                                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-br-sm'
                                  : 'glass-2 border border-blue-100/50 text-gray-700 rounded-bl-sm'
                              }`}
                            >
                              <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                            </div>
                          </div>
                        ))}
                        {isAskingQuestion && (
                          <div className="flex justify-start">
                            <div className="glass-2 border border-blue-100/50 rounded-2xl rounded-bl-sm px-4 py-2.5">
                              <div className="flex items-center gap-2 text-gray-500">
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span className="text-sm">Thinking...</span>
                              </div>
                            </div>
                          </div>
                        )}
                        <div ref={chatEndRef} />
                      </div>
                    )}

                    {/* Input */}
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={chatQuestion}
                        onChange={(e) => setChatQuestion(e.target.value)}
                        placeholder="Ask about this paper..."
                        className="flex-1 px-4 py-2.5 glass-3 border border-purple-200/50 rounded-xl text-sm focus:ring-2 focus:ring-purple-400/50"
                        onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleAskQuestion()}
                        disabled={isAskingQuestion}
                      />
                      <button
                        onClick={handleAskQuestion}
                        disabled={isAskingQuestion || !chatQuestion.trim()}
                        className="px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all disabled:opacity-50"
                      >
                        {isAskingQuestion ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>
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
                    {/* AI Chat Toggle */}
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

                    {/* Galaxy View */}
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

                  {/* View on PubMed */}
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

const LocalDBResults: React.FC<LocalDBResultsProps> = ({ results, diagnostics, onClose, onSelectResult }) => {
  const [selectedResult, setSelectedResult] = useState<PrecisionSearchResult | null>(null);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [keyPoints, setKeyPoints] = useState<string[]>([]);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);
  const [chatMode, setChatMode] = useState(false);
  const [chatQuestion, setChatQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isAskingQuestion, setIsAskingQuestion] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // Generate AI summary when result is selected
  const handleSelectResult = async (result: PrecisionSearchResult) => {
    setSelectedResult(result);
    setAiSummary(null);
    setKeyPoints([]);
    setChatHistory([]);
    setChatMode(false);
    setChatQuestion('');

    // Generate summary from the content
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
    if (!chatQuestion.trim() || !selectedResult) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: chatQuestion,
      timestamp: new Date(),
    };

    setChatHistory(prev => [...prev, userMessage]);
    const currentQuestion = chatQuestion;
    setChatQuestion('');
    setIsAskingQuestion(true);

    try {
      const response = await api.askAbstract(
        selectedResult.paper_title,
        selectedResult.content,
        currentQuestion
      );

      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
      };

      setChatHistory(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: 'Sorry, I could not process your question.',
        timestamp: new Date(),
      };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsAskingQuestion(false);
    }
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
                      {/* Matched terms */}
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
              {/* Match explanation */}
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

              {/* Q&A Section with Chat History */}
              {chatMode && (
                <div className="space-y-3">
                  <h4 className="text-xs font-bold text-blue-600 uppercase tracking-wider flex items-center gap-2">
                    <MessageSquare className="w-3.5 h-3.5" />
                    Ask AI
                  </h4>

                  {/* Chat History */}
                  {chatHistory.length > 0 && (
                    <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
                      {chatHistory.map((msg) => (
                        <div
                          key={msg.id}
                          className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div
                            className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm ${
                              msg.role === 'user'
                                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-br-sm'
                                : 'glass-2 border border-blue-100/50 text-gray-700 rounded-bl-sm'
                            }`}
                          >
                            <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                          </div>
                        </div>
                      ))}
                      {isAskingQuestion && (
                        <div className="flex justify-start">
                          <div className="glass-2 border border-blue-100/50 rounded-2xl rounded-bl-sm px-4 py-2.5">
                            <div className="flex items-center gap-2 text-gray-500">
                              <Loader2 className="w-4 h-4 animate-spin" />
                              <span className="text-sm">Thinking...</span>
                            </div>
                          </div>
                        </div>
                      )}
                      <div ref={chatEndRef} />
                    </div>
                  )}

                  {/* Input */}
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={chatQuestion}
                      onChange={(e) => setChatQuestion(e.target.value)}
                      placeholder="Ask about this content..."
                      className="flex-1 px-4 py-2.5 glass-3 border border-purple-200/50 rounded-xl text-sm focus:ring-2 focus:ring-purple-400/50"
                      onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleAskQuestion()}
                      disabled={isAskingQuestion}
                    />
                    <button
                      onClick={handleAskQuestion}
                      disabled={isAskingQuestion || !chatQuestion.trim()}
                      className="px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all disabled:opacity-50"
                    >
                      {isAskingQuestion ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="shrink-0 bg-white border-t border-purple-100/50 px-6 py-4 rounded-br-2xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {/* AI Chat Toggle */}
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

                {/* View Full Paper */}
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

interface ChatResultProps {
  response: ChatResponse;
  onClose: () => void;
}

const ChatResult: React.FC<ChatResultProps> = ({ response, onClose }) => (
  <div className="absolute top-full left-0 right-0 mt-3 glass-4 rounded-2xl shadow-2xl border border-purple-200/50 max-h-[500px] overflow-y-auto z-50 animate-appear">
    <div className="sticky top-0 glass-5 border-b border-purple-100/50 px-5 py-4 flex justify-between items-center">
      <div className="flex items-center gap-2">
        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
          <Sparkles className="w-3 h-3 text-white" />
        </div>
        <span className="text-sm font-semibold text-gray-700">AI Answer</span>
      </div>
      <button onClick={onClose} className="p-1.5 hover:bg-purple-100/50 rounded-full transition-colors">
        <X className="w-4 h-4 text-gray-500" />
      </button>
    </div>
    <div className="p-5">
      <div className="prose prose-sm max-w-none">
        <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">{response.answer}</p>
      </div>
      {response.sources.length > 0 && (
        <div className="mt-5 pt-5 border-t border-purple-100/50">
          <h5 className="text-xs font-bold text-purple-600 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span className="w-4 h-px bg-purple-300"></span>
            Sources
          </h5>
          {response.sources.map((src, idx) => (
            <div key={idx} className="text-sm text-gray-700 mb-2 flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-2 flex-shrink-0"></span>
              <div>
                <span className="font-medium">{src.paper_title}</span>
                <span className="text-gray-400 ml-2 text-xs">({src.section})</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  </div>
);

// Paper Detail Modal with Q&A and Similar Papers
interface PaperDetailModalProps {
  result: PrecisionSearchResult;
  detail: PaperDetail | null;
  isLoading: boolean;
  onClose: () => void;
}

const PaperDetailModal: React.FC<PaperDetailModalProps> = ({ result, detail, isLoading, onClose }) => {
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
      // Search for similar papers based on the paper title and key content
      const searchQuery = detail?.key_findings?.[0] || result.paper_title;
      const response = await api.search(searchQuery);
      // Filter out the current paper
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

            {/* Galaxy View - 3D visualization */}
            {result.pmid && (
              <button
                onClick={() => setShowGalaxy(true)}
                className="group px-4 py-2.5 rounded-full text-sm font-semibold transition-all flex items-center gap-2 bg-gradient-to-r from-yellow-500 to-orange-500 text-white hover:from-yellow-600 hover:to-orange-600 shadow-lg"
              >
                <Telescope className="w-4 h-4" />
                Galaxy View
              </button>
            )}

            {/* View Full Paper (primary action) */}
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
            // Could implement paper navigation here
          }}
        />
      )}
    </div>
  </div>
  );
};

// Glow Component (monet-style)
const Glow: React.FC<{ variant?: 'top' | 'center'; className?: string }> = ({ variant = 'top', className = '' }) => (
  <div className={`absolute w-full pointer-events-none ${variant === 'top' ? 'top-0' : 'top-1/2 -translate-y-1/2'} ${className}`}>
    <div className="absolute left-1/2 -translate-x-1/2 h-[256px] w-[60%] scale-[2] rounded-[50%] bg-gradient-radial from-purple-400/30 via-violet-300/20 to-transparent opacity-60 blur-3xl sm:h-[400px]" />
    <div className="absolute left-1/2 -translate-x-1/2 h-[128px] w-[40%] scale-150 rounded-[50%] bg-gradient-radial from-violet-500/20 to-transparent opacity-50 blur-2xl sm:h-[200px]" />
  </div>
);

export const Hero: React.FC = () => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<PrecisionSearchResult[] | null>(null);
  const [searchDiagnostics, setSearchDiagnostics] = useState<SearchDiagnostics | null>(null);
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Search mode state
  const [searchMode, setSearchMode] = useState<SearchMode>('local');
  const [pubmedResults, setPubmedResults] = useState<CrawlerPaper[] | null>(null);
  const [doiResult, setDoiResult] = useState<CrawlerPaper | null>(null);

  // Paper detail modal state
  const [selectedResult, setSelectedResult] = useState<PrecisionSearchResult | null>(null);
  const [paperDetail, setPaperDetail] = useState<PaperDetail | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);

  // Auto-detect DOI in query
  const isDOI = (text: string): boolean => {
    return /^10\.\d{4,}\//.test(text.trim()) || text.includes('doi.org/');
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setSearchResults(null);
    setSearchDiagnostics(null);
    setChatResponse(null);
    setPubmedResults(null);
    setDoiResult(null);

    try {
      // Auto-detect DOI and switch mode
      if (isDOI(query)) {
        const paper = await api.fetchByDOI(query);
        setDoiResult(paper);
        setPubmedResults([paper]); // Show in results list
      } else if (searchMode === 'pubmed') {
        // Real-time PubMed search
        const response = await api.searchPubMed(query, { limit: 10 });
        setPubmedResults(response.papers);
      } else if (query.trim().endsWith('?')) {
        // Question mode - use RAG
        const response = await api.ask(query);
        setChatResponse(response);
      } else {
        // Local vector DB search with precision search
        const response = await api.precisionSearch(query);
        setSearchResults(response.results);
        setSearchDiagnostics(response.diagnostics);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await api.uploadPaper(file);
      if (result.success) {
        alert(`Paper "${result.paper_title}" indexed successfully! (${result.chunks_created} chunks)`);
      } else {
        setError(result.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const closeResults = () => {
    setSearchResults(null);
    setSearchDiagnostics(null);
    setChatResponse(null);
    setPubmedResults(null);
    setDoiResult(null);
  };

  const handleSelectResult = async (result: PrecisionSearchResult) => {
    setSelectedResult(result);
    setPaperDetail(null);
    setIsLoadingDetail(true);

    try {
      // Get paper summary from API
      const summary = await api.summarize(result.paper_title);
      setPaperDetail(summary);
    } catch (err) {
      console.error('Failed to load paper detail:', err);
      // Set a basic detail if API fails
      setPaperDetail({
        title: result.paper_title,
        summary: result.content,
        key_findings: [],
      });
    } finally {
      setIsLoadingDetail(false);
    }
  };

  const closePaperDetail = () => {
    setSelectedResult(null);
    setPaperDetail(null);
  };

  return (
  <>
    <section className="relative w-full min-h-screen flex flex-col items-center justify-center overflow-hidden line-b">
      {/* Spline 3D DNA Background */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <iframe
          src='https://my.spline.design/dnaparticles-DFdgYaZFlGevB4Cghzds8lOd/'
          frameBorder='0'
          width='100%'
          height='100%'
          className="w-full h-full opacity-60 pointer-events-none"
          title="3D DNA Particles"
        ></iframe>
        {/* Enhanced gradient overlays */}
        <div className="absolute inset-0 bg-gradient-to-b from-violet-100/70 via-transparent to-purple-100/90"></div>
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-100/50 via-transparent to-pink-100/50"></div>
      </div>

      {/* Monet-style Glow Effects */}
      <Glow variant="center" className="animate-pulse-glow" />

      {/* Hero Content */}
      <div className="relative z-10 w-full max-w-4xl mx-auto px-6 text-center py-16">
        {/* Badge */}
        <div className="animate-appear opacity-0 mb-8">
          <span className="inline-flex items-center gap-2 px-4 py-2 glass-3 rounded-full border border-purple-200/50 text-sm font-medium text-gray-700 glow-white">
            <Sparkles className="w-4 h-4 text-purple-500" />
            <span className="text-gray-500">AI-Powered Research Platform</span>
            <ArrowRight className="w-3 h-3 text-purple-500" />
          </span>
        </div>

        <h1 className="animate-appear opacity-0 delay-100 text-5xl md:text-7xl font-bold tracking-tight mb-6 leading-tight">
          <span className="text-gradient-hero drop-shadow-sm">
            Biological Insight,
          </span>
          <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-500 font-serif italic">
            Starts with a Question.
          </span>
        </h1>

        <p className="animate-appear opacity-0 delay-200 text-xl md:text-2xl text-gray-600 max-w-2xl mx-auto mb-12 font-light leading-relaxed">
          Search literature, analyze data, and interpret results — in one unified platform.
        </p>

        {/* Search Mode Selector */}
        <div className="animate-appear opacity-0 delay-250 flex items-center justify-center gap-2 mb-4">
          <button
            type="button"
            onClick={() => setSearchMode('local')}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all flex items-center gap-2 ${
              searchMode === 'local'
                ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white shadow-lg'
                : 'glass-2 border border-purple-200/50 text-gray-600 hover:bg-purple-50/50'
            }`}
          >
            <Database className="w-4 h-4" />
            Local DB
          </button>
          <button
            type="button"
            onClick={() => setSearchMode('pubmed')}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all flex items-center gap-2 ${
              searchMode === 'pubmed'
                ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg'
                : 'glass-2 border border-purple-200/50 text-gray-600 hover:bg-purple-50/50'
            }`}
          >
            <Globe className="w-4 h-4" />
            PubMed Live
          </button>
          <button
            type="button"
            onClick={() => setSearchMode('doi')}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all flex items-center gap-2 ${
              searchMode === 'doi'
                ? 'bg-gradient-to-r from-orange-600 to-red-600 text-white shadow-lg'
                : 'glass-2 border border-purple-200/50 text-gray-600 hover:bg-purple-50/50'
            }`}
          >
            <Link2 className="w-4 h-4" />
            DOI/URL
          </button>
        </div>

        {/* Search Input Container */}
        <div className="animate-appear opacity-0 delay-300 relative max-w-3xl mx-auto w-full">
          <form onSubmit={handleSearch} className="relative group">
            {/* Search Icon */}
            <div className="absolute inset-y-0 left-0 pl-6 flex items-center pointer-events-none z-10">
              {isLoading ? (
                <Loader2 className="h-6 w-6 text-purple-500 animate-spin" />
              ) : searchMode === 'pubmed' ? (
                <Globe className="h-6 w-6 text-emerald-400 group-focus-within:text-emerald-600 transition-colors" />
              ) : searchMode === 'doi' ? (
                <Link2 className="h-6 w-6 text-orange-400 group-focus-within:text-orange-600 transition-colors" />
              ) : (
                <Search className="h-6 w-6 text-purple-400 group-focus-within:text-purple-600 transition-colors" />
              )}
            </div>

            {/* Input Field with Glass Effect */}
            <input
              type="text"
              className={`block w-full rounded-full border py-5 pl-16 pr-40 text-gray-900 placeholder:text-gray-400 focus:ring-2 text-lg shadow-xl glass-4 transition-all hover:shadow-2xl ${
                searchMode === 'pubmed'
                  ? 'border-emerald-200/50 focus:ring-emerald-400/50 focus:border-emerald-300 hover:border-emerald-300'
                  : searchMode === 'doi'
                  ? 'border-orange-200/50 focus:ring-orange-400/50 focus:border-orange-300 hover:border-orange-300'
                  : 'border-purple-200/50 focus:ring-purple-400/50 focus:border-purple-300 hover:border-purple-300 focus:glow-brand'
              }`}
              placeholder={
                searchMode === 'pubmed'
                  ? 'Search PubMed for papers...'
                  : searchMode === 'doi'
                  ? 'Enter DOI (e.g., 10.1038/s41586-023-...) or URL'
                  : 'Search genes, diseases, or ask a question...'
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isLoading}
              autoFocus
            />

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              className="hidden"
            />

            {/* Right Actions */}
            <div className="absolute inset-y-0 right-2 flex items-center gap-1">
              <div className="flex items-center gap-1 border-r border-purple-200/50 pr-2 mr-2 py-2">
                <button
                  type="button"
                  className="p-3 text-purple-400 hover:text-purple-600 hover:bg-purple-100/50 rounded-full transition-all"
                  title="Upload Literature (PDF)"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                >
                  <FileText className="w-5 h-5" />
                </button>
                <button
                  type="button"
                  className="p-3 text-purple-400 hover:text-purple-600 hover:bg-purple-100/50 rounded-full transition-all"
                  title="Upload RNA-seq Data"
                  disabled={isLoading}
                >
                  <Dna className="w-5 h-5" />
                </button>
              </div>

              <button
                type="submit"
                className="p-3 rounded-full bg-gradient-to-r from-violet-600 to-purple-600 text-white hover:from-violet-700 hover:to-purple-700 transition-all shadow-lg btn-glow disabled:opacity-50"
                disabled={isLoading}
              >
                {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <ArrowRight className="w-5 h-5" />}
              </button>
            </div>
          </form>

          {/* Error Message */}
          {error && (
            <div className="absolute top-full left-0 right-0 mt-3 glass-3 border border-red-200/50 text-red-700 px-4 py-3 rounded-xl text-sm animate-appear">
              {error}
            </div>
          )}

          {/* Chat Response */}
          {chatResponse && <ChatResult response={chatResponse} onClose={closeResults} />}
        </div>

        {/* Subtle indicator text - hide when results are shown */}
        {!searchResults && !chatResponse && !pubmedResults && (
          <p className="animate-appear opacity-0 delay-500 mt-10 text-sm text-gray-500 flex items-center justify-center gap-2">
            <span className="w-8 h-px bg-purple-300/50"></span>
            Try: "BRCA1 mutations" or "What causes drug resistance in cancer?"
            <span className="w-8 h-px bg-purple-300/50"></span>
          </p>
        )}
      </div>

      {/* Bottom Glow */}
      <Glow variant="top" className="bottom-0 top-auto rotate-180 opacity-40" />

    </section>

    {/* Modals rendered outside section to avoid overflow:hidden */}
    {searchResults && <LocalDBResults results={searchResults} diagnostics={searchDiagnostics} onClose={closeResults} onSelectResult={handleSelectResult} />}
    {pubmedResults && <PubMedResults papers={pubmedResults} onClose={closeResults} isLoading={isLoading} />}

    {selectedResult && (
      <PaperDetailModal
        result={selectedResult}
        detail={paperDetail}
        isLoading={isLoadingDetail}
        onClose={closePaperDetail}
      />
    )}
  </>
  );
};
