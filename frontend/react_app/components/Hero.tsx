import React, { useState, useRef } from 'react';
import { Search, FileText, Dna, ArrowRight, Loader2, X, Sparkles, BookOpen, ExternalLink, ChevronRight } from 'lucide-react';
import api, { SearchResult, ChatResponse } from '../services/client';

interface PaperDetail {
  title: string;
  summary: string;
  key_findings: string[];
  methodology?: string;
}

interface SearchResultsProps {
  results: SearchResult[];
  onClose: () => void;
  onSelectResult: (result: SearchResult) => void;
}

const SearchResults: React.FC<SearchResultsProps> = ({ results, onClose, onSelectResult }) => (
  <div className="absolute top-full left-0 right-0 mt-3 glass-4 rounded-2xl shadow-2xl border border-purple-200/50 max-h-[400px] overflow-y-auto z-50 animate-appear">
    <div className="sticky top-0 glass-5 border-b border-purple-100/50 px-5 py-4 flex justify-between items-center">
      <span className="text-sm font-semibold text-gray-700">{results.length} results found</span>
      <button onClick={onClose} className="p-1.5 hover:bg-purple-100/50 rounded-full transition-colors">
        <X className="w-4 h-4 text-gray-500" />
      </button>
    </div>
    {results.map((result, idx) => (
      <div
        key={idx}
        className="p-4 border-b border-purple-50/50 hover:bg-purple-50/50 transition-all cursor-pointer card-hover group"
        style={{ animationDelay: `${idx * 50}ms` }}
        onClick={() => onSelectResult(result)}
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-purple-600 bg-purple-100/80 px-2.5 py-1 rounded-full">
            {result.section}
          </span>
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
              {result.relevance_score.toFixed(1)}% match
            </span>
            <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-purple-500 group-hover:translate-x-0.5 transition-all" />
          </div>
        </div>
        <h4 className="text-sm font-semibold text-gray-800 mb-1.5 line-clamp-1 group-hover:text-purple-700 transition-colors">{result.paper_title}</h4>
        <p className="text-sm text-gray-600 line-clamp-2 leading-relaxed">{result.content}</p>
      </div>
    ))}
  </div>
);

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

// Paper Detail Modal
interface PaperDetailModalProps {
  result: SearchResult;
  detail: PaperDetail | null;
  isLoading: boolean;
  onClose: () => void;
}

const PaperDetailModal: React.FC<PaperDetailModalProps> = ({ result, detail, isLoading, onClose }) => (
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
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>Unable to load paper details</p>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="sticky bottom-0 glass-5 border-t border-purple-100/50 px-6 py-4 flex justify-between items-center">
        <button
          onClick={onClose}
          className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-purple-50/50 rounded-lg transition-colors"
        >
          Close
        </button>
        <button className="group px-5 py-2.5 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-full text-sm font-semibold hover:from-violet-700 hover:to-purple-700 transition-all flex items-center gap-2 shadow-lg btn-glow">
          View Full Paper
          <ExternalLink className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
        </button>
      </div>
    </div>
  </div>
);

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
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Paper detail modal state
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null);
  const [paperDetail, setPaperDetail] = useState<PaperDetail | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setSearchResults(null);
    setChatResponse(null);

    try {
      if (query.trim().endsWith('?')) {
        const response = await api.ask(query);
        setChatResponse(response);
      } else {
        const response = await api.search(query);
        setSearchResults(response.results);
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
    setChatResponse(null);
  };

  const handleSelectResult = async (result: SearchResult) => {
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
    <section className="relative w-full min-h-screen flex flex-col items-center justify-center overflow-hidden line-b">
      {/* Spline 3D DNA Background */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <iframe
          src='https://my.spline.design/dnaparticles-DFdgYaZFlGevB4Cghzds8lOd/'
          frameBorder='0'
          width='100%'
          height='100%'
          className="w-full h-full opacity-60"
          title="3D DNA Particles"
        ></iframe>
        {/* Enhanced gradient overlays */}
        <div className="absolute inset-0 bg-gradient-to-b from-violet-100/70 via-transparent to-purple-100/90"></div>
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-100/50 via-transparent to-pink-100/50"></div>
      </div>

      {/* Monet-style Glow Effects */}
      <Glow variant="center" className="animate-pulse-glow" />

      {/* Hero Content */}
      <div className="relative z-10 w-full max-w-4xl mx-auto px-6 text-center pt-32 pb-24">
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

        {/* Search Input Container */}
        <div className="animate-appear opacity-0 delay-300 relative max-w-3xl mx-auto w-full">
          <form onSubmit={handleSearch} className="relative group">
            {/* Search Icon */}
            <div className="absolute inset-y-0 left-0 pl-6 flex items-center pointer-events-none z-10">
              {isLoading ? (
                <Loader2 className="h-6 w-6 text-purple-500 animate-spin" />
              ) : (
                <Search className="h-6 w-6 text-purple-400 group-focus-within:text-purple-600 transition-colors" />
              )}
            </div>

            {/* Input Field with Glass Effect */}
            <input
              type="text"
              className="block w-full rounded-full border border-purple-200/50 py-5 pl-16 pr-40 text-gray-900 placeholder:text-gray-400 focus:ring-2 focus:ring-purple-400/50 focus:border-purple-300 text-lg shadow-xl glass-4 transition-all hover:shadow-2xl hover:border-purple-300 focus:glow-brand"
              placeholder="Search genes, diseases, or ask a question..."
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

          {/* Search Results */}
          {searchResults && <SearchResults results={searchResults} onClose={closeResults} onSelectResult={handleSelectResult} />}

          {/* Chat Response */}
          {chatResponse && <ChatResult response={chatResponse} onClose={closeResults} />}
        </div>

        {/* Subtle indicator text */}
        <p className="animate-appear opacity-0 delay-500 mt-10 text-sm text-gray-500 flex items-center justify-center gap-2">
          <span className="w-8 h-px bg-purple-300/50"></span>
          Try: "BRCA1 mutations" or "What causes drug resistance in cancer?"
          <span className="w-8 h-px bg-purple-300/50"></span>
        </p>
      </div>

      {/* Bottom Glow */}
      <Glow variant="top" className="bottom-0 top-auto rotate-180 opacity-40" />

      {/* Paper Detail Modal */}
      {selectedResult && (
        <PaperDetailModal
          result={selectedResult}
          detail={paperDetail}
          isLoading={isLoadingDetail}
          onClose={closePaperDetail}
        />
      )}
    </section>
  );
};
