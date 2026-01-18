import React, { useState, useRef, useCallback } from 'react';
import { Search, FileText, Dna, ArrowRight, Loader2, Sparkles, Globe, Link2, Database, Flame, Upload, FileSpreadsheet } from 'lucide-react';
import api, { ChatResponse, PrecisionSearchResult, SearchDiagnostics } from '../services/client';
import { PubMedResults } from './PubMedResults';
import { LocalDBResults } from './LocalDBResults';
import { PaperDetailModal } from './PaperDetailModal';
import { ChatResult } from './ChatResult';
import { Glow } from './Glow';
import ResearchTrends from './ResearchTrends';
import { useLanguage } from '../contexts/LanguageContext';
import { RNAseqUploadModal } from './RNAseqUploadModal';
import { PipelineProgress } from './PipelineProgress';
import type { ReviewPaper } from '../App';

// Paper detail interface
interface PaperDetail {
  title: string;
  summary: string;
  key_findings: string[];
  methodology?: string;
}

// Search modes
type SearchMode = 'local' | 'pubmed' | 'doi';

// Search filter options
interface SearchFilters {
  sort: 'relevance' | 'pub_date';
  yearRange: 'all' | '1' | '3' | '5';
  limit: number;
}

interface HeroProps {
  onAddToReview?: (paper: ReviewPaper) => void;
  onAddMultipleToReview?: (papers: ReviewPaper[]) => void;
  reviewPapersCount?: number;
  onOpenReview?: () => void;
}

export const Hero: React.FC<HeroProps> = ({
  onAddToReview,
  onAddMultipleToReview,
  reviewPapersCount = 0,
  onOpenReview
}) => {
  const { t, language } = useLanguage();
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<PrecisionSearchResult[] | null>(null);
  const [searchDiagnostics, setSearchDiagnostics] = useState<SearchDiagnostics | null>(null);
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Search mode state
  const [searchMode, setSearchMode] = useState<SearchMode>('local');
  const [pubmedResults, setPubmedResults] = useState<any[] | null>(null);
  const [doiResult, setDoiResult] = useState<any | null>(null);
  const [wasQueryKorean, setWasQueryKorean] = useState(false);
  const [lastSearchQuery, setLastSearchQuery] = useState('');
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({
    sort: 'relevance',
    yearRange: 'all',
    limit: 20
  });

  // Paper detail modal state
  const [selectedResult, setSelectedResult] = useState<PrecisionSearchResult | null>(null);
  const [paperDetail, setPaperDetail] = useState<PaperDetail | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);

  // Research Trends modal state
  const [showResearchTrends, setShowResearchTrends] = useState(false);
  const [researchTrendsTab, setResearchTrendsTab] = useState<'hot-topics' | 'trend-evolution' | 'trending-papers'>('hot-topics');

  // RNA-seq analysis modal state
  const [showRNAseqModal, setShowRNAseqModal] = useState(false);
  const [rnaseqJobId, setRnaseqJobId] = useState<string | null>(null);

  // Drag and drop state for RNA-seq files
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [rnaseqFile, setRnaseqFile] = useState<File | null>(null);

  // Reference for CSV/TSV file input
  const rnaseqFileInputRef = useRef<HTMLInputElement>(null);

  // Auto-detect DOI in query
  const isDOI = (text: string): boolean => {
    return /^10\.\d{4,}\//.test(text.trim()) || text.includes('doi.org/');
  };

  // Auto-detect RNA-seq related keywords in query
  const isRNAseqQuery = (text: string): boolean => {
    const lowerText = text.toLowerCase().trim();
    const rnaseqPatterns = [
      'rna-seq', 'rnaseq', 'rna seq',
      'transcriptome', 'deseq', 'deg analysis',
      '발현 분석', '전사체', '유전자 발현'
    ];
    return rnaseqPatterns.some(pattern => lowerText.includes(pattern));
  };

  // Translate papers in background (non-blocking)
  const translatePapersInBackground = async (papers: any[]) => {
    for (let i = 0; i < papers.length; i++) {
      const paper = papers[i];
      if (paper.title && !paper.title_ko) {
        try {
          const translated = await api.translateText(paper.title, 'ko');
          if (translated) {
            setPubmedResults(prev => {
              if (!prev) return prev;
              const updated = [...prev];
              if (updated[i]) {
                updated[i] = { ...updated[i], title_ko: translated };
              }
              return updated;
            });
          }
        } catch (e) {
          // Ignore translation errors
        }
      }
    }
  };

  // Calculate min year based on year range filter
  const getMinYear = (yearRange: string): number | undefined => {
    if (yearRange === 'all') return undefined;
    const currentYear = new Date().getFullYear();
    return currentYear - parseInt(yearRange);
  };

  // Re-search with new filters
  const handleFilterChange = async (newFilters: SearchFilters) => {
    setSearchFilters(newFilters);
    if (!lastSearchQuery) return;

    setIsLoading(true);
    try {
      const response = await api.searchPubMed(lastSearchQuery, {
        limit: newFilters.limit,
        sort: newFilters.sort,
        minYear: getMinYear(newFilters.yearRange)
      });
      setPubmedResults(response.papers);
    } catch (err) {
      console.error('Filter search error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    // Auto-detect RNA-seq keywords and open upload modal
    if (isRNAseqQuery(query)) {
      setShowRNAseqModal(true);
      return;
    }

    setIsLoading(true);
    setError(null);
    setSearchResults(null);
    setSearchDiagnostics(null);
    setChatResponse(null);
    setPubmedResults(null);
    setDoiResult(null);

    try {
      const searchQuery = query.trim();
      setLastSearchQuery(searchQuery);  // Store for filter re-search

      // Auto-detect DOI and switch mode
      if (isDOI(query)) {
        const paper = await api.fetchByDOI(query);
        setDoiResult(paper);
        setPubmedResults([paper]); // Show in results list
      } else if (searchMode === 'pubmed') {
        // Real-time PubMed search (fast - no result translation)
        const response = await api.searchPubMed(searchQuery, {
          limit: searchFilters.limit,
          sort: searchFilters.sort,
          minYear: getMinYear(searchFilters.yearRange)
        });
        // Track if original query was Korean for summary language
        const isKorean = api.containsKorean(searchQuery) || (response.was_translated ?? false);
        setWasQueryKorean(isKorean);
        // Show translation info if query was translated
        if (response.was_translated && response.query_translated) {
          console.log(`번역됨: "${searchQuery}" → "${response.query_translated}"`);
        }
        // Show results immediately
        setPubmedResults(response.papers);

        // Translate titles/abstracts in background (non-blocking)
        if (isKorean && response.papers?.length > 0) {
          translatePapersInBackground(response.papers);
        }
      } else if (query.trim().endsWith('?')) {
        // Question mode - use RAG
        const response = await api.ask(query);
        setChatResponse(response);
      } else {
        // Local vector DB search - translate Korean if needed
        let localSearchQuery = searchQuery;
        if (api.containsKorean(searchQuery)) {
          try {
            const translated = await api.translateQuery(searchQuery);
            if (translated.translated && translated.translated !== searchQuery) {
              localSearchQuery = translated.translated;
              console.log(`번역됨: "${searchQuery}" → "${localSearchQuery}"`);
            }
          } catch (err) {
            console.warn('Translation failed, using original:', err);
          }
        }
        // Request more results (200) for disease-only queries to show all papers
        const response = await api.precisionSearch(localSearchQuery, 'auto', { topK: 200 });
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

  // Check if file is a potential RNA-seq count matrix (CSV/TSV)
  const isRNAseqFile = (file: File): boolean => {
    const name = file.name.toLowerCase();
    return name.endsWith('.csv') || name.endsWith('.tsv') || name.endsWith('.txt');
  };

  // Handle drag events for RNA-seq file drop
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.types.includes('Files')) {
      setIsDraggingFile(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only set to false if leaving the drop zone entirely
    if (e.currentTarget.contains(e.relatedTarget as Node)) {
      return;
    }
    setIsDraggingFile(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (isRNAseqFile(file)) {
        // Open RNA-seq modal with the file pre-loaded
        setRnaseqFile(file);
        setShowRNAseqModal(true);
      } else if (file.name.endsWith('.pdf')) {
        // Handle PDF upload as before
        if (fileInputRef.current) {
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(file);
          fileInputRef.current.files = dataTransfer.files;
          handleFileUpload({ target: fileInputRef.current } as React.ChangeEvent<HTMLInputElement>);
        }
      }
    }
  }, []);

  // Handle RNA-seq file selection from input
  const handleRNAseqFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && isRNAseqFile(file)) {
      setRnaseqFile(file);
      setShowRNAseqModal(true);
    }
    // Reset input
    if (rnaseqFileInputRef.current) {
      rnaseqFileInputRef.current.value = '';
    }
  };

  return (
  <>
    <section
      className={`relative w-full min-h-screen flex flex-col items-center justify-center overflow-hidden line-b transition-all ${
        isDraggingFile ? 'bg-purple-50/50' : ''
      }`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* Drag overlay for RNA-seq files */}
      {isDraggingFile && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-purple-100/80 backdrop-blur-sm pointer-events-none">
          <div className="flex flex-col items-center gap-4 p-8 rounded-3xl bg-white/90 shadow-2xl border-2 border-dashed border-purple-400">
            <div className="p-4 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600">
              <FileSpreadsheet className="w-12 h-12 text-white" />
            </div>
            <div className="text-center">
              <p className="text-xl font-bold text-gray-800">
                {language === 'ko' ? 'RNA-seq 데이터 드롭' : 'Drop RNA-seq Data'}
              </p>
              <p className="text-sm text-gray-500 mt-1">
                {language === 'ko' ? 'CSV/TSV count matrix 파일' : 'CSV/TSV count matrix file'}
              </p>
            </div>
          </div>
        </div>
      )}

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
            <span className="text-gray-500">{t.heroBadge}</span>
            <ArrowRight className="w-3 h-3 text-purple-500" />
          </span>
        </div>

        <h1 className="animate-appear opacity-0 delay-100 text-5xl md:text-7xl font-bold tracking-tight mb-6 leading-tight">
          <span className="text-gradient-hero drop-shadow-sm">
            {t.heroTitle1}
          </span>
          <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-500 font-serif italic">
            {t.heroTitle2}
          </span>
        </h1>

        <p className="animate-appear opacity-0 delay-200 text-xl md:text-2xl text-gray-600 max-w-2xl mx-auto mb-12 font-light leading-relaxed">
          {t.heroSubtitle}
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
            {t.localDB}
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
            {t.pubmedLive}
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
            {t.doiUrl}
          </button>

          {/* Research Trends Hub - Single Button */}
          <div className="w-px h-6 bg-gray-300 mx-1" />
          <button
            type="button"
            onClick={() => {
              setResearchTrendsTab('hot-topics');
              setShowResearchTrends(true);
            }}
            className="px-4 py-2 rounded-full text-sm font-medium transition-all flex items-center gap-2 bg-gradient-to-r from-orange-500 to-pink-500 text-white shadow-lg hover:from-orange-600 hover:to-pink-600"
          >
            <Flame className="w-4 h-4" />
            {t.researchTrends}
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
              className={`block w-full rounded-full border py-5 pl-16 pr-56 text-gray-900 placeholder:text-gray-400 focus:ring-2 text-lg shadow-xl glass-4 transition-all hover:shadow-2xl ${
                searchMode === 'pubmed'
                  ? 'border-emerald-200/50 focus:ring-emerald-400/50 focus:border-emerald-300 hover:border-emerald-300'
                  : searchMode === 'doi'
                  ? 'border-orange-200/50 focus:ring-orange-400/50 focus:border-orange-300 hover:border-orange-300'
                  : 'border-purple-200/50 focus:ring-purple-400/50 focus:border-purple-300 hover:border-purple-300 focus:glow-brand'
              }`}
              placeholder={
                searchMode === 'pubmed'
                  ? t.searchPlaceholderPubmed
                  : searchMode === 'doi'
                  ? t.searchPlaceholderDoi
                  : t.searchPlaceholder
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isLoading}
              autoFocus
            />

            {/* Hidden file inputs */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              className="hidden"
            />
            <input
              ref={rnaseqFileInputRef}
              type="file"
              accept=".csv,.tsv,.txt"
              onChange={handleRNAseqFileSelect}
              className="hidden"
            />

            {/* Right Actions */}
            <div className="absolute inset-y-0 right-2 flex items-center gap-1">
              <div className="flex items-center gap-1 border-r border-purple-200/50 pr-2 mr-2 py-2">
                <button
                  type="button"
                  className="flex items-center gap-1.5 px-3 py-2 text-purple-500 hover:text-purple-700 hover:bg-purple-100/50 rounded-full transition-all text-sm font-medium"
                  title={t.uploadPdf}
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                >
                  <FileText className="w-4 h-4" />
                  <span className="hidden sm:inline">{t.uploadPdfShort}</span>
                </button>
                <button
                  type="button"
                  className="flex items-center gap-1.5 px-3 py-2 text-purple-500 hover:text-purple-700 hover:bg-purple-100/50 rounded-full transition-all text-sm font-medium"
                  title={t.uploadRnaseq}
                  disabled={isLoading}
                  onClick={() => rnaseqFileInputRef.current?.click()}
                >
                  <Dna className="w-4 h-4" />
                  <span className="hidden sm:inline">{t.uploadRnaseqShort}</span>
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
            {t.heroHint}
            <span className="w-8 h-px bg-purple-300/50"></span>
          </p>
        )}
      </div>

      {/* Bottom Glow */}
      <Glow variant="top" className="bottom-0 top-auto rotate-180 opacity-40" />

    </section>

    {/* Modals rendered outside section to avoid overflow:hidden */}
    {searchResults && <LocalDBResults results={searchResults} diagnostics={searchDiagnostics} onClose={closeResults} onSelectResult={handleSelectResult} query={lastSearchQuery} />}
    {pubmedResults && (
        <PubMedResults
          papers={pubmedResults}
          onClose={closeResults}
          isLoading={isLoading}
          onAddToReview={onAddToReview}
          onAddAllToReview={onAddMultipleToReview}
          useKorean={wasQueryKorean}
          query={lastSearchQuery}
          onSearch={handleFilterChange}
          filters={searchFilters}
        />
      )}

    {selectedResult && (
      <PaperDetailModal
        result={selectedResult}
        detail={paperDetail}
        isLoading={isLoadingDetail}
        onClose={closePaperDetail}
      />
    )}

    {/* Research Trends Hub Modal */}
    {showResearchTrends && (
      <ResearchTrends
        onClose={() => setShowResearchTrends(false)}
        initialTab={researchTrendsTab}
      />
    )}

    {/* RNA-seq Upload Modal */}
    <RNAseqUploadModal
      isOpen={showRNAseqModal}
      onClose={() => {
        setShowRNAseqModal(false);
        setRnaseqFile(null);
      }}
      onAnalysisStart={(jobId) => {
        setRnaseqJobId(jobId);
        setShowRNAseqModal(false);
        setRnaseqFile(null);
      }}
      initialFile={rnaseqFile}
    />

    {/* Pipeline Progress Modal */}
    {rnaseqJobId && (
      <PipelineProgress
        jobId={rnaseqJobId}
        onClose={() => setRnaseqJobId(null)}
        onViewReport={(jobId) => {
          window.open(`http://localhost:8000/api/rnaseq/report/${jobId}`, '_blank');
        }}
      />
    )}
  </>
  );
};
