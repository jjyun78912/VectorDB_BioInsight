import React, { useState, useEffect } from 'react';
import {
  TrendingUp, Clock, Users, ExternalLink, ChevronRight,
  Loader2, RefreshCw, Flame, Star, BookOpen, Globe,
  FileText, Quote, AlertCircle
} from 'lucide-react';
import api, { CrawlerPaper } from '../services/client';

// Category display names
const CATEGORY_LABELS: Record<string, string> = {
  oncology: 'Oncology',
  immunotherapy: 'Immunotherapy',
  gene_therapy: 'Gene Therapy',
  neurology: 'Neurology',
  infectious_disease: 'Infectious Disease',
  ai_medicine: 'AI in Medicine',
  genomics: 'Genomics',
  drug_discovery: 'Drug Discovery',
};

// Category colors
const CATEGORY_COLORS: Record<string, string> = {
  oncology: 'from-red-500 to-orange-500',
  immunotherapy: 'from-purple-500 to-pink-500',
  gene_therapy: 'from-green-500 to-teal-500',
  neurology: 'from-blue-500 to-indigo-500',
  infectious_disease: 'from-yellow-500 to-orange-500',
  ai_medicine: 'from-cyan-500 to-blue-500',
  genomics: 'from-violet-500 to-purple-500',
  drug_discovery: 'from-emerald-500 to-green-500',
};

export const TrendingPapers: React.FC = () => {
  const [papers, setPapers] = useState<CrawlerPaper[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>('oncology');
  const [categories, setCategories] = useState<string[]>(['oncology']);
  const [isCached, setIsCached] = useState(false);

  useEffect(() => {
    loadCategories();
  }, []);

  useEffect(() => {
    loadTrendingPapers();
  }, [activeCategory]);

  const loadCategories = async () => {
    try {
      const response = await api.getTrendingCategories();
      setCategories(response.categories);
    } catch (err) {
      console.error('Failed to load categories:', err);
      // Use default categories
      setCategories(Object.keys(CATEGORY_LABELS));
    }
  };

  const loadTrendingPapers = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await api.getTrendingPapers(activeCategory, 5);
      setPapers(response.papers);
      setIsCached(response.cached);
    } catch (err) {
      console.error('Failed to load trending papers:', err);
      setError('Failed to fetch trending papers. Please try again.');
      setPapers([]);
    } finally {
      setIsLoading(false);
    }
  };

  const getTrendIcon = (score: number) => {
    if (score >= 80) return <Flame className="w-4 h-4 text-orange-500" />;
    if (score >= 60) return <TrendingUp className="w-4 h-4 text-red-500" />;
    return <Star className="w-4 h-4 text-yellow-500" />;
  };

  const formatCitations = (count: number): string => {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}k`;
    }
    return count.toString();
  };

  const openPaper = (paper: CrawlerPaper) => {
    const url = paper.url || (paper.doi ? `https://doi.org/${paper.doi}` : null);
    if (url) {
      window.open(url, '_blank');
    }
  };

  return (
    <section className="relative py-16 pb-24 bg-white isolate">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-purple-50/50 via-orange-50/30 to-white pointer-events-none" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div className="space-y-3">
            <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-orange-50 rounded-full border border-orange-200/50 text-sm font-medium text-orange-600">
              <Globe className="w-3.5 h-3.5" />
              Live from PubMed
            </span>
            <h2 className="text-2xl md:text-3xl font-bold text-gray-900">
              Trending Research Papers
            </h2>
            <p className="text-gray-600">
              Real-time trending papers from PubMed and Semantic Scholar
            </p>
          </div>

          <div className="flex items-center gap-3">
            {isCached && (
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                Cached
              </span>
            )}
            <button
              onClick={loadTrendingPapers}
              disabled={isLoading}
              className="inline-flex items-center gap-2 px-4 py-2 glass-3 border border-purple-200/50 rounded-lg text-sm font-medium text-gray-700 hover:bg-purple-50/50 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* Category Tabs */}
        <div className="flex flex-wrap gap-2 mb-6 pb-4 border-b border-purple-100/50">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                activeCategory === cat
                  ? `bg-gradient-to-r ${CATEGORY_COLORS[cat] || 'from-orange-500 to-red-500'} text-white shadow-lg`
                  : 'glass-2 border border-purple-200/50 text-gray-600 hover:bg-purple-50/50'
              }`}
            >
              {CATEGORY_LABELS[cat] || cat}
            </button>
          ))}
        </div>

        {/* Error State */}
        {error && (
          <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl mb-6">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
            <p className="text-red-700">{error}</p>
            <button
              onClick={loadTrendingPapers}
              className="ml-auto text-sm font-medium text-red-600 hover:text-red-700"
            >
              Retry
            </button>
          </div>
        )}

        {/* Loading State */}
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-16">
            <Loader2 className="w-10 h-10 text-orange-500 animate-spin mb-4" />
            <p className="text-gray-600">Fetching trending papers from PubMed...</p>
          </div>
        ) : papers.length === 0 ? (
          <div className="text-center py-16">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No trending papers found for this category.</p>
          </div>
        ) : (
          /* Papers List */
          <div className="grid gap-4">
            {papers.map((paper, idx) => (
              <div
                key={paper.id}
                className="group glass-3 rounded-2xl border border-purple-100/50 p-5 card-hover animate-appear"
                style={{ animationDelay: `${idx * 100}ms` }}
              >
                <div className="flex items-start gap-4">
                  {/* Rank/Trend Badge */}
                  <div className="flex-shrink-0 w-14 h-14 bg-gradient-to-br from-orange-100 to-red-100 rounded-xl flex flex-col items-center justify-center">
                    {getTrendIcon(paper.trend_score)}
                    <span className="text-xs font-bold text-orange-600 mt-1">#{idx + 1}</span>
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <h3
                          className="font-semibold text-gray-900 mb-1 line-clamp-2 group-hover:text-purple-700 transition-colors cursor-pointer"
                          onClick={() => openPaper(paper)}
                        >
                          {paper.title}
                        </h3>
                        <div className="flex flex-wrap items-center gap-3 text-sm text-gray-500 mb-2">
                          <span className="flex items-center gap-1">
                            <Users className="w-3.5 h-3.5" />
                            {paper.authors.slice(0, 2).join(', ')}
                            {paper.authors.length > 2 && ' et al.'}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3.5 h-3.5" />
                            {paper.year}
                          </span>
                          {paper.journal && (
                            <span className="flex items-center gap-1">
                              <BookOpen className="w-3.5 h-3.5" />
                              <span className="truncate max-w-[200px]">{paper.journal}</span>
                            </span>
                          )}
                        </div>
                        {paper.abstract && (
                          <p className="text-sm text-gray-600 line-clamp-2">{paper.abstract}</p>
                        )}
                      </div>

                      {/* Stats */}
                      <div className="flex-shrink-0 flex flex-col items-end gap-2">
                        <span className={`text-xs font-medium text-white bg-gradient-to-r ${CATEGORY_COLORS[activeCategory] || 'from-orange-500 to-red-500'} px-2.5 py-1 rounded-full`}>
                          {CATEGORY_LABELS[activeCategory]}
                        </span>
                        {paper.citation_count > 0 && (
                          <span className="flex items-center gap-1 text-xs text-gray-500">
                            <Quote className="w-3 h-3" />
                            {formatCitations(paper.citation_count)} citations
                          </span>
                        )}
                        <div className="flex items-center gap-1.5">
                          <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className={`h-full bg-gradient-to-r ${CATEGORY_COLORS[activeCategory] || 'from-orange-400 to-red-500'} rounded-full transition-all`}
                              style={{ width: `${Math.min(100, paper.trend_score)}%` }}
                            />
                          </div>
                          <span className="text-xs font-medium text-orange-600">
                            {paper.trend_score.toFixed(0)}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-4 mt-3 pt-3 border-t border-purple-50/50">
                      <button
                        className="text-sm font-medium text-purple-600 hover:text-purple-700 flex items-center gap-1 transition-colors"
                        onClick={() => openPaper(paper)}
                      >
                        View Paper
                        <ExternalLink className="w-3.5 h-3.5" />
                      </button>
                      {paper.doi && (
                        <button
                          className="text-sm text-gray-500 hover:text-gray-700 transition-colors"
                          onClick={() => navigator.clipboard.writeText(`https://doi.org/${paper.doi}`)}
                        >
                          Copy DOI
                        </button>
                      )}
                      {paper.pmid && (
                        <span className="text-xs text-gray-400">
                          PMID: {paper.pmid}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-8 pb-8">
          <p className="text-sm text-gray-500">
            Data sourced from PubMed E-utilities & Semantic Scholar APIs
          </p>
        </div>
      </div>
    </section>
  );
};
