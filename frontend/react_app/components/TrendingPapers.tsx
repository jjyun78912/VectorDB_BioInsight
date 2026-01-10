import React, { useState, useEffect } from 'react';
import {
  TrendingUp, Clock, Users, ExternalLink, ChevronRight,
  Loader2, RefreshCw, Flame, BookOpen,
  FileText, Quote, AlertCircle, Sparkles
} from 'lucide-react';
import api, { TrendingResponse, CrawlerPaper } from '../services/client';
import { useLanguage } from '../contexts/LanguageContext';

// Category configuration with colors (names will be translated)
const getCategoryConfig = (t: any) => [
  { id: 'oncology', name: t.catOncology, color: 'from-violet-500 to-purple-600' },
  { id: 'immunotherapy', name: t.catImmunotherapy, color: 'from-red-500 to-orange-500' },
  { id: 'gene_therapy', name: t.catGeneTherapy, color: 'from-cyan-500 to-blue-500' },
  { id: 'neurology', name: t.catNeurology, color: 'from-pink-500 to-rose-500' },
  { id: 'cardiology', name: t.catCardiology, color: 'from-red-500 to-pink-500' },
  { id: 'infectious_disease', name: t.catInfectiousDisease, color: 'from-green-500 to-emerald-500' },
  { id: 'metabolic', name: t.catMetabolic, color: 'from-amber-500 to-orange-500' },
  { id: 'rare_disease', name: t.catRareDisease, color: 'from-purple-500 to-indigo-500' },
];

export const TrendingPapers: React.FC = () => {
  const { t } = useLanguage();
  const CATEGORIES = getCategoryConfig(t);
  const [selectedCategory, setSelectedCategory] = useState(CATEGORIES[0].id);
  const [data, setData] = useState<TrendingResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadTrendingPapers(selectedCategory);
  }, [selectedCategory]);

  const loadTrendingPapers = async (category: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await api.getTrendingPapers(category, 10);
      setData(response);
    } catch (err) {
      console.error('Failed to load trending papers:', err);
      setError('Failed to fetch trending papers. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const openPaper = (paper: CrawlerPaper) => {
    const url = paper.url || (paper.doi ? `https://doi.org/${paper.doi}` : null);
    if (url) {
      window.open(url, '_blank');
    }
  };

  const formatCitations = (count: number): string => {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}k`;
    }
    return count.toString();
  };

  const getCurrentCategory = (categoryId: string) => {
    return CATEGORIES.find(c => c.id === categoryId) || CATEGORIES[0];
  };

  const currentCategory = getCurrentCategory(selectedCategory);

  return (
    <section className="relative py-16 pb-24 bg-white isolate">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-purple-50/50 via-orange-50/30 to-white pointer-events-none" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div className="space-y-3">
            <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-purple-50 to-orange-50 rounded-full border border-purple-200/50 text-sm font-medium text-purple-600">
              <Sparkles className="w-3.5 h-3.5" />
              {t.realTimeFromPubmed}
            </span>
            <h2 className="text-2xl md:text-3xl font-bold text-gray-900">
              {t.trendingResearchPapers}
            </h2>
            <p className="text-gray-600">
              {t.trendingPapersSubtitle} <span className="font-medium text-purple-600">{t.multipleResearchAreas}</span>
            </p>
          </div>

          <div className="flex items-center gap-3">
            {data?.cached && (
              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                {t.cached}
              </span>
            )}
            <button
              onClick={() => loadTrendingPapers(selectedCategory)}
              disabled={isLoading}
              className="inline-flex items-center gap-2 px-4 py-2 glass-3 border border-purple-200/50 rounded-lg text-sm font-medium text-gray-700 hover:bg-purple-50/50 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              {t.refresh}
            </button>
          </div>
        </div>

        {/* Category Tabs */}
        <div className="flex flex-wrap gap-2 mb-8">
          {CATEGORIES.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`
                inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all
                ${selectedCategory === category.id
                  ? `bg-gradient-to-r ${category.color} text-white shadow-lg shadow-purple-200/50`
                  : 'bg-white border border-gray-200 text-gray-700 hover:border-purple-300 hover:bg-purple-50/50'
                }
              `}
            >
              {category.name}
            </button>
          ))}
        </div>

        {/* Error State */}
        {error && (
          <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl mb-6">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
            <p className="text-red-700">{error}</p>
            <button
              onClick={() => loadTrendingPapers(selectedCategory)}
              className="ml-auto text-sm font-medium text-red-600 hover:text-red-700"
            >
              Retry
            </button>
          </div>
        )}

        {/* Loading State */}
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-16">
            <Loader2 className="w-10 h-10 text-purple-500 animate-spin mb-4" />
            <p className="text-gray-600">{t.fetchingTrending}</p>
          </div>
        ) : !data || data.papers.length === 0 ? (
          <div className="text-center py-16">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">{t.noTrendingPapers} {currentCategory.name}. {t.tryRefreshing}</p>
          </div>
        ) : (
          /* Papers Grid */
          <div className="space-y-4">
            {/* Category Header Banner */}
            <div className={`p-4 rounded-xl bg-gradient-to-r ${currentCategory.color} text-white mb-6`}>
              <div className="flex items-center gap-3">
                <div>
                  <h3 className="font-bold text-lg">{currentCategory.name}</h3>
                  <p className="text-sm opacity-90">{data.papers.length} {t.papers}</p>
                </div>
              </div>
            </div>

            {/* Paper Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {data.papers.map((paper, idx) => (
                <div
                  key={paper.id}
                  className="group p-5 bg-white rounded-xl border border-gray-200 hover:border-purple-300 hover:shadow-lg transition-all"
                >
                  <div className="flex items-start gap-4">
                    {/* Rank Badge */}
                    <div className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center bg-gradient-to-r ${currentCategory.color} text-white font-bold`}>
                      #{idx + 1}
                    </div>

                    <div className="flex-1 min-w-0">
                      {/* Title */}
                      <h4
                        className="font-semibold text-gray-900 mb-2 group-hover:text-purple-700 cursor-pointer line-clamp-2"
                        onClick={() => openPaper(paper)}
                      >
                        {paper.title}
                      </h4>

                      {/* Meta Info */}
                      <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500 mb-3">
                        <span className="flex items-center gap-1">
                          <Users className="w-3 h-3" />
                          {paper.authors.slice(0, 2).join(', ')}
                          {paper.authors.length > 2 && ' et al.'}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {paper.year}
                        </span>
                        {paper.journal && (
                          <span className="flex items-center gap-1">
                            <BookOpen className="w-3 h-3" />
                            <span className="truncate max-w-[120px]">{paper.journal}</span>
                          </span>
                        )}
                        {paper.citation_count > 0 && (
                          <span className="flex items-center gap-1 text-amber-600">
                            <Quote className="w-3 h-3" />
                            {formatCitations(paper.citation_count)} {t.citations}
                          </span>
                        )}
                      </div>

                      {/* Trend Metrics */}
                      {paper.trend_score > 0 && (
                        <div className="flex flex-wrap items-center gap-2 mb-3">
                          <div className="flex items-center gap-1 px-2 py-1 bg-orange-50 rounded-full">
                            <Flame className="w-3 h-3 text-orange-500" />
                            <span className="text-xs font-medium text-orange-600">
                              {t.trend}: {paper.trend_score.toFixed(1)}
                            </span>
                          </div>
                          {paper.citation_velocity && paper.citation_velocity > 0 && (
                            <div className="flex items-center gap-1 px-2 py-1 bg-green-50 rounded-full">
                              <TrendingUp className="w-3 h-3 text-green-500" />
                              <span className="text-xs font-medium text-green-600">
                                {t.velocity}: {paper.citation_velocity.toFixed(2)}x
                              </span>
                            </div>
                          )}
                          {paper.publication_surge && paper.publication_surge > 1 && (
                            <div className="flex items-center gap-1 px-2 py-1 bg-purple-50 rounded-full">
                              <Flame className="w-3 h-3 text-purple-500" />
                              <span className="text-xs font-medium text-purple-600">
                                {t.surge}: {paper.publication_surge.toFixed(2)}x
                              </span>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Keywords */}
                      {paper.keywords.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mb-3">
                          {paper.keywords.slice(0, 4).map((kw, i) => (
                            <span
                              key={i}
                              className="px-2 py-0.5 text-xs bg-purple-100 text-purple-700 rounded-full"
                            >
                              {kw}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Abstract Preview */}
                      {paper.abstract && (
                        <p className="text-sm text-gray-600 line-clamp-2 mb-3">{paper.abstract}</p>
                      )}

                      {/* Actions */}
                      <div className="flex items-center gap-4">
                        <button
                          className="text-sm font-medium text-purple-600 hover:text-purple-700 flex items-center gap-1"
                          onClick={() => openPaper(paper)}
                        >
                          {t.viewPaper}
                          <ExternalLink className="w-3 h-3" />
                        </button>
                        {paper.pmid && (
                          <span className="text-xs text-gray-400">PMID: {paper.pmid}</span>
                        )}
                        {paper.doi && (
                          <span className="text-xs text-gray-400 truncate max-w-[150px]">
                            DOI: {paper.doi}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Arrow */}
                    <ChevronRight className="w-5 h-5 text-gray-300 group-hover:text-purple-400 flex-shrink-0" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-8 pb-8">
          <p className="text-sm text-gray-500">
            {t.dataSourcedFrom} • {t.updatedRealTime}
          </p>
        </div>
      </div>
    </section>
  );
};
