import React, { useState, useEffect, useRef } from 'react';
import {
  X, Newspaper, TrendingUp, TrendingDown, Minus,
  ChevronRight, ExternalLink, Calendar, BarChart3,
  Sparkles, RefreshCw, ChevronDown, ChevronUp, ArrowLeft,
  FileText, Download
} from 'lucide-react';

interface TrendItem {
  keyword: string;
  count: number;
  change_label: string;
  trend_indicator: string;
  category?: string;
  why_hot?: string;
  is_predefined: boolean;
  is_emerging: boolean;
}

interface ArticleItem {
  title: string;
  summary?: string;
  content?: string;
  hook?: string;
  source?: string;
  pmid?: string;
  doi?: string;
  journal?: string;
  source_journal?: string;
  pub_date?: string;
  insight?: string;
}

interface BriefingData {
  issue_number: number;
  date: string;
  total_papers_analyzed: number;
  trends: TrendItem[];
  articles: { [key: string]: ArticleItem[] };
  articles_by_trend?: { [key: string]: ArticleItem[] };
  editor_comment: string;
}

interface DailyBriefingProps {
  isOpen: boolean;
  onClose: () => void;
}

type ViewMode = 'topics' | 'newsletter';

export const DailyBriefing: React.FC<DailyBriefingProps> = ({ isOpen, onClose }) => {
  const [briefing, setBriefing] = useState<BriefingData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedTrend, setExpandedTrend] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('newsletter');
  const [newsletterHtml, setNewsletterHtml] = useState<string | null>(null);
  const [newsletterInfo, setNewsletterInfo] = useState<{date: string; issue_number: number; version: string} | null>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    if (isOpen) {
      fetchBriefing();
      fetchNewsletterHtml();
    }
  }, [isOpen]);

  const fetchBriefing = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/briefing/latest');
      if (!response.ok) {
        throw new Error('Failed to fetch briefing');
      }
      const data = await response.json();
      setBriefing(data);
      // Auto-expand first trend
      if (data.trends && data.trends.length > 0) {
        setExpandedTrend(data.trends[0].keyword);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load briefing');
    } finally {
      setLoading(false);
    }
  };

  const fetchNewsletterHtml = async () => {
    try {
      const response = await fetch('/api/briefing/html/latest');
      if (response.ok) {
        const data = await response.json();
        setNewsletterHtml(data.html);
        setNewsletterInfo({
          date: data.date,
          issue_number: data.issue_number,
          version: data.version
        });
      }
    } catch (err) {
      console.error('Failed to fetch newsletter HTML:', err);
    }
  };

  const downloadNewsletter = () => {
    if (!newsletterHtml) return;

    // Create a blob and download
    const blob = new Blob([newsletterHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `BIO_Daily_Briefing_${newsletterInfo?.issue_number || 'latest'}_${newsletterInfo?.date || 'today'}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const openInNewTab = () => {
    if (!newsletterHtml) return;
    const newWindow = window.open('', '_blank');
    if (newWindow) {
      newWindow.document.write(newsletterHtml);
      newWindow.document.close();
    }
  };

  const getTrendIcon = (indicator: string) => {
    if (indicator.includes('🔥') || indicator.includes('⬆️')) {
      return <TrendingUp className="w-4 h-4 text-green-500" />;
    } else if (indicator.includes('⬇️')) {
      return <TrendingDown className="w-4 h-4 text-red-500" />;
    } else if (indicator.includes('🆕')) {
      return <Sparkles className="w-4 h-4 text-yellow-500" />;
    }
    return <Minus className="w-4 h-4 text-gray-400" />;
  };

  const formatDate = (dateStr: string) => {
    if (dateStr.includes('년')) {
      return dateStr;
    }
    try {
      const year = dateStr.slice(0, 4);
      const month = dateStr.slice(4, 6);
      const day = dateStr.slice(6, 8);
      return `${year}년 ${month}월 ${day}일`;
    } catch {
      return dateStr;
    }
  };

  const getArticles = (keyword: string): ArticleItem[] => {
    if (!briefing) return [];
    return briefing.articles[keyword] || briefing.articles_by_trend?.[keyword] || [];
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gradient-to-r from-violet-600 to-purple-600 px-6 py-4 shadow-lg">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-white" />
            </button>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                <Newspaper className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">BIO Daily Briefing</h1>
                {briefing && (
                  <p className="text-sm text-white/70">
                    Issue #{briefing.issue_number} | {formatDate(briefing.date)}
                  </p>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* View Mode Toggle */}
            <div className="flex items-center bg-white/10 rounded-lg p-1">
              <button
                onClick={() => setViewMode('newsletter')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  viewMode === 'newsletter'
                    ? 'bg-white text-purple-600'
                    : 'text-white/70 hover:text-white'
                }`}
              >
                <FileText className="w-4 h-4" />
                Newsletter
              </button>
              <button
                onClick={() => setViewMode('topics')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  viewMode === 'topics'
                    ? 'bg-white text-purple-600'
                    : 'text-white/70 hover:text-white'
                }`}
              >
                <TrendingUp className="w-4 h-4" />
                Topics
              </button>
            </div>

            {viewMode === 'newsletter' && newsletterHtml && (
              <>
                <button
                  onClick={openInNewTab}
                  className="flex items-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors text-white text-sm"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open
                </button>
                <button
                  onClick={downloadNewsletter}
                  className="flex items-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors text-white text-sm"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
              </>
            )}

            <button
              onClick={() => { fetchBriefing(); fetchNewsletterHtml(); }}
              disabled={loading}
              className="flex items-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors text-white text-sm"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-white" />
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="overflow-y-auto h-[calc(100vh-72px)]">
        <div className="max-w-6xl mx-auto px-6 py-8">
          {loading && (
            <div className="flex flex-col items-center justify-center py-32">
              <RefreshCw className="w-10 h-10 text-purple-400 animate-spin mb-4" />
              <p className="text-gray-400 text-lg">Loading briefing...</p>
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center justify-center py-32">
              <p className="text-red-400 text-lg mb-4">{error}</p>
              <button
                onClick={fetchBriefing}
                className="px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
              >
                Try Again
              </button>
            </div>
          )}

          {/* Newsletter View */}
          {viewMode === 'newsletter' && !loading && (
            <div className="h-full">
              {newsletterHtml ? (
                <div className="bg-white rounded-2xl overflow-hidden shadow-2xl">
                  <iframe
                    ref={iframeRef}
                    srcDoc={newsletterHtml}
                    className="w-full min-h-[calc(100vh-180px)]"
                    style={{ border: 'none' }}
                    title="BIO Daily Briefing Newsletter"
                  />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-32 bg-slate-800/50 rounded-2xl border border-slate-700/50">
                  <FileText className="w-16 h-16 text-gray-600 mb-4" />
                  <p className="text-gray-400 text-lg mb-2">Newsletter not available</p>
                  <p className="text-gray-500 text-sm">Generate a newsletter first or switch to Topics view</p>
                </div>
              )}
            </div>
          )}

          {/* Topics View */}
          {viewMode === 'topics' && briefing && !loading && (
            <div className="space-y-8">
              {/* Stats Bar */}
              <div className="flex flex-wrap items-center gap-6 p-6 bg-slate-800/50 rounded-2xl border border-slate-700/50">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <BarChart3 className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Papers Analyzed</p>
                    <p className="text-2xl font-bold text-white">{briefing.total_papers_analyzed}</p>
                  </div>
                </div>
                <div className="w-px h-12 bg-slate-600 hidden sm:block" />
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <Calendar className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Time Range</p>
                    <p className="text-lg font-semibold text-white">Last 48 hours</p>
                  </div>
                </div>
                <div className="w-px h-12 bg-slate-600 hidden sm:block" />
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                    <TrendingUp className="w-5 h-5 text-green-400" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Hot Topics</p>
                    <p className="text-lg font-semibold text-white">{briefing.trends.filter(t => t.is_predefined).length} tracked</p>
                  </div>
                </div>
              </div>

              {/* Editor Comment */}
              {briefing.editor_comment && (
                <div className="p-6 bg-gradient-to-r from-purple-900/40 to-violet-900/40 rounded-2xl border border-purple-500/30">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 bg-purple-500/30 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Sparkles className="w-5 h-5 text-purple-300" />
                    </div>
                    <div>
                      <h2 className="font-bold text-purple-300 text-lg mb-3">Editor's Analysis</h2>
                      <div className="text-gray-300 leading-relaxed whitespace-pre-line prose prose-invert prose-sm max-w-none">
                        {briefing.editor_comment.replace(/##\s*/g, '').replace(/\*\*/g, '')}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="grid lg:grid-cols-3 gap-8">
                {/* Left: Hot Topics List */}
                <div className="lg:col-span-1">
                  <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-green-400" />
                    Hot Topics
                  </h2>

                  <div className="space-y-2">
                    {briefing.trends
                      .filter(t => t.is_predefined && !t.is_emerging)
                      .map((trend, idx) => (
                        <button
                          key={trend.keyword}
                          onClick={() => setExpandedTrend(expandedTrend === trend.keyword ? null : trend.keyword)}
                          className={`w-full p-4 rounded-xl border transition-all text-left ${
                            expandedTrend === trend.keyword
                              ? 'bg-purple-600/30 border-purple-500/50'
                              : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-700/50'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <span className={`w-7 h-7 rounded-lg flex items-center justify-center text-sm font-bold ${
                                expandedTrend === trend.keyword
                                  ? 'bg-purple-500 text-white'
                                  : 'bg-slate-700 text-gray-300'
                              }`}>
                                {idx + 1}
                              </span>
                              <div>
                                <h4 className="font-semibold text-white">{trend.keyword}</h4>
                                {trend.category && (
                                  <span className="text-xs text-gray-500">{trend.category}</span>
                                )}
                              </div>
                            </div>
                            <div className="text-right">
                              <span className="text-gray-400 text-sm">{trend.count}</span>
                              <ChevronRight className={`w-4 h-4 text-gray-500 transition-transform ${
                                expandedTrend === trend.keyword ? 'rotate-90' : ''
                              }`} />
                            </div>
                          </div>
                        </button>
                      ))}
                  </div>

                  {/* Emerging Trends */}
                  {briefing.trends.filter(t => t.is_emerging).length > 0 && (
                    <div className="mt-6">
                      <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-yellow-400" />
                        Emerging
                      </h3>
                      <div className="space-y-2">
                        {briefing.trends
                          .filter(t => t.is_emerging)
                          .map((trend) => (
                            <button
                              key={trend.keyword}
                              onClick={() => setExpandedTrend(expandedTrend === trend.keyword ? null : trend.keyword)}
                              className={`w-full p-3 rounded-xl border transition-all text-left ${
                                expandedTrend === trend.keyword
                                  ? 'bg-yellow-600/20 border-yellow-500/50'
                                  : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-700/50'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <Sparkles className="w-4 h-4 text-yellow-400" />
                                  <span className="font-medium text-white">{trend.keyword}</span>
                                </div>
                                <span className="text-yellow-400 text-sm">{trend.count}</span>
                              </div>
                            </button>
                          ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Right: Articles Detail */}
                <div className="lg:col-span-2">
                  {expandedTrend ? (
                    <div>
                      <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-bold text-white">{expandedTrend}</h2>
                        {briefing.trends.find(t => t.keyword === expandedTrend)?.why_hot && (
                          <span className="text-sm text-purple-300 bg-purple-900/30 px-3 py-1 rounded-full">
                            {briefing.trends.find(t => t.keyword === expandedTrend)?.why_hot}
                          </span>
                        )}
                      </div>

                      <div className="space-y-4">
                        {getArticles(expandedTrend).map((article, aIdx) => (
                          <div
                            key={aIdx}
                            className="p-6 bg-slate-800/70 rounded-2xl border border-slate-700/50 hover:border-purple-500/30 transition-colors"
                          >
                            {article.hook && (
                              <p className="text-purple-400 text-sm mb-2">{article.hook}</p>
                            )}
                            <h3 className="text-lg font-semibold text-white mb-3">{article.title}</h3>
                            <p className="text-gray-400 leading-relaxed mb-4">
                              {article.content || article.summary}
                            </p>
                            {article.insight && (
                              <p className="text-sm text-green-400 bg-green-900/20 px-4 py-2 rounded-lg mb-4">
                                💡 {article.insight}
                              </p>
                            )}
                            <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500">
                              {(article.source_journal || article.journal) && (
                                <span>{article.source_journal || article.journal}</span>
                              )}
                              {article.pub_date && <span>{article.pub_date}</span>}
                              {article.pmid && (
                                <a
                                  href={`https://pubmed.ncbi.nlm.nih.gov/${article.pmid}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="flex items-center gap-1 text-purple-400 hover:text-purple-300"
                                >
                                  PMID: {article.pmid}
                                  <ExternalLink className="w-3 h-3" />
                                </a>
                              )}
                              {article.doi && (
                                <a
                                  href={`https://doi.org/${article.doi}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="flex items-center gap-1 text-blue-400 hover:text-blue-300"
                                >
                                  DOI
                                  <ExternalLink className="w-3 h-3" />
                                </a>
                              )}
                            </div>
                          </div>
                        ))}

                        {getArticles(expandedTrend).length === 0 && (
                          <div className="p-8 bg-slate-800/50 rounded-2xl border border-slate-700/50 text-center">
                            <p className="text-gray-500">No articles available for this topic</p>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-64 bg-slate-800/30 rounded-2xl border border-slate-700/50">
                      <p className="text-gray-500">Select a topic to view articles</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DailyBriefing;
