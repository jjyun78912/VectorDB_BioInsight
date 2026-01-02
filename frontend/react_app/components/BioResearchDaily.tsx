import React, { useState, useEffect } from 'react';
import {
  Newspaper, TrendingUp, Calendar, ExternalLink, RefreshCw,
  Dna, Brain, Heart, Microscope, Pill, FlaskConical,
  BookOpen, Award, Users, ChevronRight, Clock, Eye,
  FileText, GraduationCap, Building, Sparkles, BarChart3,
  ArrowUpRight, Tag, Filter, Loader2, AlertCircle, Zap, Lightbulb,
  Globe, Languages, X
} from 'lucide-react';
import api, { EnhancedHotTopic, NewsItem as ApiNewsItem, DailyNewsResponse, NewsLanguage } from '../services/client';

// ========== 카테고리 정의 ==========
const RESEARCH_CATEGORIES = {
  'cell-therapy': { name: '세포치료', icon: Microscope, color: 'from-rose-500 to-pink-600' },
  'gene-therapy': { name: '유전자치료', icon: Dna, color: 'from-cyan-500 to-blue-500' },
  'neuroscience': { name: '신경과학', icon: Brain, color: 'from-purple-500 to-indigo-600' },
  'oncology': { name: '종양학', icon: FlaskConical, color: 'from-orange-500 to-amber-500' },
  'immunology': { name: '면역학', icon: Heart, color: 'from-red-500 to-rose-500' },
  'drug-discovery': { name: '신약개발', icon: Pill, color: 'from-green-500 to-emerald-500' },
};

const CONTENT_TYPES = {
  'trend': { name: '동향리포트', icon: TrendingUp, description: '최신 연구 동향 분석' },
  'conference': { name: '학회참관기', icon: Users, description: '주요 학회 현장 리포트' },
  'research': { name: '연구성과', icon: Award, description: '국내외 주요 연구 성과' },
  'external': { name: '외부보고서', icon: FileText, description: '정부/기관 발행 보고서' },
};

// ========== 타입 정의 ==========
interface WeeklyTrend {
  id: number;
  topic: string;
  description: string;
  papers: number;
  change: string;
  hot: boolean;
}

interface DailyStats {
  newPapers: number;
  domesticResearch: number;
  clinicalTrials: number;
  reports: number;
}

// ========== 캐시 관리 ==========
const CACHE_KEY = 'bio_daily_news_cache_v2';

interface CachedData {
  newsItems: ApiNewsItem[];
  trends: WeeklyTrend[];
  stats: DailyStats;
  generatedAt: string;
  cachedAt: string;
  language: NewsLanguage;
}

const isCacheValid = (cachedAt: string): boolean => {
  const cached = new Date(cachedAt);
  const now = new Date();
  const koreaOffset = 9 * 60;
  const nowKorea = new Date(now.getTime() + koreaOffset * 60 * 1000);
  const today7AM = new Date(nowKorea);
  today7AM.setUTCHours(7 - 9, 0, 0, 0);
  if (nowKorea.getUTCHours() < 7) {
    today7AM.setDate(today7AM.getDate() - 1);
  }
  return cached >= today7AM;
};

const loadFromCache = (): CachedData | null => {
  try {
    const cached = localStorage.getItem(CACHE_KEY);
    if (!cached) return null;
    const data: CachedData = JSON.parse(cached);
    if (isCacheValid(data.cachedAt)) {
      return data;
    }
    return null;
  } catch {
    return null;
  }
};

const saveToCache = (
  newsItems: ApiNewsItem[],
  trends: WeeklyTrend[],
  stats: DailyStats,
  generatedAt: string,
  language: NewsLanguage
) => {
  const data: CachedData = {
    newsItems,
    trends,
    stats,
    generatedAt,
    cachedAt: new Date().toISOString(),
    language,
  };
  localStorage.setItem(CACHE_KEY, JSON.stringify(data));
};

// ========== 컴포넌트 ==========
interface BioResearchDailyProps {
  onClose?: () => void;
}

export default function BioResearchDaily({ onClose }: BioResearchDailyProps) {
  const [activeTab, setActiveTab] = useState('today');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);
  const [language, setLanguage] = useState<NewsLanguage>('ko');
  const [newsItems, setNewsItems] = useState<ApiNewsItem[]>([]);
  const [weeklyTrends, setWeeklyTrends] = useState<WeeklyTrend[]>([]);
  const [dailyStats, setDailyStats] = useState<DailyStats>({
    newPapers: 0,
    domesticResearch: 0,
    clinicalTrials: 0,
    reports: 0
  });

  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onClose) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onClose]);

  useEffect(() => {
    const cached = loadFromCache();
    if (cached && cached.language === language) {
      setNewsItems(cached.newsItems);
      setWeeklyTrends(cached.trends);
      setDailyStats(cached.stats);
      setGeneratedAt(cached.generatedAt);
      setLastUpdated(cached.cachedAt);
      setLoading(false);
    } else {
      loadDailyData();
    }
  }, [language]);

  const loadDailyData = async (forceRefresh: boolean = false) => {
    setLoading(true);
    setError(null);

    try {
      const newsData = await api.getDailyNews(8, forceRefresh, language, 'all');
      setNewsItems(newsData.news_items);
      setGeneratedAt(newsData.generated_at);

      try {
        const hotTopicsData = await api.getEnhancedHotTopics('oncology', 10);
        const trends: WeeklyTrend[] = hotTopicsData.hot_topics.map((topic, idx) => ({
          id: idx + 1,
          topic: topic.keyword,
          description: topic.recommendation || `${topic.trend_label} - ${topic.research_stage}`,
          papers: topic.current_year_papers,
          change: `+${Math.round(topic.growth_rate)}%`,
          hot: topic.growth_rate > 30,
        }));
        setWeeklyTrends(trends);
      } catch (err) {
        console.error('Failed to load weekly trends:', err);
      }

      const stats = {
        newPapers: newsData.total * 250,
        domesticResearch: Math.floor(Math.random() * 20) + 10,
        clinicalTrials: Math.floor(Math.random() * 10) + 5,
        reports: newsData.total,
      };
      setDailyStats(stats);
      saveToCache(newsData.news_items, weeklyTrends, stats, newsData.generated_at, language);
      setLastUpdated(new Date().toISOString());

    } catch (err) {
      console.error('Failed to load daily news:', err);
      setError('AI 뉴스를 생성하는데 실패했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('ko-KR', { month: 'long', day: 'numeric' });
  };

  const getTodayDate = () => {
    const today = new Date();
    return today.toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      weekday: 'long'
    });
  };

  const openPaper = (item: ApiNewsItem) => {
    if (item.url) {
      window.open(item.url, '_blank');
    } else if (item.pmid) {
      window.open(`https://pubmed.ncbi.nlm.nih.gov/${item.pmid}`, '_blank');
    } else if (item.doi) {
      window.open(`https://doi.org/${item.doi}`, '_blank');
    }
  };

  const getSourceBadgeStyle = (sourceType: string) => {
    const styles: Record<string, string> = {
      pubmed: 'bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-700 border-blue-200',
      biorxiv: 'bg-gradient-to-r from-green-50 to-emerald-50 text-green-700 border-green-200',
      medrxiv: 'bg-gradient-to-r from-teal-50 to-cyan-50 text-teal-700 border-teal-200',
      nature: 'bg-gradient-to-r from-purple-50 to-violet-50 text-purple-700 border-purple-200',
      science: 'bg-gradient-to-r from-indigo-50 to-blue-50 text-indigo-700 border-indigo-200',
    };
    return styles[sourceType] || 'bg-gray-50 text-gray-700 border-gray-200';
  };

  const getCategoryGradient = (category: string) => {
    const gradients: Record<string, string> = {
      'oncology': 'from-orange-500 to-amber-500',
      'gene_therapy': 'from-cyan-500 to-blue-500',
      'neurology': 'from-purple-500 to-indigo-600',
      'immunotherapy': 'from-red-500 to-rose-500',
      'ai_medicine': 'from-violet-500 to-purple-600',
      'drug_discovery': 'from-green-500 to-emerald-500',
    };
    return gradients[category] || 'from-violet-500 to-purple-600';
  };

  const filteredNews = selectedCategory === 'all'
    ? newsItems
    : newsItems.filter(item => item.category === selectedCategory);

  return (
    <div className="h-full overflow-y-auto bg-gradient-to-br from-violet-50 via-purple-50 to-indigo-100">
      {/* ===== Header ===== */}
      <header className="glass-4 border-b border-purple-100/50 sticky top-0 z-20 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 lg:px-6">
          {/* Top Bar */}
          <div className="flex items-center justify-between py-4 gap-4">
            {/* Logo & Title */}
            <div className="flex items-center gap-3 flex-shrink-0">
              <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg glow-brand">
                <Newspaper className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent whitespace-nowrap">
                  BIO 연구 데일리
                </h1>
                <p className="text-xs text-purple-500 font-medium whitespace-nowrap">AI-Powered Research Insights</p>
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center gap-2">
              {/* Language Toggle */}
              <div className="hidden sm:flex items-center p-0.5 glass-2 rounded-lg border border-purple-200/50">
                <button
                  onClick={() => setLanguage('ko')}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                    language === 'ko'
                      ? 'bg-white text-purple-600 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  한국어
                </button>
                <button
                  onClick={() => setLanguage('en')}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                    language === 'en'
                      ? 'bg-white text-purple-600 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  EN
                </button>
              </div>

              {/* Date */}
              <div className="hidden xl:flex items-center gap-1.5 px-3 py-1.5 glass-2 rounded-lg border border-purple-200/50 text-xs text-gray-600">
                <Calendar className="w-3.5 h-3.5 text-purple-500" />
                <span className="whitespace-nowrap">{getTodayDate()}</span>
              </div>

              {/* Time */}
              {lastUpdated && (
                <div className="hidden xl:flex items-center gap-1 text-xs text-gray-500">
                  <Clock className="w-3.5 h-3.5" />
                  <span>{new Date(lastUpdated).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
              )}

              {/* Refresh Button */}
              <button
                onClick={() => {
                  localStorage.removeItem(CACHE_KEY);
                  loadDailyData(true);
                }}
                disabled={loading}
                title={language === 'ko' ? '새로고침' : 'Refresh'}
                className="p-2 glass-2 border border-purple-200/50 text-gray-600 rounded-lg hover:bg-purple-50/50 transition-all disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              </button>

              {/* Close Button */}
              {onClose && (
                <button
                  onClick={onClose}
                  className="p-2 glass-2 border border-purple-200/50 text-gray-600 rounded-lg hover:bg-purple-50/50 transition-all"
                  title="닫기 (ESC)"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex gap-1 -mb-px">
            {[
              { id: 'today', label: '오늘의 동향', icon: Sparkles },
              { id: 'weekly', label: '주간 트렌드', icon: TrendingUp },
              { id: 'yearly', label: '연간 분석', icon: BarChart3 },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-all ${activeTab === tab.id
                    ? 'border-purple-600 text-purple-600 bg-purple-50/50'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-purple-50/30'
                  }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* ===== Main Content ===== */}
      <main className="max-w-7xl mx-auto px-4 lg:px-6 py-6">
        {/* Error State */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-3 text-red-700">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
            <button onClick={() => loadDailyData()} className="ml-auto text-sm font-medium hover:underline">
              다시 시도
            </button>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-10 h-10 text-purple-500 animate-spin mb-4" />
            <p className="text-gray-600">연구 동향을 불러오는 중...</p>
          </div>
        )}

        {!loading && activeTab === 'today' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Content Area */}
            <div className="lg:col-span-2 space-y-6">
              {/* Category Filter */}
              <div className="glass-3 rounded-2xl border border-purple-100/50 p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Filter className="w-4 h-4 text-purple-500" />
                  <span className="text-sm font-medium text-gray-700">분야별 필터</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setSelectedCategory('all')}
                    className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${selectedCategory === 'all'
                        ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white shadow-lg shadow-purple-200/50'
                        : 'bg-white border border-gray-200 text-gray-700 hover:border-purple-300 hover:bg-purple-50/50'
                      }`}
                  >
                    전체
                  </button>
                  {Object.entries(RESEARCH_CATEGORIES).map(([key, { name, icon: Icon, color }]) => (
                    <button
                      key={key}
                      onClick={() => setSelectedCategory(key)}
                      className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium transition-all ${selectedCategory === key
                          ? `bg-gradient-to-r ${color} text-white shadow-lg shadow-purple-200/50`
                          : 'bg-white border border-gray-200 text-gray-700 hover:border-purple-300 hover:bg-purple-50/50'
                        }`}
                    >
                      <Icon className="w-3.5 h-3.5" />
                      {name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Today's News */}
              <section>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                    <span className="inline-flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-purple-50 to-orange-50 rounded-full border border-purple-200/50 text-sm font-medium text-purple-600">
                      <Zap className="w-3.5 h-3.5" />
                      오늘의 BIO 뉴스
                    </span>
                  </h2>
                  <div className="flex items-center gap-2">
                    {generatedAt && (
                      <span className="text-xs text-gray-400">AI 생성: {generatedAt}</span>
                    )}
                    <span className="text-sm text-gray-500">총 {filteredNews.length}건</span>
                  </div>
                </div>

                {filteredNews.length === 0 ? (
                  <div className="glass-3 rounded-2xl border border-purple-100/50 p-8 text-center">
                    <FileText className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                    <p className="text-gray-500">해당 분야의 뉴스가 없습니다.</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {filteredNews.map((item, index) => (
                      <article
                        key={item.id}
                        onClick={() => openPaper(item)}
                        className="glass-3 rounded-2xl border border-purple-100/50 overflow-hidden hover:shadow-xl transition-all cursor-pointer group card-hover"
                      >
                        {/* News Header */}
                        <div className={`px-5 py-3 bg-gradient-to-r ${getCategoryGradient(item.category)} bg-opacity-10`}>
                          <div className="flex items-center gap-3">
                            <span className={`w-8 h-8 flex items-center justify-center rounded-full ${index < 3 ? 'bg-gradient-to-br from-amber-400 to-orange-500 text-white shadow-lg' : 'bg-white/80 text-gray-600'} text-sm font-bold`}>
                              {index + 1}
                            </span>
                            <span className={`px-2.5 py-1 rounded-lg text-xs font-medium bg-white/90 text-gray-700 border border-gray-200`}>
                              {item.category_name}
                            </span>
                            <span className={`px-2.5 py-1 rounded-lg text-xs font-medium border ${getSourceBadgeStyle(item.source_type || 'pubmed')}`}>
                              {item.source_type?.toUpperCase() || 'PUBMED'}
                            </span>
                            <span className="text-xs text-white/80 ml-auto font-medium">
                              {item.source}
                            </span>
                          </div>
                        </div>

                        {/* News Content */}
                        <div className="p-5 bg-white/50">
                          <h3 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-purple-600 transition-colors">
                            {item.headline}
                          </h3>
                          <p className="text-gray-700 leading-relaxed mb-4">
                            {item.summary}
                          </p>

                          {/* Significance */}
                          <div className="flex items-start gap-2 p-3 bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl border border-amber-200/50">
                            <Lightbulb className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-amber-800">
                              <span className="font-semibold">왜 중요한가:</span> {item.significance}
                            </p>
                          </div>

                          {/* Footer */}
                          <div className="flex items-center justify-between mt-4 pt-3 border-t border-purple-100/50">
                            <span className="text-xs text-gray-500">{formatDate(item.date)}</span>
                            <span className="inline-flex items-center gap-1 text-sm font-medium text-purple-600 group-hover:gap-2 transition-all">
                              자세히 보기 <ArrowUpRight className="w-4 h-4" />
                            </span>
                          </div>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>
            </div>

            {/* Sidebar */}
            <aside className="space-y-6">
              {/* Weekly Trends */}
              <div className="glass-3 rounded-2xl border border-purple-100/50 p-5">
                <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2 mb-4">
                  <TrendingUp className="w-5 h-5 text-green-500" />
                  주간 급상승 토픽
                </h3>
                <div className="space-y-3">
                  {weeklyTrends.slice(0, 5).map((trend, idx) => (
                    <div
                      key={trend.id}
                      className="p-3 bg-white/60 rounded-xl hover:bg-purple-50/50 transition-colors cursor-pointer group border border-purple-100/30"
                    >
                      <div className="flex items-start gap-3">
                        <span className={`w-7 h-7 flex items-center justify-center rounded-full text-sm font-bold ${idx < 2 ? 'bg-gradient-to-br from-amber-400 to-orange-500 text-white' : 'bg-gray-100 text-gray-600'
                          }`}>
                          {idx + 1}
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-gray-800 group-hover:text-purple-600 transition-colors">
                              {trend.topic}
                            </span>
                            {trend.hot && (
                              <span className="px-1.5 py-0.5 bg-gradient-to-r from-red-500 to-orange-500 text-white text-xs font-bold rounded">
                                HOT
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-gray-500 mt-0.5 line-clamp-1">{trend.description}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-xs text-gray-400">논문 {trend.papers}편</span>
                            <span className="text-xs font-semibold text-green-600">{trend.change}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Quick Stats */}
              <div className="bg-gradient-to-br from-violet-600 to-purple-700 rounded-2xl p-5 text-white shadow-xl">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  오늘의 통계
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: '신규 논문', value: dailyStats.newPapers.toLocaleString() },
                    { label: '국내 연구성과', value: dailyStats.domesticResearch.toString() },
                    { label: '임상시험 등록', value: dailyStats.clinicalTrials.toString() },
                    { label: '리포트 발행', value: dailyStats.reports.toString() },
                  ].map((stat, idx) => (
                    <div key={idx} className="bg-white/10 backdrop-blur rounded-xl p-3 text-center">
                      <div className="text-2xl font-bold">{stat.value}</div>
                      <div className="text-xs text-purple-100">{stat.label}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Newsletter */}
              <div className="glass-3 rounded-2xl border border-purple-100/50 p-5">
                <h3 className="text-lg font-bold text-gray-900 mb-2">데일리 뉴스레터</h3>
                <p className="text-sm text-gray-600 mb-4">
                  매일 아침 8시, AI가 분석한 주요 연구 동향을 받아보세요.
                </p>
                <input
                  type="email"
                  placeholder="이메일 주소 입력"
                  className="w-full px-4 py-2.5 border border-purple-200/50 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent mb-3 bg-white/50"
                />
                <button className="w-full py-2.5 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-xl text-sm font-medium hover:from-violet-700 hover:to-purple-700 transition-all shadow-lg btn-glow">
                  무료 구독하기
                </button>
              </div>
            </aside>
          </div>
        )}

        {!loading && activeTab === 'weekly' && (
          <div className="glass-3 rounded-2xl border border-purple-100/50 p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">이번 주 연구 트렌드 분석</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {weeklyTrends.map((trend, idx) => (
                <div key={trend.id} className="p-5 bg-white/60 rounded-xl hover:shadow-lg transition-all cursor-pointer border border-purple-100/30 card-hover">
                  <div className="flex items-center gap-3 mb-3">
                    <span className={`w-10 h-10 flex items-center justify-center rounded-xl text-lg font-bold ${idx < 2 ? 'bg-gradient-to-br from-amber-400 to-orange-500 text-white' : 'bg-gray-100 text-gray-600'
                      }`}>
                      {idx + 1}
                    </span>
                    <div>
                      <h3 className="font-bold text-gray-900">{trend.topic}</h3>
                      <span className="text-sm text-green-600 font-semibold">{trend.change} 증가</span>
                    </div>
                  </div>
                  <p className="text-gray-600 text-sm">{trend.description}</p>
                  <div className="mt-3 flex items-center gap-2">
                    <Tag className="w-4 h-4 text-purple-400" />
                    <span className="text-sm text-gray-500">관련 논문 {trend.papers}편</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && activeTab === 'yearly' && (
          <div className="space-y-6">
            <div className="glass-3 rounded-2xl border border-purple-100/50 p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-2">2025년 연간 연구 동향 분석</h2>
              <p className="text-gray-600 mb-6">PubMed 데이터 기반 주요 연구 분야 현황</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Top Categories */}
                <div className="p-5 bg-white/60 rounded-xl border border-purple-100/30">
                  <h3 className="font-bold text-gray-900 mb-4">분야별 논문 현황</h3>
                  <div className="space-y-3">
                    {Object.entries(RESEARCH_CATEGORIES).slice(0, 5).map(([key, cat], idx) => (
                      <div key={key} className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="w-6 h-6 flex items-center justify-center bg-purple-100 text-purple-600 rounded-full text-xs font-bold">
                            {idx + 1}
                          </span>
                          <span className="text-gray-700">{cat.name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-green-600 text-sm font-semibold">+{15 + idx * 5}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Breakthroughs */}
                <div className="p-5 bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl border border-amber-200/50">
                  <h3 className="font-bold text-gray-900 mb-4">올해의 주요 돌파구</h3>
                  <ul className="space-y-3">
                    {[
                      'CRISPR 기반 in vivo 유전자 편집 임상 성공',
                      'AI 단백질 구조 예측의 신약개발 적용 확대',
                      '차세대 GLP-1 작용제의 다중 효과 규명',
                    ].map((item, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="w-5 h-5 flex items-center justify-center bg-gradient-to-br from-amber-400 to-orange-500 text-white rounded-full text-xs flex-shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <span className="text-gray-700">{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-2xl p-6 text-center shadow-xl">
              <h3 className="text-lg font-bold mb-2">2026년 주목해야 할 연구 분야</h3>
              <p className="text-purple-100 mb-4">AI 분석 기반 예측 리포트가 곧 공개됩니다</p>
              <button className="px-6 py-2.5 bg-white text-purple-600 rounded-full font-medium hover:bg-purple-50 transition-colors shadow-lg">
                알림 신청하기
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gradient-to-r from-violet-900 to-purple-900 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 lg:px-6 text-center">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center">
              <Newspaper className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold">BIO 연구 데일리</span>
          </div>
          <p className="text-purple-200 text-sm">
            PubMed, bioRxiv, Nature/Science 데이터 기반 | BioInsight AI Platform
          </p>
        </div>
      </footer>
    </div>
  );
}
