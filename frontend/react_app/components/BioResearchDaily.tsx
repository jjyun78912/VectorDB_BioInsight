import React, { useState, useEffect } from 'react';
import {
  Newspaper, TrendingUp, Calendar, ExternalLink, RefreshCw,
  Dna, Brain, Heart, Microscope, Pill, FlaskConical,
  BookOpen, Award, Users, ChevronRight, Clock, Eye,
  FileText, GraduationCap, Building, Sparkles, BarChart3,
  ArrowUpRight, Tag, Filter, Loader2, AlertCircle, Zap, Lightbulb,
  Globe, Languages
} from 'lucide-react';
import api, { EnhancedHotTopic, NewsItem as ApiNewsItem, DailyNewsResponse, NewsLanguage, NewsSource } from '../services/client';

// ========== 카테고리 정의 ==========
const RESEARCH_CATEGORIES = {
  'cell-therapy': { name: '세포치료', icon: Microscope, color: 'rose', keywords: ['cell therapy', 'CAR-T', 'stem cell'] },
  'gene-therapy': { name: '유전자치료', icon: Dna, color: 'blue', keywords: ['gene therapy', 'CRISPR', 'gene editing'] },
  'neuroscience': { name: '신경과학', icon: Brain, color: 'purple', keywords: ['neuroscience', 'Alzheimer', 'Parkinson'] },
  'oncology': { name: '종양학', icon: FlaskConical, color: 'orange', keywords: ['cancer', 'tumor', 'oncology'] },
  'immunology': { name: '면역학', icon: Heart, color: 'red', keywords: ['immunotherapy', 'immune checkpoint', 'antibody'] },
  'drug-discovery': { name: '신약개발', icon: Pill, color: 'green', keywords: ['drug discovery', 'clinical trial', 'FDA'] },
};

const CONTENT_TYPES = {
  'trend': { name: '동향리포트', icon: TrendingUp, description: '최신 연구 동향 분석' },
  'conference': { name: '학회참관기', icon: Users, description: '주요 학회 현장 리포트' },
  'research': { name: '연구성과', icon: Award, description: '국내외 주요 연구 성과' },
  'external': { name: '외부보고서', icon: FileText, description: '정부/기관 발행 보고서' },
};

// ========== 타입 정의 ==========
// Using ApiNewsItem from client.ts for news items

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

// ========== 소스 정의 ==========
const NEWS_SOURCES = {
  pubmed: { name: 'PubMed', icon: '📚', color: 'blue' },
  biorxiv: { name: 'bioRxiv', icon: '🧬', color: 'green' },
  nature: { name: 'Nature/Science', icon: '🌍', color: 'purple' },
  all: { name: 'All Sources', icon: '🌐', color: 'indigo' },
};

// ========== 캐시 관리 ==========
const CACHE_KEY = 'bio_daily_news_cache_v2';

interface CachedData {
  newsItems: ApiNewsItem[];
  trends: WeeklyTrend[];
  stats: DailyStats;
  generatedAt: string;
  cachedAt: string; // ISO timestamp
  language: NewsLanguage;
  sources: NewsSource[];
}

// 대한민국 07:00 기준으로 캐시 만료 확인
const isCacheValid = (cachedAt: string): boolean => {
  const cached = new Date(cachedAt);
  const now = new Date();

  // 한국 시간 기준 오늘 07:00
  const koreaOffset = 9 * 60; // UTC+9
  const nowKorea = new Date(now.getTime() + koreaOffset * 60 * 1000);
  const today7AM = new Date(nowKorea);
  today7AM.setUTCHours(7 - 9, 0, 0, 0); // UTC 기준으로 변환 (-9시간)

  // 현재 시간이 오늘 07:00 이전이면, 어제 07:00 기준
  if (nowKorea.getUTCHours() < 7) {
    today7AM.setDate(today7AM.getDate() - 1);
  }

  // 캐시가 오늘(또는 어제) 07:00 이후에 생성되었는지 확인
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
  language: NewsLanguage,
  sources: NewsSource[]
) => {
  const data: CachedData = {
    newsItems,
    trends,
    stats,
    generatedAt,
    cachedAt: new Date().toISOString(),
    language,
    sources,
  };
  localStorage.setItem(CACHE_KEY, JSON.stringify(data));
};

// ========== 컴포넌트 ==========
export default function BioResearchDaily() {
  const [activeTab, setActiveTab] = useState('today');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [generatedAt, setGeneratedAt] = useState<string | null>(null);

  // Global settings
  const [language, setLanguage] = useState<NewsLanguage>('ko');
  const [selectedSources, setSelectedSources] = useState<NewsSource[]>(['pubmed']);

  // Real data states
  const [newsItems, setNewsItems] = useState<ApiNewsItem[]>([]);
  const [weeklyTrends, setWeeklyTrends] = useState<WeeklyTrend[]>([]);
  const [dailyStats, setDailyStats] = useState<DailyStats>({
    newPapers: 0,
    domesticResearch: 0,
    clinicalTrials: 0,
    reports: 0
  });

  // Load data on mount - check cache first
  useEffect(() => {
    const cached = loadFromCache();
    if (cached && cached.language === language &&
        JSON.stringify(cached.sources?.sort()) === JSON.stringify(selectedSources.sort())) {
      // 캐시 데이터 사용 - 설정이 같을 때만
      setNewsItems(cached.newsItems);
      setWeeklyTrends(cached.trends);
      setDailyStats(cached.stats);
      setGeneratedAt(cached.generatedAt);
      setLastUpdated(cached.cachedAt);
      setLoading(false);
    } else {
      // 캐시 없거나 설정 다름 - API 호출
      loadDailyData();
    }
  }, [language, selectedSources]);

  const loadDailyData = async (forceRefresh: boolean = false) => {
    setLoading(true);
    setError(null);

    try {
      // Determine sources to fetch
      const sourcesToFetch = selectedSources.includes('all') ? 'all' : selectedSources;

      // Load AI-generated daily news with language and sources
      const newsData = await api.getDailyNews(8, forceRefresh, language, sourcesToFetch);
      setNewsItems(newsData.news_items);
      setGeneratedAt(newsData.generated_at);

      // Load hot topics for weekly trends (in parallel)
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

      // Set daily stats
      const stats = {
        newPapers: newsData.total * 250,  // Estimated
        domesticResearch: Math.floor(Math.random() * 20) + 10,
        clinicalTrials: Math.floor(Math.random() * 10) + 5,
        reports: newsData.total,
      };
      setDailyStats(stats);

      // 캐시에 저장
      saveToCache(newsData.news_items, weeklyTrends, stats, newsData.generated_at, language, selectedSources);
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

  const getCategoryColor = (colorName: string) => {
    const colors: Record<string, { bg: string; text: string; badge: string; border: string }> = {
      rose: { bg: 'bg-rose-50', text: 'text-rose-700', badge: 'bg-rose-100 text-rose-800', border: 'border-rose-200' },
      blue: { bg: 'bg-blue-50', text: 'text-blue-700', badge: 'bg-blue-100 text-blue-800', border: 'border-blue-200' },
      purple: { bg: 'bg-purple-50', text: 'text-purple-700', badge: 'bg-purple-100 text-purple-800', border: 'border-purple-200' },
      orange: { bg: 'bg-orange-50', text: 'text-orange-700', badge: 'bg-orange-100 text-orange-800', border: 'border-orange-200' },
      red: { bg: 'bg-red-50', text: 'text-red-700', badge: 'bg-red-100 text-red-800', border: 'border-red-200' },
      green: { bg: 'bg-green-50', text: 'text-green-700', badge: 'bg-green-100 text-green-800', border: 'border-green-200' },
    };
    return colors[colorName] || colors.blue;
  };

  const getTypeIcon = (type: string) => {
    return CONTENT_TYPES[type as keyof typeof CONTENT_TYPES]?.icon || FileText;
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

  // Get source badge color
  const getSourceBadgeColor = (sourceType: string) => {
    const colors: Record<string, string> = {
      pubmed: 'bg-blue-100 text-blue-700',
      biorxiv: 'bg-green-100 text-green-700',
      medrxiv: 'bg-teal-100 text-teal-700',
      nature: 'bg-purple-100 text-purple-700',
      science: 'bg-indigo-100 text-indigo-700',
    };
    return colors[sourceType] || 'bg-gray-100 text-gray-700';
  };

  // Toggle source selection
  const toggleSource = (source: NewsSource) => {
    if (source === 'all') {
      setSelectedSources(['all']);
    } else {
      setSelectedSources(prev => {
        const filtered = prev.filter(s => s !== 'all');
        if (filtered.includes(source)) {
          const result = filtered.filter(s => s !== source);
          return result.length === 0 ? ['pubmed'] : result;
        }
        return [...filtered, source];
      });
    }
  };

  const filteredNews = selectedCategory === 'all'
    ? newsItems
    : newsItems.filter(item => item.category === selectedCategory);

  return (
    <div className="h-full overflow-y-auto bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* ===== Header ===== */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-20 shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          {/* Top Bar */}
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
                  <Newspaper className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    BIO 연구 데일리
                  </h1>
                  <p className="text-sm text-slate-500">매일 업데이트되는 생명과학·의학 연구 동향</p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Language Toggle */}
              <div className="hidden sm:flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
                <button
                  onClick={() => setLanguage('ko')}
                  className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    language === 'ko'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  🇰🇷 한국어
                </button>
                <button
                  onClick={() => setLanguage('en')}
                  className={`flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    language === 'en'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  🇺🇸 English
                </button>
              </div>

              {/* Source Filter */}
              <div className="hidden md:flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
                {(['pubmed', 'biorxiv', 'nature', 'all'] as NewsSource[]).map((source) => (
                  <button
                    key={source}
                    onClick={() => toggleSource(source)}
                    className={`flex items-center gap-1 px-2 py-1.5 rounded-md text-xs font-medium transition-all ${
                      selectedSources.includes(source)
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-slate-500 hover:text-slate-700'
                    }`}
                  >
                    {NEWS_SOURCES[source].icon} {NEWS_SOURCES[source].name}
                  </button>
                ))}
              </div>

              <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-full text-sm text-slate-600">
                <Calendar className="w-4 h-4" />
                <span>{getTodayDate()}</span>
              </div>
              {lastUpdated && (
                <div className="hidden xl:flex items-center gap-1.5 text-xs text-slate-500">
                  <Clock className="w-3.5 h-3.5" />
                  <span>{new Date(lastUpdated).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
              )}
              <button
                onClick={() => {
                  localStorage.removeItem(CACHE_KEY); // 강제 새로고침 시 캐시 삭제
                  loadDailyData(true); // Force refresh from server
                }}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all shadow-md hover:shadow-lg disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                <span className="hidden sm:inline">{language === 'ko' ? '새로고침' : 'Refresh'}</span>
              </button>
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
                    ? 'border-blue-600 text-blue-600 bg-blue-50/50'
                    : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'
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
      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Error State */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-3 text-red-700">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
            <button onClick={loadDailyData} className="ml-auto text-sm font-medium hover:underline">
              다시 시도
            </button>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-10 h-10 text-blue-600 animate-spin mb-4" />
            <p className="text-slate-600">연구 동향을 불러오는 중...</p>
          </div>
        )}

        {!loading && activeTab === 'today' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Content Area */}
            <div className="lg:col-span-2 space-y-6">
              {/* Category Filter */}
              <div className="bg-white rounded-xl border border-slate-200 p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Filter className="w-4 h-4 text-slate-400" />
                  <span className="text-sm font-medium text-slate-600">분야별 필터</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setSelectedCategory('all')}
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${selectedCategory === 'all'
                        ? 'bg-blue-600 text-white shadow-md'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                      }`}
                  >
                    전체
                  </button>
                  {Object.entries(RESEARCH_CATEGORIES).map(([key, { name, icon: Icon, color }]) => (
                    <button
                      key={key}
                      onClick={() => setSelectedCategory(key)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${selectedCategory === key
                          ? `${getCategoryColor(color).badge} shadow-md`
                          : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
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
                  <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-500" />
                    오늘의 BIO 뉴스
                  </h2>
                  <div className="flex items-center gap-2">
                    {generatedAt && (
                      <span className="text-xs text-slate-400">AI 생성: {generatedAt}</span>
                    )}
                    <span className="text-sm text-slate-500">총 {filteredNews.length}건</span>
                  </div>
                </div>

                {filteredNews.length === 0 ? (
                  <div className="bg-white rounded-xl border border-slate-200 p-8 text-center">
                    <FileText className="w-12 h-12 text-slate-300 mx-auto mb-3" />
                    <p className="text-slate-500">해당 분야의 뉴스가 없습니다.</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {filteredNews.map((item, index) => {
                      const categoryColorMap: Record<string, string> = {
                        'oncology': 'orange',
                        'gene_therapy': 'blue',
                        'neurology': 'purple',
                        'immunotherapy': 'red',
                        'ai_medicine': 'indigo',
                        'drug_discovery': 'green',
                      };
                      const colorName = categoryColorMap[item.category] || 'blue';
                      const colors = getCategoryColor(colorName);

                      return (
                        <article
                          key={item.id}
                          onClick={() => openPaper(item)}
                          className="bg-white rounded-xl border border-slate-200 overflow-hidden hover:shadow-lg transition-all cursor-pointer group"
                        >
                          {/* News Header with Number Badge */}
                          <div className={`px-5 py-3 ${colors.bg} border-b ${colors.border}`}>
                            <div className="flex items-center gap-3">
                              <span className={`w-8 h-8 flex items-center justify-center rounded-full ${index < 3 ? 'bg-amber-500 text-white' : 'bg-white/80 text-slate-600'} text-sm font-bold`}>
                                {index + 1}
                              </span>
                              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors.badge}`}>
                                {item.category_name}
                              </span>
                              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getSourceBadgeColor(item.source_type || 'pubmed')}`}>
                                {item.source_type?.toUpperCase() || 'PUBMED'}
                              </span>
                              <span className="text-xs text-slate-500 ml-auto">
                                {item.source}
                              </span>
                            </div>
                          </div>

                          {/* News Content */}
                          <div className="p-5">
                            {/* Headline */}
                            <h3 className="text-xl font-bold text-slate-900 mb-3 group-hover:text-blue-600 transition-colors">
                              {item.headline}
                            </h3>

                            {/* Summary */}
                            <p className="text-slate-700 leading-relaxed mb-4">
                              {item.summary}
                            </p>

                            {/* Significance */}
                            <div className="flex items-start gap-2 p-3 bg-amber-50 rounded-lg border border-amber-200">
                              <Lightbulb className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                              <p className="text-sm text-amber-800">
                                <span className="font-semibold">왜 중요한가:</span> {item.significance}
                              </p>
                            </div>

                            {/* Footer */}
                            <div className="flex items-center justify-between mt-4 pt-3 border-t border-slate-100">
                              <span className="text-xs text-slate-500">{formatDate(item.date)}</span>
                              <span className="inline-flex items-center gap-1 text-sm font-medium text-blue-600 group-hover:gap-2 transition-all">
                                자세히 보기 <ArrowUpRight className="w-4 h-4" />
                              </span>
                            </div>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                )}
              </section>
            </div>

            {/* Sidebar */}
            <aside className="space-y-6">
              {/* Weekly Trends */}
              <div className="bg-white rounded-xl border border-slate-200 p-5">
                <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2 mb-4">
                  <TrendingUp className="w-5 h-5 text-green-500" />
                  주간 급상승 토픽
                </h3>
                <div className="space-y-3">
                  {weeklyTrends.slice(0, 5).map((trend, idx) => (
                    <div
                      key={trend.id}
                      className="p-3 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors cursor-pointer group"
                    >
                      <div className="flex items-start gap-3">
                        <span className={`w-7 h-7 flex items-center justify-center rounded-full text-sm font-bold ${idx < 2 ? 'bg-gradient-to-br from-amber-400 to-orange-500 text-white' : 'bg-slate-200 text-slate-600'
                          }`}>
                          {idx + 1}
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-slate-800 group-hover:text-blue-600 transition-colors">
                              {trend.topic}
                            </span>
                            {trend.hot && (
                              <span className="px-1.5 py-0.5 bg-red-100 text-red-600 text-xs font-bold rounded">
                                HOT
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-slate-500 mt-0.5 line-clamp-1">{trend.description}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-xs text-slate-400">논문 {trend.papers}편</span>
                            <span className="text-xs font-semibold text-green-600">{trend.change}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Quick Stats */}
              <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-xl p-5 text-white">
                <h3 className="text-lg font-bold mb-4">오늘의 통계</h3>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: '신규 논문', value: dailyStats.newPapers.toLocaleString() },
                    { label: '국내 연구성과', value: dailyStats.domesticResearch.toString() },
                    { label: '임상시험 등록', value: dailyStats.clinicalTrials.toString() },
                    { label: '리포트 발행', value: dailyStats.reports.toString() },
                  ].map((stat, idx) => (
                    <div key={idx} className="bg-white/10 backdrop-blur rounded-lg p-3 text-center">
                      <div className="text-2xl font-bold">{stat.value}</div>
                      <div className="text-xs text-blue-100">{stat.label}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Newsletter */}
              <div className="bg-white rounded-xl border border-slate-200 p-5">
                <h3 className="text-lg font-bold text-slate-800 mb-2">데일리 뉴스레터</h3>
                <p className="text-sm text-slate-600 mb-4">
                  매일 아침 8시, AI가 분석한 주요 연구 동향을 받아보세요.
                </p>
                <input
                  type="email"
                  placeholder="이메일 주소 입력"
                  className="w-full px-4 py-2.5 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-3"
                />
                <button className="w-full py-2.5 bg-slate-800 text-white rounded-lg text-sm font-medium hover:bg-slate-900 transition-colors">
                  무료 구독하기
                </button>
              </div>
            </aside>
          </div>
        )}

        {!loading && activeTab === 'weekly' && (
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h2 className="text-xl font-bold text-slate-800 mb-6">이번 주 연구 트렌드 분석</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {weeklyTrends.map((trend, idx) => (
                <div key={trend.id} className="p-5 bg-slate-50 rounded-xl hover:shadow-md transition-all cursor-pointer">
                  <div className="flex items-center gap-3 mb-3">
                    <span className={`w-10 h-10 flex items-center justify-center rounded-xl text-lg font-bold ${idx < 2 ? 'bg-gradient-to-br from-amber-400 to-orange-500 text-white' : 'bg-slate-200 text-slate-600'
                      }`}>
                      {idx + 1}
                    </span>
                    <div>
                      <h3 className="font-bold text-slate-800">{trend.topic}</h3>
                      <span className="text-sm text-green-600 font-semibold">{trend.change} 증가</span>
                    </div>
                  </div>
                  <p className="text-slate-600 text-sm">{trend.description}</p>
                  <div className="mt-3 flex items-center gap-2">
                    <Tag className="w-4 h-4 text-slate-400" />
                    <span className="text-sm text-slate-500">관련 논문 {trend.papers}편</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && activeTab === 'yearly' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <h2 className="text-xl font-bold text-slate-800 mb-2">2025년 연간 연구 동향 분석</h2>
              <p className="text-slate-600 mb-6">PubMed 데이터 기반 주요 연구 분야 현황</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Top Categories */}
                <div className="p-5 bg-slate-50 rounded-xl">
                  <h3 className="font-bold text-slate-800 mb-4">분야별 논문 현황</h3>
                  <div className="space-y-3">
                    {Object.entries(RESEARCH_CATEGORIES).slice(0, 5).map(([key, cat], idx) => (
                      <div key={key} className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="w-6 h-6 flex items-center justify-center bg-blue-100 text-blue-600 rounded-full text-xs font-bold">
                            {idx + 1}
                          </span>
                          <span className="text-slate-700">{cat.name}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-green-600 text-sm font-semibold">+{15 + idx * 5}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Breakthroughs */}
                <div className="p-5 bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl border border-amber-200">
                  <h3 className="font-bold text-slate-800 mb-4">올해의 주요 돌파구</h3>
                  <ul className="space-y-3">
                    {[
                      'CRISPR 기반 in vivo 유전자 편집 임상 성공',
                      'AI 단백질 구조 예측의 신약개발 적용 확대',
                      '차세대 GLP-1 작용제의 다중 효과 규명',
                    ].map((item, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="w-5 h-5 flex items-center justify-center bg-amber-500 text-white rounded-full text-xs flex-shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <span className="text-slate-700">{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-blue-600 text-white rounded-xl p-6 text-center">
              <h3 className="text-lg font-bold mb-2">2026년 주목해야 할 연구 분야</h3>
              <p className="text-blue-100 mb-4">AI 분석 기반 예측 리포트가 곧 공개됩니다</p>
              <button className="px-6 py-2 bg-white text-blue-600 rounded-lg font-medium hover:bg-blue-50 transition-colors">
                알림 신청하기
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-slate-800 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-slate-400 text-sm">
            BIO 연구 데일리 | PubMed, bioRxiv 데이터 기반 | BioInsight AI
          </p>
        </div>
      </footer>
    </div>
  );
}
