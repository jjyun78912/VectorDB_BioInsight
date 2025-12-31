/**
 * ResearchTrends - Unified Research Trend Analysis Hub
 *
 * Combines three perspectives into one cohesive experience:
 * - Tab 1: Hot Topics - "What should I watch now?"
 * - Tab 2: Trend Evolution - "Is this keyword trustworthy?"
 * - Tab 3: Trending Papers - "Show me the key papers"
 */

import { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts';
import {
  Flame,
  TrendingUp,
  FileText,
  X,
  Loader2,
  Search,
  Plus,
  RefreshCw,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  BookOpen,
  Beaker,
  Users,
  Clock,
  Quote,
  Lightbulb,
  ArrowUp,
  ArrowDown,
  BarChart3,
  LineChartIcon,
  Rocket,
  FlaskConical,
  Brain,
  Dna,
  Bug,
  Cpu,
  Info,
  MessageSquare,
  Sparkles,
} from 'lucide-react';
import api, {
  TrendAnalysisResponse,
  KeywordTrend,
  ValidatedTrend,
  EnhancedHotTopic,
  EnhancedHotTopicsResponse,
  CrawlerPaper,
} from '../services/client';
import { useLanguage } from '../contexts/LanguageContext';

// ============================================================================
// Constants
// ============================================================================

const CHART_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

const DOMAINS = [
  { key: 'oncology', label: 'Oncology', labelKo: '종양학', icon: Beaker, color: 'rose' },
  { key: 'neuroscience', label: 'Neuroscience', labelKo: '신경과학', icon: Brain, color: 'purple' },
  { key: 'genomics', label: 'Genomics', labelKo: '유전체학', icon: Dna, color: 'emerald' },
  { key: 'infectious_disease', label: 'Infectious Disease', labelKo: '감염병', icon: Bug, color: 'amber' },
  { key: 'ai_medicine', label: 'AI in Medicine', labelKo: 'AI 의학', icon: Cpu, color: 'blue' },
];

const COLOR_CLASSES: Record<string, { bg: string; text: string; gradient: string }> = {
  rose: { bg: 'bg-rose-100', text: 'text-rose-600', gradient: 'from-rose-500 to-pink-600' },
  purple: { bg: 'bg-purple-100', text: 'text-purple-600', gradient: 'from-purple-500 to-indigo-600' },
  emerald: { bg: 'bg-emerald-100', text: 'text-emerald-600', gradient: 'from-emerald-500 to-teal-600' },
  amber: { bg: 'bg-amber-100', text: 'text-amber-600', gradient: 'from-amber-500 to-orange-600' },
  blue: { bg: 'bg-blue-100', text: 'text-blue-600', gradient: 'from-blue-500 to-cyan-600' },
};

const PAPER_CATEGORIES: Record<string, string> = {
  oncology: 'Oncology',
  immunotherapy: 'Immunotherapy',
  gene_therapy: 'Gene Therapy',
  neurology: 'Neurology',
  infectious_disease: 'Infectious Disease',
  ai_medicine: 'AI in Medicine',
  genomics: 'Genomics',
  drug_discovery: 'Drug Discovery',
};

// ============================================================================
// Types
// ============================================================================

type TabType = 'hot-topics' | 'trend-evolution' | 'trending-papers';

interface ResearchTrendsProps {
  onClose?: () => void;
  initialTab?: TabType;
  onAskAboutPaper?: (paper: CrawlerPaper, question: string) => void;
}

// ============================================================================
// Main Component
// ============================================================================

export default function ResearchTrends({
  onClose,
  initialTab = 'hot-topics',
  onAskAboutPaper
}: ResearchTrendsProps) {
  const { language } = useLanguage();
  const [activeTab, setActiveTab] = useState<TabType>(initialTab);

  const t = {
    title: language === 'ko' ? '연구 트렌드 허브' : 'Research Trends Hub',
    hotTopics: language === 'ko' ? '핫 토픽' : 'Hot Topics',
    hotTopicsDesc: language === 'ko' ? '지금 주목받는 키워드' : 'Keywords to watch now',
    trendEvolution: language === 'ko' ? '트렌드 진화' : 'Trend Evolution',
    trendEvolutionDesc: language === 'ko' ? '5년 추이 & 검증' : '5-Year trends & validation',
    trendingPapers: language === 'ko' ? '트렌딩 논문' : 'Trending Papers',
    trendingPapersDesc: language === 'ko' ? '핵심 논문 큐레이션' : 'Key paper curation',
  };

  const tabs = [
    { id: 'hot-topics' as TabType, label: t.hotTopics, desc: t.hotTopicsDesc, icon: Flame },
    { id: 'trend-evolution' as TabType, label: t.trendEvolution, desc: t.trendEvolutionDesc, icon: TrendingUp },
    { id: 'trending-papers' as TabType, label: t.trendingPapers, desc: t.trendingPapersDesc, icon: FileText },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="relative w-full max-w-6xl h-[85vh] bg-white rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-purple-50 to-indigo-50">
          <div>
            <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-600" />
              {t.title}
            </h2>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          )}
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b bg-gray-50">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-all border-b-2 ${
                  isActive
                    ? 'border-purple-600 text-purple-600 bg-white'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Icon className="w-4 h-4" />
                <div className="text-left">
                  <div>{tab.label}</div>
                  <div className={`text-xs ${isActive ? 'text-purple-400' : 'text-gray-400'}`}>
                    {tab.desc}
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden">
          {activeTab === 'hot-topics' && <HotTopicsTab language={language} />}
          {activeTab === 'trend-evolution' && <TrendEvolutionTab language={language} />}
          {activeTab === 'trending-papers' && (
            <TrendingPapersTab language={language} onAskAboutPaper={onAskAboutPaper} />
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Tab 1: Hot Topics
// ============================================================================

function HotTopicsTab({ language }: { language: string }) {
  const [selectedDomain, setSelectedDomain] = useState('oncology');
  const [hotTopicsData, setHotTopicsData] = useState<EnhancedHotTopicsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const t = {
    loading: language === 'ko' ? '분석 중...' : 'Analyzing...',
    error: language === 'ko' ? '데이터 로드 실패' : 'Failed to load data',
    noData: language === 'ko' ? '데이터 없음' : 'No data',
    papers: language === 'ko' ? '편' : 'papers',
    trials: language === 'ko' ? '임상시험' : 'trials',
    totalScore: language === 'ko' ? '종합' : 'Total',
    rising: language === 'ko' ? '급상승' : 'Rising',
    interest: language === 'ko' ? '관심도' : 'Interest',
    activity: language === 'ko' ? '활동성' : 'Activity',
    future: language === 'ko' ? '미래성' : 'Future',
    searchPubmed: language === 'ko' ? 'PubMed' : 'PubMed',
  };

  const fetchHotTopics = async (domain: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getEnhancedHotTopics(domain, 10);
      setHotTopicsData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHotTopics(selectedDomain);
  }, [selectedDomain]);

  const currentDomain = DOMAINS.find(d => d.key === selectedDomain) || DOMAINS[0];
  const colorClass = COLOR_CLASSES[currentDomain.color];

  return (
    <div className="h-full flex flex-col">
      {/* Domain Tabs */}
      <div className="flex overflow-x-auto p-3 gap-2 border-b bg-gray-50 flex-shrink-0">
        {DOMAINS.map((domain) => {
          const Icon = domain.icon;
          const isActive = selectedDomain === domain.key;
          const dColorClass = COLOR_CLASSES[domain.color];

          return (
            <button
              key={domain.key}
              onClick={() => setSelectedDomain(domain.key)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-all ${
                isActive
                  ? `bg-gradient-to-r ${dColorClass.gradient} text-white shadow-md`
                  : `${dColorClass.bg} ${dColorClass.text} hover:shadow-sm`
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {language === 'ko' ? domain.labelKo : domain.label}
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex flex-col items-center justify-center py-16 gap-3">
            <Loader2 className={`w-8 h-8 ${colorClass.text} animate-spin`} />
            <p className="text-gray-500 text-sm">{t.loading}</p>
          </div>
        ) : hotTopicsData && hotTopicsData.hot_topics.length > 0 ? (
          <div className="space-y-3">
            {hotTopicsData.hot_topics.map((topic, idx) => (
              <HotTopicCard
                key={topic.keyword}
                topic={topic}
                rank={idx + 1}
                colorClass={colorClass}
                t={t}
              />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-16 text-gray-500">
            <BarChart3 className="w-12 h-12 mb-3 text-gray-300" />
            <p className="text-sm">{t.noData}</p>
          </div>
        )}
      </div>
    </div>
  );
}

function HotTopicCard({
  topic,
  rank,
  colorClass,
  t,
}: {
  topic: EnhancedHotTopic;
  rank: number;
  colorClass: { bg: string; text: string; gradient: string };
  t: Record<string, string>;
}) {
  const [expanded, setExpanded] = useState(false);
  const isTopThree = rank <= 3;

  const handleSearchPubMed = (e: React.MouseEvent) => {
    e.stopPropagation();
    window.open(`https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(topic.keyword)}`, '_blank');
  };

  return (
    <div
      className={`bg-white rounded-xl border shadow-sm hover:shadow-md transition-all cursor-pointer ${
        isTopThree ? 'ring-1 ring-offset-1' : ''
      }`}
      style={{
        ['--tw-ring-color' as string]: isTopThree
          ? rank === 1 ? '#f59e0b' : rank === 2 ? '#9ca3af' : '#cd7f32'
          : undefined,
      }}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="p-3">
        <div className="flex items-center gap-3">
          {/* Rank */}
          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
            rank === 1 ? 'bg-gradient-to-br from-yellow-400 to-amber-500 text-white' :
            rank === 2 ? 'bg-gradient-to-br from-gray-300 to-gray-400 text-white' :
            rank === 3 ? 'bg-gradient-to-br from-orange-300 to-orange-400 text-white' :
            `${colorClass.bg} ${colorClass.text}`
          }`}>
            {rank}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-semibold text-gray-900 text-sm">{topic.keyword}</h3>
              <span className="text-xs px-1.5 py-0.5 rounded bg-gray-100">{topic.trend_label}</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
              <span className="flex items-center gap-1">
                <TrendingUp className={`w-3 h-3 ${topic.growth_rate >= 20 ? 'text-green-500' : 'text-gray-400'}`} />
                {topic.growth_rate > 0 ? '+' : ''}{topic.growth_rate.toFixed(1)}%
              </span>
              <span>{topic.current_year_papers.toLocaleString()} {t.papers}</span>
              {topic.clinical_trials > 0 && (
                <span className="text-blue-600">{topic.clinical_trials} {t.trials}</span>
              )}
            </div>
          </div>

          {/* Score */}
          <div className="text-right flex-shrink-0">
            <div className={`text-xl font-bold ${
              topic.scores.total_score >= 70 ? 'text-green-600' :
              topic.scores.total_score >= 50 ? 'text-yellow-600' : 'text-gray-500'
            }`}>
              {topic.scores.total_score.toFixed(0)}
            </div>
            <div className="text-xs text-gray-400">{t.totalScore}</div>
          </div>

          {/* Expand Icon */}
          <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`} />
        </div>
      </div>

      {/* Expanded */}
      {expanded && (
        <div className="px-3 pb-3 pt-2 border-t bg-gray-50/50">
          <div className="grid grid-cols-4 gap-2 mb-3">
            <ScoreBar score={topic.scores.rising_score} label={t.rising} color="text-orange-500" />
            <ScoreBar score={topic.scores.interest_score} label={t.interest} color="text-blue-500" />
            <ScoreBar score={topic.scores.activity_score} label={t.activity} color="text-green-500" />
            <ScoreBar score={topic.scores.future_score} label={t.future} color="text-purple-500" />
          </div>
          <p className="text-xs text-gray-600 italic mb-2">{topic.recommendation}</p>
          <button
            onClick={handleSearchPubMed}
            className={`text-xs px-3 py-1.5 ${colorClass.bg} ${colorClass.text} rounded-lg font-medium hover:opacity-80`}
          >
            <ExternalLink className="w-3 h-3 inline mr-1" />
            {t.searchPubmed}
          </button>
        </div>
      )}
    </div>
  );
}

function ScoreBar({ score, label, color }: { score: number; label: string; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-500">{label}</span>
        <span className={`font-medium ${color}`}>{score.toFixed(0)}</span>
      </div>
      <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${
            score >= 70 ? 'bg-green-500' : score >= 40 ? 'bg-yellow-500' : 'bg-gray-400'
          }`}
          style={{ width: `${score}%` }}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Tab 2: Trend Evolution
// ============================================================================

function TrendEvolutionTab({ language }: { language: string }) {
  const [keywords, setKeywords] = useState<string[]>([]);
  const [validatedKeywords, setValidatedKeywords] = useState<Map<string, ValidatedTrend>>(new Map());
  const [newKeyword, setNewKeyword] = useState('');
  const [trendData, setTrendData] = useState<TrendAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingValidation, setLoadingValidation] = useState(false);
  const [chartType, setChartType] = useState<'line' | 'bar'>('line');
  const [expandedKeyword, setExpandedKeyword] = useState<string | null>(null);

  const t = {
    loading: language === 'ko' ? '검증된 키워드 로딩 중...' : 'Loading validated keywords...',
    analyze: language === 'ko' ? '분석' : 'Analyze',
    analyzing: language === 'ko' ? '분석 중...' : 'Analyzing...',
    placeholder: language === 'ko' ? '키워드 추가...' : 'Add keyword...',
    maxKeywords: language === 'ko' ? '최대 5개' : 'Max 5',
    lineChart: language === 'ko' ? '선형' : 'Line',
    barChart: language === 'ko' ? '막대' : 'Bar',
    totalPapers: language === 'ko' ? '총 논문' : 'Total Papers',
    growth5yr: language === 'ko' ? '5년 성장' : '5Y Growth',
    high: language === 'ko' ? '높음' : 'High',
    medium: language === 'ko' ? '보통' : 'Medium',
    emerging: language === 'ko' ? '신흥' : 'Emerging',
    uncertain: language === 'ko' ? '불확실' : 'Uncertain',
    evidence: language === 'ko' ? '검증 근거' : 'Evidence',
    viewDetails: language === 'ko' ? '상세' : 'Details',
  };

  useEffect(() => {
    loadValidatedDefaults();
  }, []);

  const loadValidatedDefaults = async () => {
    setLoadingValidation(true);
    try {
      const response = await api.getValidatedDefaults();
      const keywordList = response.trends.map(t => t.keyword);
      setKeywords(keywordList);

      const validationMap = new Map<string, ValidatedTrend>();
      response.trends.forEach(trend => validationMap.set(trend.keyword, trend));
      setValidatedKeywords(validationMap);

      if (keywordList.length > 0) {
        analyzeTrends(keywordList);
      }
    } catch (err) {
      console.error('Failed to load defaults:', err);
      const fallback = ['CRISPR', 'CAR-T therapy', 'mRNA vaccine'];
      setKeywords(fallback);
      analyzeTrends(fallback);
    } finally {
      setLoadingValidation(false);
    }
  };

  const analyzeTrends = async (keywordList: string[]) => {
    if (keywordList.length === 0) return;
    setLoading(true);
    try {
      const data = await api.analyzeTrends(keywordList);
      setTrendData(data);
    } catch (err) {
      console.error('Analysis failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const addKeyword = async () => {
    const trimmed = newKeyword.trim();
    if (trimmed && keywords.length < 5 && !keywords.includes(trimmed)) {
      const newKeywords = [...keywords, trimmed];
      setKeywords(newKeywords);
      setNewKeyword('');

      try {
        const validated = await api.validateKeyword(trimmed);
        setValidatedKeywords(new Map(validatedKeywords.set(trimmed, validated)));
      } catch (err) {
        console.error('Validation failed:', err);
      }
    }
  };

  const removeKeyword = (keyword: string) => {
    setKeywords(keywords.filter(k => k !== keyword));
    const newMap = new Map(validatedKeywords);
    newMap.delete(keyword);
    setValidatedKeywords(newMap);
  };

  const getConfidenceBadge = (level: string) => {
    switch (level) {
      case 'high': return { bg: 'bg-green-100', text: 'text-green-700', label: t.high };
      case 'medium': return { bg: 'bg-yellow-100', text: 'text-yellow-700', label: t.medium };
      case 'emerging': return { bg: 'bg-orange-100', text: 'text-orange-700', label: t.emerging };
      default: return { bg: 'bg-gray-100', text: 'text-gray-700', label: t.uncertain };
    }
  };

  const chartData = trendData?.years.map(year => {
    const dataPoint: Record<string, number | string> = { year };
    trendData.trends.forEach(trend => {
      const yearData = trend.yearly_counts.find(yc => yc.year === year);
      dataPoint[trend.keyword] = yearData?.count || 0;
    });
    return dataPoint;
  }) || [];

  return (
    <div className="h-full flex flex-col">
      {/* Keyword Bar */}
      <div className="p-3 border-b bg-gray-50 flex-shrink-0">
        <div className="flex gap-2 items-center flex-wrap">
          {keywords.map((keyword, idx) => {
            const validation = validatedKeywords.get(keyword);
            const confidence = validation ? getConfidenceBadge(validation.confidence_level) : null;

            return (
              <span
                key={keyword}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium"
                style={{
                  backgroundColor: `${CHART_COLORS[idx]}20`,
                  color: CHART_COLORS[idx],
                  borderColor: CHART_COLORS[idx],
                  borderWidth: 1,
                }}
              >
                {keyword}
                <span className={`w-5 h-4 flex items-center justify-center text-xs rounded-full ${
                  validation ? `${confidence?.bg} ${confidence?.text}` : 'bg-gray-100 text-gray-400'
                }`}>
                  {validation ? validation.confidence_emoji : '·'}
                </span>
                <button onClick={() => removeKeyword(keyword)} className="hover:opacity-70">
                  <X className="w-3 h-3" />
                </button>
              </span>
            );
          })}

          {keywords.length < 5 && (
            <div className="flex items-center gap-1">
              <input
                type="text"
                value={newKeyword}
                onChange={(e) => setNewKeyword(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addKeyword()}
                placeholder={t.placeholder}
                className="px-2 py-1 text-xs border rounded-lg focus:ring-1 focus:ring-blue-500 outline-none w-32"
              />
              <button
                onClick={addKeyword}
                disabled={!newKeyword.trim()}
                className="p-1 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 disabled:opacity-50"
              >
                <Plus className="w-3 h-3" />
              </button>
            </div>
          )}

          <button
            onClick={() => analyzeTrends(keywords)}
            disabled={loading || keywords.length === 0}
            className="ml-auto px-3 py-1 bg-blue-600 text-white rounded-lg text-xs hover:bg-blue-700 disabled:opacity-50 flex items-center gap-1"
          >
            {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Search className="w-3 h-3" />}
            {loading ? t.analyzing : t.analyze}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {loadingValidation || loading ? (
          <div className="flex flex-col items-center justify-center py-16 gap-3">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            <p className="text-gray-500 text-sm">{t.loading}</p>
          </div>
        ) : trendData && trendData.trends.length > 0 ? (
          <div className="space-y-4">
            {/* Chart Toggle */}
            <div className="flex justify-end gap-1">
              <button
                onClick={() => setChartType('line')}
                className={`px-2 py-1 rounded text-xs ${chartType === 'line' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'}`}
              >
                <LineChartIcon className="w-3 h-3 inline mr-1" />
                {t.lineChart}
              </button>
              <button
                onClick={() => setChartType('bar')}
                className={`px-2 py-1 rounded text-xs ${chartType === 'bar' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'}`}
              >
                <BarChart3 className="w-3 h-3 inline mr-1" />
                {t.barChart}
              </button>
            </div>

            {/* Chart */}
            <div className="h-64 bg-white rounded-xl border p-3">
              <ResponsiveContainer width="100%" height="100%">
                {chartType === 'line' ? (
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => v.toLocaleString()} />
                    <Tooltip formatter={(value: number) => value.toLocaleString()} />
                    <Legend />
                    {trendData.trends.map((trend, idx) => (
                      <Line key={trend.keyword} type="monotone" dataKey={trend.keyword} stroke={CHART_COLORS[idx]} strokeWidth={2} dot={{ r: 3 }} />
                    ))}
                  </LineChart>
                ) : (
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => v.toLocaleString()} />
                    <Tooltip formatter={(value: number) => value.toLocaleString()} />
                    <Legend />
                    {trendData.trends.map((trend, idx) => (
                      <Bar key={trend.keyword} dataKey={trend.keyword} fill={CHART_COLORS[idx]} radius={[2, 2, 0, 0]} />
                    ))}
                  </BarChart>
                )}
              </ResponsiveContainer>
            </div>

            {/* Keyword Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {trendData.trends.map((trend, idx) => (
                <TrendCard
                  key={trend.keyword}
                  trend={trend}
                  validation={validatedKeywords.get(trend.keyword)}
                  color={CHART_COLORS[idx]}
                  t={t}
                  isExpanded={expandedKeyword === trend.keyword}
                  onToggle={() => setExpandedKeyword(expandedKeyword === trend.keyword ? null : trend.keyword)}
                />
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function TrendCard({
  trend,
  validation,
  color,
  t,
  isExpanded,
  onToggle,
}: {
  trend: KeywordTrend;
  validation?: ValidatedTrend;
  color: string;
  t: Record<string, string>;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const getConfidenceBadge = (level: string) => {
    switch (level) {
      case 'high': return { bg: 'bg-green-100', text: 'text-green-700', label: t.high };
      case 'medium': return { bg: 'bg-yellow-100', text: 'text-yellow-700', label: t.medium };
      case 'emerging': return { bg: 'bg-orange-100', text: 'text-orange-700', label: t.emerging };
      default: return { bg: 'bg-gray-100', text: 'text-gray-700', label: t.uncertain };
    }
  };

  const confidence = validation ? getConfidenceBadge(validation.confidence_level) : null;

  return (
    <div
      className="bg-white rounded-xl border hover:shadow-md transition-shadow overflow-hidden"
      style={{ borderLeftColor: color, borderLeftWidth: 3 }}
    >
      <div className="p-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-gray-900 text-sm">{trend.keyword}</h3>
          {validation && (
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${confidence?.bg} ${confidence?.text}`}>
              {validation.confidence_emoji} {confidence?.label}
            </span>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <p className="text-gray-500">{t.totalPapers}</p>
            <p className="font-bold text-gray-900">{trend.total_count.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-gray-500">{t.growth5yr}</p>
            <p className={`font-bold flex items-center gap-1 ${
              (trend.growth_5yr || 0) > 0 ? 'text-green-600' : (trend.growth_5yr || 0) < 0 ? 'text-red-600' : 'text-gray-600'
            }`}>
              {(trend.growth_5yr || 0) > 0 && <ArrowUp className="w-3 h-3" />}
              {(trend.growth_5yr || 0) < 0 && <ArrowDown className="w-3 h-3" />}
              {trend.growth_5yr !== null ? `${trend.growth_5yr}%` : '-'}
            </p>
          </div>
        </div>

        {validation && (
          <button
            onClick={onToggle}
            className="mt-2 w-full flex items-center justify-center gap-1 text-xs text-blue-600 hover:text-blue-700"
          >
            {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            {t.viewDetails}
          </button>
        )}
      </div>

      {isExpanded && validation && (
        <div className="px-3 pb-3 pt-2 border-t bg-gray-50 text-xs">
          <h4 className="font-semibold text-gray-700 mb-2 flex items-center gap-1">
            <Info className="w-3 h-3" />
            {t.evidence}
          </h4>
          <ul className="space-y-1">
            {validation.evidence_summary.slice(0, 3).map((evidence, idx) => (
              <li key={idx} className="text-gray-600 pl-2 border-l-2 border-blue-200">
                {evidence}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Tab 3: Trending Papers
// ============================================================================

function TrendingPapersTab({
  language,
  onAskAboutPaper
}: {
  language: string;
  onAskAboutPaper?: (paper: CrawlerPaper, question: string) => void;
}) {
  const [papers, setPapers] = useState<CrawlerPaper[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeCategory, setActiveCategory] = useState('oncology');
  const [categories, setCategories] = useState<string[]>(Object.keys(PAPER_CATEGORIES));

  const t = {
    loading: language === 'ko' ? '논문 로딩 중...' : 'Loading papers...',
    noPapers: language === 'ko' ? '논문 없음' : 'No papers found',
    viewPaper: language === 'ko' ? '논문 보기' : 'View Paper',
    askAI: language === 'ko' ? 'AI에게 질문' : 'Ask AI',
    whyImportant: language === 'ko' ? '이 논문이 왜 중요한가?' : 'Why is this paper important?',
    keyFindings: language === 'ko' ? '핵심 발견은?' : 'What are the key findings?',
  };

  useEffect(() => {
    loadCategories();
  }, []);

  useEffect(() => {
    loadPapers();
  }, [activeCategory]);

  const loadCategories = async () => {
    try {
      const response = await api.getTrendingCategories();
      setCategories(response.categories);
    } catch (err) {
      console.error('Failed to load categories:', err);
    }
  };

  const loadPapers = async () => {
    setIsLoading(true);
    try {
      const response = await api.getTrendingPapers(activeCategory, 8);
      setPapers(response.papers);
    } catch (err) {
      console.error('Failed to load papers:', err);
      setPapers([]);
    } finally {
      setIsLoading(false);
    }
  };

  const openPaper = (paper: CrawlerPaper) => {
    const url = paper.url || (paper.doi ? `https://doi.org/${paper.doi}` : null);
    if (url) window.open(url, '_blank');
  };

  const handleAskQuestion = (paper: CrawlerPaper, question: string) => {
    if (onAskAboutPaper) {
      onAskAboutPaper(paper, question);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Category Tabs */}
      <div className="flex overflow-x-auto p-3 gap-2 border-b bg-gray-50 flex-shrink-0">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-all ${
              activeCategory === cat
                ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white shadow-md'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {PAPER_CATEGORIES[cat] || cat}
          </button>
        ))}
      </div>

      {/* Papers */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-16 gap-3">
            <Loader2 className="w-8 h-8 text-orange-500 animate-spin" />
            <p className="text-gray-500 text-sm">{t.loading}</p>
          </div>
        ) : papers.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-gray-500">
            <FileText className="w-12 h-12 mb-3 text-gray-300" />
            <p className="text-sm">{t.noPapers}</p>
          </div>
        ) : (
          <div className="space-y-3">
            {papers.map((paper, idx) => (
              <PaperCard
                key={paper.id}
                paper={paper}
                rank={idx + 1}
                t={t}
                onOpen={() => openPaper(paper)}
                onAsk={(q) => handleAskQuestion(paper, q)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function PaperCard({
  paper,
  rank,
  t,
  onOpen,
  onAsk,
}: {
  paper: CrawlerPaper;
  rank: number;
  t: Record<string, string>;
  onOpen: () => void;
  onAsk: (question: string) => void;
}) {
  const [showQuestions, setShowQuestions] = useState(false);

  return (
    <div className="bg-white rounded-xl border shadow-sm hover:shadow-md transition-all p-3">
      <div className="flex gap-3">
        {/* Rank */}
        <div className="w-8 h-8 bg-gradient-to-br from-orange-100 to-red-100 rounded-lg flex items-center justify-center flex-shrink-0">
          <span className="text-xs font-bold text-orange-600">#{rank}</span>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <h3
            className="font-semibold text-gray-900 text-sm line-clamp-2 hover:text-purple-700 cursor-pointer"
            onClick={onOpen}
          >
            {paper.title}
          </h3>
          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500 mt-1">
            <span className="flex items-center gap-1">
              <Users className="w-3 h-3" />
              {paper.authors.slice(0, 2).join(', ')}
              {paper.authors.length > 2 && ' et al.'}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {paper.year}
            </span>
            {paper.citation_count > 0 && (
              <span className="flex items-center gap-1">
                <Quote className="w-3 h-3" />
                {paper.citation_count}
              </span>
            )}
          </div>

          {paper.abstract && (
            <p className="text-xs text-gray-600 line-clamp-2 mt-1">{paper.abstract}</p>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 mt-2">
            <button
              onClick={onOpen}
              className="text-xs px-2 py-1 bg-purple-100 text-purple-600 rounded-lg hover:bg-purple-200 flex items-center gap-1"
            >
              <ExternalLink className="w-3 h-3" />
              {t.viewPaper}
            </button>
            <button
              onClick={() => setShowQuestions(!showQuestions)}
              className="text-xs px-2 py-1 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 flex items-center gap-1"
            >
              <MessageSquare className="w-3 h-3" />
              {t.askAI}
            </button>
          </div>

          {/* AI Questions */}
          {showQuestions && (
            <div className="mt-2 flex gap-2 flex-wrap">
              <button
                onClick={() => onAsk(t.whyImportant)}
                className="text-xs px-2 py-1 border border-blue-200 text-blue-600 rounded-full hover:bg-blue-50"
              >
                {t.whyImportant}
              </button>
              <button
                onClick={() => onAsk(t.keyFindings)}
                className="text-xs px-2 py-1 border border-blue-200 text-blue-600 rounded-full hover:bg-blue-50"
              >
                {t.keyFindings}
              </button>
            </div>
          )}
        </div>

        {/* Trend Score */}
        <div className="flex-shrink-0 text-right">
          <div className="w-10 h-10 bg-gradient-to-br from-orange-400 to-red-500 rounded-lg flex items-center justify-center">
            <span className="text-xs font-bold text-white">{paper.trend_score.toFixed(0)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
