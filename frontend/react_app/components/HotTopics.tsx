/**
 * Enhanced HotTopics Component - Multi-dimensional Research Trend Analysis
 *
 * Dimensions:
 * - Rising Score: YoY growth rate
 * - Interest Score: Citation velocity, researcher attention
 * - Activity Score: Publication volume + clinical trials
 * - Future Score: Research gaps ("future research needed" patterns)
 */

import { useState, useEffect } from 'react';
import {
  TrendingUp,
  Flame,
  Loader2,
  X,
  RefreshCw,
  ExternalLink,
  Beaker,
  Brain,
  Dna,
  Bug,
  Cpu,
  ArrowUpRight,
  BarChart2,
  FlaskConical,
  Lightbulb,
  Users,
  Rocket,
  Info,
} from 'lucide-react';
import api, { EnhancedHotTopicsResponse, EnhancedHotTopic, MultiDimensionalScore } from '../services/client';
import { useLanguage } from '../contexts/LanguageContext';

// Domain configuration
const DOMAINS = [
  { key: 'oncology', label: 'Oncology', labelKo: '종양학', icon: Beaker, color: 'rose' },
  { key: 'neuroscience', label: 'Neuroscience', labelKo: '신경과학', icon: Brain, color: 'purple' },
  { key: 'genomics', label: 'Genomics', labelKo: '유전체학', icon: Dna, color: 'emerald' },
  { key: 'infectious_disease', label: 'Infectious Disease', labelKo: '감염병', icon: Bug, color: 'amber' },
  { key: 'ai_medicine', label: 'AI in Medicine', labelKo: 'AI 의학', icon: Cpu, color: 'blue' },
];

const COLOR_CLASSES: Record<string, { bg: string; text: string; border: string; gradient: string }> = {
  rose: { bg: 'bg-rose-100', text: 'text-rose-600', border: 'border-rose-200', gradient: 'from-rose-500 to-pink-600' },
  purple: { bg: 'bg-purple-100', text: 'text-purple-600', border: 'border-purple-200', gradient: 'from-purple-500 to-indigo-600' },
  emerald: { bg: 'bg-emerald-100', text: 'text-emerald-600', border: 'border-emerald-200', gradient: 'from-emerald-500 to-teal-600' },
  amber: { bg: 'bg-amber-100', text: 'text-amber-600', border: 'border-amber-200', gradient: 'from-amber-500 to-orange-600' },
  blue: { bg: 'bg-blue-100', text: 'text-blue-600', border: 'border-blue-200', gradient: 'from-blue-500 to-cyan-600' },
};

interface HotTopicsProps {
  onClose?: () => void;
}

export default function HotTopics({ onClose }: HotTopicsProps) {
  const { language } = useLanguage();
  const [selectedDomain, setSelectedDomain] = useState('oncology');
  const [hotTopicsData, setHotTopicsData] = useState<EnhancedHotTopicsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showMethodology, setShowMethodology] = useState(false);

  // Translations
  const t = {
    title: language === 'ko' ? '급상승 연구 주제' : 'Hot Research Topics',
    subtitle: language === 'ko'
      ? '다차원 분석 기반 연구 트렌드'
      : 'Multi-dimensional Research Trend Analysis',
    loading: language === 'ko' ? '분석 중... (약 30초 소요)' : 'Analyzing... (takes ~30s)',
    error: language === 'ko' ? '데이터 로드 실패' : 'Failed to load data',
    refresh: language === 'ko' ? '새로고침' : 'Refresh',
    papers: language === 'ko' ? '편' : 'papers',
    searchPubmed: language === 'ko' ? 'PubMed 검색' : 'Search PubMed',
    noData: language === 'ko' ? '데이터가 없습니다' : 'No data available',
    methodology: language === 'ko' ? '분석 방법론' : 'Methodology',
    clinicalTrials: language === 'ko' ? '임상시험' : 'Clinical Trials',
    futureResearch: language === 'ko' ? '연구 갭' : 'Research Gaps',

    // Score dimensions
    rising: language === 'ko' ? '급상승도' : 'Rising',
    interest: language === 'ko' ? '관심도' : 'Interest',
    activity: language === 'ko' ? '활동성' : 'Activity',
    future: language === 'ko' ? '미래성' : 'Future',
    totalScore: language === 'ko' ? '종합 점수' : 'Total Score',

    // Insights
    recommendation: language === 'ko' ? '인사이트' : 'Insight',
    researchStage: language === 'ko' ? '연구 단계' : 'Research Stage',
  };

  // Fetch enhanced hot topics
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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="relative w-full max-w-6xl max-h-[90vh] bg-white rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className={`flex items-center justify-between p-6 border-b bg-gradient-to-r ${colorClass.gradient} text-white`}>
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-2">
              <Flame className="w-6 h-6" />
              {t.title}
            </h2>
            <p className="text-sm opacity-90 mt-1">{t.subtitle}</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowMethodology(!showMethodology)}
              className={`p-2 rounded-full transition-colors ${showMethodology ? 'bg-white/30' : 'hover:bg-white/20'}`}
              title={t.methodology}
            >
              <Info className="w-5 h-5" />
            </button>
            <button
              onClick={() => fetchHotTopics(selectedDomain)}
              disabled={loading}
              className="p-2 hover:bg-white/20 rounded-full transition-colors"
              title={t.refresh}
            >
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            {onClose && (
              <button onClick={onClose} className="p-2 hover:bg-white/20 rounded-full transition-colors">
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Methodology Panel */}
        {showMethodology && hotTopicsData && (
          <div className="bg-gray-50 border-b px-6 py-4 text-sm">
            <div className="flex items-start gap-4">
              <div className="flex-1">
                <h4 className="font-semibold text-gray-900 mb-2">{t.methodology}</h4>
                <p className="text-gray-600">{hotTopicsData.methodology}</p>
              </div>
              <div className="text-right text-gray-500">
                <p>Last updated: {hotTopicsData.last_updated}</p>
                <p>Period: {hotTopicsData.analysis_period}</p>
              </div>
            </div>
          </div>
        )}

        {/* Domain Tabs */}
        <div className="flex overflow-x-auto p-4 gap-2 border-b bg-gray-50">
          {DOMAINS.map((domain) => {
            const Icon = domain.icon;
            const isActive = selectedDomain === domain.key;
            const dColorClass = COLOR_CLASSES[domain.color];

            return (
              <button
                key={domain.key}
                onClick={() => setSelectedDomain(domain.key)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? `bg-gradient-to-r ${dColorClass.gradient} text-white shadow-lg`
                    : `${dColorClass.bg} ${dColorClass.text} hover:shadow-md`
                }`}
              >
                <Icon className="w-4 h-4" />
                {language === 'ko' ? domain.labelKo : domain.label}
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              {error}
            </div>
          )}

          {loading ? (
            <div className="flex flex-col items-center justify-center py-16 gap-4">
              <Loader2 className={`w-10 h-10 ${colorClass.text} animate-spin`} />
              <p className="text-gray-500">{t.loading}</p>
              <p className="text-xs text-gray-400">Fetching PubMed, ClinicalTrials.gov, and research gap data...</p>
            </div>
          ) : hotTopicsData && hotTopicsData.hot_topics.length > 0 ? (
            <div className="space-y-4">
              {hotTopicsData.hot_topics.map((topic, idx) => (
                <EnhancedTopicCard
                  key={topic.keyword}
                  topic={topic}
                  rank={idx + 1}
                  colorClass={colorClass}
                  t={t}
                  language={language}
                />
              ))}
            </div>
          ) : !loading && (
            <div className="flex flex-col items-center justify-center py-16 text-gray-500">
              <BarChart2 className="w-16 h-16 mb-4 text-gray-300" />
              <p>{t.noData}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Score bar component
function ScoreBar({ score, label, icon: Icon, color }: { score: number; label: string; icon: React.ElementType; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <Icon className={`w-4 h-4 ${color}`} />
      <div className="flex-1">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-gray-600">{label}</span>
          <span className="font-medium">{score.toFixed(0)}</span>
        </div>
        <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              score >= 70 ? 'bg-green-500' : score >= 40 ? 'bg-yellow-500' : 'bg-gray-400'
            }`}
            style={{ width: `${score}%` }}
          />
        </div>
      </div>
    </div>
  );
}

// Enhanced topic card
function EnhancedTopicCard({
  topic,
  rank,
  colorClass,
  t,
  language
}: {
  topic: EnhancedHotTopic;
  rank: number;
  colorClass: { bg: string; text: string; border: string; gradient: string };
  t: Record<string, string>;
  language: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const isTopThree = rank <= 3;

  const handleSearchPubMed = () => {
    const query = encodeURIComponent(topic.keyword);
    window.open(`https://pubmed.ncbi.nlm.nih.gov/?term=${query}`, '_blank');
  };

  const handleSearchClinicalTrials = () => {
    const query = encodeURIComponent(topic.keyword);
    window.open(`https://clinicaltrials.gov/search?term=${query}`, '_blank');
  };

  return (
    <div
      className={`bg-white rounded-xl border shadow-sm hover:shadow-lg transition-all ${
        isTopThree ? 'ring-2 ring-offset-1' : ''
      }`}
      style={{
        ['--tw-ring-color' as string]: isTopThree
          ? rank === 1 ? '#f59e0b' : rank === 2 ? '#9ca3af' : '#cd7f32'
          : undefined,
      }}
    >
      {/* Main Row */}
      <div
        className="p-4 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-start gap-4">
          {/* Rank Badge */}
          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold shadow-lg flex-shrink-0 ${
            rank === 1
              ? 'bg-gradient-to-br from-yellow-400 to-amber-500 text-white'
              : rank === 2
              ? 'bg-gradient-to-br from-gray-300 to-gray-400 text-white'
              : rank === 3
              ? 'bg-gradient-to-br from-orange-300 to-orange-400 text-white'
              : `${colorClass.bg} ${colorClass.text}`
          }`}>
            {rank}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <h3 className="font-bold text-gray-900 text-lg">{topic.keyword}</h3>
              <span className="text-sm px-2 py-0.5 rounded-full bg-gray-100">
                {topic.trend_label}
              </span>
            </div>

            {/* Quick Stats */}
            <div className="flex flex-wrap items-center gap-3 text-sm text-gray-600">
              <span className="flex items-center gap-1">
                <TrendingUp className={`w-4 h-4 ${topic.growth_rate >= 20 ? 'text-green-500' : 'text-gray-400'}`} />
                {topic.growth_rate > 0 ? '+' : ''}{topic.growth_rate.toFixed(1)}% YoY
              </span>
              <span>
                {topic.current_year_papers.toLocaleString()} papers (2025)
              </span>
              {topic.clinical_trials > 0 && (
                <span className="flex items-center gap-1 text-blue-600">
                  <FlaskConical className="w-4 h-4" />
                  {topic.clinical_trials} trials
                </span>
              )}
            </div>

            {/* Recommendation */}
            <p className="mt-2 text-sm text-gray-500 italic">
              {topic.recommendation}
            </p>
          </div>

          {/* Total Score */}
          <div className="text-right flex-shrink-0">
            <div className={`text-3xl font-bold ${
              topic.scores.total_score >= 70 ? 'text-green-600' :
              topic.scores.total_score >= 50 ? 'text-yellow-600' : 'text-gray-500'
            }`}>
              {topic.scores.total_score.toFixed(0)}
            </div>
            <div className="text-xs text-gray-400">{t.totalScore}</div>
          </div>
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="px-4 pb-4 pt-2 border-t bg-gray-50/50 animate-appear">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Score Breakdown */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Score Breakdown</h4>
              <ScoreBar
                score={topic.scores.rising_score}
                label={t.rising}
                icon={Rocket}
                color="text-orange-500"
              />
              <ScoreBar
                score={topic.scores.interest_score}
                label={t.interest}
                icon={Users}
                color="text-blue-500"
              />
              <ScoreBar
                score={topic.scores.activity_score}
                label={t.activity}
                icon={FlaskConical}
                color="text-green-500"
              />
              <ScoreBar
                score={topic.scores.future_score}
                label={t.future}
                icon={Lightbulb}
                color="text-purple-500"
              />
            </div>

            {/* Metrics & Actions */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Details</h4>
              <div className="grid grid-cols-2 gap-3 text-sm mb-4">
                <div className="bg-white p-3 rounded-lg border">
                  <p className="text-gray-500 text-xs">2025 Papers</p>
                  <p className="font-bold text-lg">{topic.current_year_papers.toLocaleString()}</p>
                </div>
                <div className="bg-white p-3 rounded-lg border">
                  <p className="text-gray-500 text-xs">2024 Papers</p>
                  <p className="font-bold text-lg">{topic.previous_year_papers.toLocaleString()}</p>
                </div>
                <div className="bg-white p-3 rounded-lg border">
                  <p className="text-gray-500 text-xs">{t.clinicalTrials}</p>
                  <p className="font-bold text-lg text-blue-600">{topic.clinical_trials}</p>
                </div>
                <div className="bg-white p-3 rounded-lg border">
                  <p className="text-gray-500 text-xs">{t.futureResearch}</p>
                  <p className="font-bold text-lg text-purple-600">{topic.future_mentions}</p>
                </div>
              </div>

              <div className="mb-3">
                <span className="text-xs text-gray-500">{t.researchStage}:</span>
                <span className="ml-2 text-sm font-medium text-gray-700">{topic.research_stage}</span>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                <button
                  onClick={handleSearchPubMed}
                  className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 ${colorClass.bg} ${colorClass.text} rounded-lg text-sm font-medium hover:opacity-80 transition-opacity`}
                >
                  <ExternalLink className="w-4 h-4" />
                  {t.searchPubmed}
                </button>
                {topic.clinical_trials > 0 && (
                  <button
                    onClick={handleSearchClinicalTrials}
                    className="flex items-center justify-center gap-2 px-3 py-2 bg-blue-100 text-blue-600 rounded-lg text-sm font-medium hover:opacity-80 transition-opacity"
                  >
                    <FlaskConical className="w-4 h-4" />
                    Trials
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
