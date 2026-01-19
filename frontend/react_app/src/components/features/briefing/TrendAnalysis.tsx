/**
 * TrendAnalysis Component - Research Trend Visualization with Validation
 *
 * Features:
 * - 5-year keyword trend line chart
 * - Multiple keyword comparison
 * - Growth rate indicators
 * - Trend direction badges (rising/stable/declining)
 * - **Validation badges with confidence levels**
 * - **Evidence-based trend justification**
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
  TrendingUp,
  TrendingDown,
  Minus,
  Search,
  Plus,
  X,
  Loader2,
  BarChart3,
  LineChartIcon,
  ArrowUp,
  ArrowDown,
  ShieldCheck,
  BookOpen,
  Beaker,
  FileText,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Info,
} from 'lucide-react';
import api, { TrendAnalysisResponse, KeywordTrend, ValidatedTrend, ValidatedTrendsResponse } from '../services/client';
import { useLanguage } from '../contexts/LanguageContext';

// Chart colors for up to 5 keywords
const CHART_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
];

interface TrendAnalysisProps {
  onClose?: () => void;
}

export default function TrendAnalysis({ onClose }: TrendAnalysisProps) {
  const { language } = useLanguage();
  const [keywords, setKeywords] = useState<string[]>([]);
  const [validatedKeywords, setValidatedKeywords] = useState<Map<string, ValidatedTrend>>(new Map());
  const [newKeyword, setNewKeyword] = useState('');
  const [trendData, setTrendData] = useState<TrendAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingValidation, setLoadingValidation] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chartType, setChartType] = useState<'line' | 'bar'>('line');
  const [showValidationDetails, setShowValidationDetails] = useState<string | null>(null);

  // Translations
  const t = {
    title: language === 'ko' ? '연구 트렌드 분석' : 'Research Trend Analysis',
    subtitle: language === 'ko'
      ? 'PubMed 5년 키워드 트렌드 - 자동 검증 시스템'
      : 'PubMed 5-Year Keyword Trends - Auto-Validated',
    addKeyword: language === 'ko' ? '키워드 추가' : 'Add keyword',
    analyze: language === 'ko' ? '분석' : 'Analyze',
    analyzing: language === 'ko' ? '분석 중...' : 'Analyzing...',
    totalPapers: language === 'ko' ? '총 논문 수' : 'Total Papers',
    growth5yr: language === 'ko' ? '5년 성장률' : '5-Year Growth',
    peakYear: language === 'ko' ? '최대 발행 연도' : 'Peak Year',
    rising: language === 'ko' ? '상승' : 'Rising',
    stable: language === 'ko' ? '안정' : 'Stable',
    declining: language === 'ko' ? '하락' : 'Declining',
    yearlyChange: language === 'ko' ? '연간 변화율' : 'YoY Change',
    lineChart: language === 'ko' ? '선 차트' : 'Line Chart',
    barChart: language === 'ko' ? '막대 차트' : 'Bar Chart',
    publications: language === 'ko' ? '논문 수' : 'Publications',
    year: language === 'ko' ? '연도' : 'Year',
    placeholder: language === 'ko' ? '예: CAR-T, immunotherapy' : 'e.g., CAR-T, immunotherapy',
    maxKeywords: language === 'ko' ? '최대 5개 키워드 (검증된 기본값 로딩 중...)' : 'Max 5 keywords (Loading validated defaults...)',
    maxKeywordsReady: language === 'ko' ? '최대 5개 키워드 (검증 완료)' : 'Max 5 keywords (Validated)',
    noData: language === 'ko' ? '검증된 키워드를 불러오는 중...' : 'Loading validated keywords...',
    error: language === 'ko' ? '분석 중 오류 발생' : 'Analysis failed',
    validated: language === 'ko' ? '검증됨' : 'Validated',
    confidence: language === 'ko' ? '신뢰도' : 'Confidence',
    high: language === 'ko' ? '높음' : 'High',
    medium: language === 'ko' ? '보통' : 'Medium',
    emerging: language === 'ko' ? '신흥' : 'Emerging',
    uncertain: language === 'ko' ? '불확실' : 'Uncertain',
    evidence: language === 'ko' ? '검증 근거' : 'Validation Evidence',
    clinicalTrials: language === 'ko' ? '활성 임상시험' : 'Active Clinical Trials',
    reviews: language === 'ko' ? '체계적 문헌고찰' : 'Systematic Reviews',
    journals: language === 'ko' ? '고영향력 저널' : 'High-IF Journals',
    viewDetails: language === 'ko' ? '상세 보기' : 'View Details',
    hideDetails: language === 'ko' ? '접기' : 'Hide',
    validationScore: language === 'ko' ? '검증 점수' : 'Validation Score',
    publicationScore: language === 'ko' ? '출판 점수' : 'Publication',
    diversityScore: language === 'ko' ? '다양성 점수' : 'Diversity',
    reviewScore: language === 'ko' ? '리뷰 점수' : 'Reviews',
    clinicalScore: language === 'ko' ? '임상 점수' : 'Clinical',
    gapScore: language === 'ko' ? '연구갭 점수' : 'Research Gap',
  };

  // Load validated defaults on mount
  useEffect(() => {
    loadValidatedDefaults();
  }, []);

  // Load validated default keywords
  const loadValidatedDefaults = async () => {
    setLoadingValidation(true);
    try {
      const response = await api.getValidatedDefaults();
      const keywordList = response.trends.map(t => t.keyword);
      setKeywords(keywordList);

      // Store validation data
      const validationMap = new Map<string, ValidatedTrend>();
      response.trends.forEach(trend => {
        validationMap.set(trend.keyword, trend);
      });
      setValidatedKeywords(validationMap);

      // Auto-analyze after loading defaults
      if (keywordList.length > 0) {
        analyzeTrendsWithKeywords(keywordList);
      }
    } catch (err) {
      console.error('Failed to load validated defaults:', err);
      // Fallback to hardcoded defaults
      const fallbackKeywords = ['CRISPR', 'CAR-T therapy', 'mRNA vaccine', 'AlphaFold', 'single-cell RNA-seq'];
      setKeywords(fallbackKeywords);
      analyzeTrendsWithKeywords(fallbackKeywords);
    } finally {
      setLoadingValidation(false);
    }
  };

  // Analyze trends with specific keywords
  const analyzeTrendsWithKeywords = async (keywordList: string[]) => {
    if (keywordList.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      const data = await api.analyzeTrends(keywordList);
      setTrendData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.error);
    } finally {
      setLoading(false);
    }
  };

  // Analyze trends
  const analyzeTrends = async () => {
    await analyzeTrendsWithKeywords(keywords);
  };

  // Add keyword with validation
  const addKeyword = async () => {
    const trimmed = newKeyword.trim();
    if (trimmed && keywords.length < 5 && !keywords.includes(trimmed)) {
      setKeywords([...keywords, trimmed]);
      setNewKeyword('');

      // Validate the new keyword
      try {
        const validated = await api.validateKeyword(trimmed);
        setValidatedKeywords(new Map(validatedKeywords.set(trimmed, validated)));
      } catch (err) {
        console.error('Validation failed for:', trimmed);
      }
    }
  };

  // Remove keyword
  const removeKeyword = (keyword: string) => {
    setKeywords(keywords.filter(k => k !== keyword));
    const newMap = new Map(validatedKeywords);
    newMap.delete(keyword);
    setValidatedKeywords(newMap);
  };

  // Handle Enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addKeyword();
    }
  };

  // Prepare chart data
  const chartData = trendData?.years.map(year => {
    const dataPoint: Record<string, number | string> = { year };
    trendData.trends.forEach(trend => {
      const yearData = trend.yearly_counts.find(yc => yc.year === year);
      dataPoint[trend.keyword] = yearData?.count || 0;
    });
    return dataPoint;
  }) || [];

  // Get confidence badge colors
  const getConfidenceBadge = (level: string) => {
    switch (level) {
      case 'high':
        return { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-200', label: t.high };
      case 'medium':
        return { bg: 'bg-yellow-100', text: 'text-yellow-700', border: 'border-yellow-200', label: t.medium };
      case 'emerging':
        return { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-200', label: t.emerging };
      default:
        return { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200', label: t.uncertain };
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="relative w-full max-w-6xl h-[85vh] bg-white rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b bg-gradient-to-r from-blue-50 to-indigo-50">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-blue-600" />
              {t.title}
              <span className="ml-2 px-2 py-0.5 bg-green-100 text-green-700 text-xs font-medium rounded-full flex items-center gap-1">
                <ShieldCheck className="w-3 h-3" />
                {t.validated}
              </span>
            </h2>
            <p className="text-sm text-gray-600 mt-1">{t.subtitle}</p>
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

        {/* Search Bar */}
        <div className="p-4 border-b bg-gray-50">
          <div className="flex gap-2 items-center flex-wrap">
            {/* Existing keywords as chips with validation badges */}
            {keywords.map((keyword, idx) => {
              const validation = validatedKeywords.get(keyword);
              const confidence = validation ? getConfidenceBadge(validation.confidence_level) : null;

              return (
                <span
                  key={keyword}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-sm font-medium"
                  style={{
                    backgroundColor: `${CHART_COLORS[idx]}20`,
                    color: CHART_COLORS[idx],
                    borderColor: CHART_COLORS[idx],
                    borderWidth: 1,
                  }}
                >
                  {keyword}
                  {/* Fixed-size badge placeholder to prevent layout shift */}
                  <span className={`ml-1 w-6 h-5 flex items-center justify-center text-xs rounded-full ${
                    validation
                      ? `${confidence?.bg} ${confidence?.text}`
                      : 'bg-gray-100 text-gray-400'
                  }`}>
                    {validation ? validation.confidence_emoji : '·'}
                  </span>
                  <button
                    onClick={() => removeKeyword(keyword)}
                    className="ml-1 hover:opacity-70"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              );
            })}

            {/* Add keyword input */}
            {keywords.length < 5 && (
              <div className="flex items-center gap-1">
                <input
                  type="text"
                  value={newKeyword}
                  onChange={(e) => setNewKeyword(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={t.placeholder}
                  className="px-3 py-1.5 text-sm border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none w-48"
                />
                <button
                  onClick={addKeyword}
                  disabled={!newKeyword.trim()}
                  className="p-1.5 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Analyze button */}
            <button
              onClick={analyzeTrends}
              disabled={loading || keywords.length === 0}
              className="ml-auto px-4 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {t.analyzing}
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  {t.analyze}
                </>
              )}
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {loadingValidation ? t.maxKeywords : t.maxKeywordsReady}
          </p>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              {error}
            </div>
          )}

          {trendData && trendData.trends.length > 0 ? (
            <div className="space-y-6">
              {/* Chart Type Toggle */}
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setChartType('line')}
                  className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm ${
                    chartType === 'line'
                      ? 'bg-blue-100 text-blue-700'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  <LineChartIcon className="w-4 h-4" />
                  {t.lineChart}
                </button>
                <button
                  onClick={() => setChartType('bar')}
                  className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm ${
                    chartType === 'bar'
                      ? 'bg-blue-100 text-blue-700'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                  {t.barChart}
                </button>
              </div>

              {/* Chart */}
              <div className="h-80 bg-white rounded-xl border p-4">
                <ResponsiveContainer width="100%" height="100%">
                  {chartType === 'line' ? (
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="year"
                        tick={{ fontSize: 12 }}
                        axisLine={{ stroke: '#e5e7eb' }}
                      />
                      <YAxis
                        tick={{ fontSize: 12 }}
                        axisLine={{ stroke: '#e5e7eb' }}
                        tickFormatter={(value) => value.toLocaleString()}
                      />
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), t.publications]}
                        labelFormatter={(label) => `${t.year}: ${label}`}
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                        }}
                      />
                      <Legend />
                      {trendData.trends.map((trend, idx) => (
                        <Line
                          key={trend.keyword}
                          type="monotone"
                          dataKey={trend.keyword}
                          stroke={CHART_COLORS[idx]}
                          strokeWidth={2}
                          dot={{ fill: CHART_COLORS[idx], strokeWidth: 2 }}
                          activeDot={{ r: 6 }}
                        />
                      ))}
                    </LineChart>
                  ) : (
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="year"
                        tick={{ fontSize: 12 }}
                        axisLine={{ stroke: '#e5e7eb' }}
                      />
                      <YAxis
                        tick={{ fontSize: 12 }}
                        axisLine={{ stroke: '#e5e7eb' }}
                        tickFormatter={(value) => value.toLocaleString()}
                      />
                      <Tooltip
                        formatter={(value: number) => [value.toLocaleString(), t.publications]}
                        labelFormatter={(label) => `${t.year}: ${label}`}
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                        }}
                      />
                      <Legend />
                      {trendData.trends.map((trend, idx) => (
                        <Bar
                          key={trend.keyword}
                          dataKey={trend.keyword}
                          fill={CHART_COLORS[idx]}
                          radius={[4, 4, 0, 0]}
                        />
                      ))}
                    </BarChart>
                  )}
                </ResponsiveContainer>
              </div>

              {/* Trend Summary Cards with Validation */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {trendData.trends.map((trend, idx) => (
                  <ValidatedTrendCard
                    key={trend.keyword}
                    trend={trend}
                    validation={validatedKeywords.get(trend.keyword)}
                    color={CHART_COLORS[idx]}
                    t={t}
                    isExpanded={showValidationDetails === trend.keyword}
                    onToggle={() => setShowValidationDetails(
                      showValidationDetails === trend.keyword ? null : trend.keyword
                    )}
                  />
                ))}
              </div>
            </div>
          ) : loading || loadingValidation ? (
            <div className="flex flex-col items-center justify-center py-16 text-gray-500">
              <Loader2 className="w-12 h-12 mb-4 text-blue-500 animate-spin" />
              <p>{t.noData}</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-16 text-gray-500">
              <BarChart3 className="w-16 h-16 mb-4 text-gray-300" />
              <p>{t.noData}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Validated Trend Card Component
function ValidatedTrendCard({
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
  const latestGrowth = trend.yearly_counts[trend.yearly_counts.length - 1]?.growth_rate;

  const getConfidenceBadge = (level: string) => {
    switch (level) {
      case 'high':
        return { bg: 'bg-green-100', text: 'text-green-700', label: t.high };
      case 'medium':
        return { bg: 'bg-yellow-100', text: 'text-yellow-700', label: t.medium };
      case 'emerging':
        return { bg: 'bg-orange-100', text: 'text-orange-700', label: t.emerging };
      default:
        return { bg: 'bg-gray-100', text: 'text-gray-700', label: t.uncertain };
    }
  };

  const confidence = validation ? getConfidenceBadge(validation.confidence_level) : null;

  // Radar chart data for validation scores
  const radarData = validation ? [
    { subject: t.publicationScore, score: validation.publication_score, fullMark: 100 },
    { subject: t.diversityScore, score: validation.diversity_score, fullMark: 100 },
    { subject: t.reviewScore, score: validation.review_score, fullMark: 100 },
    { subject: t.clinicalScore, score: validation.clinical_score, fullMark: 100 },
    { subject: t.gapScore, score: validation.gap_score, fullMark: 100 },
  ] : [];

  return (
    <div
      className="bg-white rounded-xl border hover:shadow-md transition-shadow overflow-hidden"
      style={{ borderLeftColor: color, borderLeftWidth: 4 }}
    >
      <div className="p-4">
        {/* Header with validation badge */}
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold text-gray-900">{trend.keyword}</h3>
          <div className="flex items-center gap-2">
            {validation && (
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium flex items-center gap-1 ${confidence?.bg} ${confidence?.text}`}>
                {validation.confidence_emoji} {confidence?.label}
              </span>
            )}
            <span
              className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                trend.trend_direction === 'rising'
                  ? 'bg-green-100 text-green-700'
                  : trend.trend_direction === 'declining'
                  ? 'bg-red-100 text-red-700'
                  : 'bg-gray-100 text-gray-700'
              }`}
            >
              {trend.trend_direction === 'rising' ? t.rising :
               trend.trend_direction === 'declining' ? t.declining : t.stable}
            </span>
          </div>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-gray-500">{t.totalPapers}</p>
            <p className="font-bold text-lg text-gray-900">
              {trend.total_count.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-gray-500">{t.growth5yr}</p>
            <p className={`font-bold text-lg flex items-center gap-1 ${
              (trend.growth_5yr || 0) > 0 ? 'text-green-600' :
              (trend.growth_5yr || 0) < 0 ? 'text-red-600' : 'text-gray-600'
            }`}>
              {(trend.growth_5yr || 0) > 0 && <ArrowUp className="w-4 h-4" />}
              {(trend.growth_5yr || 0) < 0 && <ArrowDown className="w-4 h-4" />}
              {trend.growth_5yr !== null ? `${trend.growth_5yr}%` : '-'}
            </p>
          </div>
        </div>

        {/* Validation metrics preview */}
        {validation && (
          <div className="mt-3 pt-3 border-t">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-1">
                  <Beaker className="w-3 h-3" />
                  {validation.active_clinical_trials}
                </span>
                <span className="flex items-center gap-1">
                  <BookOpen className="w-3 h-3" />
                  {validation.systematic_reviews + validation.meta_analyses}
                </span>
                <span className="flex items-center gap-1">
                  <FileText className="w-3 h-3" />
                  {validation.unique_journals}
                </span>
              </div>
              <span className="font-medium text-blue-600">
                {validation.total_score.toFixed(0)}점
              </span>
            </div>
          </div>
        )}

        {/* Expand button */}
        {validation && (
          <button
            onClick={onToggle}
            className="mt-3 w-full flex items-center justify-center gap-1 text-xs text-blue-600 hover:text-blue-700"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-3 h-3" />
                {t.hideDetails}
              </>
            ) : (
              <>
                <ChevronDown className="w-3 h-3" />
                {t.viewDetails}
              </>
            )}
          </button>
        )}
      </div>

      {/* Expanded validation details */}
      {isExpanded && validation && (
        <div className="px-4 pb-4 pt-2 border-t bg-gray-50">
          {/* Radar Chart */}
          <div className="h-48 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis
                  dataKey="subject"
                  tick={{ fontSize: 10, fill: '#6b7280' }}
                />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 100]}
                  tick={{ fontSize: 8 }}
                />
                <Radar
                  name={t.validationScore}
                  dataKey="score"
                  stroke={color}
                  fill={color}
                  fillOpacity={0.3}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Evidence summary */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-gray-700 flex items-center gap-1">
              <Info className="w-3 h-3" />
              {t.evidence}
            </h4>
            <ul className="space-y-1">
              {validation.evidence_summary.map((evidence, idx) => (
                <li key={idx} className="text-xs text-gray-600 pl-2 border-l-2 border-blue-200">
                  {evidence}
                </li>
              ))}
            </ul>
          </div>

          {/* Detailed metrics */}
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-500">{t.clinicalTrials}</span>
              <span className="font-medium">{validation.active_clinical_trials}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">{t.reviews}</span>
              <span className="font-medium">{validation.systematic_reviews}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">{t.journals}</span>
              <span className="font-medium">{validation.high_if_journals}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">{t.yearlyChange}</span>
              <span className="font-medium">{validation.growth_rate_yoy.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
