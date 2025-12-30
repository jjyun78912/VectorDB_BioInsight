/**
 * TrendAnalysis Component - Research Trend Visualization
 *
 * Features:
 * - 5-year keyword trend line chart
 * - Multiple keyword comparison
 * - Growth rate indicators
 * - Trend direction badges (rising/stable/declining)
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
} from 'lucide-react';
import api, { TrendAnalysisResponse, KeywordTrend } from '../services/client';
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
  const [keywords, setKeywords] = useState<string[]>(['CRISPR']);
  const [newKeyword, setNewKeyword] = useState('');
  const [trendData, setTrendData] = useState<TrendAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chartType, setChartType] = useState<'line' | 'bar'>('line');

  // Translations
  const t = {
    title: language === 'ko' ? '연구 트렌드 분석' : 'Research Trend Analysis',
    subtitle: language === 'ko'
      ? 'PubMed 5년 키워드 트렌드 (2021-2025)'
      : 'PubMed 5-Year Keyword Trends (2021-2025)',
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
    maxKeywords: language === 'ko' ? '최대 5개 키워드' : 'Max 5 keywords',
    noData: language === 'ko' ? '키워드를 입력하고 분석 버튼을 클릭하세요' : 'Enter keywords and click Analyze',
    error: language === 'ko' ? '분석 중 오류 발생' : 'Analysis failed',
  };

  // Analyze trends
  const analyzeTrends = async () => {
    if (keywords.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      const data = await api.analyzeTrends(keywords);
      setTrendData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.error);
    } finally {
      setLoading(false);
    }
  };

  // Add keyword
  const addKeyword = () => {
    const trimmed = newKeyword.trim();
    if (trimmed && keywords.length < 5 && !keywords.includes(trimmed)) {
      setKeywords([...keywords, trimmed]);
      setNewKeyword('');
    }
  };

  // Remove keyword
  const removeKeyword = (keyword: string) => {
    setKeywords(keywords.filter(k => k !== keyword));
  };

  // Handle Enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addKeyword();
    }
  };

  // Auto-analyze on mount
  useEffect(() => {
    analyzeTrends();
  }, []);

  // Prepare chart data
  const chartData = trendData?.years.map(year => {
    const dataPoint: Record<string, number | string> = { year };
    trendData.trends.forEach(trend => {
      const yearData = trend.yearly_counts.find(yc => yc.year === year);
      dataPoint[trend.keyword] = yearData?.count || 0;
    });
    return dataPoint;
  }) || [];

  // Trend direction icon
  const TrendIcon = ({ direction }: { direction: string }) => {
    switch (direction) {
      case 'rising':
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'declining':
        return <TrendingDown className="w-4 h-4 text-red-500" />;
      default:
        return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  // Trend badge
  const TrendBadge = ({ direction }: { direction: string }) => {
    const colors = {
      rising: 'bg-green-100 text-green-700 border-green-200',
      declining: 'bg-red-100 text-red-700 border-red-200',
      stable: 'bg-gray-100 text-gray-700 border-gray-200',
    };
    const labels = {
      rising: t.rising,
      declining: t.declining,
      stable: t.stable,
    };

    return (
      <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${colors[direction as keyof typeof colors]}`}>
        {labels[direction as keyof typeof labels]}
      </span>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="relative w-full max-w-5xl max-h-[90vh] bg-white rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b bg-gradient-to-r from-blue-50 to-indigo-50">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-blue-600" />
              {t.title}
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
            {/* Existing keywords as chips */}
            {keywords.map((keyword, idx) => (
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
                <button
                  onClick={() => removeKeyword(keyword)}
                  className="ml-1 hover:opacity-70"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}

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
          <p className="text-xs text-gray-500 mt-2">{t.maxKeywords}</p>
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

              {/* Trend Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {trendData.trends.map((trend, idx) => (
                  <TrendCard key={trend.keyword} trend={trend} color={CHART_COLORS[idx]} t={t} />
                ))}
              </div>
            </div>
          ) : !loading && (
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

// Trend summary card component
function TrendCard({
  trend,
  color,
  t
}: {
  trend: KeywordTrend;
  color: string;
  t: Record<string, string>;
}) {
  const latestGrowth = trend.yearly_counts[trend.yearly_counts.length - 1]?.growth_rate;

  return (
    <div
      className="bg-white rounded-xl border p-4 hover:shadow-md transition-shadow"
      style={{ borderLeftColor: color, borderLeftWidth: 4 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-900">{trend.keyword}</h3>
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
        <div>
          <p className="text-gray-500">{t.peakYear}</p>
          <p className="font-bold text-gray-900">{trend.peak_year || '-'}</p>
        </div>
        <div>
          <p className="text-gray-500">{t.yearlyChange}</p>
          <p className={`font-bold flex items-center gap-1 ${
            (latestGrowth || 0) > 0 ? 'text-green-600' :
            (latestGrowth || 0) < 0 ? 'text-red-600' : 'text-gray-600'
          }`}>
            {latestGrowth !== null ? `${latestGrowth > 0 ? '+' : ''}${latestGrowth}%` : '-'}
          </p>
        </div>
      </div>
    </div>
  );
}
