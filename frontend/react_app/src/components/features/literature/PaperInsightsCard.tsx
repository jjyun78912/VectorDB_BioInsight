/**
 * PaperInsightsCard - Quick paper evaluation display component.
 *
 * Shows at-a-glance insights:
 * - Bottom Line: One-sentence takeaway
 * - Quality Score: Study design & methodology badge
 * - Key Outcomes: Effect sizes with interpretation
 * - Population: Sample size & demographics
 */
import React, { useState, useEffect } from 'react';
import {
  Lightbulb, Award, Target, Users, ChevronDown, ChevronUp,
  AlertTriangle, CheckCircle, Info, Loader2, TrendingUp, TrendingDown
} from 'lucide-react';
import api, { PaperInsightsResponse, QualityResponse } from '../services/client';

interface PaperInsightsCardProps {
  title: string;
  abstract: string;
  // Auto-fetch mode: if true, fetches insights on mount
  autoFetch?: boolean;
  // Pre-loaded insights (for batch loading)
  preloadedInsights?: PaperInsightsResponse;
  // Compact mode: only show bottom line and quality badge
  compact?: boolean;
}

// Quality score color coding
const getQualityColor = (score: number | null): string => {
  if (!score) return 'bg-gray-100 text-gray-600';
  if (score >= 8) return 'bg-green-100 text-green-700';
  if (score >= 5) return 'bg-yellow-100 text-yellow-700';
  return 'bg-red-100 text-red-600';
};

// Clinical relevance badge color
const getRelevanceColor = (relevance: string | null): string => {
  switch (relevance) {
    case 'High': return 'bg-purple-100 text-purple-700';
    case 'Medium': return 'bg-blue-100 text-blue-700';
    case 'Low': return 'bg-gray-100 text-gray-600';
    default: return 'bg-gray-100 text-gray-600';
  }
};

// Bias risk indicator
const BiasRiskBadge: React.FC<{ risk: string | null }> = ({ risk }) => {
  if (!risk) return null;

  const config: Record<string, { color: string; icon: React.ElementType }> = {
    'Low': { color: 'text-green-600', icon: CheckCircle },
    'Medium': { color: 'text-yellow-600', icon: Info },
    'High': { color: 'text-red-500', icon: AlertTriangle },
    'Unclear': { color: 'text-gray-500', icon: Info },
  };

  const { color, icon: Icon } = config[risk] || config['Unclear'];

  return (
    <span className={`inline-flex items-center gap-1 text-xs ${color}`}>
      <Icon className="w-3 h-3" />
      {risk} bias risk
    </span>
  );
};

// Quality Badge component
const QualityBadge: React.FC<{ quality: QualityResponse | null }> = ({ quality }) => {
  if (!quality) return null;

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg ${getQualityColor(quality.quality_score)}`}>
      <Award className="w-4 h-4" />
      <div className="flex flex-col">
        <span className="text-xs font-medium">{quality.design || 'Study'}</span>
        <span className="text-[10px] opacity-75">
          Score: {quality.quality_score?.toFixed(1) || '-'}/10
          {quality.sample_size && ` | n=${quality.sample_size.toLocaleString()}`}
        </span>
      </div>
    </div>
  );
};

export const PaperInsightsCard: React.FC<PaperInsightsCardProps> = ({
  title,
  abstract,
  autoFetch = true,
  preloadedInsights,
  compact = false,
}) => {
  const [insights, setInsights] = useState<PaperInsightsResponse | null>(preloadedInsights || null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (preloadedInsights) {
      setInsights(preloadedInsights);
      return;
    }

    if (autoFetch && title && abstract) {
      fetchInsights();
    }
  }, [title, abstract, autoFetch, preloadedInsights]);

  const fetchInsights = async () => {
    if (!title || !abstract) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await api.getPaperInsights(title, abstract);
      setInsights(data);
    } catch (err) {
      console.error('Failed to fetch paper insights:', err);
      setError('Could not analyze paper');
    } finally {
      setIsLoading(false);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500 py-2">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Analyzing paper...</span>
      </div>
    );
  }

  // Error or no data
  if (error || !insights) {
    return null; // Silently fail - don't show errors in UI
  }

  const { bottom_line, quality, key_outcomes, population } = insights;

  // Compact mode: just show bottom line and quality badge
  if (compact) {
    return (
      <div className="space-y-2">
        {/* Bottom Line */}
        {bottom_line?.summary && (
          <div className="flex items-start gap-2 p-2.5 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-200/50">
            <Lightbulb className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-gray-800 font-medium leading-snug">
              {bottom_line.summary}
            </p>
          </div>
        )}

        {/* Quality & Relevance Badges */}
        <div className="flex flex-wrap items-center gap-2">
          <QualityBadge quality={quality} />
          {bottom_line?.clinical_relevance && (
            <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getRelevanceColor(bottom_line.clinical_relevance)}`}>
              {bottom_line.clinical_relevance} Relevance
            </span>
          )}
          {bottom_line?.action_type && (
            <span className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 rounded-full text-xs text-gray-600">
              {bottom_line.action_type}
            </span>
          )}
        </div>
      </div>
    );
  }

  // Full mode: show all insights
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden bg-white">
      {/* Header with Bottom Line */}
      <div className="p-4 bg-gradient-to-r from-amber-50 to-orange-50 border-b border-amber-200/50">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-amber-100 rounded-lg">
            <Lightbulb className="w-5 h-5 text-amber-600" />
          </div>
          <div className="flex-1">
            <h4 className="text-xs font-semibold text-amber-700 uppercase tracking-wide mb-1">
              Bottom Line
            </h4>
            {bottom_line?.summary ? (
              <p className="text-gray-800 font-medium leading-relaxed">
                {bottom_line.summary}
              </p>
            ) : (
              <p className="text-gray-500 italic">
                Could not extract bottom line
              </p>
            )}

            {/* Tags */}
            <div className="flex flex-wrap items-center gap-2 mt-2">
              {bottom_line?.clinical_relevance && (
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${getRelevanceColor(bottom_line.clinical_relevance)}`}>
                  {bottom_line.clinical_relevance} Relevance
                </span>
              )}
              {bottom_line?.action_type && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-white/70 rounded-full text-xs text-gray-600">
                  {bottom_line.action_type}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Quick Stats Row */}
      <div className="flex flex-wrap items-center gap-3 p-3 bg-gray-50 border-b border-gray-100">
        <QualityBadge quality={quality} />
        {quality && <BiasRiskBadge risk={quality.bias_risk} />}
        {population?.n && (
          <span className="inline-flex items-center gap-1 text-xs text-gray-600">
            <Users className="w-3 h-3" />
            n={population.n.toLocaleString()}
          </span>
        )}
      </div>

      {/* Expandable Details */}
      <div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full flex items-center justify-between px-4 py-2.5 text-sm text-gray-600 hover:bg-gray-50 transition-colors"
        >
          <span className="font-medium">
            {key_outcomes.length > 0
              ? `${key_outcomes.length} Key Outcome${key_outcomes.length > 1 ? 's' : ''}`
              : 'More Details'}
          </span>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>

        {isExpanded && (
          <div className="p-4 pt-0 space-y-4">
            {/* Key Outcomes */}
            {key_outcomes.length > 0 && (
              <div>
                <h5 className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                  <Target className="w-3.5 h-3.5" />
                  Key Outcomes
                </h5>
                <div className="space-y-2">
                  {key_outcomes.map((outcome, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-2.5 bg-gray-50 rounded-lg"
                    >
                      <div>
                        <span className="text-sm font-medium text-gray-800">
                          {outcome.outcome}
                        </span>
                        <span className="text-xs text-gray-500 ml-2">
                          {outcome.metric} = {outcome.value.toFixed(2)}
                          {outcome.ci && ` (${outcome.ci})`}
                        </span>
                      </div>
                      {outcome.interpretation && (
                        <span className={`inline-flex items-center gap-1 text-xs font-medium ${
                          outcome.value < 1 ? 'text-green-600' : 'text-red-500'
                        }`}>
                          {outcome.value < 1 ? (
                            <TrendingDown className="w-3 h-3" />
                          ) : (
                            <TrendingUp className="w-3 h-3" />
                          )}
                          {outcome.interpretation}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Population */}
            {population && (population.n || population.condition || population.age) && (
              <div>
                <h5 className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                  <Users className="w-3.5 h-3.5" />
                  Study Population
                </h5>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  {population.condition && (
                    <div className="p-2 bg-gray-50 rounded">
                      <span className="text-gray-500 text-xs">Condition</span>
                      <p className="font-medium text-gray-800">{population.condition}</p>
                    </div>
                  )}
                  {population.n && (
                    <div className="p-2 bg-gray-50 rounded">
                      <span className="text-gray-500 text-xs">Sample Size</span>
                      <p className="font-medium text-gray-800">{population.n.toLocaleString()}</p>
                    </div>
                  )}
                  {population.age && (
                    <div className="p-2 bg-gray-50 rounded">
                      <span className="text-gray-500 text-xs">Age</span>
                      <p className="font-medium text-gray-800">{population.age}</p>
                    </div>
                  )}
                  {population.setting && (
                    <div className="p-2 bg-gray-50 rounded">
                      <span className="text-gray-500 text-xs">Setting</span>
                      <p className="font-medium text-gray-800">{population.setting}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Quality Details */}
            {quality && (quality.strengths.length > 0 || quality.limitations.length > 0) && (
              <div className="grid grid-cols-2 gap-3">
                {quality.strengths.length > 0 && (
                  <div>
                    <h5 className="text-xs font-medium text-green-700 mb-1">Strengths</h5>
                    <ul className="text-xs text-gray-600 space-y-0.5">
                      {quality.strengths.map((s, i) => (
                        <li key={i} className="flex items-start gap-1">
                          <CheckCircle className="w-3 h-3 text-green-500 mt-0.5 flex-shrink-0" />
                          {s}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {quality.limitations.length > 0 && (
                  <div>
                    <h5 className="text-xs font-medium text-amber-700 mb-1">Limitations</h5>
                    <ul className="text-xs text-gray-600 space-y-0.5">
                      {quality.limitations.map((l, i) => (
                        <li key={i} className="flex items-start gap-1">
                          <AlertTriangle className="w-3 h-3 text-amber-500 mt-0.5 flex-shrink-0" />
                          {l}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Compact inline version for use in paper lists
 */
export const PaperInsightsBadges: React.FC<{
  title: string;
  abstract: string;
}> = ({ title, abstract }) => {
  const [insights, setInsights] = useState<PaperInsightsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchQuality = async () => {
      try {
        // Just get quality score - faster
        const data = await api.getQualityScore(title, abstract);
        setInsights({
          bottom_line: null,
          quality: data,
          key_outcomes: [],
          population: null
        });
      } catch (err) {
        console.error('Failed to fetch quality:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchQuality();
  }, [title, abstract]);

  if (isLoading) {
    return <span className="text-xs text-gray-400">Analyzing...</span>;
  }

  if (!insights?.quality) {
    return null;
  }

  const q = insights.quality;

  return (
    <div className="flex items-center gap-2">
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${getQualityColor(q.quality_score)}`}>
        <Award className="w-3 h-3" />
        {q.quality_score?.toFixed(1) || '-'}
      </span>
      {q.design && (
        <span className="text-xs text-gray-500">{q.design}</span>
      )}
    </div>
  );
};

export default PaperInsightsCard;
