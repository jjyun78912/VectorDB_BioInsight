import React, { useState, useEffect, useCallback } from 'react';
import { X, Activity, Share2, GitBranch, Database, BarChart2, FileText, CheckCircle2, Loader2, AlertCircle, ExternalLink, Download } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

// Agent info matching backend
const AGENTS = [
  { id: 'agent1_deg', name: 'DEG Analysis', nameKo: 'DEG 분석', icon: Activity, description: 'DESeq2 차등 발현 분석', descriptionKo: 'DESeq2 차등 발현 유전자 분석' },
  { id: 'agent2_network', name: 'Network Analysis', nameKo: '네트워크 분석', icon: Share2, description: 'Hub gene detection', descriptionKo: 'Hub gene 탐지' },
  { id: 'agent3_pathway', name: 'Pathway Enrichment', nameKo: '경로 농축', icon: GitBranch, description: 'GO/KEGG enrichment', descriptionKo: 'GO/KEGG 경로 농축' },
  { id: 'agent4_validation', name: 'DB Validation', nameKo: 'DB 검증', icon: Database, description: 'DisGeNET, OMIM, COSMIC', descriptionKo: 'DisGeNET, OMIM, COSMIC 검증' },
  { id: 'agent5_visualization', name: 'Visualization', nameKo: '시각화', icon: BarChart2, description: 'Volcano, Heatmap, Network', descriptionKo: 'Volcano plot, Heatmap, Network' },
  { id: 'agent6_report', name: 'Report Generation', nameKo: '리포트 생성', icon: FileText, description: 'HTML report', descriptionKo: 'HTML 보고서 생성' },
];

interface SSEMessage {
  type: 'pipeline_start' | 'agent_start' | 'agent_progress' | 'agent_complete' | 'agent_error' | 'pipeline_complete' | 'pipeline_error' | 'complete' | 'final';
  agent?: string;
  name?: string;
  description?: string;
  progress?: number;
  result_summary?: any;
  error?: string;
  status?: string;
  completed_agents?: string[];
  failed_agents?: string[];
  report_path?: string;
  run_dir?: string;
  timestamp?: string;
}

type AgentStatus = 'pending' | 'running' | 'completed' | 'error';

interface PipelineProgressProps {
  jobId: string;
  onClose: () => void;
  onViewReport?: (jobId: string) => void;
}

export const PipelineProgress: React.FC<PipelineProgressProps> = ({
  jobId,
  onClose,
  onViewReport
}) => {
  const { language } = useLanguage();
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({});
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [pipelineStatus, setPipelineStatus] = useState<'running' | 'completed' | 'error'>('running');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [resultSummaries, setResultSummaries] = useState<Record<string, any>>({});
  const [reportPath, setReportPath] = useState<string | null>(null);

  // Connect to SSE stream
  useEffect(() => {
    const eventSource = new EventSource(`http://localhost:8000/api/rnaseq/stream/${jobId}`);

    eventSource.onmessage = (event) => {
      try {
        const data: SSEMessage = JSON.parse(event.data);

        switch (data.type) {
          case 'pipeline_start':
            // Initialize all agents as pending
            const initialStatuses: Record<string, AgentStatus> = {};
            AGENTS.forEach(a => { initialStatuses[a.id] = 'pending'; });
            setAgentStatuses(initialStatuses);
            break;

          case 'agent_start':
            if (data.agent) {
              setCurrentAgent(data.agent);
              setAgentStatuses(prev => ({ ...prev, [data.agent!]: 'running' }));
              if (data.progress !== undefined) setProgress(data.progress);
            }
            break;

          case 'agent_complete':
            if (data.agent) {
              setAgentStatuses(prev => ({ ...prev, [data.agent!]: 'completed' }));
              if (data.progress !== undefined) setProgress(data.progress);
              if (data.result_summary) {
                setResultSummaries(prev => ({ ...prev, [data.agent!]: data.result_summary }));
              }
            }
            break;

          case 'agent_error':
            if (data.agent) {
              setAgentStatuses(prev => ({ ...prev, [data.agent!]: 'error' }));
              setErrorMessage(data.error || 'Unknown error');
            }
            break;

          case 'pipeline_complete':
            setPipelineStatus('completed');
            setProgress(100);
            setCurrentAgent(null);
            if (data.report_path) setReportPath(data.report_path);
            break;

          case 'pipeline_error':
            setPipelineStatus('error');
            setErrorMessage(data.error || 'Pipeline failed');
            break;

          case 'final':
          case 'complete':
            eventSource.close();
            break;
        }
      } catch (err) {
        console.error('Error parsing SSE message:', err);
      }
    };

    eventSource.onerror = () => {
      console.error('SSE connection error');
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [jobId]);

  const handleViewReport = () => {
    if (onViewReport) {
      onViewReport(jobId);
    } else {
      // Open in new tab
      window.open(`http://localhost:8000/api/rnaseq/report/${jobId}`, '_blank');
    }
  };

  const t = {
    title: language === 'ko' ? 'RNA-seq 분석 진행 중' : 'RNA-seq Analysis in Progress',
    completed: language === 'ko' ? '분석 완료' : 'Analysis Complete',
    error: language === 'ko' ? '분석 실패' : 'Analysis Failed',
    viewReport: language === 'ko' ? '리포트 보기' : 'View Report',
    downloadReport: language === 'ko' ? '리포트 다운로드' : 'Download Report',
    close: language === 'ko' ? '닫기' : 'Close',
    processing: language === 'ko' ? '처리 중...' : 'Processing...',
  };

  const getAgentIcon = (agentId: string, status: AgentStatus) => {
    const agent = AGENTS.find(a => a.id === agentId);
    if (!agent) return null;

    const IconComponent = agent.icon;
    const baseClasses = "w-5 h-5";

    switch (status) {
      case 'completed':
        return <CheckCircle2 className={`${baseClasses} text-green-500`} />;
      case 'running':
        return <Loader2 className={`${baseClasses} text-purple-500 animate-spin`} />;
      case 'error':
        return <AlertCircle className={`${baseClasses} text-red-500`} />;
      default:
        return <IconComponent className={`${baseClasses} text-gray-300`} />;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-appear">
      <div className="relative w-full max-w-lg glass-4 rounded-3xl shadow-2xl border border-purple-200/50 overflow-hidden">
        {/* Close button */}
        {pipelineStatus !== 'running' && (
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-2 rounded-full hover:bg-gray-100 transition-colors z-10"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        )}

        {/* Header */}
        <div className="p-6 pb-4">
          <h2 className="text-xl font-bold text-gray-800">
            {pipelineStatus === 'completed' ? t.completed : pipelineStatus === 'error' ? t.error : t.title}
          </h2>
          <p className="text-sm text-gray-500 mt-1">Job ID: {jobId}</p>
        </div>

        {/* Progress Bar */}
        <div className="px-6">
          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 ${
                pipelineStatus === 'error' ? 'bg-red-500' : pipelineStatus === 'completed' ? 'bg-green-500' : 'bg-gradient-to-r from-purple-500 to-indigo-500'
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-right text-sm text-gray-500 mt-1">{progress}%</p>
        </div>

        {/* Agent List */}
        <div className="p-6 space-y-3">
          {AGENTS.map((agent) => {
            const status = agentStatuses[agent.id] || 'pending';
            const isActive = currentAgent === agent.id;
            const summary = resultSummaries[agent.id];

            return (
              <div
                key={agent.id}
                className={`flex items-center gap-3 p-3 rounded-xl transition-all ${
                  isActive ? 'bg-purple-50 border border-purple-200' :
                  status === 'completed' ? 'bg-green-50/50' :
                  status === 'error' ? 'bg-red-50/50' :
                  'bg-gray-50/50'
                }`}
              >
                {getAgentIcon(agent.id, status)}
                <div className="flex-1 min-w-0">
                  <p className={`font-medium text-sm ${
                    isActive ? 'text-purple-700' :
                    status === 'completed' ? 'text-green-700' :
                    status === 'error' ? 'text-red-700' :
                    'text-gray-500'
                  }`}>
                    {language === 'ko' ? agent.nameKo : agent.name}
                  </p>
                  <p className="text-xs text-gray-400 truncate">
                    {isActive ? t.processing : language === 'ko' ? agent.descriptionKo : agent.description}
                  </p>
                  {summary && (
                    <p className="text-xs text-green-600 mt-0.5">
                      {Object.entries(summary).map(([k, v]) => `${k}: ${v}`).join(', ')}
                    </p>
                  )}
                </div>
                {isActive && (
                  <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                )}
              </div>
            );
          })}
        </div>

        {/* Error Message */}
        {errorMessage && (
          <div className="mx-6 mb-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm">
            {errorMessage}
          </div>
        )}

        {/* Actions */}
        {pipelineStatus === 'completed' && (
          <div className="p-6 pt-2 flex gap-3">
            <button
              onClick={handleViewReport}
              className="flex-1 py-3 px-4 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-medium flex items-center justify-center gap-2 hover:from-purple-700 hover:to-indigo-700 transition-all shadow-lg"
            >
              <ExternalLink className="w-4 h-4" />
              {t.viewReport}
            </button>
            <a
              href={`http://localhost:8000/api/rnaseq/report/${jobId}`}
              download={`rnaseq_report_${jobId}.html`}
              className="py-3 px-4 rounded-xl border border-gray-200 text-gray-700 font-medium flex items-center justify-center gap-2 hover:bg-gray-50 transition-all"
            >
              <Download className="w-4 h-4" />
            </a>
          </div>
        )}

        {pipelineStatus !== 'running' && (
          <div className="p-6 pt-0">
            <button
              onClick={onClose}
              className="w-full py-2.5 px-4 rounded-xl border border-gray-200 text-gray-600 font-medium hover:bg-gray-50 transition-all"
            >
              {t.close}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default PipelineProgress;
