import React, { useState, useRef, useCallback, useEffect } from 'react';
import { X, Upload, FileSpreadsheet, Dna, ArrowRight, Loader2, AlertCircle, CheckCircle2, Info, Users, ChevronDown } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

// Cancer type options
const CANCER_TYPES = [
  { value: 'breast_cancer', label: 'Breast Cancer', labelKo: '유방암' },
  { value: 'lung_cancer', label: 'Lung Cancer', labelKo: '폐암' },
  { value: 'colorectal_cancer', label: 'Colorectal Cancer', labelKo: '대장암' },
  { value: 'pancreatic_cancer', label: 'Pancreatic Cancer', labelKo: '췌장암' },
  { value: 'liver_cancer', label: 'Liver Cancer', labelKo: '간암' },
  { value: 'glioblastoma', label: 'Glioblastoma', labelKo: '교모세포종' },
  { value: 'blood_cancer', label: 'Blood Cancer', labelKo: '혈액암' },
  { value: 'unknown', label: 'Other / Unknown', labelKo: '기타 / 미상' },
];

interface SampleInfo {
  sample_id: string;
  condition: string;
}

interface RNAseqUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAnalysisStart: (jobId: string) => void;
  initialFile?: File | null;
}

export const RNAseqUploadModal: React.FC<RNAseqUploadModalProps> = ({
  isOpen,
  onClose,
  onAnalysisStart,
  initialFile
}) => {
  const { language } = useLanguage();
  const [countMatrixFile, setCountMatrixFile] = useState<File | null>(null);
  const [metadataFile, setMetadataFile] = useState<File | null>(null);
  const [cancerType, setCancerType] = useState('unknown');
  const [studyName, setStudyName] = useState('');
  const [conditionColumn, setConditionColumn] = useState('condition');
  const [treatmentLabel, setTreatmentLabel] = useState('tumor');
  const [controlLabel, setControlLabel] = useState('normal');
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Auto-metadata state
  const [useAutoMetadata, setUseAutoMetadata] = useState(true);
  const [samples, setSamples] = useState<SampleInfo[]>([]);
  const [isLoadingSamples, setIsLoadingSamples] = useState(false);
  const [geneCount, setGeneCount] = useState(0);

  const countInputRef = useRef<HTMLInputElement>(null);
  const metadataInputRef = useRef<HTMLInputElement>(null);

  // Auto-load initial file when provided
  useEffect(() => {
    if (initialFile && isOpen) {
      setCountMatrixFile(initialFile);
      if (useAutoMetadata) {
        previewSamples(initialFile);
      }
    }
  }, [initialFile, isOpen]);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setCountMatrixFile(null);
      setMetadataFile(null);
      setSamples([]);
      setGeneCount(0);
      setError(null);
      setCancerType('unknown');
      setStudyName('');
    }
  }, [isOpen]);

  // Preview samples from count matrix
  const previewSamples = async (file: File) => {
    // Check file size (max 50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      setError(language === 'ko'
        ? `파일이 너무 큽니다 (${(file.size / 1024 / 1024).toFixed(1)}MB). 최대 50MB까지 지원됩니다.`
        : `File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum 50MB allowed.`
      );
      return;
    }

    setIsLoadingSamples(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('count_matrix', file);

      // Use AbortController for timeout (180 seconds for large files)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 180000);

      const response = await fetch('http://localhost:8000/api/rnaseq/preview-samples', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Preview failed');
      }

      const result = await response.json();
      setSamples(result.samples);
      setGeneCount(result.gene_count);
      setTreatmentLabel(result.suggested_treatment);
      setControlLabel(result.suggested_control);

    } catch (err) {
      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError(language === 'ko'
            ? '파일 처리 시간이 초과되었습니다. 더 작은 파일을 시도해주세요.'
            : 'File processing timed out. Please try a smaller file.'
          );
        } else if (err.message === 'Failed to fetch') {
          setError(language === 'ko'
            ? '서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인하세요. (http://localhost:8000)'
            : 'Cannot connect to server. Please check if backend is running. (http://localhost:8000)'
          );
        } else {
          setError(err.message);
        }
      } else {
        setError('Failed to preview samples');
      }
      setSamples([]);
    } finally {
      setIsLoadingSamples(false);
    }
  };

  // Update sample condition
  const updateSampleCondition = (sampleId: string, condition: string) => {
    setSamples(prev => prev.map(s =>
      s.sample_id === sampleId ? { ...s, condition } : s
    ));
  };

  // Set all samples to a condition
  const setAllSamplesCondition = (condition: string) => {
    setSamples(prev => prev.map(s => ({ ...s, condition })));
  };

  const handleDrop = useCallback((e: React.DragEvent, type: 'count' | 'metadata') => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.csv') || file.name.endsWith('.tsv') || file.name.endsWith('.txt'))) {
      if (type === 'count') {
        setCountMatrixFile(file);
        if (useAutoMetadata) {
          previewSamples(file);
        }
      } else {
        setMetadataFile(file);
      }
    }
  }, [useAutoMetadata]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleUpload = async () => {
    // Check required files based on mode
    if (!countMatrixFile) {
      setError(language === 'ko' ? 'Count Matrix 파일을 업로드해주세요.' : 'Please upload count matrix file.');
      return;
    }

    if (useAutoMetadata) {
      // Validate sample conditions
      const treatmentCount = samples.filter(s => s.condition === 'treatment').length;
      const controlCount = samples.filter(s => s.condition === 'control').length;

      if (treatmentCount === 0 || controlCount === 0) {
        setError(language === 'ko'
          ? `Treatment(${treatmentCount})과 Control(${controlCount}) 샘플이 각각 1개 이상 필요합니다.`
          : `Need at least 1 treatment (${treatmentCount}) and 1 control (${controlCount}) sample.`
        );
        return;
      }
    } else if (!metadataFile) {
      setError(language === 'ko' ? '메타데이터 파일을 업로드해주세요.' : 'Please upload metadata file.');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      let response: Response;

      if (useAutoMetadata && samples.length > 0) {
        // Use auto-metadata endpoint
        const formData = new FormData();
        formData.append('count_matrix', countMatrixFile);

        // Build sample conditions map
        const sampleConditions: Record<string, string> = {};
        samples.forEach(s => {
          sampleConditions[s.sample_id] = s.condition;
        });
        formData.append('sample_conditions', JSON.stringify(sampleConditions));
        formData.append('cancer_type', cancerType);
        formData.append('study_name', studyName || `RNA-seq Analysis`);
        formData.append('treatment_label', treatmentLabel);
        formData.append('control_label', controlLabel);

        response = await fetch('http://localhost:8000/api/rnaseq/upload-with-auto-metadata', {
          method: 'POST',
          body: formData
        });
      } else {
        // Use traditional upload with metadata file
        const formData = new FormData();
        formData.append('count_matrix', countMatrixFile);
        formData.append('metadata', metadataFile!);
        formData.append('cancer_type', cancerType);
        formData.append('study_name', studyName || `RNA-seq Analysis`);
        formData.append('condition_column', conditionColumn);
        formData.append('treatment_label', treatmentLabel);
        formData.append('control_label', controlLabel);

        response = await fetch('http://localhost:8000/api/rnaseq/upload', {
          method: 'POST',
          body: formData
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();

      // Start analysis
      const startResponse = await fetch(`http://localhost:8000/api/rnaseq/start/${result.job_id}`, {
        method: 'POST'
      });

      if (!startResponse.ok) {
        throw new Error('Failed to start analysis');
      }

      onAnalysisStart(result.job_id);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  if (!isOpen) return null;

  const t = {
    title: language === 'ko' ? 'RNA-seq 데이터 분석' : 'RNA-seq Data Analysis',
    subtitle: language === 'ko' ? '6-Agent 파이프라인으로 유전자 발현 데이터 분석' : 'Analyze gene expression data with 6-Agent pipeline',
    countMatrix: language === 'ko' ? 'Count Matrix' : 'Count Matrix',
    countMatrixDesc: language === 'ko' ? 'gene_id 열 + 샘플별 raw count' : 'gene_id column + sample columns with raw counts',
    metadata: language === 'ko' ? '메타데이터' : 'Metadata',
    metadataDesc: language === 'ko' ? 'sample_id, condition (tumor/normal) 열 필수' : 'Required: sample_id, condition (tumor/normal) columns',
    cancerType: language === 'ko' ? '암 종류' : 'Cancer Type',
    studyName: language === 'ko' ? '연구명 (선택)' : 'Study Name (Optional)',
    studyNamePlaceholder: language === 'ko' ? '예: GSE12345 폐암 RNA-seq' : 'e.g., GSE12345 Lung Cancer RNA-seq',
    advancedOptions: language === 'ko' ? '고급 옵션' : 'Advanced Options',
    conditionColumn: language === 'ko' ? 'Condition 열 이름' : 'Condition Column Name',
    treatmentLabel: language === 'ko' ? 'Treatment 라벨' : 'Treatment Label',
    controlLabel: language === 'ko' ? 'Control 라벨' : 'Control Label',
    startAnalysis: language === 'ko' ? '분석 시작' : 'Start Analysis',
    uploading: language === 'ko' ? '업로드 중...' : 'Uploading...',
    dragDrop: language === 'ko' ? '파일을 드래그하거나 클릭하여 선택' : 'Drag & drop or click to select',
    fileSelected: language === 'ko' ? '파일 선택됨' : 'File selected',
    autoMetadata: language === 'ko' ? '샘플 조건 자동 설정' : 'Auto-detect sample conditions',
    manualMetadata: language === 'ko' ? '메타데이터 파일 업로드' : 'Upload metadata file',
    sampleConditions: language === 'ko' ? '샘플 조건 선택' : 'Select Sample Conditions',
    treatment: language === 'ko' ? 'Treatment (Tumor)' : 'Treatment (Tumor)',
    control: language === 'ko' ? 'Control (Normal)' : 'Control (Normal)',
    unknown: language === 'ko' ? '미정' : 'Unknown',
    setAllTreatment: language === 'ko' ? '전체 Treatment' : 'All Treatment',
    setAllControl: language === 'ko' ? '전체 Control' : 'All Control',
    genesDetected: language === 'ko' ? '유전자 감지됨' : 'genes detected',
    samplesDetected: language === 'ko' ? '샘플 감지됨' : 'samples detected',
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-appear">
      <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto glass-4 rounded-3xl shadow-2xl border border-purple-200/50">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 rounded-full hover:bg-gray-100 transition-colors z-10"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>

        {/* Header */}
        <div className="p-6 pb-4 border-b border-purple-100/50">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 text-white">
              <Dna className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-800">{t.title}</h2>
              <p className="text-sm text-gray-500">{t.subtitle}</p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-5">
          {/* Count Matrix Upload */}
          <div
            className={`relative p-4 rounded-xl border-2 border-dashed transition-all cursor-pointer hover:border-purple-400 hover:bg-purple-50/30 ${
              countMatrixFile ? 'border-green-400 bg-green-50/30' : 'border-gray-300'
            }`}
            onClick={() => countInputRef.current?.click()}
            onDrop={(e) => handleDrop(e, 'count')}
            onDragOver={handleDragOver}
          >
            <input
              ref={countInputRef}
              type="file"
              accept=".csv,.tsv,.txt"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  setCountMatrixFile(file);
                  if (useAutoMetadata) {
                    previewSamples(file);
                  }
                }
              }}
            />
            <div className="text-center">
              {countMatrixFile ? (
                <CheckCircle2 className="w-8 h-8 text-green-500 mx-auto mb-2" />
              ) : (
                <FileSpreadsheet className="w-8 h-8 text-purple-400 mx-auto mb-2" />
              )}
              <p className="font-medium text-gray-700">{t.countMatrix}</p>
              <p className="text-xs text-gray-500 mt-1">{t.countMatrixDesc}</p>
              {countMatrixFile ? (
                <div className="mt-2">
                  <p className="text-sm text-green-600 font-medium">{countMatrixFile.name}</p>
                  {geneCount > 0 && (
                    <p className="text-xs text-gray-500 mt-1">
                      {geneCount.toLocaleString()} {t.genesDetected} · {samples.length} {t.samplesDetected}
                    </p>
                  )}
                </div>
              ) : (
                <p className="text-xs text-gray-400 mt-2">{t.dragDrop}</p>
              )}
            </div>
          </div>

          {/* Metadata Mode Toggle */}
          <div className="flex items-center gap-4 p-3 rounded-xl bg-gray-50/50 border border-gray-100">
            <button
              type="button"
              onClick={() => setUseAutoMetadata(true)}
              className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                useAutoMetadata
                  ? 'bg-purple-600 text-white shadow-md'
                  : 'bg-white text-gray-600 hover:bg-gray-100'
              }`}
            >
              {t.autoMetadata}
            </button>
            <button
              type="button"
              onClick={() => setUseAutoMetadata(false)}
              className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                !useAutoMetadata
                  ? 'bg-purple-600 text-white shadow-md'
                  : 'bg-white text-gray-600 hover:bg-gray-100'
              }`}
            >
              {t.manualMetadata}
            </button>
          </div>

          {/* Sample Conditions Selection (Auto Mode) */}
          {useAutoMetadata && samples.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-gray-700 flex items-center gap-2">
                  <Users className="w-4 h-4" />
                  {t.sampleConditions}
                </p>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setAllSamplesCondition('treatment')}
                    className="px-2 py-1 text-xs rounded-md bg-red-100 text-red-700 hover:bg-red-200 transition-colors"
                  >
                    {t.setAllTreatment}
                  </button>
                  <button
                    type="button"
                    onClick={() => setAllSamplesCondition('control')}
                    className="px-2 py-1 text-xs rounded-md bg-blue-100 text-blue-700 hover:bg-blue-200 transition-colors"
                  >
                    {t.setAllControl}
                  </button>
                </div>
              </div>

              <div className="max-h-48 overflow-y-auto rounded-xl border border-gray-200 bg-white">
                {samples.map((sample, idx) => (
                  <div
                    key={sample.sample_id}
                    className={`flex items-center justify-between px-3 py-2 ${
                      idx !== samples.length - 1 ? 'border-b border-gray-100' : ''
                    }`}
                  >
                    <span className="text-sm text-gray-700 font-mono truncate max-w-[200px]" title={sample.sample_id}>
                      {sample.sample_id}
                    </span>
                    <select
                      value={sample.condition}
                      onChange={(e) => updateSampleCondition(sample.sample_id, e.target.value)}
                      className={`px-2 py-1 text-xs rounded-md border transition-colors ${
                        sample.condition === 'treatment'
                          ? 'bg-red-50 border-red-300 text-red-700'
                          : sample.condition === 'control'
                          ? 'bg-blue-50 border-blue-300 text-blue-700'
                          : 'bg-gray-50 border-gray-300 text-gray-600'
                      }`}
                    >
                      <option value="treatment">{t.treatment}</option>
                      <option value="control">{t.control}</option>
                      <option value="unknown">{t.unknown}</option>
                    </select>
                  </div>
                ))}
              </div>

              {/* Sample count summary */}
              <div className="flex justify-center gap-4 text-xs">
                <span className="px-2 py-1 rounded-full bg-red-100 text-red-700">
                  Treatment: {samples.filter(s => s.condition === 'treatment').length}
                </span>
                <span className="px-2 py-1 rounded-full bg-blue-100 text-blue-700">
                  Control: {samples.filter(s => s.condition === 'control').length}
                </span>
                {samples.filter(s => s.condition === 'unknown').length > 0 && (
                  <span className="px-2 py-1 rounded-full bg-gray-100 text-gray-600">
                    Unknown: {samples.filter(s => s.condition === 'unknown').length}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Loading samples indicator */}
          {isLoadingSamples && (
            <div className="flex flex-col items-center justify-center py-4 gap-2">
              <div className="flex items-center">
                <Loader2 className="w-5 h-5 animate-spin text-purple-500 mr-2" />
                <span className="text-sm text-gray-500">
                  {language === 'ko' ? '샘플 정보 분석 중...' : 'Analyzing samples...'}
                </span>
              </div>
              {countMatrixFile && (
                <span className="text-xs text-gray-400">
                  {(countMatrixFile.size / 1024 / 1024).toFixed(1)}MB - {language === 'ko' ? '큰 파일은 시간이 걸릴 수 있습니다' : 'Large files may take longer'}
                </span>
              )}
            </div>
          )}

          {/* Manual Metadata Upload */}
          {!useAutoMetadata && (
            <div
              className={`relative p-4 rounded-xl border-2 border-dashed transition-all cursor-pointer hover:border-purple-400 hover:bg-purple-50/30 ${
                metadataFile ? 'border-green-400 bg-green-50/30' : 'border-gray-300'
              }`}
              onClick={() => metadataInputRef.current?.click()}
              onDrop={(e) => handleDrop(e, 'metadata')}
              onDragOver={handleDragOver}
            >
              <input
                ref={metadataInputRef}
                type="file"
                accept=".csv,.tsv,.txt"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && setMetadataFile(e.target.files[0])}
              />
              <div className="text-center">
                {metadataFile ? (
                  <CheckCircle2 className="w-8 h-8 text-green-500 mx-auto mb-2" />
                ) : (
                  <FileSpreadsheet className="w-8 h-8 text-indigo-400 mx-auto mb-2" />
                )}
                <p className="font-medium text-gray-700">{t.metadata}</p>
                <p className="text-xs text-gray-500 mt-1">{t.metadataDesc}</p>
                {metadataFile ? (
                  <p className="text-sm text-green-600 mt-2 font-medium">{metadataFile.name}</p>
                ) : (
                  <p className="text-xs text-gray-400 mt-2">{t.dragDrop}</p>
                )}
              </div>
            </div>
          )}

          {/* Cancer Type & Study Name */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">{t.cancerType}</label>
              <select
                value={cancerType}
                onChange={(e) => setCancerType(e.target.value)}
                className="w-full px-3 py-2.5 rounded-xl border border-gray-200 bg-white/80 focus:ring-2 focus:ring-purple-400 focus:border-purple-400 text-sm"
              >
                {CANCER_TYPES.map((type) => (
                  <option key={type.value} value={type.value}>
                    {language === 'ko' ? type.labelKo : type.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">{t.studyName}</label>
              <input
                type="text"
                value={studyName}
                onChange={(e) => setStudyName(e.target.value)}
                placeholder={t.studyNamePlaceholder}
                className="w-full px-3 py-2.5 rounded-xl border border-gray-200 bg-white/80 focus:ring-2 focus:ring-purple-400 focus:border-purple-400 text-sm placeholder:text-gray-400"
              />
            </div>
          </div>

          {/* Advanced Options Toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-700 transition-colors"
          >
            <Info className="w-4 h-4" />
            {t.advancedOptions}
            <ArrowRight className={`w-3 h-3 transition-transform ${showAdvanced ? 'rotate-90' : ''}`} />
          </button>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="grid grid-cols-3 gap-3 p-4 rounded-xl bg-gray-50/50 border border-gray-100">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">{t.conditionColumn}</label>
                <input
                  type="text"
                  value={conditionColumn}
                  onChange={(e) => setConditionColumn(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-lg border border-gray-200 bg-white text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">{t.treatmentLabel}</label>
                <input
                  type="text"
                  value={treatmentLabel}
                  onChange={(e) => setTreatmentLabel(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-lg border border-gray-200 bg-white text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">{t.controlLabel}</label>
                <input
                  type="text"
                  value={controlLabel}
                  onChange={(e) => setControlLabel(e.target.value)}
                  className="w-full px-2 py-1.5 rounded-lg border border-gray-200 bg-white text-sm"
                />
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="flex items-center gap-2 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 pt-4 border-t border-purple-100/50">
          <button
            onClick={handleUpload}
            disabled={
              !countMatrixFile ||
              isUploading ||
              isLoadingSamples ||
              (useAutoMetadata && samples.length === 0) ||
              (!useAutoMetadata && !metadataFile)
            }
            className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-medium flex items-center justify-center gap-2 hover:from-purple-700 hover:to-indigo-700 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                {t.uploading}
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                {t.startAnalysis}
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default RNAseqUploadModal;
