import React, { useState, useRef, useCallback } from 'react';
import { X, Upload, FileSpreadsheet, Dna, ArrowRight, Loader2, AlertCircle, CheckCircle2, Info } from 'lucide-react';
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

interface RNAseqUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAnalysisStart: (jobId: string) => void;
}

export const RNAseqUploadModal: React.FC<RNAseqUploadModalProps> = ({
  isOpen,
  onClose,
  onAnalysisStart
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

  const countInputRef = useRef<HTMLInputElement>(null);
  const metadataInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent, type: 'count' | 'metadata') => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.csv') || file.name.endsWith('.tsv') || file.name.endsWith('.txt'))) {
      if (type === 'count') {
        setCountMatrixFile(file);
      } else {
        setMetadataFile(file);
      }
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleUpload = async () => {
    if (!countMatrixFile || !metadataFile) {
      setError(language === 'ko' ? '모든 파일을 업로드해주세요.' : 'Please upload all required files.');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('count_matrix', countMatrixFile);
      formData.append('metadata', metadataFile);
      formData.append('cancer_type', cancerType);
      formData.append('study_name', studyName || `RNA-seq Analysis`);
      formData.append('condition_column', conditionColumn);
      formData.append('treatment_label', treatmentLabel);
      formData.append('control_label', controlLabel);

      const response = await fetch('http://localhost:8000/api/rnaseq/upload', {
        method: 'POST',
        body: formData
      });

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
          {/* File Upload Areas */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                onChange={(e) => e.target.files?.[0] && setCountMatrixFile(e.target.files[0])}
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
                  <p className="text-sm text-green-600 mt-2 font-medium">{countMatrixFile.name}</p>
                ) : (
                  <p className="text-xs text-gray-400 mt-2">{t.dragDrop}</p>
                )}
              </div>
            </div>

            {/* Metadata Upload */}
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
          </div>

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
            disabled={!countMatrixFile || !metadataFile || isUploading}
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
