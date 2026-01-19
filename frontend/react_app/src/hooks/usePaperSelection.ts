import { useState } from 'react';

export function usePaperSelection<T>() {
  const [selectedPaper, setSelectedPaper] = useState<T | null>(null);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [keyPoints, setKeyPoints] = useState<string[]>([]);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);

  const selectPaper = (paper: T) => {
    setSelectedPaper(paper);
    setAiSummary(null);
    setKeyPoints([]);
    setIsLoadingSummary(true);
  };

  const setSummary = (summary: string, points: string[] = []) => {
    setAiSummary(summary);
    setKeyPoints(points);
    setIsLoadingSummary(false);
  };

  const clearSelection = () => {
    setSelectedPaper(null);
    setAiSummary(null);
    setKeyPoints([]);
    setIsLoadingSummary(false);
  };

  return {
    selectedPaper,
    setSelectedPaper,
    aiSummary,
    setAiSummary,
    keyPoints,
    setKeyPoints,
    isLoadingSummary,
    setIsLoadingSummary,
    selectPaper,
    setSummary,
    clearSelection,
  };
}
