/**
 * Chat Query Hooks
 * React Query hooks for RAG chat functionality
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { chatApi } from '@/services/api';
import { useChatStore } from '@/stores';

// Query keys
export const chatKeys = {
  all: ['chat'] as const,
  session: (sessionId: string) => [...chatKeys.all, 'session', sessionId] as const,
  insights: (title: string) => [...chatKeys.all, 'insights', title] as const,
  bottomLine: (title: string) => [...chatKeys.all, 'bottomLine', title] as const,
};

/**
 * Mutation hook for asking questions (RAG)
 */
export function useAskQuestion() {
  const { addMessage, setLoading, setError } = useChatStore();

  return useMutation({
    mutationFn: ({
      question,
      domain,
      options,
    }: {
      question: string;
      domain?: string;
      options?: { section?: string; topK?: number };
    }) => chatApi.ask(question, domain, options),
    onMutate: ({ question }) => {
      addMessage({ role: 'user', content: question });
      setLoading(true);
    },
    onSuccess: (data) => {
      addMessage({
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        confidence: data.confidence,
      });
      setLoading(false);
    },
    onError: (error) => {
      const message = error instanceof Error ? error.message : 'Failed to get answer';
      setError(message);
    },
  });
}

/**
 * Mutation hook for asking agent (session-based)
 */
export function useAskAgent() {
  const { addMessage, setLoading, setError } = useChatStore();

  return useMutation({
    mutationFn: ({
      sessionId,
      question,
      topK,
    }: {
      sessionId: string;
      question: string;
      topK?: number;
    }) => chatApi.askAgent(sessionId, question, topK),
    onMutate: ({ question }) => {
      addMessage({ role: 'user', content: question });
      setLoading(true);
    },
    onSuccess: (data) => {
      addMessage({
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        confidence: data.confidence,
      });
      setLoading(false);
    },
    onError: (error) => {
      const message = error instanceof Error ? error.message : 'Failed to get answer';
      setError(message);
    },
  });
}

/**
 * Mutation hook for uploading PDF to agent
 */
export function useUploadToAgent() {
  const { setSession, setLoading, setError } = useChatStore();

  return useMutation({
    mutationFn: (file: File) => chatApi.uploadToAgent(file),
    onMutate: () => {
      setLoading(true);
    },
    onSuccess: (data) => {
      setSession(data.session_id, data.paper_title);
      setLoading(false);
    },
    onError: (error) => {
      const message = error instanceof Error ? error.message : 'Failed to upload PDF';
      setError(message);
    },
  });
}

/**
 * Hook for getting agent session info
 */
export function useAgentSession(sessionId: string, enabled = true) {
  return useQuery({
    queryKey: chatKeys.session(sessionId),
    queryFn: () => chatApi.getAgentSession(sessionId),
    enabled: enabled && !!sessionId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Mutation hook for deleting agent session
 */
export function useDeleteAgentSession() {
  const queryClient = useQueryClient();
  const { clearSession } = useChatStore();

  return useMutation({
    mutationFn: (sessionId: string) => chatApi.deleteAgentSession(sessionId),
    onSuccess: (_, sessionId) => {
      queryClient.invalidateQueries({ queryKey: chatKeys.session(sessionId) });
      clearSession();
    },
  });
}

/**
 * Mutation hook for summarizing abstract
 */
export function useSummarizeAbstract() {
  return useMutation({
    mutationFn: ({
      title,
      abstract,
      language,
    }: {
      title: string;
      abstract: string;
      language?: string;
    }) => chatApi.summarizeAbstract(title, abstract, language),
  });
}

/**
 * Mutation hook for asking about abstract
 */
export function useAskAbstract() {
  return useMutation({
    mutationFn: ({
      title,
      abstract,
      question,
    }: {
      title: string;
      abstract: string;
      question: string;
    }) => chatApi.askAbstract(title, abstract, question),
  });
}

/**
 * Hook for getting paper insights
 */
export function usePaperInsights(title: string, abstract: string, fullText?: string, enabled = true) {
  return useQuery({
    queryKey: chatKeys.insights(title),
    queryFn: () => chatApi.getPaperInsights(title, abstract, fullText),
    enabled: enabled && title.length > 0 && abstract.length > 0,
    staleTime: 30 * 60 * 1000, // 30 minutes
  });
}

/**
 * Mutation hook for getting paper insights (imperative)
 */
export function useGetPaperInsights() {
  return useMutation({
    mutationFn: ({
      title,
      abstract,
      fullText,
    }: {
      title: string;
      abstract: string;
      fullText?: string;
    }) => chatApi.getPaperInsights(title, abstract, fullText),
  });
}

/**
 * Hook for getting bottom line only
 */
export function useBottomLine(title: string, abstract: string, enabled = true) {
  return useQuery({
    queryKey: chatKeys.bottomLine(title),
    queryFn: () => chatApi.getBottomLine(title, abstract),
    enabled: enabled && title.length > 0 && abstract.length > 0,
    staleTime: 30 * 60 * 1000,
  });
}

/**
 * Mutation hook for getting quality score
 */
export function useGetQualityScore() {
  return useMutation({
    mutationFn: ({
      title,
      abstract,
      fullText,
    }: {
      title: string;
      abstract: string;
      fullText?: string;
    }) => chatApi.getQualityScore(title, abstract, fullText),
  });
}
