/**
 * Chat API
 * RAG-based Q&A, paper summarization, and agent sessions
 */

import { httpClient } from './http-client';
import type {
  ChatResponse,
  AgentChatResponse,
  AgentSession,
  AgentUploadResponse,
  PaperSummaryResponse,
  PaperInsightsResponse,
  BottomLineResponse,
  QualityResponse,
  TranslationResponse,
  AbstractSummaryResponse,
  AbstractAskResponse,
  Source,
} from '@/types/api.types';

export interface AskOptions {
  section?: string;
  topK?: number;
}

export const chatApi = {
  /**
   * Ask a question using RAG
   */
  ask: (
    question: string,
    domain: string = 'pheochromocytoma',
    options?: AskOptions
  ): Promise<ChatResponse> => {
    return httpClient.post('/chat/ask', {
      question,
      domain,
      section: options?.section,
      top_k: options?.topK || 5,
    });
  },

  /**
   * Summarize a paper
   */
  summarize: (
    title: string,
    domain: string = 'pheochromocytoma',
    brief: boolean = false
  ): Promise<PaperSummaryResponse> => {
    return httpClient.post('/chat/summarize', { title, domain, brief });
  },

  /**
   * Summarize a paper from its abstract (for papers not in local DB)
   */
  summarizeAbstract: (
    title: string,
    abstract: string,
    language: string = 'en'
  ): Promise<AbstractSummaryResponse> => {
    return httpClient.post('/chat/summarize-abstract', { title, abstract, language });
  },

  /**
   * Ask a question about a paper based on its abstract
   */
  askAbstract: (
    title: string,
    abstract: string,
    question: string
  ): Promise<AbstractAskResponse> => {
    return httpClient.post('/chat/ask-abstract', { title, abstract, question });
  },

  /**
   * Translate Korean search query to English
   */
  translateQuery: (text: string): Promise<TranslationResponse> => {
    return httpClient.post('/chat/translate', { text });
  },

  /**
   * Translate text to target language
   */
  translateText: async (text: string, targetLang: 'ko' | 'en'): Promise<string | null> => {
    try {
      const data = await httpClient.post<{ translated?: string }>('/chat/translate', {
        text,
        target_lang: targetLang,
      });
      return data.translated || null;
    } catch {
      return null;
    }
  },

  /**
   * Check if text contains Korean characters
   */
  containsKorean: (text: string): boolean => {
    return /[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]/.test(text);
  },

  // ============== Paper Insights ==============

  /**
   * Get comprehensive paper insights
   */
  getPaperInsights: (
    title: string,
    abstract: string,
    fullText?: string
  ): Promise<PaperInsightsResponse> => {
    return httpClient.post('/chat/insights', {
      title,
      abstract,
      full_text: fullText,
    });
  },

  /**
   * Get just the bottom line summary (fastest)
   */
  getBottomLine: (title: string, abstract: string): Promise<BottomLineResponse> => {
    return httpClient.post('/chat/insights/bottom-line', { title, abstract });
  },

  /**
   * Get quality assessment (fast, rule-based)
   */
  getQualityScore: (
    title: string,
    abstract: string,
    fullText?: string
  ): Promise<QualityResponse> => {
    return httpClient.post('/chat/insights/quality', {
      title,
      abstract,
      full_text: fullText,
    });
  },

  // ============== Paper Agent (Session-based) ==============

  /**
   * Upload a PDF to create a dedicated chat agent
   */
  uploadToAgent: (file: File): Promise<AgentUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    return httpClient.postForm('/chat/agent/upload', formData);
  },

  /**
   * Ask a question to a paper-specific agent
   */
  askAgent: (
    sessionId: string,
    question: string,
    topK: number = 5
  ): Promise<AgentChatResponse> => {
    return httpClient.post('/chat/agent/ask', {
      session_id: sessionId,
      question,
      top_k: topK,
    });
  },

  /**
   * Get agent session info
   */
  getAgentSession: (sessionId: string): Promise<AgentSession> => {
    return httpClient.get(`/chat/agent/session/${sessionId}`);
  },

  /**
   * Delete agent session
   */
  deleteAgentSession: (sessionId: string): Promise<void> => {
    return httpClient.delete(`/chat/agent/session/${sessionId}`);
  },
};

export default chatApi;
