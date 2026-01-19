/**
 * Chat Store
 * Global state for chat/RAG functionality
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { Source, ChatResponse } from '@/types/api.types';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  confidence?: number;
  timestamp: number;
}

interface ChatState {
  // Messages
  messages: Message[];

  // Current session
  sessionId: string | null;
  paperTitle: string | null;

  // UI state
  isLoading: boolean;
  error: string | null;

  // Actions
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void;
  setSession: (sessionId: string, paperTitle: string) => void;
  clearSession: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearMessages: () => void;
  reset: () => void;
}

const generateId = () => `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;

const initialState = {
  messages: [],
  sessionId: null,
  paperTitle: null,
  isLoading: false,
  error: null,
};

export const useChatStore = create<ChatState>()(
  devtools(
    (set, get) => ({
      ...initialState,

      addMessage: (message) => {
        const newMessage: Message = {
          ...message,
          id: generateId(),
          timestamp: Date.now(),
        };
        set((state) => ({
          messages: [...state.messages, newMessage],
          error: null,
        }));
      },

      setSession: (sessionId, paperTitle) =>
        set({
          sessionId,
          paperTitle,
          messages: [],
          error: null,
        }),

      clearSession: () =>
        set({
          sessionId: null,
          paperTitle: null,
          messages: [],
        }),

      setLoading: (isLoading) => set({ isLoading }),

      setError: (error) => set({ error, isLoading: false }),

      clearMessages: () => set({ messages: [] }),

      reset: () => set(initialState),
    }),
    { name: 'ChatStore' }
  )
);
