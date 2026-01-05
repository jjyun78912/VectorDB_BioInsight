import { useState, useRef, useEffect } from 'react';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function useChat() {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when chat history updates
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const sendMessage = async (handler: (q: string) => Promise<string>) => {
    if (!question.trim()) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: question,
      timestamp: new Date(),
    };

    setChatHistory(prev => [...prev, userMessage]);
    const currentQuestion = question;
    setQuestion('');
    setIsLoading(true);

    try {
      const answerContent = await handler(currentQuestion);
      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: answerContent,
        timestamp: new Date(),
      };
      setChatHistory(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: 'Sorry, I could not process your question.',
        timestamp: new Date(),
      };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const addSystemMessage = (content: string) => {
    const systemMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'assistant',
      content,
      timestamp: new Date(),
    };
    setChatHistory([systemMessage]);
  };

  const clearHistory = () => {
    setChatHistory([]);
    setQuestion('');
  };

  return {
    chatHistory,
    question,
    setQuestion,
    isLoading,
    sendMessage,
    addSystemMessage,
    clearHistory,
    chatEndRef,
  };
}
