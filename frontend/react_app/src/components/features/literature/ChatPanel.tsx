import React from 'react';
import { MessageSquare, Send, Loader2 } from 'lucide-react';
import type { ChatMessage } from '../hooks/useChat';

interface ChatPanelProps {
  chatHistory: ChatMessage[];
  chatQuestion: string;
  onQuestionChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  chatEndRef: React.RefObject<HTMLDivElement>;
  placeholder?: string;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  chatHistory,
  chatQuestion,
  onQuestionChange,
  onSubmit,
  isLoading,
  chatEndRef,
  placeholder = "Ask about this paper..."
}) => {
  return (
    <div className="space-y-3">
      <h4 className="text-xs font-bold text-blue-600 uppercase tracking-wider flex items-center gap-2">
        <MessageSquare className="w-3.5 h-3.5" />
        Ask AI
      </h4>

      {/* Chat History */}
      {chatHistory.length > 0 && (
        <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
          {chatHistory.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-br-sm'
                    : 'glass-2 border border-blue-100/50 text-gray-700 rounded-bl-sm'
                }`}
              >
                <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="glass-2 border border-blue-100/50 rounded-2xl rounded-bl-sm px-4 py-2.5">
                <div className="flex items-center gap-2 text-gray-500">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
      )}

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={chatQuestion}
          onChange={(e) => onQuestionChange(e.target.value)}
          placeholder={placeholder}
          className="flex-1 px-4 py-2.5 glass-3 border border-purple-200/50 rounded-xl text-sm focus:ring-2 focus:ring-purple-400/50"
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && onSubmit()}
          disabled={isLoading}
        />
        <button
          onClick={onSubmit}
          disabled={isLoading || !chatQuestion.trim()}
          className="px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-all disabled:opacity-50"
        >
          {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
};
