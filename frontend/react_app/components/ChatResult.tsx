import React from 'react';
import { Sparkles, X } from 'lucide-react';
import type { ChatResponse } from '../services/client';

interface ChatResultProps {
  response: ChatResponse;
  onClose: () => void;
}

export const ChatResult: React.FC<ChatResultProps> = ({ response, onClose }) => (
  <div className="absolute top-full left-0 right-0 mt-3 glass-4 rounded-2xl shadow-2xl border border-purple-200/50 max-h-[500px] overflow-y-auto z-50 animate-appear">
    <div className="sticky top-0 glass-5 border-b border-purple-100/50 px-5 py-4 flex justify-between items-center">
      <div className="flex items-center gap-2">
        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
          <Sparkles className="w-3 h-3 text-white" />
        </div>
        <span className="text-sm font-semibold text-gray-700">AI Answer</span>
      </div>
      <button onClick={onClose} className="p-1.5 hover:bg-purple-100/50 rounded-full transition-colors">
        <X className="w-4 h-4 text-gray-500" />
      </button>
    </div>
    <div className="p-5">
      <div className="prose prose-sm max-w-none">
        <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">{response.answer}</p>
      </div>
      {response.sources.length > 0 && (
        <div className="mt-5 pt-5 border-t border-purple-100/50">
          <h5 className="text-xs font-bold text-purple-600 uppercase tracking-wider mb-3 flex items-center gap-2">
            <span className="w-4 h-px bg-purple-300"></span>
            Sources
          </h5>
          {response.sources.map((src, idx) => (
            <div key={idx} className="text-sm text-gray-700 mb-2 flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-2 flex-shrink-0"></span>
              <div>
                <span className="font-medium">{src.paper_title}</span>
                <span className="text-gray-400 ml-2 text-xs">({src.section})</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  </div>
);
