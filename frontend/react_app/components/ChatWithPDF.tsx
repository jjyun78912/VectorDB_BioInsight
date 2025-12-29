import React, { useState, useRef, useEffect } from 'react';
import {
  Upload, FileText, Send, X, Sparkles, Loader2, Copy, Check,
  ChevronDown, Trash2, Plus, MessageSquare, BookOpen, Lightbulb
} from 'lucide-react';
import api from '../services/client';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: { paper_title: string; section: string; excerpt: string }[];
  timestamp: Date;
}

interface UploadedPaper {
  id: string;
  name: string;
  title: string;
  chunks: number;
  uploadedAt: Date;
  sessionId: string;  // Agent session ID for this paper
}

interface ChatWithPDFProps {
  isOpen: boolean;
  onClose: () => void;
}

const SUGGESTED_QUESTIONS = [
  "What are the main findings of this paper?",
  "What methodology was used in this study?",
  "What are the limitations mentioned?",
  "Summarize the key conclusions",
  "What future research is suggested?"
];

export const ChatWithPDF: React.FC<ChatWithPDFProps> = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedPapers, setUploadedPapers] = useState<UploadedPaper[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedPaper, setSelectedPaper] = useState<UploadedPaper | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle file upload - Using Agent API
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      // Use the new Agent API for paper-specific chat
      const result = await api.uploadToAgent(file);
      if (result.success) {
        const newPaper: UploadedPaper = {
          id: `paper-${Date.now()}`,
          name: file.name,
          title: result.paper_title || file.name,
          chunks: result.chunks_indexed,
          uploadedAt: new Date(),
          sessionId: result.session_id,  // Store the agent session ID
        };
        setUploadedPapers([...uploadedPapers, newPaper]);
        setSelectedPaper(newPaper);

        // Add welcome message
        setMessages([{
          id: `msg-${Date.now()}`,
          role: 'assistant',
          content: `I've processed "${result.paper_title}". The paper has been indexed into ${result.chunks_indexed} searchable sections. You can now ask me any questions about this paper!`,
          timestamp: new Date(),
        }]);
        setShowSuggestions(true);
      } else {
        // Handle upload failure
        setMessages([{
          id: `msg-${Date.now()}`,
          role: 'assistant',
          content: `Sorry, I couldn't process this PDF. ${result.message}`,
          timestamp: new Date(),
        }]);
      }
    } catch (err) {
      console.error('Upload failed:', err);
      setMessages([{
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, there was an error uploading the PDF. Please try again.',
        timestamp: new Date(),
      }]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Send message - Using Agent API
  const handleSend = async (question?: string) => {
    const messageText = question || input.trim();
    if (!messageText || !selectedPaper) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: messageText,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setShowSuggestions(false);

    try {
      // Use the Agent API with paper's session ID
      const response = await api.askAgent(selectedPaper.sessionId, messageText);

      // Build sources with proper formatting
      const sources = response.sources?.map(s => ({
        paper_title: s.paper_title,
        section: s.section,
        excerpt: s.excerpt,
      }));

      const assistantMessage: Message = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: response.answer,
        sources: sources,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: Message = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your question. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Copy message
  const copyMessage = (id: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Clear chat
  const clearChat = () => {
    setMessages([]);
    setShowSuggestions(true);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] bg-gradient-to-br from-violet-50 via-purple-50 to-indigo-100 flex">
      {/* Sidebar - Papers List */}
      <div className="w-72 glass-4 border-r border-purple-100/50 flex flex-col">
        <div className="p-4 border-b border-purple-100/50">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
                <MessageSquare className="w-4 h-4 text-white" />
              </div>
              <span className="font-bold text-gray-900">Chat with PDF</span>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-purple-100/50 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>

          {/* Upload Button */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full px-4 py-3 glass-3 border border-dashed border-purple-300 rounded-xl text-sm font-medium text-purple-600 hover:bg-purple-50/50 transition-colors flex items-center justify-center gap-2"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Upload PDF
              </>
            )}
          </button>
        </div>

        {/* Papers List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Your Papers</p>
          {uploadedPapers.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="w-10 h-10 text-gray-300 mx-auto mb-2" />
              <p className="text-sm text-gray-500">No papers uploaded yet</p>
            </div>
          ) : (
            uploadedPapers.map((paper) => (
              <button
                key={paper.id}
                onClick={() => setSelectedPaper(paper)}
                className={`w-full text-left p-3 rounded-xl transition-all ${
                  selectedPaper?.id === paper.id
                    ? 'bg-gradient-to-r from-violet-500/10 to-purple-500/10 border border-purple-300/50'
                    : 'glass-2 border border-transparent hover:border-purple-200/50'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                    selectedPaper?.id === paper.id
                      ? 'bg-gradient-to-br from-violet-500 to-purple-600'
                      : 'bg-purple-100'
                  }`}>
                    <FileText className={`w-4 h-4 ${selectedPaper?.id === paper.id ? 'text-white' : 'text-purple-600'}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">{paper.title}</p>
                    <p className="text-xs text-gray-500">{paper.chunks} chunks indexed</p>
                  </div>
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Chat Header */}
        <header className="glass-3 border-b border-purple-100/50 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {selectedPaper ? (
              <>
                <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center">
                  <FileText className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="font-semibold text-gray-900 line-clamp-1">{selectedPaper.title}</h2>
                  <p className="text-xs text-gray-500">{selectedPaper.chunks} sections available</p>
                </div>
              </>
            ) : (
              <h2 className="font-semibold text-gray-900">Select or upload a paper to start</h2>
            )}
          </div>
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="px-3 py-1.5 text-sm text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors flex items-center gap-1"
            >
              <Trash2 className="w-4 h-4" />
              Clear chat
            </button>
          )}
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && !selectedPaper && (
            <div className="flex flex-col items-center justify-center h-full gap-6">
              <div className="w-24 h-24 glass-3 rounded-3xl flex items-center justify-center">
                <BookOpen className="w-12 h-12 text-purple-400" />
              </div>
              <div className="text-center">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Chat with your research papers</h3>
                <p className="text-gray-500 max-w-md">
                  Upload a PDF and ask questions to get instant, accurate answers with citations from the paper
                </p>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-4 animate-appear ${message.role === 'user' ? 'justify-end' : ''}`}
            >
              {message.role === 'assistant' && (
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
              )}
              <div className={`max-w-2xl ${message.role === 'user' ? 'order-first' : ''}`}>
                <div
                  className={`rounded-2xl p-4 ${
                    message.role === 'user'
                      ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-tr-sm'
                      : 'glass-3 border border-purple-100/50 text-gray-800 rounded-tl-sm'
                  }`}
                >
                  <p className="leading-relaxed whitespace-pre-wrap">{message.content}</p>
                </div>

                {/* Sources */}
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 space-y-2">
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1">
                      <BookOpen className="w-3 h-3" />
                      Sources
                    </p>
                    {message.sources.map((source, idx) => (
                      <div key={idx} className="glass-2 rounded-lg p-3 border border-purple-100/50">
                        <p className="text-xs font-medium text-purple-600 mb-1">{source.section}</p>
                        <p className="text-sm text-gray-600 line-clamp-2">{source.excerpt}</p>
                      </div>
                    ))}
                  </div>
                )}

                {/* Actions */}
                <div className="mt-2 flex items-center gap-2">
                  <button
                    onClick={() => copyMessage(message.id, message.content)}
                    className="p-1.5 text-gray-400 hover:text-purple-600 rounded transition-colors"
                  >
                    {copiedId === message.id ? (
                      <Check className="w-4 h-4 text-emerald-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                  <span className="text-xs text-gray-400">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              </div>
              {message.role === 'user' && (
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center flex-shrink-0">
                  <span className="text-white font-semibold text-sm">You</span>
                </div>
              )}
            </div>
          ))}

          {/* Suggested Questions */}
          {showSuggestions && selectedPaper && messages.length <= 1 && (
            <div className="pt-4">
              <p className="text-sm font-medium text-gray-600 mb-3 flex items-center gap-2">
                <Lightbulb className="w-4 h-4 text-yellow-500" />
                Suggested questions
              </p>
              <div className="flex flex-wrap gap-2">
                {SUGGESTED_QUESTIONS.map((question, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(question)}
                    className="px-4 py-2 glass-3 border border-purple-200/50 rounded-full text-sm text-gray-700 hover:bg-purple-50/50 hover:border-purple-300 transition-all"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Loading indicator */}
          {isLoading && (
            <div className="flex gap-4 animate-appear">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div className="glass-3 border border-purple-100/50 rounded-2xl rounded-tl-sm p-4">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-purple-500 animate-spin" />
                  <span className="text-gray-600">Analyzing paper...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="glass-3 border-t border-purple-100/50 p-4">
          <form
            onSubmit={(e) => { e.preventDefault(); handleSend(); }}
            className="max-w-4xl mx-auto relative"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={selectedPaper ? "Ask a question about this paper..." : "Upload a paper first..."}
              disabled={!selectedPaper || isLoading}
              className="w-full pl-4 pr-14 py-4 glass-4 border border-purple-200/50 rounded-xl text-gray-900 placeholder:text-gray-400 focus:ring-2 focus:ring-purple-400/50 focus:border-purple-300 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!input.trim() || !selectedPaper || isLoading}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-3 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-lg hover:from-violet-700 hover:to-purple-700 transition-all disabled:opacity-50"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
