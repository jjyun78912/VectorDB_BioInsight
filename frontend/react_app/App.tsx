import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { FeatureSuite } from './components/FeatureSuite';
import { TrendingPapers } from './components/TrendingPapers';
import { CtaSection } from './components/CtaSection';
import { Footer } from './components/Footer';
import { KnowledgeGraph } from './components/KnowledgeGraph';
import { LiteratureReview } from './components/LiteratureReview';
import { ChatWithPDF } from './components/ChatWithPDF';
import { ResearchLibrary } from './components/ResearchLibrary';
import { DailyBriefing } from './components/DailyBriefing';
import { LanguageProvider } from './contexts/LanguageContext';
import { useLanguage } from './contexts/LanguageContext';
import {
  Network, Sparkles, ArrowRight, BookOpen, MessageSquare,
  Library, FileSearch, Zap, Newspaper
} from 'lucide-react';

// Paper type for Literature Review
export interface ReviewPaper {
  id: string;
  title: string;
  authors: string[];
  year: number;
  journal?: string;
  abstract?: string;
  doi?: string;
  pmid?: string;
  relevance?: number;
}

const AppContent: React.FC = () => {
  const [showGraph, setShowGraph] = useState(false);
  const [showLiteratureReview, setShowLiteratureReview] = useState(false);
  const [showChatWithPDF, setShowChatWithPDF] = useState(false);
  const [showLibrary, setShowLibrary] = useState(false);
  const [showBriefing, setShowBriefing] = useState(false);

  // Language context
  const { t } = useLanguage();

  // Papers for Literature Review (collected from Hero search)
  const [reviewPapers, setReviewPapers] = useState<ReviewPaper[]>([]);

  // Add paper to Literature Review
  const addToReview = (paper: ReviewPaper) => {
    setReviewPapers(prev => {
      // Avoid duplicates
      if (prev.some(p => p.id === paper.id || p.title === paper.title)) {
        return prev;
      }
      return [...prev, paper];
    });
  };

  // Add multiple papers to Literature Review
  const addMultipleToReview = (papers: ReviewPaper[]) => {
    setReviewPapers(prev => {
      const newPapers = papers.filter(p =>
        !prev.some(existing => existing.id === p.id || existing.title === p.title)
      );
      return [...prev, ...newPapers];
    });
    // Open Literature Review after adding
    setShowLiteratureReview(true);
  };

  // Remove paper from review
  const removeFromReview = (paperId: string) => {
    setReviewPapers(prev => prev.filter(p => p.id !== paperId));
  };

  // Clear all papers from review
  const clearReview = () => {
    setReviewPapers([]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50 via-purple-50 to-indigo-100 text-gray-900 selection:bg-purple-200">
      <Navbar />
      <main>
        <Hero
          onAddToReview={addToReview}
          onAddMultipleToReview={addMultipleToReview}
          reviewPapersCount={reviewPapers.length}
          onOpenReview={() => setShowLiteratureReview(true)}
        />
        <TrendingPapers />
        <FeatureSuite />

        {/* Quick Access Tools Section */}
        <section className="relative py-20 overflow-hidden">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[80%] h-[60%] bg-gradient-radial from-purple-200/30 via-violet-100/20 to-transparent blur-3xl pointer-events-none" />

          <div className="relative z-10 max-w-7xl mx-auto px-6">
            <div className="text-center mb-12">
              <span className="inline-flex items-center gap-2 px-3 py-1.5 glass-2 rounded-full border border-purple-200/50 text-sm font-medium text-purple-600 mb-4">
                <Zap className="w-3.5 h-3.5" />
                Research Tools
              </span>
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
                Powerful Tools at Your Fingertips
              </h2>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Access AI-powered research tools designed to accelerate your scientific discoveries
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
              {/* Literature Review */}
              <button
                onClick={() => setShowLiteratureReview(true)}
                className="group glass-3 rounded-2xl border border-purple-100/50 p-6 text-left card-hover"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <FileSearch className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-bold text-gray-900 mb-2">Literature Review</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Search and analyze papers with AI-generated summaries and table view
                </p>
                <span className="inline-flex items-center gap-1 text-sm font-medium text-purple-600 group-hover:gap-2 transition-all">
                  Open Tool <ArrowRight className="w-4 h-4" />
                </span>
              </button>

              {/* Chat with PDF */}
              <button
                onClick={() => setShowChatWithPDF(true)}
                className="group glass-3 rounded-2xl border border-purple-100/50 p-6 text-left card-hover"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <MessageSquare className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-bold text-gray-900 mb-2">Chat with PDF</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Upload papers and ask questions to get instant answers with citations
                </p>
                <span className="inline-flex items-center gap-1 text-sm font-medium text-emerald-600 group-hover:gap-2 transition-all">
                  Open Tool <ArrowRight className="w-4 h-4" />
                </span>
              </button>

              {/* Research Library */}
              <button
                onClick={() => setShowLibrary(true)}
                className="group glass-3 rounded-2xl border border-purple-100/50 p-6 text-left card-hover"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Library className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-bold text-gray-900 mb-2">Research Library</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Organize your papers into collections with tags and annotations
                </p>
                <span className="inline-flex items-center gap-1 text-sm font-medium text-orange-600 group-hover:gap-2 transition-all">
                  Open Tool <ArrowRight className="w-4 h-4" />
                </span>
              </button>

              {/* Knowledge Graph */}
              <button
                onClick={() => setShowGraph(true)}
                className="group glass-3 rounded-2xl border border-purple-100/50 p-6 text-left card-hover"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Network className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-bold text-gray-900 mb-2">Knowledge Graph</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Visualize connections between genes, diseases, and research in 3D
                </p>
                <span className="inline-flex items-center gap-1 text-sm font-medium text-indigo-600 group-hover:gap-2 transition-all">
                  Open Tool <ArrowRight className="w-4 h-4" />
                </span>
              </button>

              {/* Daily Briefing */}
              <button
                onClick={() => setShowBriefing(true)}
                className="group glass-3 rounded-2xl border border-purple-100/50 p-6 text-left card-hover"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-violet-600 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Newspaper className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-bold text-gray-900 mb-2">Daily Briefing</h3>
                <p className="text-sm text-gray-600 mb-4">
                  AI-curated daily bio/healthcare research trends and news
                </p>
                <span className="inline-flex items-center gap-1 text-sm font-medium text-purple-600 group-hover:gap-2 transition-all">
                  Open Tool <ArrowRight className="w-4 h-4" />
                </span>
              </button>
            </div>
          </div>
        </section>

        {/* Knowledge Graph Section */}
        <section className="relative py-24 overflow-hidden line-b">
          {/* Background Glow */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[70%] h-[60%] bg-gradient-radial from-indigo-200/40 via-purple-100/20 to-transparent blur-3xl pointer-events-none animate-pulse-glow"></div>

          <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
            <span className="inline-flex items-center gap-2 px-3 py-1.5 glass-2 rounded-full border border-purple-200/50 text-sm font-medium text-purple-600 mb-6 animate-appear">
              <Network className="w-3.5 h-3.5" />
              3D Visualization
            </span>

            <h2 className="text-3xl md:text-5xl font-bold mb-6 text-gray-900 animate-appear delay-100">
              Explore the <span className="text-gradient-brand">Knowledge Universe</span>
            </h2>

            <p className="text-lg text-gray-600 mb-10 max-w-2xl mx-auto leading-relaxed animate-appear delay-200">
              Visualize connections between genes, diseases, pathways, and research papers
              in an interactive 3D space. Discover hidden relationships in your research.
            </p>

            <button
              onClick={() => setShowGraph(true)}
              className="group inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-violet-600 to-purple-600 rounded-full text-white font-semibold hover:from-violet-700 hover:to-purple-700 transition-all shadow-xl btn-glow animate-appear delay-300"
            >
              <Network className="w-5 h-5" />
              Launch Knowledge Graph
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>

            {/* Feature Cards */}
            <div className="mt-16 grid md:grid-cols-3 gap-6 animate-appear delay-500">
              {[
                { title: 'Gene Networks', desc: 'Explore gene-gene interactions and pathways' },
                { title: 'Disease Links', desc: 'Connect diseases to genetic markers' },
                { title: 'Paper Clusters', desc: 'See research paper relationships' }
              ].map((item, idx) => (
                <div key={idx} className="glass-3 border border-purple-100/50 rounded-2xl p-6 card-hover">
                  <div className="w-10 h-10 mx-auto mb-4 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-xl flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-purple-600" />
                  </div>
                  <h3 className="font-semibold text-gray-800 mb-2">{item.title}</h3>
                  <p className="text-sm text-gray-500">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <CtaSection />
      </main>
      <Footer />

      {/* Modals */}
      <KnowledgeGraph isOpen={showGraph} onClose={() => setShowGraph(false)} />
      <LiteratureReview
        isOpen={showLiteratureReview}
        onClose={() => setShowLiteratureReview(false)}
        papers={reviewPapers}
        onRemovePaper={removeFromReview}
        onClearAll={clearReview}
      />
      <ChatWithPDF isOpen={showChatWithPDF} onClose={() => setShowChatWithPDF(false)} />
      <ResearchLibrary isOpen={showLibrary} onClose={() => setShowLibrary(false)} />
      <DailyBriefing isOpen={showBriefing} onClose={() => setShowBriefing(false)} />

    </div>
  );
};

// Wrap with LanguageProvider
const App: React.FC = () => {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
};

export default App;
