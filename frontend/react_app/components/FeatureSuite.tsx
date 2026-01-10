import React, { useState } from 'react';
import { Search, FileText, TrendingUp, MessageSquare, Sparkles, BrainCircuit, Dna, FlaskConical, ArrowRight } from 'lucide-react';
import { FeatureTab } from '../types';
import { useLanguage } from '../contexts/LanguageContext';

// Define tabs dynamically based on translations
const getTranslatedTabs = (t: any): FeatureTab[] => [
  {
    id: 'paper',
    label: t.featureTabPaper,
    icon: FileText,
    color: 'from-violet-500 to-purple-500',
    title: t.paperAnalysisTitle,
    description: t.paperAnalysisDesc,
    benefits: [
      t.paperBenefit1,
      t.paperBenefit2,
      t.paperBenefit3,
      t.paperBenefit4
    ],
    mockupData: {
      type: 'mail',
      items: [
        { id: 1, title: 'SDHB Mutations in PCC', subtitle: 'Genetic basis of paraganglioma', tag: '94%', content: 'Key finding: Novel SDHB variant identified...' },
        { id: 2, title: 'VHL Syndrome Review', subtitle: 'von Hippel-Lindau associations', tag: '89%', content: 'Methods: Cohort study of 234 patients...' },
        { id: 3, title: 'RET Proto-oncogene', subtitle: 'MEN2 syndrome markers', tag: '85%', content: 'Conclusion: RET mutations predict...' },
      ]
    }
  },
  {
    id: 'rnaseq',
    label: t.featureTabRnaseq,
    icon: Dna,
    color: 'from-emerald-500 to-teal-500',
    title: t.rnaseqTitle,
    description: t.rnaseqDesc,
    benefits: [
      t.rnaseqBenefit1,
      t.rnaseqBenefit2,
      t.rnaseqBenefit3,
      t.rnaseqBenefit4
    ],
    mockupData: {
      type: 'chart',
      items: [
        { id: 1, title: 'Volcano Plot', subtitle: '2,847 DEGs identified', content: 'FDR < 0.05, |log2FC| > 1' },
        { id: 2, title: 'Top Pathways', subtitle: 'KEGG Enrichment', content: 'p53 signaling, Cell cycle, Apoptosis' },
      ]
    }
  },
  {
    id: 'ml',
    label: t.featureTabMl,
    icon: FlaskConical,
    color: 'from-orange-500 to-red-500',
    title: t.mlTitle,
    description: t.mlDesc,
    benefits: [
      t.mlBenefit1,
      t.mlBenefit2,
      t.mlBenefit3,
      t.mlBenefit4
    ],
    mockupData: {
      type: 'model',
      items: [
        { id: 1, title: 'Model Performance', subtitle: 'XGBoost Classifier', content: 'ROC-AUC: 0.94 | Accuracy: 91.2%' },
      ]
    }
  },
  {
    id: 'assistant',
    label: t.featureTabAssistant,
    icon: MessageSquare,
    color: 'from-pink-500 to-rose-500',
    title: t.assistantTitle,
    description: t.assistantDesc,
    benefits: [
      t.assistantBenefit1,
      t.assistantBenefit2,
      t.assistantBenefit3,
      t.assistantBenefit4
    ],
    mockupData: {
      type: 'chat',
      items: [
        { id: 1, title: 'Research Assistant', subtitle: 'AI-Powered', content: "Based on your DEG results and 5 indexed papers, the upregulated genes (BRCA1, TP53, ATM) are strongly associated with DNA repair pathways. This aligns with Chen et al. (2023) findings on...", avatar: '' },
      ]
    }
  }
];

export const FeatureSuite: React.FC = () => {
  const { t } = useLanguage();
  const TABS = getTranslatedTabs(t);
  const [activeTabId, setActiveTabId] = useState('paper');
  const activeTab = TABS.find(tab => tab.id === activeTabId) || TABS[0];

  return (
    <section className="py-24 relative overflow-hidden line-b">
      {/* Background Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[80%] h-[60%] bg-gradient-radial from-purple-200/30 via-violet-100/20 to-transparent blur-3xl pointer-events-none" />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Header */}
        <div className="mb-16 text-center md:text-left">
          <span className="inline-flex items-center gap-2 px-3 py-1.5 glass-2 rounded-full border border-purple-200/50 text-sm font-medium text-purple-600 mb-4">
            <Sparkles className="w-3.5 h-3.5" />
            {t.allInOnePlatform}
          </span>
          <h2 className="text-3xl md:text-5xl font-bold mb-4 text-gray-900">
            {t.yourBioInsight} <span className="text-gradient-brand">{t.aiSuite}</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl">
            {t.platformSubtitle}
          </p>

          {/* Tabs */}
          <div className="mt-10 grid grid-cols-2 md:grid-cols-4 gap-2 md:gap-3 p-1.5 glass-3 rounded-2xl border border-purple-100/50 shadow-lg">
            {TABS.map((tab) => {
              const isActive = tab.id === activeTabId;
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTabId(tab.id)}
                  className={`flex items-center justify-center gap-2.5 py-4 rounded-xl transition-all duration-300 ${
                    isActive
                      ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white shadow-lg glow-brand scale-[1.02]'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-purple-50/50'
                  }`}
                >
                  <Icon className={`w-5 h-5 ${isActive ? 'text-white' : ''}`} />
                  <span className="font-semibold text-sm md:text-base">{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Content Area */}
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-20 items-center min-h-[600px]">

          {/* Left: Text Content */}
          <div className="order-2 lg:order-1 animate-appear" key={activeTab.id}>
            <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${activeTab.color} mb-8 flex items-center justify-center shadow-xl glow-brand`}>
              <activeTab.icon className="w-7 h-7 text-white" />
            </div>

            <h3 className="text-3xl md:text-4xl font-bold mb-6 leading-tight text-gray-900">
              {activeTab.title}
            </h3>

            <p className="text-lg text-gray-600 mb-8 leading-relaxed">
              {activeTab.description}
            </p>

            <a href="#" className="inline-flex items-center gap-2 text-purple-600 hover:text-purple-700 font-semibold mb-10 transition-colors group">
              {t.learnMoreAbout} {activeTab.label}
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </a>

            <ul className="space-y-4">
              {activeTab.benefits.map((benefit, i) => (
                <li key={i} className="flex items-start gap-3 text-gray-700">
                  <div className="mt-1.5 w-5 h-5 rounded-full bg-purple-100 border border-purple-200 flex items-center justify-center flex-shrink-0">
                    <div className="w-2 h-2 rounded-full bg-gradient-to-br from-violet-500 to-purple-600"></div>
                  </div>
                  <span className="text-base font-medium">{benefit}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Right: UI Mockup */}
          <div className="order-1 lg:order-2 relative h-full flex items-center justify-center">
            {/* Abstract Background Shapes */}
            <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-gradient-to-tr ${activeTab.color} opacity-10 blur-[80px] rounded-full animate-pulse-glow`}></div>

            {/* Glass Card Container */}
            <div className="relative w-full aspect-square md:aspect-[4/3] glass-4 border border-purple-100/50 rounded-3xl p-6 shadow-2xl overflow-hidden group card-hover">

              {/* Fake UI Header */}
              <div className="flex items-center justify-between mb-6 border-b border-purple-100/30 pb-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-400/80"></div>
                  <div className="w-3 h-3 rounded-full bg-yellow-400/80"></div>
                  <div className="w-3 h-3 rounded-full bg-green-400/80"></div>
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-400 font-mono glass-2 px-3 py-1 rounded-full">
                  <Sparkles className="w-3 h-3 text-purple-400" />
                  BioInsight AI v2.0
                </div>
              </div>

              {/* Dynamic Content based on type */}
              <div className="space-y-3 relative z-10">
                {activeTab.mockupData.items.map((item, idx) => (
                   <MockupItem key={item.id} item={item} index={idx} type={activeTab.mockupData.type} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

// Helper for UI Items
const MockupItem: React.FC<{ item: any, index: number, type: string }> = ({ item, index, type }) => {
  if (type === 'mail') {
    return (
      <div
        className="glass-2 hover:glass-3 border border-purple-100/50 rounded-xl p-4 transition-all duration-300 cursor-pointer card-hover animate-appear"
        style={{ animationDelay: `${index * 100}ms` }}
      >
        <div className="flex justify-between items-start mb-2">
          <span className="font-semibold text-sm text-gray-800">{item.title}</span>
          {item.tag && (
            <span className="text-[10px] px-2.5 py-1 rounded-full uppercase tracking-wider font-bold bg-gradient-to-r from-emerald-100 to-teal-100 text-emerald-600 border border-emerald-200/50">
              {item.tag} match
            </span>
          )}
        </div>
        <div className="text-sm font-medium text-gray-600 mb-1.5 truncate">{item.subtitle}</div>
        <div className="text-xs text-gray-400 truncate">{item.content}</div>
      </div>
    );
  }

  if (type === 'chat') {
    return (
      <div className="space-y-4 pt-4 animate-appear">
        <div className="flex gap-4">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center flex-shrink-0 shadow-lg glow-brand">
            <BrainCircuit className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1">
             <div className="flex items-baseline gap-2 mb-2">
                <span className="font-bold text-sm text-gray-800">BioInsight AI</span>
                <span className="text-xs text-gray-400">Just now</span>
             </div>
             <div className="glass-2 rounded-2xl rounded-tl-sm p-4 text-sm leading-relaxed border border-purple-100/50 text-gray-700">
                {item.content}
             </div>
             <div className="mt-3 flex gap-2">
                <span className="px-3 py-1.5 rounded-full glass-2 text-emerald-600 text-xs border border-emerald-200/50 flex items-center gap-1.5 font-medium">
                  <Sparkles className="w-3 h-3" /> 5 sources cited
                </span>
                <span className="px-3 py-1.5 rounded-full glass-2 text-purple-600 text-xs border border-purple-200/50 font-medium">High confidence</span>
             </div>
          </div>
        </div>
      </div>
    );
  }

  if (type === 'chart' || type === 'model') {
    return (
      <div className="glass-2 border border-purple-100/50 rounded-xl p-4 flex gap-4 items-center card-hover animate-appear" style={{ animationDelay: `${index * 100}ms` }}>
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${type === 'chart' ? 'from-emerald-400 to-teal-500' : 'from-orange-400 to-red-500'} flex items-center justify-center shadow-lg`}>
          {type === 'chart' ? <TrendingUp className="w-6 h-6 text-white" /> : <FlaskConical className="w-6 h-6 text-white" />}
        </div>
        <div className="flex-1">
          <div className="text-sm font-bold text-gray-800">{item.title}</div>
          <div className="text-xs text-gray-500 mb-1">{item.subtitle}</div>
          <div className="text-xs text-purple-600 font-medium">{item.content}</div>
        </div>
      </div>
    );
  }

  // Default Fallback generic card
  return (
    <div className="glass-2 border border-purple-100/50 rounded-xl p-4 flex gap-4 items-center card-hover animate-appear">
      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
        <BrainCircuit className="w-5 h-5 text-white" />
      </div>
      <div>
        <div className="text-sm font-semibold text-gray-800">{item.title}</div>
        <div className="text-xs text-gray-500">{item.content}</div>
      </div>
    </div>
  );
};
