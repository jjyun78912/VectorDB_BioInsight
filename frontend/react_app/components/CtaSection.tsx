import React from 'react';
import { ArrowRight, Clock, Zap, TrendingUp, Sparkles } from 'lucide-react';

export const CtaSection: React.FC = () => {
  return (
    <section className="relative py-32 overflow-hidden line-b">
      {/* Background Glow Effects */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-[50%] h-[50%] bg-gradient-radial from-purple-200/40 via-violet-100/20 to-transparent blur-3xl animate-pulse-glow"></div>
        <div className="absolute bottom-1/4 right-1/4 w-[40%] h-[40%] bg-gradient-radial from-indigo-200/30 via-blue-100/10 to-transparent blur-3xl animate-pulse-glow" style={{ animationDelay: '1.5s' }}></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        {/* Impact Stats */}
        <div className="grid md:grid-cols-3 gap-6 mb-24">
          {[
            {
              icon: Clock,
              color: 'from-violet-500 to-purple-500',
              stat: '80-90%',
              label: 'Reduction in paper analysis time',
              detail: 'From 2-4 hours to 10-30 minutes'
            },
            {
              icon: Zap,
              color: 'from-emerald-500 to-teal-500',
              stat: '90%+',
              label: 'Faster RNA-seq analysis',
              detail: 'From 1-2 weeks to hours'
            },
            {
              icon: TrendingUp,
              color: 'from-orange-500 to-red-500',
              stat: 'Zero',
              label: 'Coding required for ML',
              detail: 'Auto-generated models & SHAP analysis'
            }
          ].map((item, idx) => (
            <div
              key={idx}
              className="text-center p-8 glass-3 rounded-3xl border border-purple-100/50 card-hover animate-appear"
              style={{ animationDelay: `${idx * 100}ms` }}
            >
              <div className={`w-14 h-14 mx-auto mb-5 bg-gradient-to-br ${item.color} rounded-2xl flex items-center justify-center shadow-lg glow-brand`}>
                <item.icon className="w-7 h-7 text-white" />
              </div>
              <div className="text-4xl font-bold text-gradient-hero mb-2">{item.stat}</div>
              <p className="text-gray-700 font-medium mb-1">{item.label}</p>
              <p className="text-sm text-gray-500">{item.detail}</p>
            </div>
          ))}
        </div>

        {/* CTA Content */}
        <div className="grid md:grid-cols-2 gap-12 lg:gap-20 items-center">
          <div className="animate-appear delay-300">
            <span className="inline-flex items-center gap-2 px-3 py-1.5 glass-2 rounded-full border border-purple-200/50 text-sm font-medium text-purple-600 mb-6">
              <Sparkles className="w-3.5 h-3.5" />
              Transform Your Research
            </span>
            <h2 className="text-4xl md:text-5xl font-bold mb-6 text-gray-900 leading-tight">
              The first Bio AI platform that{' '}
              <span className="text-gradient-brand">integrates everything.</span>
            </h2>
            <p className="text-lg text-gray-600 mb-10 leading-relaxed">
              Not just an analysis tool — BioInsight AI combines data analysis, literature knowledge, and ML experiments in one unified platform. Think faster, discover deeper, and focus on what only you can do.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <button className="group px-8 py-4 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-full font-semibold hover:from-violet-700 hover:to-purple-700 transition-all shadow-xl btn-glow flex items-center justify-center gap-2">
                Get Started Free
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
              <button className="px-8 py-4 glass-3 border border-purple-200/50 text-purple-700 hover:bg-purple-50/50 transition-all rounded-full font-medium flex items-center justify-center gap-2 card-hover">
                Request Demo
              </button>
            </div>
          </div>

          <div className="relative animate-appear delay-500">
            {/* Decorative Glow */}
            <div className="absolute -inset-4 bg-gradient-to-br from-violet-200/50 to-purple-200/50 rounded-3xl blur-2xl opacity-60"></div>

            <div className="relative h-[400px] rounded-3xl overflow-hidden glass-3 border border-purple-200/50 shadow-2xl flex items-center justify-center group">
              <img
                src="https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?q=80&w=1000&auto=format&fit=crop"
                alt="Research lab"
                className="absolute inset-0 w-full h-full object-cover opacity-50 group-hover:scale-105 transition-transform duration-700"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-violet-900/70 via-purple-600/30 to-transparent"></div>

              {/* Content Overlay */}
              <div className="relative z-10 p-10 text-center">
                <div className="inline-flex items-center gap-2 px-4 py-2 glass-2 rounded-full border border-white/20 text-sm font-medium text-white/90 mb-6">
                  <Sparkles className="w-4 h-4" />
                  AI-Powered
                </div>
                <span className="block text-5xl md:text-6xl font-serif italic text-white mb-2 drop-shadow-lg">Research</span>
                <span className="block text-5xl md:text-6xl font-serif italic text-white/80 drop-shadow-lg">Reimagined</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
