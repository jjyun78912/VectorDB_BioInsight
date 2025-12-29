import React from 'react';
import { Linkedin, Twitter, Github, Mail, Sparkles, ArrowUpRight } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="relative bg-gradient-to-b from-gray-900 via-gray-900 to-black text-white pt-24 pb-12 overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[60%] h-[300px] bg-gradient-radial from-purple-600/10 via-violet-600/5 to-transparent blur-3xl pointer-events-none"></div>

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-12 mb-20">
          <div className="lg:col-span-2">
             <div className="flex items-center gap-3 mb-6 group cursor-pointer">
                <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform">
                   <Sparkles className="w-5 h-5 text-white" />
                </div>
                <div className="flex flex-col">
                  <span className="font-bold tracking-tight text-lg">BIOINSIGHT</span>
                  <span className="text-[10px] text-purple-400 font-medium -mt-1 tracking-widest">AI PLATFORM</span>
                </div>
             </div>
             <p className="text-gray-400 mb-4 max-w-sm leading-relaxed">
               The first Bio AI platform that integrates data analysis, literature knowledge, and ML experiments in one unified workspace.
             </p>
             <p className="text-sm text-gray-500">
               Empowering researchers to think faster and discover deeper.
             </p>
          </div>

          <div>
            <h4 className="font-semibold mb-6 text-white flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
              Platform
            </h4>
            <ul className="space-y-4 text-sm text-gray-400">
              {['Paper Analysis', 'RNA-seq Analysis', 'ML Prediction', 'AI Assistant'].map((item) => (
                <li key={item}>
                  <a href="#" className="hover:text-purple-400 transition-colors flex items-center gap-1 group">
                    {item}
                    <ArrowUpRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-6 text-white flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              Resources
            </h4>
            <ul className="space-y-4 text-sm text-gray-400">
              {['Documentation', 'API Reference', 'Tutorials', 'Research Blog'].map((item) => (
                <li key={item}>
                  <a href="#" className="hover:text-emerald-400 transition-colors flex items-center gap-1 group">
                    {item}
                    <ArrowUpRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-6 text-white flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-orange-500"></span>
              Company
            </h4>
            <ul className="space-y-4 text-sm text-gray-400">
              {['About Us', 'Careers', 'Contact', 'Privacy Policy'].map((item) => (
                <li key={item}>
                  <a href="#" className="hover:text-orange-400 transition-colors flex items-center gap-1 group">
                    {item}
                    <ArrowUpRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Social Links */}
        <div className="flex flex-col md:flex-row justify-between items-center pt-8 border-t border-gray-800/50">
          <div className="flex items-center gap-4 mb-4 md:mb-0">
            {[
              { icon: Twitter, href: '#' },
              { icon: Linkedin, href: '#' },
              { icon: Github, href: '#' },
              { icon: Mail, href: '#' }
            ].map((social, idx) => (
              <a
                key={idx}
                href={social.href}
                className="w-10 h-10 rounded-xl bg-gray-800/50 hover:bg-purple-600/20 border border-gray-700/50 hover:border-purple-500/50 flex items-center justify-center text-gray-400 hover:text-purple-400 transition-all"
              >
                <social.icon size={18} />
              </a>
            ))}
          </div>
          <p className="text-sm text-gray-500">
            © 2024 BioInsight AI. All rights reserved.
          </p>
        </div>

        {/* Giant Logo Watermark */}
        <div className="mt-16 text-center select-none pointer-events-none">
           <h1 className="text-[12vw] font-bold bg-gradient-to-b from-gray-800/30 to-transparent bg-clip-text text-transparent leading-none tracking-tight">BIOINSIGHT</h1>
        </div>
      </div>
    </footer>
  );
};
