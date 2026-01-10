import React, { useState, useEffect } from 'react';
import { Menu, X, ArrowRight, Sparkles, Globe } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

export const Navbar: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const { language, setLanguage, t } = useLanguage();

  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'ko' : 'en');
  };

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        isScrolled
          ? 'glass-4 border-b border-purple-100/50 shadow-lg'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3 cursor-pointer group">
          <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg glow-brand group-hover:scale-105 transition-transform">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-lg tracking-tight text-gray-900">BIOINSIGHT</span>
            <span className="text-[10px] text-purple-500 font-medium -mt-1 tracking-widest">AI PLATFORM</span>
          </div>
        </div>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center gap-1">
          {[
            { key: 'navProduct', label: t.navProduct },
            { key: 'navEnterprise', label: t.navEnterprise },
            { key: 'navPricing', label: t.navPricing },
            { key: 'navDocs', label: t.navDocs },
          ].map((item) => (
            <a
              key={item.key}
              href="#"
              className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-purple-50/50 rounded-lg transition-all"
            >
              {item.label}
            </a>
          ))}
        </div>

        {/* Desktop Actions */}
        <div className="hidden md:flex items-center gap-3">
          {/* Language Toggle */}
          <button
            onClick={toggleLanguage}
            className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-purple-50/50 rounded-lg transition-all"
            title={language === 'en' ? 'Switch to Korean' : '영어로 전환'}
          >
            <Globe className="w-4 h-4" />
            <span>{language === 'en' ? 'EN' : 'KO'}</span>
          </button>
          <button className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-purple-50/50 rounded-lg transition-all">
            {t.logIn}
          </button>
          <button className="group px-5 py-2.5 bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-full text-sm font-semibold hover:from-violet-700 hover:to-purple-700 transition-all flex items-center gap-2 shadow-lg btn-glow">
            {t.getStarted}
            <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
          </button>
        </div>

        {/* Mobile Menu Toggle */}
        <button
          className="md:hidden p-2 text-gray-900 hover:bg-purple-50/50 rounded-lg transition-colors"
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        >
          {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="md:hidden glass-5 absolute top-20 left-0 w-full p-6 flex flex-col gap-2 border-b border-purple-100/50 shadow-xl animate-appear">
          {[
            { key: 'navProduct', label: t.navProduct },
            { key: 'navEnterprise', label: t.navEnterprise },
            { key: 'navPricing', label: t.navPricing },
            { key: 'navDocs', label: t.navDocs },
          ].map((item) => (
            <a
              key={item.key}
              href="#"
              className="px-4 py-3 text-lg font-medium text-gray-700 hover:text-purple-600 hover:bg-purple-50/50 rounded-xl transition-all"
            >
              {item.label}
            </a>
          ))}
          <div className="h-px bg-purple-100/50 w-full my-3"></div>
          {/* Language Toggle - Mobile */}
          <button
            onClick={toggleLanguage}
            className="flex items-center gap-2 px-4 py-3 text-lg font-medium text-gray-700 hover:text-purple-600 hover:bg-purple-50/50 rounded-xl transition-all"
          >
            <Globe className="w-5 h-5" />
            {language === 'en' ? 'English (EN)' : '한국어 (KO)'}
          </button>
          <button className="px-4 py-3 text-left text-lg font-medium text-gray-900 hover:bg-purple-50/50 rounded-xl transition-colors">
            {t.logIn}
          </button>
          <button className="mt-2 bg-gradient-to-r from-violet-600 to-purple-600 text-white w-full py-3.5 rounded-xl text-lg font-semibold hover:from-violet-700 hover:to-purple-700 transition-all shadow-lg flex items-center justify-center gap-2">
            {t.getStarted}
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      )}
    </nav>
  );
};
