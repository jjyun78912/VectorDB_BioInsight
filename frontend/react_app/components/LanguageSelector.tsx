/**
 * Language Selector Component
 * Toggle between English and Korean
 */
import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { Language } from '../i18n/translations';

const LanguageSelector: React.FC = () => {
  const { language, setLanguage } = useLanguage();

  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'ko' : 'en');
  };

  return (
    <button
      onClick={toggleLanguage}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        padding: '6px 12px',
        background: 'rgba(255, 255, 255, 0.1)',
        border: '1px solid rgba(255, 255, 255, 0.2)',
        borderRadius: '20px',
        color: '#fff',
        fontSize: '13px',
        fontWeight: 500,
        cursor: 'pointer',
        transition: 'all 0.2s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.3)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
      }}
      title={language === 'en' ? 'Switch to Korean' : '영어로 전환'}
    >
      <span style={{ fontSize: '16px' }}>
        {language === 'en' ? '🇺🇸' : '🇰🇷'}
      </span>
      <span>{language === 'en' ? 'EN' : 'KO'}</span>
      <svg
        width="12"
        height="12"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        style={{ opacity: 0.7 }}
      >
        <path d="M7 10l5 5 5-5" />
      </svg>
    </button>
  );
};

export default LanguageSelector;
