import React from 'react';

const LOGOS = [
  'RIVIAN', 'eventbrite', 'OpenAI', 'zoom', 'Brex', 'ATLASSIAN'
];

export const TrustedBy: React.FC = () => {
  return (
    <section className="py-12 border-b border-white/5 bg-[#2A2450]">
      <div className="max-w-7xl mx-auto px-6 text-center">
        <p className="text-sm text-purple-200/50 mb-8 font-medium">Trusted by the most innovative companies in the world</p>
        <div className="flex flex-wrap justify-center items-center gap-8 md:gap-16 opacity-50 grayscale hover:grayscale-0 transition-all duration-500">
          {LOGOS.map((logo, i) => (
            <span key={i} className="text-xl md:text-2xl font-bold font-sans tracking-tight text-white/70 hover:text-white cursor-default">
              {logo}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
};