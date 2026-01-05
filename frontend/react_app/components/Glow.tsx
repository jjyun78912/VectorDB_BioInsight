import React from 'react';

interface GlowProps {
  variant?: 'top' | 'center';
  className?: string;
}

export const Glow: React.FC<GlowProps> = ({ variant = 'top', className = '' }) => (
  <div className={`absolute w-full pointer-events-none ${variant === 'top' ? 'top-0' : 'top-1/2 -translate-y-1/2'} ${className}`}>
    <div className="absolute left-1/2 -translate-x-1/2 h-[256px] w-[60%] scale-[2] rounded-[50%] bg-gradient-radial from-purple-400/30 via-violet-300/20 to-transparent opacity-60 blur-3xl sm:h-[400px]" />
    <div className="absolute left-1/2 -translate-x-1/2 h-[128px] w-[40%] scale-150 rounded-[50%] bg-gradient-radial from-violet-500/20 to-transparent opacity-50 blur-2xl sm:h-[200px]" />
  </div>
);
