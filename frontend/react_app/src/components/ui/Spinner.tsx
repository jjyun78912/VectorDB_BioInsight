import React from 'react';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

export interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  label?: string;
}

const sizeStyles = {
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-8 h-8',
  xl: 'w-12 h-12',
};

export const Spinner: React.FC<SpinnerProps> = ({
  size = 'md',
  className,
  label,
}) => {
  return (
    <div className="flex flex-col items-center justify-center gap-2">
      <Loader2
        className={cn('animate-spin text-purple-600', sizeStyles[size], className)}
      />
      {label && <p className="text-sm text-gray-500">{label}</p>}
    </div>
  );
};

Spinner.displayName = 'Spinner';

// Full page loading spinner
export const PageSpinner: React.FC<{ label?: string }> = ({ label = 'Loading...' }) => {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <Spinner size="xl" label={label} />
    </div>
  );
};

PageSpinner.displayName = 'PageSpinner';
