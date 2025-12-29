import React, { useState, useRef, useEffect } from 'react';
import { BookOpen, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';

export interface CitedSource {
  citation_index: number;
  paper_title: string;
  section: string;
  excerpt: string;
  full_content?: string;
  relevance_score: number;
}

interface CitationRendererProps {
  content: string;
  sources: CitedSource[];
  onSourceClick?: (index: number) => void;
}

interface TooltipPosition {
  x: number;
  y: number;
}

/**
 * Parse text content and convert [1], [2] etc. citations into interactive elements
 */
export const CitationRenderer: React.FC<CitationRendererProps> = ({
  content,
  sources,
  onSourceClick,
}) => {
  const [hoveredCitation, setHoveredCitation] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState<TooltipPosition>({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Parse content and extract citations
  const parseContent = (text: string): (string | { type: 'citation'; index: number })[] => {
    const parts: (string | { type: 'citation'; index: number })[] = [];
    // Match [1], [2], [1][2], etc.
    const regex = /\[(\d+)\]/g;
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
      // Add text before citation
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }
      // Add citation
      parts.push({ type: 'citation', index: parseInt(match[1], 10) });
      lastIndex = regex.lastIndex;
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }

    return parts;
  };

  const handleCitationHover = (
    e: React.MouseEvent<HTMLSpanElement>,
    index: number
  ) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const containerRect = containerRef.current?.getBoundingClientRect();

    if (containerRect) {
      setTooltipPos({
        x: rect.left - containerRect.left + rect.width / 2,
        y: rect.top - containerRect.top,
      });
    }
    setHoveredCitation(index);
  };

  const handleCitationLeave = () => {
    setHoveredCitation(null);
  };

  const getSourceByIndex = (index: number): CitedSource | undefined => {
    return sources.find((s) => s.citation_index === index);
  };

  const parsedContent = parseContent(content);

  return (
    <div ref={containerRef} className="relative">
      {/* Main content with inline citations */}
      <p className="leading-relaxed whitespace-pre-wrap">
        {parsedContent.map((part, idx) => {
          if (typeof part === 'string') {
            return <span key={idx}>{part}</span>;
          }

          const source = getSourceByIndex(part.index);
          const isHovered = hoveredCitation === part.index;

          return (
            <span
              key={idx}
              className={`
                inline-flex items-center justify-center
                min-w-[1.25rem] h-5 px-1
                text-xs font-semibold
                bg-gradient-to-r from-violet-100 to-purple-100
                text-purple-700 rounded
                cursor-pointer
                transition-all duration-200
                hover:from-violet-200 hover:to-purple-200
                hover:scale-110
                ${isHovered ? 'ring-2 ring-purple-400 ring-offset-1' : ''}
              `}
              onMouseEnter={(e) => handleCitationHover(e, part.index)}
              onMouseLeave={handleCitationLeave}
              onClick={() => onSourceClick?.(part.index)}
            >
              {part.index}
            </span>
          );
        })}
      </p>

      {/* Tooltip */}
      {hoveredCitation !== null && (
        <CitationTooltip
          source={getSourceByIndex(hoveredCitation)}
          position={tooltipPos}
          citationIndex={hoveredCitation}
        />
      )}
    </div>
  );
};

interface CitationTooltipProps {
  source: CitedSource | undefined;
  position: TooltipPosition;
  citationIndex: number;
}

const CitationTooltip: React.FC<CitationTooltipProps> = ({
  source,
  position,
  citationIndex,
}) => {
  if (!source) {
    return (
      <div
        className="absolute z-50 w-64 p-3 bg-white/95 backdrop-blur-md rounded-xl shadow-xl border border-purple-100 transform -translate-x-1/2 -translate-y-full"
        style={{
          left: position.x,
          top: position.y - 8,
        }}
      >
        <p className="text-sm text-gray-500">Source [{citationIndex}] not found</p>
      </div>
    );
  }

  return (
    <div
      className="absolute z-50 w-80 p-4 bg-white/95 backdrop-blur-md rounded-xl shadow-2xl border border-purple-200 transform -translate-x-1/2 -translate-y-full animate-appear"
      style={{
        left: position.x,
        top: position.y - 12,
      }}
    >
      {/* Arrow */}
      <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-full">
        <div className="w-3 h-3 bg-white border-r border-b border-purple-200 transform rotate-45 -translate-y-1.5" />
      </div>

      {/* Header */}
      <div className="flex items-start gap-2 mb-2">
        <div className="flex items-center justify-center w-6 h-6 rounded-full bg-gradient-to-r from-violet-500 to-purple-600 text-white text-xs font-bold flex-shrink-0">
          {source.citation_index}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-purple-600 uppercase tracking-wide">
            {source.section}
          </p>
        </div>
      </div>

      {/* Content */}
      <p className="text-sm text-gray-700 leading-relaxed line-clamp-4">
        {source.excerpt}
      </p>

      {/* Relevance score */}
      <div className="mt-2 flex items-center gap-2">
        <div className="flex-1 h-1 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-violet-500 to-purple-500 rounded-full"
            style={{ width: `${source.relevance_score}%` }}
          />
        </div>
        <span className="text-xs text-gray-500">
          {source.relevance_score.toFixed(0)}% relevant
        </span>
      </div>
    </div>
  );
};

/**
 * Source Cards Component - Displays expandable source cards below the answer
 */
interface SourceCardsProps {
  sources: CitedSource[];
  highlightedIndex?: number | null;
  onSourceClick?: (index: number) => void;
}

export const SourceCards: React.FC<SourceCardsProps> = ({
  sources,
  highlightedIndex,
  onSourceClick,
}) => {
  const [expandedCard, setExpandedCard] = useState<number | null>(null);
  const cardRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

  // Scroll to highlighted card
  useEffect(() => {
    if (highlightedIndex !== null && cardRefs.current[highlightedIndex]) {
      cardRefs.current[highlightedIndex]?.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }
  }, [highlightedIndex]);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
        <BookOpen className="w-3.5 h-3.5" />
        Sources ({sources.length})
      </p>

      <div className="space-y-2">
        {sources.map((source) => {
          const isExpanded = expandedCard === source.citation_index;
          const isHighlighted = highlightedIndex === source.citation_index;

          return (
            <div
              key={source.citation_index}
              ref={(el) => (cardRefs.current[source.citation_index] = el)}
              className={`
                rounded-xl overflow-hidden transition-all duration-300
                ${isHighlighted
                  ? 'ring-2 ring-purple-400 ring-offset-2 bg-gradient-to-r from-violet-50 to-purple-50'
                  : 'glass-2 hover:bg-purple-50/30'
                }
                border border-purple-100/50
              `}
            >
              {/* Card Header - Always visible */}
              <button
                onClick={() => {
                  setExpandedCard(isExpanded ? null : source.citation_index);
                  onSourceClick?.(source.citation_index);
                }}
                className="w-full p-3 text-left flex items-start gap-3"
              >
                {/* Citation badge */}
                <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 text-white text-sm font-bold flex-shrink-0 shadow-sm">
                  {source.citation_index}
                </div>

                {/* Content preview */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <p className="text-xs font-semibold text-purple-600">
                      {source.section}
                    </p>
                    <div className="flex-1" />
                    <span className="text-xs text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">
                      {source.relevance_score.toFixed(0)}%
                    </span>
                  </div>
                  <p className={`text-sm text-gray-600 ${isExpanded ? '' : 'line-clamp-2'}`}>
                    {isExpanded ? (source.full_content || source.excerpt) : source.excerpt}
                  </p>
                </div>

                {/* Expand/collapse icon */}
                <div className="flex-shrink-0 text-gray-400">
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </div>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default CitationRenderer;
