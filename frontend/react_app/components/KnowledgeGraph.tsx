import React, { useRef, useCallback, useEffect, useState } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import { X, Loader2, ZoomIn, ZoomOut, RotateCcw, Info, Telescope } from 'lucide-react';

interface GraphNode {
  id: string;
  name: string;
  type: string;
  size: number;
  color: string;
  metadata?: Record<string, any>;
  x?: number;
  y?: number;
  z?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  strength: number;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  stats: {
    total_nodes: number;
    total_links: number;
    total_papers: number;
    total_keywords: number;
    node_types: Record<string, number>;
  };
}

// Similar papers visualization data
interface SimilarPaperViz {
  pmid: string;
  title: string;
  x: number;
  y: number;
  similarity_score: number;
}

interface SimilarPapersData {
  source_paper: string;
  source_pmid: string;
  similar_papers: Array<{
    pmid: string;
    title: string;
    abstract: string;
    similarity_score: number;
    common_keywords: string[];
    year?: string;
  }>;
  visualization?: {
    source: { pmid: string; x: number; y: number };
    papers: SimilarPaperViz[];
  };
}

type ViewMode = 'knowledge' | 'similar';

interface KnowledgeGraphProps {
  isOpen: boolean;
  onClose: () => void;
  mode?: ViewMode;
  sourcePmid?: string;
  domain?: string;
  onPaperClick?: (pmid: string) => void;
}

const NODE_COLORS: Record<string, string> = {
  paper: '#ff6b6b',
  gene: '#ffd93d',
  disease: '#6bcb77',
  pathway: '#4d96ff',
  clinical: '#ff9f43',
  biomarker: '#a66cff',
  keyword: '#95d5b2',
  source: '#ffd700',  // Gold for source paper
  similar: '#60a5fa', // Blue for similar papers
};

export const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({
  isOpen,
  onClose,
  mode = 'knowledge',
  sourcePmid,
  domain = 'pancreatic_cancer',
  onPaperClick
}) => {
  const fgRef = useRef<any>();
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [similarData, setSimilarData] = useState<SimilarPapersData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);

  // Fetch data based on mode
  useEffect(() => {
    if (!isOpen) return;

    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        if (mode === 'similar' && sourcePmid) {
          // Fetch similar papers
          const response = await fetch(
            `/api/search/similar/${sourcePmid}?domain=${domain}&top_k=10&include_visualization=true`
          );
          if (!response.ok) throw new Error('Failed to fetch similar papers');
          const data: SimilarPapersData = await response.json();
          setSimilarData(data);

          // Convert to graph format
          const graphFromSimilar = convertSimilarToGraph(data);
          setGraphData(graphFromSimilar);
        } else {
          // Original knowledge graph fetch
          const response = await fetch('/api/graph/?domain=pheochromocytoma&include_papers=true&min_connections=1');
          if (!response.ok) throw new Error('Failed to fetch graph data');
          const data = await response.json();
          setGraphData(data);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [isOpen, mode, sourcePmid, domain]);

  // Convert similar papers data to graph format
  const convertSimilarToGraph = (data: SimilarPapersData): GraphData => {
    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];

    if (!data.visualization) {
      // Fallback: create circular layout
      const sourceNode: GraphNode = {
        id: data.source_pmid,
        name: data.source_paper,
        type: 'source',
        size: 8,
        color: NODE_COLORS.source,
        metadata: { isSource: true },
        x: 0,
        y: 0,
        z: 0
      };
      nodes.push(sourceNode);

      data.similar_papers.forEach((paper, index) => {
        const angle = (2 * Math.PI * index) / data.similar_papers.length;
        const distance = 150 - (paper.similarity_score / 100) * 50; // Higher similarity = closer

        nodes.push({
          id: paper.pmid,
          name: paper.title,
          type: 'similar',
          size: 3 + (paper.similarity_score / 100) * 3,
          color: NODE_COLORS.similar,
          metadata: {
            abstract: paper.abstract,
            similarity: paper.similarity_score,
            year: paper.year,
            common_keywords: paper.common_keywords
          },
          x: Math.cos(angle) * distance,
          y: Math.sin(angle) * distance,
          z: (Math.random() - 0.5) * 50
        });

        links.push({
          source: data.source_pmid,
          target: paper.pmid,
          strength: paper.similarity_score / 100
        });
      });
    } else {
      // Use t-SNE coordinates from API
      const scale = 150; // Scale factor for visualization

      const sourceNode: GraphNode = {
        id: data.source_pmid,
        name: data.source_paper,
        type: 'source',
        size: 8,
        color: NODE_COLORS.source,
        metadata: { isSource: true },
        x: data.visualization.source.x * scale,
        y: data.visualization.source.y * scale,
        z: 0
      };
      nodes.push(sourceNode);

      data.visualization.papers.forEach((paper) => {
        const paperInfo = data.similar_papers.find(p => p.pmid === paper.pmid);

        nodes.push({
          id: paper.pmid,
          name: paper.title,
          type: 'similar',
          size: 3 + (paper.similarity_score / 100) * 3,
          color: getColorBySimilarity(paper.similarity_score),
          metadata: {
            abstract: paperInfo?.abstract || '',
            similarity: paper.similarity_score,
            year: paperInfo?.year,
            common_keywords: paperInfo?.common_keywords || []
          },
          x: paper.x * scale,
          y: paper.y * scale,
          z: (Math.random() - 0.5) * 30 // Small Z variation for depth
        });

        links.push({
          source: data.source_pmid,
          target: paper.pmid,
          strength: paper.similarity_score / 100
        });
      });
    }

    return {
      nodes,
      links,
      stats: {
        total_nodes: nodes.length,
        total_links: links.length,
        total_papers: nodes.length,
        total_keywords: 0,
        node_types: { source: 1, similar: nodes.length - 1 }
      }
    };
  };

  // Get color based on similarity score
  const getColorBySimilarity = (score: number): string => {
    if (score >= 95) return '#22c55e'; // Green - very similar
    if (score >= 90) return '#84cc16'; // Lime
    if (score >= 85) return '#eab308'; // Yellow
    if (score >= 80) return '#f97316'; // Orange
    return '#60a5fa'; // Blue - less similar
  };

  // Handle node click
  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);

    // Zoom to node
    if (fgRef.current) {
      const distance = 100;
      const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0);

      fgRef.current.cameraPosition(
        {
          x: (node.x || 0) * distRatio,
          y: (node.y || 0) * distRatio,
          z: (node.z || 0) * distRatio
        },
        node,
        2000
      );
    }

    // In similar mode, allow navigation to paper
    if (mode === 'similar' && node.type === 'similar' && onPaperClick) {
      // Don't navigate immediately, show info first
    }
  }, [mode, onPaperClick]);

  // Navigate to paper detail
  const handleNavigateToPaper = useCallback((pmid: string) => {
    if (onPaperClick) {
      onPaperClick(pmid);
      onClose();
    }
  }, [onPaperClick, onClose]);

  // Handle node hover
  const handleNodeHover = useCallback((node: GraphNode | null) => {
    setHoveredNode(node);
    document.body.style.cursor = node ? 'pointer' : 'default';
  }, []);

  // Reset camera
  const resetCamera = useCallback(() => {
    if (fgRef.current) {
      fgRef.current.cameraPosition({ x: 0, y: 0, z: 500 }, { x: 0, y: 0, z: 0 }, 1000);
    }
    setSelectedNode(null);
  }, []);

  // Zoom controls
  const zoomIn = useCallback(() => {
    if (fgRef.current) {
      const { x, y, z } = fgRef.current.cameraPosition();
      fgRef.current.cameraPosition({ x: x * 0.7, y: y * 0.7, z: z * 0.7 }, null, 500);
    }
  }, []);

  const zoomOut = useCallback(() => {
    if (fgRef.current) {
      const { x, y, z } = fgRef.current.cameraPosition();
      fgRef.current.cameraPosition({ x: x * 1.3, y: y * 1.3, z: z * 1.3 }, null, 500);
    }
  }, []);

  // Custom node rendering with sprites
  const nodeThreeObject = useCallback((node: GraphNode) => {
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: createNodeTexture(node),
        transparent: true,
        depthWrite: false,
      })
    );
    sprite.scale.set(node.size * 8, node.size * 8, 1);
    return sprite;
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-gradient-to-b from-black/80 to-transparent p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            {mode === 'similar' ? (
              <>
                <Telescope className="w-6 h-6 text-yellow-400" />
                <h2 className="text-xl font-bold text-white">Paper Galaxy</h2>
              </>
            ) : (
              <h2 className="text-xl font-bold text-white">Knowledge Universe</h2>
            )}
            {graphData && (
              <div className="flex gap-2 text-sm text-gray-400">
                {mode === 'similar' ? (
                  <>
                    <span className="text-yellow-400">1</span>
                    <span>source +</span>
                    <span className="text-blue-400">{graphData.stats.total_nodes - 1}</span>
                    <span>similar papers</span>
                  </>
                ) : (
                  <>
                    <span>{graphData.stats.total_nodes} nodes</span>
                    <span>|</span>
                    <span>{graphData.stats.total_links} connections</span>
                  </>
                )}
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="absolute bottom-6 left-6 z-10 flex flex-col gap-2">
        <button
          onClick={zoomIn}
          className="p-3 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors backdrop-blur-sm"
          title="Zoom In"
        >
          <ZoomIn className="w-5 h-5" />
        </button>
        <button
          onClick={zoomOut}
          className="p-3 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors backdrop-blur-sm"
          title="Zoom Out"
        >
          <ZoomOut className="w-5 h-5" />
        </button>
        <button
          onClick={resetCamera}
          className="p-3 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors backdrop-blur-sm"
          title="Reset View"
        >
          <RotateCcw className="w-5 h-5" />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-6 right-6 z-10 bg-black/60 backdrop-blur-sm rounded-xl p-4 border border-white/10">
        {mode === 'similar' ? (
          <>
            <h3 className="text-sm font-semibold text-white mb-3">Similarity Score</h3>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#ffd700' }} />
                <span className="text-xs text-gray-300">Source Paper</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#22c55e' }} />
                <span className="text-xs text-gray-300">95%+ (Very Similar)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#84cc16' }} />
                <span className="text-xs text-gray-300">90-95%</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#eab308' }} />
                <span className="text-xs text-gray-300">85-90%</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#f97316' }} />
                <span className="text-xs text-gray-300">80-85%</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#60a5fa' }} />
                <span className="text-xs text-gray-300">&lt;80%</span>
              </div>
            </div>
          </>
        ) : (
          <>
            <h3 className="text-sm font-semibold text-white mb-3">Node Types</h3>
            <div className="space-y-2">
              {Object.entries(NODE_COLORS).filter(([type]) => !['source', 'similar'].includes(type)).map(([type, color]) => (
                <div key={type} className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs text-gray-300 capitalize">{type}</span>
                  {graphData && (
                    <span className="text-xs text-gray-500">
                      ({graphData.stats.node_types[type] || 0})
                    </span>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Selected Node Info */}
      {selectedNode && (
        <div className="absolute top-20 right-6 z-10 bg-black/80 backdrop-blur-sm rounded-xl p-4 border border-white/10 max-w-sm">
          <div className="flex items-start justify-between mb-2">
            <div
              className="w-3 h-3 rounded-full mt-1"
              style={{ backgroundColor: selectedNode.color }}
            />
            <button
              onClick={() => setSelectedNode(null)}
              className="p-1 hover:bg-white/10 rounded"
            >
              <X className="w-4 h-4 text-gray-400" />
            </button>
          </div>
          <h4 className="text-white font-semibold mb-1 line-clamp-2">{selectedNode.name}</h4>

          {mode === 'similar' && selectedNode.metadata ? (
            <div className="space-y-2">
              {selectedNode.type === 'source' ? (
                <p className="text-xs text-yellow-400">Source Paper</p>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">Similarity:</span>
                    <span className="text-sm font-semibold" style={{ color: selectedNode.color }}>
                      {selectedNode.metadata.similarity?.toFixed(1)}%
                    </span>
                  </div>
                  {selectedNode.metadata.year && (
                    <p className="text-xs text-gray-500">Year: {selectedNode.metadata.year}</p>
                  )}
                  {selectedNode.metadata.abstract && (
                    <p className="text-xs text-gray-400 line-clamp-3 mt-2">
                      {selectedNode.metadata.abstract}
                    </p>
                  )}
                  {selectedNode.metadata.common_keywords?.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {selectedNode.metadata.common_keywords.map((kw: string, i: number) => (
                        <span key={i} className="px-2 py-0.5 bg-purple-500/20 text-purple-300 text-xs rounded-full">
                          {kw}
                        </span>
                      ))}
                    </div>
                  )}
                  {onPaperClick && (
                    <button
                      onClick={() => handleNavigateToPaper(selectedNode.id)}
                      className="w-full mt-3 px-3 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 text-sm rounded-lg transition-colors"
                    >
                      View Paper Details
                    </button>
                  )}
                </>
              )}
            </div>
          ) : (
            <>
              <p className="text-xs text-gray-400 capitalize mb-2">Type: {selectedNode.type}</p>
              {selectedNode.metadata && (
                <div className="text-xs text-gray-500">
                  {selectedNode.metadata.occurrences && (
                    <p>Appears in {selectedNode.metadata.occurrences} papers</p>
                  )}
                  {selectedNode.metadata.doi && (
                    <p className="truncate">DOI: {selectedNode.metadata.doi}</p>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Hover tooltip */}
      {hoveredNode && !selectedNode && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-10 bg-black/80 backdrop-blur-sm rounded-lg px-3 py-2 border border-white/10">
          <span className="text-sm text-white">{hoveredNode.name}</span>
          <span className="text-xs text-gray-400 ml-2 capitalize">({hoveredNode.type})</span>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <div className="text-center">
            <Loader2 className="w-12 h-12 text-purple-500 animate-spin mx-auto mb-4" />
            <p className="text-white">
              {mode === 'similar' ? 'Finding similar papers...' : 'Building knowledge universe...'}
            </p>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <div className="text-center text-red-400">
            <p>Error: {error}</p>
            <button
              onClick={onClose}
              className="mt-4 px-4 py-2 bg-red-500/20 rounded-lg hover:bg-red-500/30"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* 3D Graph */}
      {graphData && !loading && (
        <ForceGraph3D
          ref={fgRef}
          graphData={graphData}
          nodeId="id"
          nodeLabel={(node: GraphNode) => `${node.name} (${node.type})`}
          nodeColor={(node: GraphNode) => node.color}
          nodeVal={(node: GraphNode) => node.size * 2}
          nodeThreeObject={nodeThreeObject}
          nodeThreeObjectExtend={false}
          linkColor={() => 'rgba(255, 255, 255, 0.15)'}
          linkWidth={(link: GraphLink) => link.strength * 2}
          linkOpacity={0.3}
          linkDirectionalParticles={2}
          linkDirectionalParticleSpeed={0.005}
          linkDirectionalParticleWidth={1}
          linkDirectionalParticleColor={() => '#a78bfa'}
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
          backgroundColor="#000011"
          showNavInfo={false}
          enableNodeDrag={true}
          enableNavigationControls={true}
          controlType="orbit"
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          warmupTicks={100}
          cooldownTicks={200}
        />
      )}

      {/* Instructions */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 text-center text-gray-500 text-xs">
        <p>
          {mode === 'similar'
            ? 'Drag to rotate | Scroll to zoom | Click paper to view details'
            : 'Drag to rotate | Scroll to zoom | Click node for details'}
        </p>
      </div>
    </div>
  );
};

// Create a canvas texture for node labels
function createNodeTexture(node: GraphNode): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  const size = 128;
  canvas.width = size;
  canvas.height = size;

  // Draw glow effect
  const gradient = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  gradient.addColorStop(0, node.color);
  gradient.addColorStop(0.3, node.color);
  gradient.addColorStop(0.5, hexToRgba(node.color, 0.5));
  gradient.addColorStop(1, 'transparent');

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(size/2, size/2, size/2, 0, Math.PI * 2);
  ctx.fill();

  // Draw center dot
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(size/2, size/2, size * 0.15, 0, Math.PI * 2);
  ctx.fill();

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function hexToRgba(hex: string, alpha: number): string {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return `rgba(255, 255, 255, ${alpha})`;
  return `rgba(${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}, ${alpha})`;
}

export default KnowledgeGraph;
