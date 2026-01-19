/**
 * GeneNetworkGraph.tsx
 *
 * Obsidian-style 3D gene network visualization for RNA-seq analysis results.
 *
 * Features:
 * - Interactive 3D force-directed graph
 * - Hub gene highlighting with glow effects
 * - Expression direction color coding (up/down)
 * - Pathway clustering
 * - Gene search and filtering
 * - Click-to-focus with gene details panel
 * - Correlation strength visualization
 */

import React, { useRef, useCallback, useEffect, useState, useMemo } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import {
  X, Loader2, ZoomIn, ZoomOut, RotateCcw, Search, Filter,
  TrendingUp, TrendingDown, Target, Database, Dna, Network,
  ChevronDown, Eye, EyeOff, Sparkles, Info
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════
// Type Definitions
// ═══════════════════════════════════════════════════════════════

interface NetworkNode {
  id: string;
  gene_symbol: string | null;
  log2FC: number;
  padj: number;
  direction: string;
  is_hub: boolean;
  hub_score: number;
  degree: number;
  betweenness: number;
  eigenvector: number;
  pathway_count: number;
  db_matched: boolean;
  db_sources: string[];
  confidence: string;
  tags: string[];
  // 3D positioning (computed)
  x?: number;
  y?: number;
  z?: number;
  // Computed display properties
  color?: string;
  size?: number;
}

interface NetworkEdge {
  source: string | NetworkNode;
  target: string | NetworkNode;
  correlation: number;
  abs_correlation: number;
}

interface NetworkGraphData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  stats: {
    total_nodes: number;
    total_edges: number;
    hub_count: number;
    up_regulated: number;
    down_regulated: number;
    db_matched_count: number;
    avg_correlation: number;
    analysis_id: string;
  };
}

interface AnalysisInfo {
  id: string;
  name: string;
  path: string;
  created_at: string;
  node_count: number;
  edge_count: number;
  hub_count: number;
}

interface GeneNetworkGraphProps {
  isOpen: boolean;
  onClose: () => void;
  initialAnalysisId?: string;
}

// ═══════════════════════════════════════════════════════════════
// Color Schemes (Obsidian-inspired)
// ═══════════════════════════════════════════════════════════════

const COLORS = {
  // Node colors by expression direction
  up: '#ef4444',        // Red - upregulated
  down: '#3b82f6',      // Blue - downregulated
  unchanged: '#6b7280', // Gray - unchanged

  // Hub gene highlight
  hubGlow: '#fbbf24',   // Amber glow for hub genes
  hubCore: '#f59e0b',   // Amber core

  // DB validated
  dbValidated: '#22c55e', // Green border for DB-matched

  // Confidence levels
  high: '#22c55e',      // Green
  medium: '#eab308',    // Yellow
  low: '#ef4444',       // Red
  novel: '#a855f7',     // Purple for novel candidates

  // Edges
  positiveCorr: '#4ade80', // Green - positive correlation
  negativeCorr: '#f87171', // Red - negative correlation

  // Background
  background: '#0a0a0f',
};

// ═══════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════

export const GeneNetworkGraph: React.FC<GeneNetworkGraphProps> = ({
  isOpen,
  onClose,
  initialAnalysisId
}) => {
  const fgRef = useRef<any>();

  // State
  const [analyses, setAnalyses] = useState<AnalysisInfo[]>([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(initialAnalysisId || null);
  const [graphData, setGraphData] = useState<NetworkGraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // UI State
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<NetworkNode | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showAnalysisPicker, setShowAnalysisPicker] = useState(false);

  // Filter State
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    hubOnly: false,
    dbMatchedOnly: false,
    minCorrelation: 0.7,
    showUpRegulated: true,
    showDownRegulated: true,
  });

  // ═══════════════════════════════════════════════════════════════
  // Data Fetching
  // ═══════════════════════════════════════════════════════════════

  // Fetch available analyses on mount
  useEffect(() => {
    if (!isOpen) return;

    const fetchAnalyses = async () => {
      try {
        const response = await fetch('/api/rnaseq/analyses');
        if (!response.ok) throw new Error('Failed to fetch analyses');
        const data = await response.json();
        setAnalyses(data);

        // Auto-select first analysis if none selected
        if (!selectedAnalysis && data.length > 0) {
          setSelectedAnalysis(data[0].id);
        }
      } catch (err) {
        console.error('Error fetching analyses:', err);
      }
    };

    fetchAnalyses();
  }, [isOpen]);

  // Fetch network data when analysis is selected
  useEffect(() => {
    if (!isOpen || !selectedAnalysis) return;

    const fetchNetwork = async () => {
      setLoading(true);
      setError(null);

      try {
        const params = new URLSearchParams({
          max_nodes: '500',
          max_edges: '2000',
          hub_only: filters.hubOnly.toString(),
          min_correlation: filters.minCorrelation.toString(),
        });

        const response = await fetch(
          `/api/rnaseq/network/${selectedAnalysis}?${params}`
        );

        if (!response.ok) throw new Error('Failed to fetch network data');
        const data: NetworkGraphData = await response.json();

        // Process nodes for visualization
        const processedNodes = data.nodes.map(node => ({
          ...node,
          color: getNodeColor(node),
          size: getNodeSize(node),
        }));

        setGraphData({
          ...data,
          nodes: processedNodes,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchNetwork();
  }, [isOpen, selectedAnalysis, filters.hubOnly, filters.minCorrelation]);

  // ═══════════════════════════════════════════════════════════════
  // Node Appearance Helpers
  // ═══════════════════════════════════════════════════════════════

  const getNodeColor = (node: NetworkNode): string => {
    if (node.is_hub) {
      return COLORS.hubCore;
    }
    if (node.direction === 'up') return COLORS.up;
    if (node.direction === 'down') return COLORS.down;
    return COLORS.unchanged;
  };

  const getNodeSize = (node: NetworkNode): number => {
    const baseSize = 3;
    const hubBonus = node.is_hub ? 4 : 0;
    const degreeBonus = Math.min(node.degree / 100, 3);
    const expressionBonus = Math.min(Math.abs(node.log2FC) / 2, 2);

    return baseSize + hubBonus + degreeBonus + expressionBonus;
  };

  // ═══════════════════════════════════════════════════════════════
  // Filtered Data
  // ═══════════════════════════════════════════════════════════════

  const filteredData = useMemo(() => {
    if (!graphData) return null;

    let filteredNodes = graphData.nodes;

    // Apply direction filters
    if (!filters.showUpRegulated) {
      filteredNodes = filteredNodes.filter(n => n.direction !== 'up');
    }
    if (!filters.showDownRegulated) {
      filteredNodes = filteredNodes.filter(n => n.direction !== 'down');
    }

    // Apply DB matched filter
    if (filters.dbMatchedOnly) {
      filteredNodes = filteredNodes.filter(n => n.db_matched);
    }

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filteredNodes = filteredNodes.filter(n =>
        n.id.toLowerCase().includes(query) ||
        (n.gene_symbol && n.gene_symbol.toLowerCase().includes(query))
      );
    }

    // Filter edges to only include remaining nodes
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = graphData.edges.filter(e => {
      const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
      const targetId = typeof e.target === 'string' ? e.target : e.target.id;
      return nodeIds.has(sourceId) && nodeIds.has(targetId);
    });

    return {
      ...graphData,
      nodes: filteredNodes,
      links: filteredEdges, // Note: react-force-graph uses 'links'
    };
  }, [graphData, filters, searchQuery]);

  // ═══════════════════════════════════════════════════════════════
  // Event Handlers
  // ═══════════════════════════════════════════════════════════════

  const handleNodeClick = useCallback((node: NetworkNode) => {
    setSelectedNode(node);

    // Zoom to node
    if (fgRef.current) {
      const distance = 150;
      const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0);

      fgRef.current.cameraPosition(
        {
          x: (node.x || 0) * distRatio,
          y: (node.y || 0) * distRatio,
          z: (node.z || 0) * distRatio
        },
        node,
        1500
      );
    }
  }, []);

  const handleNodeHover = useCallback((node: NetworkNode | null) => {
    setHoveredNode(node);
    document.body.style.cursor = node ? 'pointer' : 'default';
  }, []);

  const resetCamera = useCallback(() => {
    if (fgRef.current) {
      fgRef.current.cameraPosition({ x: 0, y: 0, z: 600 }, { x: 0, y: 0, z: 0 }, 1000);
    }
    setSelectedNode(null);
  }, []);

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

  // Focus on searched gene
  const focusOnGene = useCallback((geneId: string) => {
    if (!filteredData) return;

    const node = filteredData.nodes.find(n =>
      n.id === geneId || n.gene_symbol === geneId
    );

    if (node) {
      handleNodeClick(node);
    }
  }, [filteredData, handleNodeClick]);

  // ═══════════════════════════════════════════════════════════════
  // Custom Node Rendering (Obsidian Style)
  // ═══════════════════════════════════════════════════════════════

  const nodeThreeObject = useCallback((node: NetworkNode) => {
    const group = new THREE.Group();

    // Create glow sphere for hub genes
    if (node.is_hub) {
      const glowGeometry = new THREE.SphereGeometry((node.size || 5) * 2, 16, 16);
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: COLORS.hubGlow,
        transparent: true,
        opacity: 0.15,
      });
      const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
      group.add(glowMesh);

      // Outer glow ring
      const ringGeometry = new THREE.RingGeometry(
        (node.size || 5) * 1.8,
        (node.size || 5) * 2.2,
        32
      );
      const ringMaterial = new THREE.MeshBasicMaterial({
        color: COLORS.hubGlow,
        transparent: true,
        opacity: 0.3,
        side: THREE.DoubleSide,
      });
      const ringMesh = new THREE.Mesh(ringGeometry, ringMaterial);
      group.add(ringMesh);
    }

    // Main node sphere
    const geometry = new THREE.SphereGeometry(node.size || 5, 16, 16);
    const material = new THREE.MeshPhongMaterial({
      color: node.color || COLORS.unchanged,
      emissive: node.is_hub ? COLORS.hubGlow : node.color,
      emissiveIntensity: node.is_hub ? 0.3 : 0.1,
      shininess: 50,
    });
    const sphere = new THREE.Mesh(geometry, material);
    group.add(sphere);

    // DB validated indicator (green ring)
    if (node.db_matched) {
      const dbRingGeometry = new THREE.TorusGeometry(
        (node.size || 5) * 1.3,
        0.5,
        8,
        32
      );
      const dbRingMaterial = new THREE.MeshBasicMaterial({
        color: COLORS.dbValidated,
        transparent: true,
        opacity: 0.8,
      });
      const dbRing = new THREE.Mesh(dbRingGeometry, dbRingMaterial);
      dbRing.rotation.x = Math.PI / 2;
      group.add(dbRing);
    }

    return group;
  }, []);

  // ═══════════════════════════════════════════════════════════════
  // Link (Edge) Rendering
  // ═══════════════════════════════════════════════════════════════

  const linkColor = useCallback((link: NetworkEdge) => {
    if (link.correlation > 0) {
      return `rgba(74, 222, 128, ${Math.min(link.abs_correlation, 0.8)})`;
    }
    return `rgba(248, 113, 113, ${Math.min(link.abs_correlation, 0.8)})`;
  }, []);

  const linkWidth = useCallback((link: NetworkEdge) => {
    return 0.5 + link.abs_correlation * 2;
  }, []);

  // ═══════════════════════════════════════════════════════════════
  // Render
  // ═══════════════════════════════════════════════════════════════

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-[#0a0a0f]">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-20 bg-gradient-to-b from-black/90 to-transparent p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Dna className="w-6 h-6 text-amber-400" />
              <h2 className="text-xl font-bold text-white">Gene Network</h2>
            </div>

            {/* Analysis Picker */}
            <div className="relative">
              <button
                onClick={() => setShowAnalysisPicker(!showAnalysisPicker)}
                className="flex items-center gap-2 px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-lg text-sm text-gray-300 transition-colors"
              >
                <Database className="w-4 h-4" />
                {selectedAnalysis ? analyses.find(a => a.id === selectedAnalysis)?.name || selectedAnalysis : 'Select Analysis'}
                <ChevronDown className="w-4 h-4" />
              </button>

              {showAnalysisPicker && (
                <div className="absolute top-full left-0 mt-2 w-72 bg-gray-900 border border-gray-700 rounded-xl shadow-2xl overflow-hidden">
                  <div className="p-2 border-b border-gray-700">
                    <p className="text-xs text-gray-500 px-2">Available Analyses</p>
                  </div>
                  <div className="max-h-60 overflow-y-auto">
                    {analyses.map(analysis => (
                      <button
                        key={analysis.id}
                        onClick={() => {
                          setSelectedAnalysis(analysis.id);
                          setShowAnalysisPicker(false);
                        }}
                        className={`w-full px-4 py-3 text-left hover:bg-gray-800 transition-colors ${
                          selectedAnalysis === analysis.id ? 'bg-gray-800' : ''
                        }`}
                      >
                        <p className="text-sm text-white font-medium">{analysis.name}</p>
                        <p className="text-xs text-gray-500 mt-1">
                          {analysis.node_count} genes | {analysis.edge_count} edges | {analysis.hub_count} hubs
                        </p>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Stats */}
            {graphData && (
              <div className="flex gap-4 text-sm text-gray-400">
                <span className="flex items-center gap-1">
                  <Target className="w-4 h-4 text-amber-400" />
                  {graphData.stats.hub_count} hubs
                </span>
                <span className="flex items-center gap-1">
                  <TrendingUp className="w-4 h-4 text-red-400" />
                  {graphData.stats.up_regulated}
                </span>
                <span className="flex items-center gap-1">
                  <TrendingDown className="w-4 h-4 text-blue-400" />
                  {graphData.stats.down_regulated}
                </span>
                <span className="flex items-center gap-1">
                  <Network className="w-4 h-4" />
                  {graphData.stats.total_edges} edges
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && searchQuery) {
                    focusOnGene(searchQuery);
                  }
                }}
                placeholder="Search gene..."
                className="pl-10 pr-4 py-2 bg-white/10 border border-gray-700 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-amber-500 w-48"
              />
            </div>

            {/* Filter Toggle */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`p-2 rounded-lg transition-colors ${
                showFilters ? 'bg-amber-500/20 text-amber-400' : 'bg-white/10 text-gray-400 hover:bg-white/20'
              }`}
            >
              <Filter className="w-5 h-5" />
            </button>

            {/* Close */}
            <button
              onClick={onClose}
              className="p-2 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Filter Panel */}
        {showFilters && (
          <div className="mt-4 max-w-7xl mx-auto">
            <div className="bg-gray-900/80 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
              <div className="flex flex-wrap gap-4">
                {/* Hub Only */}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filters.hubOnly}
                    onChange={(e) => setFilters(f => ({ ...f, hubOnly: e.target.checked }))}
                    className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-amber-500 focus:ring-amber-500"
                  />
                  <span className="text-sm text-gray-300">Hub genes only</span>
                </label>

                {/* DB Matched */}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filters.dbMatchedOnly}
                    onChange={(e) => setFilters(f => ({ ...f, dbMatchedOnly: e.target.checked }))}
                    className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-green-500 focus:ring-green-500"
                  />
                  <span className="text-sm text-gray-300">DB validated only</span>
                </label>

                {/* Up Regulated */}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filters.showUpRegulated}
                    onChange={(e) => setFilters(f => ({ ...f, showUpRegulated: e.target.checked }))}
                    className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-red-500 focus:ring-red-500"
                  />
                  <span className="text-sm text-gray-300">Upregulated</span>
                </label>

                {/* Down Regulated */}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filters.showDownRegulated}
                    onChange={(e) => setFilters(f => ({ ...f, showDownRegulated: e.target.checked }))}
                    className="w-4 h-4 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-300">Downregulated</span>
                </label>

                {/* Correlation Threshold */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-400">Min Correlation:</span>
                  <input
                    type="range"
                    min="0.5"
                    max="0.95"
                    step="0.05"
                    value={filters.minCorrelation}
                    onChange={(e) => setFilters(f => ({ ...f, minCorrelation: parseFloat(e.target.value) }))}
                    className="w-24 accent-amber-500"
                  />
                  <span className="text-sm text-white">{filters.minCorrelation.toFixed(2)}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Zoom Controls */}
      <div className="absolute bottom-6 left-6 z-20 flex flex-col gap-2">
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
      <div className="absolute bottom-6 right-6 z-20 bg-black/70 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
        <h3 className="text-sm font-semibold text-white mb-3">Legend</h3>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-amber-500 shadow-lg shadow-amber-500/50" />
            <span className="text-xs text-gray-300">Hub Gene</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.up }} />
            <span className="text-xs text-gray-300">Upregulated</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.down }} />
            <span className="text-xs text-gray-300">Downregulated</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full border-2" style={{ borderColor: COLORS.dbValidated }} />
            <span className="text-xs text-gray-300">DB Validated</span>
          </div>
          <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-700">
            <div className="w-6 h-0.5 bg-green-400" />
            <span className="text-xs text-gray-300">+ Correlation</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-0.5 bg-red-400" />
            <span className="text-xs text-gray-300">- Correlation</span>
          </div>
        </div>
      </div>

      {/* Selected Node Details */}
      {selectedNode && (
        <div className="absolute top-24 right-6 z-20 w-80 bg-gray-900/95 backdrop-blur-sm rounded-xl border border-gray-700 overflow-hidden">
          <div className="p-4 border-b border-gray-700 bg-gradient-to-r from-gray-800/50 to-transparent">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: selectedNode.color }}
                />
                <h3 className="font-bold text-white">
                  {selectedNode.gene_symbol || selectedNode.id.split('.')[0]}
                </h3>
                {selectedNode.is_hub && (
                  <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded-full flex items-center gap-1">
                    <Sparkles className="w-3 h-3" /> Hub
                  </span>
                )}
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="p-1 hover:bg-white/10 rounded"
              >
                <X className="w-4 h-4 text-gray-400" />
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-1">{selectedNode.id}</p>
          </div>

          <div className="p-4 space-y-4">
            {/* Expression */}
            <div>
              <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">Expression</h4>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-gray-800 rounded-lg p-2">
                  <p className="text-xs text-gray-500">log2FC</p>
                  <p className={`text-lg font-bold ${
                    selectedNode.log2FC > 0 ? 'text-red-400' : 'text-blue-400'
                  }`}>
                    {selectedNode.log2FC > 0 ? '+' : ''}{selectedNode.log2FC.toFixed(2)}
                  </p>
                </div>
                <div className="bg-gray-800 rounded-lg p-2">
                  <p className="text-xs text-gray-500">padj</p>
                  <p className="text-lg font-bold text-white">
                    {selectedNode.padj < 0.001 ? selectedNode.padj.toExponential(1) : selectedNode.padj.toFixed(3)}
                  </p>
                </div>
              </div>
            </div>

            {/* Network Metrics */}
            <div>
              <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">Network</h4>
              <div className="grid grid-cols-3 gap-2">
                <div className="text-center">
                  <p className="text-lg font-bold text-white">{selectedNode.degree}</p>
                  <p className="text-xs text-gray-500">Degree</p>
                </div>
                <div className="text-center">
                  <p className="text-lg font-bold text-white">{(selectedNode.hub_score * 100).toFixed(0)}%</p>
                  <p className="text-xs text-gray-500">Hub Score</p>
                </div>
                <div className="text-center">
                  <p className="text-lg font-bold text-white">{(selectedNode.betweenness * 1000).toFixed(1)}</p>
                  <p className="text-xs text-gray-500">Between.</p>
                </div>
              </div>
            </div>

            {/* Validation */}
            {selectedNode.db_matched && (
              <div>
                <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">Validation</h4>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.db_sources.map((source, i) => (
                    <span key={i} className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full">
                      {source}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Tags */}
            {selectedNode.tags.length > 0 && (
              <div>
                <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">Tags</h4>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.tags.map((tag, i) => (
                    <span key={i} className="px-2 py-1 bg-purple-500/20 text-purple-400 text-xs rounded-full">
                      {tag.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Confidence */}
            <div className="flex items-center justify-between pt-2 border-t border-gray-700">
              <span className="text-xs text-gray-500">Confidence</span>
              <span className={`px-2 py-1 text-xs rounded-full ${
                selectedNode.confidence === 'high' ? 'bg-green-500/20 text-green-400' :
                selectedNode.confidence === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                selectedNode.confidence === 'novel_candidate' ? 'bg-purple-500/20 text-purple-400' :
                'bg-gray-500/20 text-gray-400'
              }`}>
                {selectedNode.confidence.replace(/_/g, ' ')}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Hover Tooltip */}
      {hoveredNode && !selectedNode && (
        <div className="absolute top-24 left-1/2 -translate-x-1/2 z-20 bg-black/90 backdrop-blur-sm rounded-lg px-4 py-2 border border-gray-700">
          <span className="text-sm font-medium text-white">
            {hoveredNode.gene_symbol || hoveredNode.id.split('.')[0]}
          </span>
          <span className={`ml-2 text-xs ${
            hoveredNode.direction === 'up' ? 'text-red-400' : 'text-blue-400'
          }`}>
            {hoveredNode.log2FC > 0 ? '+' : ''}{hoveredNode.log2FC.toFixed(2)}
          </span>
          {hoveredNode.is_hub && (
            <span className="ml-2 text-xs text-amber-400">Hub</span>
          )}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-30">
          <div className="text-center">
            <Loader2 className="w-12 h-12 text-amber-500 animate-spin mx-auto mb-4" />
            <p className="text-white">Building gene network...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-30">
          <div className="text-center">
            <p className="text-red-400 mb-4">Error: {error}</p>
            <button
              onClick={() => setError(null)}
              className="px-4 py-2 bg-red-500/20 text-red-300 rounded-lg hover:bg-red-500/30"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* No Analysis Selected */}
      {!selectedAnalysis && !loading && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="text-center">
            <Database className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg mb-4">Select an analysis to visualize</p>
            <button
              onClick={() => setShowAnalysisPicker(true)}
              className="px-6 py-3 bg-amber-500/20 text-amber-400 rounded-xl hover:bg-amber-500/30 transition-colors"
            >
              Choose Analysis
            </button>
          </div>
        </div>
      )}

      {/* 3D Graph */}
      {filteredData && !loading && (
        <ForceGraph3D
          ref={fgRef}
          graphData={filteredData}
          nodeId="id"
          nodeLabel={(node: NetworkNode) =>
            `${node.gene_symbol || node.id.split('.')[0]} (${node.direction})`
          }
          nodeColor={(node: NetworkNode) => node.color || COLORS.unchanged}
          nodeVal={(node: NetworkNode) => (node.size || 5) * 2}
          nodeThreeObject={nodeThreeObject}
          nodeThreeObjectExtend={false}
          linkSource="source"
          linkTarget="target"
          linkColor={linkColor}
          linkWidth={linkWidth}
          linkOpacity={0.4}
          linkDirectionalParticles={1}
          linkDirectionalParticleSpeed={0.003}
          linkDirectionalParticleWidth={1}
          linkDirectionalParticleColor={() => '#fbbf24'}
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
          backgroundColor={COLORS.background}
          showNavInfo={false}
          enableNodeDrag={true}
          enableNavigationControls={true}
          controlType="orbit"
          d3AlphaDecay={0.015}
          d3VelocityDecay={0.25}
          warmupTicks={100}
          cooldownTicks={200}
        />
      )}

      {/* Instructions */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 text-center text-gray-600 text-xs">
        <p>Drag to rotate | Scroll to zoom | Click node for details | Search by gene symbol</p>
      </div>
    </div>
  );
};

export default GeneNetworkGraph;
