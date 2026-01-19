import React, { useState, useEffect } from 'react';
import {
  Library, FolderPlus, Search, Filter, Grid3X3, List, Star,
  MoreVertical, Trash2, FolderOpen, FileText, Calendar, Tag,
  X, Plus, Edit2, Check, ChevronRight, Download, BookOpen, ExternalLink
} from 'lucide-react';
import api from '../services/client';

interface Paper {
  id: string;
  title: string;
  authors: string[];
  year: number;
  journal?: string;
  abstract?: string;
  addedAt: Date;
  isStarred: boolean;
  tags: string[];
  collectionId?: string;
  doi?: string;
  pmid?: string;
}

interface Collection {
  id: string;
  name: string;
  color: string;
  paperCount: number;
  createdAt: Date;
}

interface ResearchLibraryProps {
  isOpen: boolean;
  onClose: () => void;
}

const COLORS = [
  'from-violet-500 to-purple-500',
  'from-blue-500 to-cyan-500',
  'from-emerald-500 to-teal-500',
  'from-orange-500 to-red-500',
  'from-pink-500 to-rose-500',
  'from-indigo-500 to-blue-500',
];

export const ResearchLibrary: React.FC<ResearchLibraryProps> = ({ isOpen, onClose }) => {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [collections, setCollections] = useState<Collection[]>([
    { id: 'default', name: 'All Papers', color: COLORS[0], paperCount: 0, createdAt: new Date() },
    { id: 'starred', name: 'Starred', color: COLORS[1], paperCount: 0, createdAt: new Date() },
  ]);
  const [selectedCollection, setSelectedCollection] = useState<string>('default');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [isCreatingCollection, setIsCreatingCollection] = useState(false);
  const [newCollectionName, setNewCollectionName] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Load papers from all domains on mount
  useEffect(() => {
    if (isOpen) {
      loadPapers();
    }
  }, [isOpen]);

  const loadPapers = async () => {
    setIsLoading(true);
    try {
      const domains = ['pancreatic_cancer', 'blood_cancer', 'glioblastoma', 'alzheimer', 'pcos'];
      const allPapers: Paper[] = [];

      for (const domain of domains) {
        try {
          const response = await api.listPapers(domain);
          const domainPapers = response.papers.map((p, idx) => ({
            id: `${domain}-${idx}`,
            title: p.title,
            authors: p.authors || [],
            year: p.year || 2024,
            journal: domain.replace('_', ' ').toUpperCase(),
            addedAt: new Date(),
            isStarred: false,
            tags: [domain],
            doi: p.doi,
          }));
          allPapers.push(...domainPapers);
        } catch (e) {
          console.log(`Error loading ${domain}:`, e);
        }
      }

      // Deduplicate
      const uniquePapers = allPapers.reduce((acc, paper) => {
        if (!acc.find(p => p.title === paper.title)) {
          acc.push(paper);
        }
        return acc;
      }, [] as Paper[]);

      setPapers(uniquePapers);

      // Update collection counts
      setCollections(prev => prev.map(c => ({
        ...c,
        paperCount: c.id === 'default' ? uniquePapers.length : c.id === 'starred' ? uniquePapers.filter(p => p.isStarred).length : 0
      })));
    } catch (err) {
      console.error('Failed to load papers:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Filter papers
  const filteredPapers = papers.filter(paper => {
    const matchesSearch = paper.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      paper.tags.some(t => t.toLowerCase().includes(searchQuery.toLowerCase()));

    if (selectedCollection === 'starred') {
      return matchesSearch && paper.isStarred;
    }
    if (selectedCollection !== 'default') {
      return matchesSearch && paper.collectionId === selectedCollection;
    }
    return matchesSearch;
  });

  // Toggle star
  const toggleStar = (id: string) => {
    setPapers(papers.map(p =>
      p.id === id ? { ...p, isStarred: !p.isStarred } : p
    ));
    // Update starred count
    setCollections(prev => prev.map(c =>
      c.id === 'starred' ? { ...c, paperCount: papers.filter(p => p.id === id ? !p.isStarred : p.isStarred).length } : c
    ));
  };

  // Create collection
  const createCollection = () => {
    if (!newCollectionName.trim()) return;

    const newCollection: Collection = {
      id: `col-${Date.now()}`,
      name: newCollectionName,
      color: COLORS[collections.length % COLORS.length],
      paperCount: 0,
      createdAt: new Date(),
    };

    setCollections([...collections, newCollection]);
    setNewCollectionName('');
    setIsCreatingCollection(false);
  };

  // Delete paper
  const deletePaper = (id: string) => {
    setPapers(papers.filter(p => p.id !== id));
  };

  // Open paper in new tab
  const openPaper = (paper: Paper) => {
    if (paper.doi) {
      window.open(`https://doi.org/${paper.doi}`, '_blank');
    } else if (paper.pmid) {
      window.open(`https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`, '_blank');
    } else {
      // Fallback: search on PubMed by title
      const searchUrl = `https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(paper.title)}`;
      window.open(searchUrl, '_blank');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] bg-gradient-to-br from-violet-50 via-purple-50 to-indigo-100 flex">
      {/* Sidebar */}
      <div className="w-64 glass-4 border-r border-purple-100/50 flex flex-col">
        <div className="p-4 border-b border-purple-100/50">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Library className="w-4 h-4 text-white" />
              </div>
              <span className="font-bold text-gray-900">Library</span>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-purple-100/50 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </div>

        {/* Collections */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Collections</p>
            <button
              onClick={() => setIsCreatingCollection(true)}
              className="p-1 hover:bg-purple-100/50 rounded transition-colors"
            >
              <Plus className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          {/* New Collection Input */}
          {isCreatingCollection && (
            <div className="mb-2 flex items-center gap-2">
              <input
                type="text"
                value={newCollectionName}
                onChange={(e) => setNewCollectionName(e.target.value)}
                placeholder="Collection name"
                className="flex-1 px-3 py-2 glass-3 border border-purple-200/50 rounded-lg text-sm focus:ring-2 focus:ring-purple-400/50"
                autoFocus
                onKeyDown={(e) => e.key === 'Enter' && createCollection()}
              />
              <button
                onClick={createCollection}
                className="p-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
              >
                <Check className="w-4 h-4" />
              </button>
            </div>
          )}

          <div className="space-y-1">
            {collections.map((collection) => (
              <button
                key={collection.id}
                onClick={() => setSelectedCollection(collection.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${
                  selectedCollection === collection.id
                    ? 'bg-gradient-to-r from-violet-500/10 to-purple-500/10 border border-purple-300/50'
                    : 'hover:bg-purple-50/50'
                }`}
              >
                <div className={`w-3 h-3 rounded-full bg-gradient-to-r ${collection.color}`} />
                <span className="flex-1 text-left text-sm font-medium text-gray-700">{collection.name}</span>
                <span className="text-xs text-gray-400">{collection.paperCount}</span>
              </button>
            ))}
          </div>

          {/* Tags Section */}
          <div className="mt-6">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Tags</p>
            <div className="flex flex-wrap gap-2">
              {['pancreatic_cancer', 'blood_cancer', 'glioblastoma', 'alzheimer', 'pcos'].map((tag) => (
                <button
                  key={tag}
                  onClick={() => setSearchQuery(tag)}
                  className="px-2.5 py-1 glass-2 border border-purple-200/50 rounded-full text-xs font-medium text-gray-600 hover:bg-purple-50/50 transition-colors"
                >
                  {tag.replace('_', ' ')}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="glass-3 border-b border-purple-100/50 px-6 py-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-gray-900">
              {collections.find(c => c.id === selectedCollection)?.name || 'Library'}
            </h2>
            <div className="flex items-center gap-3">
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search papers..."
                  className="pl-10 pr-4 py-2 w-64 glass-3 border border-purple-200/50 rounded-lg text-sm focus:ring-2 focus:ring-purple-400/50"
                />
              </div>

              {/* View Toggle */}
              <div className="flex items-center gap-1 glass-2 rounded-lg p-1">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded-md transition-colors ${viewMode === 'grid' ? 'bg-white shadow-sm text-purple-600' : 'text-gray-500 hover:text-gray-700'}`}
                >
                  <Grid3X3 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded-md transition-colors ${viewMode === 'list' ? 'bg-white shadow-sm text-purple-600' : 'text-gray-500 hover:text-gray-700'}`}
                >
                  <List className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Papers Grid/List */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin mx-auto mb-4" />
                <p className="text-gray-600">Loading your library...</p>
              </div>
            </div>
          ) : filteredPapers.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 gap-4">
              <div className="w-20 h-20 glass-3 rounded-2xl flex items-center justify-center">
                <BookOpen className="w-10 h-10 text-purple-400" />
              </div>
              <h3 className="text-lg font-semibold text-gray-800">No papers found</h3>
              <p className="text-gray-500">
                {searchQuery ? 'Try a different search term' : 'Add papers to your library to get started'}
              </p>
            </div>
          ) : viewMode === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredPapers.map((paper) => (
                <div
                  key={paper.id}
                  className="glass-3 rounded-2xl border border-purple-100/50 p-5 card-hover group"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex flex-wrap gap-1">
                      {paper.tags.slice(0, 2).map((tag) => (
                        <span key={tag} className="text-[10px] font-medium text-purple-600 bg-purple-100/80 px-2 py-0.5 rounded-full">
                          {tag.replace('_', ' ')}
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => toggleStar(paper.id)}
                        className={`p-1.5 rounded-lg transition-colors ${paper.isStarred ? 'text-yellow-500' : 'text-gray-400 hover:text-yellow-500'}`}
                      >
                        <Star className="w-4 h-4" fill={paper.isStarred ? 'currentColor' : 'none'} />
                      </button>
                      <button
                        onClick={() => deletePaper(paper.id)}
                        className="p-1.5 text-gray-400 hover:text-red-500 rounded-lg transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <h3
                    className="font-semibold text-gray-900 mb-2 line-clamp-2 text-sm cursor-pointer hover:text-purple-600 transition-colors"
                    onClick={() => openPaper(paper)}
                  >
                    {paper.title}
                  </h3>

                  <div className="flex items-center justify-between mt-auto">
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <Calendar className="w-3.5 h-3.5" />
                      <span>{paper.year}</span>
                    </div>
                    <button
                      onClick={() => openPaper(paper)}
                      className="p-1.5 text-purple-500 hover:text-purple-700 hover:bg-purple-100 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                      title="Open paper"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {filteredPapers.map((paper) => (
                <div
                  key={paper.id}
                  className="glass-3 rounded-xl border border-purple-100/50 p-4 flex items-center gap-4 card-hover group"
                >
                  <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <FileText className="w-5 h-5 text-purple-600" />
                  </div>

                  <div className="flex-1 min-w-0">
                    <h3
                      className="font-medium text-gray-900 truncate cursor-pointer hover:text-purple-600 transition-colors"
                      onClick={() => openPaper(paper)}
                    >
                      {paper.title}
                    </h3>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="text-xs text-gray-500">{paper.year}</span>
                      {paper.tags.slice(0, 2).map((tag) => (
                        <span key={tag} className="text-[10px] font-medium text-purple-600 bg-purple-100/80 px-2 py-0.5 rounded-full">
                          {tag.replace('_', ' ')}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => openPaper(paper)}
                      className="p-2 text-purple-500 hover:text-purple-700 hover:bg-purple-100 rounded-lg transition-colors"
                      title="Open paper"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => toggleStar(paper.id)}
                      className={`p-2 rounded-lg transition-colors ${paper.isStarred ? 'text-yellow-500' : 'text-gray-400 hover:text-yellow-500'}`}
                    >
                      <Star className="w-4 h-4" fill={paper.isStarred ? 'currentColor' : 'none'} />
                    </button>
                    <button
                      onClick={() => deletePaper(paper.id)}
                      className="p-2 text-gray-400 hover:text-red-500 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Stats Footer */}
        <div className="glass-3 border-t border-purple-100/50 px-6 py-3">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>{filteredPapers.length} papers</span>
            <span>{papers.filter(p => p.isStarred).length} starred</span>
          </div>
        </div>
      </div>
    </div>
  );
};
