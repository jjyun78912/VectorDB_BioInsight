import React from 'react';
import { X } from 'lucide-react';

interface KnowledgeGraphProps {
  isOpen: boolean;
  onClose: () => void;
}

// Temporarily disabled due to WebGPU compatibility issues
export const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-2xl p-8 max-w-md mx-4 text-center">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 hover:bg-gray-100 rounded-full"
        >
          <X className="w-5 h-5" />
        </button>
        <h3 className="text-xl font-bold mb-4">3D Knowledge Graph</h3>
        <p className="text-gray-600 mb-4">
          이 기능은 현재 브라우저 호환성 문제로 일시적으로 비활성화되어 있습니다.
        </p>
        <p className="text-sm text-gray-500">
          곧 업데이트될 예정입니다.
        </p>
        <button
          onClick={onClose}
          className="mt-6 px-6 py-2 bg-purple-600 text-white rounded-full hover:bg-purple-700"
        >
          닫기
        </button>
      </div>
    </div>
  );
};

export default KnowledgeGraph;
