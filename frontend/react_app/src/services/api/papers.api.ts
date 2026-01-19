/**
 * Papers API
 * Paper management, upload, and stats
 */

import { httpClient } from './http-client';
import type {
  PaperListResponse,
  StatsResponse,
  PaperUploadResponse,
} from '@/types/api.types';

export const papersApi = {
  /**
   * List all papers
   */
  listPapers: (domain: string = 'pheochromocytoma'): Promise<PaperListResponse> => {
    return httpClient.get('/papers', {
      params: { domain },
    });
  },

  /**
   * Get collection stats
   */
  getStats: (domain: string = 'pheochromocytoma'): Promise<StatsResponse> => {
    return httpClient.get('/papers/stats', {
      params: { domain },
    });
  },

  /**
   * Upload and index a PDF
   */
  uploadPaper: (file: File, domain: string = 'pheochromocytoma'): Promise<PaperUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('domain', domain);
    return httpClient.postForm('/papers/upload', formData);
  },
};

export default papersApi;
