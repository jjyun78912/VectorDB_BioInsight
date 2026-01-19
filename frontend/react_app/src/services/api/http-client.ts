/**
 * Base HTTP Client
 * Centralized fetch wrapper with error handling
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public statusText: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

interface RequestConfig extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
}

class HttpClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private buildUrl(endpoint: string, params?: Record<string, string | number | boolean | undefined>): string {
    const url = `${this.baseUrl}${endpoint}`;
    if (!params) return url;

    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.set(key, String(value));
      }
    });

    const queryString = searchParams.toString();
    return queryString ? `${url}?${queryString}` : url;
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const errorDetail = await response.text().catch(() => response.statusText);
      throw new ApiError(
        `Request failed (${response.status}): ${errorDetail}`,
        response.status,
        response.statusText
      );
    }
    return response.json();
  }

  async get<T>(endpoint: string, config?: RequestConfig): Promise<T> {
    const { params, ...fetchConfig } = config || {};
    const url = this.buildUrl(endpoint, params);
    const response = await fetch(url, {
      method: 'GET',
      ...fetchConfig,
    });
    return this.handleResponse<T>(response);
  }

  async post<T>(endpoint: string, body?: unknown, config?: RequestConfig): Promise<T> {
    const { params, ...fetchConfig } = config || {};
    const url = this.buildUrl(endpoint, params);
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...fetchConfig.headers,
      },
      body: body ? JSON.stringify(body) : undefined,
      ...fetchConfig,
    });
    return this.handleResponse<T>(response);
  }

  async postForm<T>(endpoint: string, formData: FormData, config?: RequestConfig): Promise<T> {
    const { params, ...fetchConfig } = config || {};
    const url = this.buildUrl(endpoint, params);
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      ...fetchConfig,
    });
    return this.handleResponse<T>(response);
  }

  async delete<T = void>(endpoint: string, config?: RequestConfig): Promise<T> {
    const { params, ...fetchConfig } = config || {};
    const url = this.buildUrl(endpoint, params);
    const response = await fetch(url, {
      method: 'DELETE',
      ...fetchConfig,
    });
    if (response.status === 204) {
      return undefined as T;
    }
    return this.handleResponse<T>(response);
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }
}

export const httpClient = new HttpClient();
export default httpClient;
