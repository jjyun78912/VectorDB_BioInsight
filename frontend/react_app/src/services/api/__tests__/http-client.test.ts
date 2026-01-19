import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { httpClient, ApiError } from '../http-client';

describe('HttpClient', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.resetAllMocks();
  });

  describe('get', () => {
    it('makes GET request with correct URL', async () => {
      const mockResponse = { data: 'test' };
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await httpClient.get('/test-endpoint');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/test-endpoint'),
        expect.objectContaining({ method: 'GET' })
      );
      expect(result).toEqual(mockResponse);
    });

    it('includes query params in URL', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await httpClient.get('/test', { params: { foo: 'bar', num: 123 } });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('foo=bar'),
        expect.anything()
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('num=123'),
        expect.anything()
      );
    });

    it('excludes undefined params', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await httpClient.get('/test', { params: { foo: 'bar', undef: undefined } });

      const calledUrl = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(calledUrl).toContain('foo=bar');
      expect(calledUrl).not.toContain('undef');
    });

    it('throws ApiError on non-ok response', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        text: () => Promise.resolve('Resource not found'),
      });

      try {
        await httpClient.get('/not-found');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError);
        expect((error as ApiError).status).toBe(404);
        expect((error as ApiError).statusText).toBe('Not Found');
      }
    });
  });

  describe('post', () => {
    it('makes POST request with JSON body', async () => {
      const mockResponse = { success: true };
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const body = { name: 'test', value: 123 };
      const result = await httpClient.post('/test-endpoint', body);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/test-endpoint'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(body),
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('handles POST without body', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      });

      await httpClient.post('/test-endpoint');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          method: 'POST',
          body: undefined,
        })
      );
    });
  });

  describe('postForm', () => {
    it('makes POST request with FormData', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ uploaded: true }),
      });

      const formData = new FormData();
      formData.append('file', new Blob(['test']), 'test.txt');

      await httpClient.postForm('/upload', formData);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/upload'),
        expect.objectContaining({
          method: 'POST',
          body: formData,
        })
      );
    });
  });

  describe('delete', () => {
    it('makes DELETE request', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ deleted: true }),
      });

      const result = await httpClient.delete('/resource/123');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/resource/123'),
        expect.objectContaining({ method: 'DELETE' })
      );
      expect(result).toEqual({ deleted: true });
    });

    it('handles 204 No Content response', async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const result = await httpClient.delete('/resource/123');

      expect(result).toBeUndefined();
    });
  });

  describe('ApiError', () => {
    it('has correct properties', () => {
      const error = new ApiError('Test error', 500, 'Internal Server Error');

      expect(error.message).toBe('Test error');
      expect(error.status).toBe(500);
      expect(error.statusText).toBe('Internal Server Error');
      expect(error.name).toBe('ApiError');
    });
  });
});
