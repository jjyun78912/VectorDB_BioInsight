import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules/', 'src/test/'],
    },
  },
  resolve: {
    alias: [
      { find: '@/components', replacement: path.resolve(__dirname, './src/components') },
      { find: '@/hooks', replacement: path.resolve(__dirname, './src/hooks') },
      { find: '@/services', replacement: path.resolve(__dirname, './src/services') },
      { find: '@/stores', replacement: path.resolve(__dirname, './src/stores') },
      { find: '@/types', replacement: path.resolve(__dirname, './src/types') },
      { find: '@/lib', replacement: path.resolve(__dirname, './src/lib') },
      { find: '@/i18n', replacement: path.resolve(__dirname, './src/i18n') },
      { find: '@', replacement: path.resolve(__dirname, '.') },
    ],
  },
});
