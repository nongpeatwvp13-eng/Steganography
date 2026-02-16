import { defineConfig } from 'vite'

export default defineConfig({
  root: 'Frontend',
  publicDir: 'public',
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:5000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  build: {
    outDir: '../dist', 
    emptyOutDir: true,
  },
  optimizeDeps: {
    include: ['chart.js']
  }
})