/**
 * API Configuration
 * Configure backend API endpoint based on environment
 */

// Determine API base URL
const getApiBaseUrl = () => {
  // In production (served by Nginx), use relative path
  if (import.meta.env.PROD) {
    return window.location.origin + '/api';
  }

  // In development, use environment variable or default to localhost
  return import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
};

export const API_BASE_URL = getApiBaseUrl();

// API Endpoints
export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL.replace('/api', '')}/health`,
  QUERY: `${API_BASE_URL}/query`,
  QUERY_STREAM: `${API_BASE_URL}/query/stream`,
  CACHE_STATS: `${API_BASE_URL}/cache/stats`,
  CACHE_CLEAR: `${API_BASE_URL}/cache`,
  COLLECTIONS: `${API_BASE_URL}/collections`,
  CREATE_COLLECTION: `${API_BASE_URL}/collections/create`,
  INGEST_TEXT: `${API_BASE_URL}/ingest/text`,
  INGEST_FILE: `${API_BASE_URL}/ingest/file`,
  MODELS: `${API_BASE_URL}/models`,
};

console.log('API Configuration:', {
  baseUrl: API_BASE_URL,
  endpoints: API_ENDPOINTS
});
