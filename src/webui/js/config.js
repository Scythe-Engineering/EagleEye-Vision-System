// Configuration constants for the web UI
// This file centralizes all URL configurations to make the code more maintainable and environment-agnostic
//
// To override these URLs in different environments:
// 1. For JavaScript: Set window.BACKEND_BASE_URL and window.DEV_SERVER_BASE_URL before importing this module
// 2. For Python (web_server.py): Set the CORS_ORIGINS environment variable
//
// Example:
//   window.BACKEND_BASE_URL = "https://api.example.com";
//   window.DEV_SERVER_BASE_URL = "https://dev.example.com";

// Backend API configuration - can be overridden by setting window.BACKEND_BASE_URL
export const BACKEND_BASE_URL = typeof window !== 'undefined' && window.BACKEND_BASE_URL
    ? window.BACKEND_BASE_URL
    : "http://localhost:5001";

// Development server configuration (used for development builds) - can be overridden by setting window.DEV_SERVER_BASE_URL
export const DEV_SERVER_BASE_URL = typeof window !== 'undefined' && window.DEV_SERVER_BASE_URL
    ? window.DEV_SERVER_BASE_URL
    : "http://localhost:5173";

// Helper function to construct backend URLs
export function buildBackendUrl(path) {
    return `${BACKEND_BASE_URL}${path.startsWith('/') ? path : '/' + path}`;
}

// Helper function to construct development server URLs
export function buildDevUrl(path) {
    return `${DEV_SERVER_BASE_URL}${path.startsWith('/') ? path : '/' + path}`;
}
