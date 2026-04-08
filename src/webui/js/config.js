// Configuration constants for the web UI
// This file centralizes all URL configurations to make the code more maintainable and environment-agnostic
//
// To override these URLs in different environments:
// 1. For JavaScript: Set window.BACKEND_BASE_URL and window.DEV_SERVER_BASE_URL before importing this module
//
// Example:
//   window.BACKEND_BASE_URL = "https://api.example.com";
//   window.DEV_SERVER_BASE_URL = "https://dev.example.com";

const browserLocation = typeof window !== "undefined" ? window.location : null;

function buildOriginForPort(port) {
    if (!browserLocation) {
        return null;
    }

    if (browserLocation.port === String(port)) {
        return browserLocation.origin;
    }

    return `${browserLocation.protocol}//${browserLocation.hostname}:${port}`;
}

// Backend API configuration - can be overridden by setting window.BACKEND_BASE_URL
export const BACKEND_BASE_URL = typeof window !== 'undefined' && window.BACKEND_BASE_URL
    ? window.BACKEND_BASE_URL
    : buildOriginForPort(5001) ?? "http://localhost:5001";

// Development server configuration (used for development builds) - can be overridden by setting window.DEV_SERVER_BASE_URL
export const DEV_SERVER_BASE_URL = typeof window !== 'undefined' && window.DEV_SERVER_BASE_URL
    ? window.DEV_SERVER_BASE_URL
    : buildOriginForPort(5173) ?? "http://localhost:5173";

// Helper function to construct backend URLs
// @param {string} path - The endpoint path (with or without leading slash)
// @returns {string} A complete URL with the backend base URL and normalized path
export function buildBackendUrl(path) {
    return `${BACKEND_BASE_URL}${path.startsWith('/') ? path : '/' + path}`;
}

// Helper function to construct development server URLs
// @param {string} path - The endpoint path (with or without leading slash)
// @returns {string} A complete URL with the development server base URL and normalized path
export function buildDevUrl(path) {
    return `${DEV_SERVER_BASE_URL}${path.startsWith('/') ? path : '/' + path}`;
}
