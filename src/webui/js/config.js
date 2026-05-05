// Web UI configuration helpers for backend URL resolution.
// This module keeps browser/server URL handling centralized.

// Configuration constants for the web UI
// This file centralizes all URL configurations to make the code more maintainable and environment-agnostic
//
// To override the backend URL, set window.BACKEND_BASE_URL before importing this module.
//
// Example:
//   window.BACKEND_BASE_URL = "https://api.example.com";

const browserLocation = typeof window !== "undefined" ? window.location : null;

/**
 * Build an origin URL for the given port when running in a browser.
 *
 * @param {number} port - The target port.
 * @returns {string|null} The origin for the port, or null when unavailable.
 */
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

/**
 * Construct a backend URL from the configured base URL and a path.
 *
 * @param {string} path - The endpoint path, with or without a leading slash.
 * @returns {string} The full backend URL.
 */
export function buildBackendUrl(path) {
    return `${BACKEND_BASE_URL}${path.startsWith('/') ? path : '/' + path}`;
}
