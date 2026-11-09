import { BACKEND_BASE_URL } from "../config.js";

/**
 * Fetches JSON from the backend and throws on HTTP errors.
 * @param {string} path - Backend path.
 * @param {RequestInit} [options={}] - Fetch options.
 * @returns {Promise<any>}
 */
export async function fetchJson(path, options = {}) {
    const response = await fetch(`${BACKEND_BASE_URL}${path}`, options);
    let data = null;
    try {
        data = await response.json();
    } catch {
        data = null;
    }

    if (!response.ok) {
        const errorMessage =
            data?.error ||
            data?.message ||
            `Request failed: ${response.status}`;
        throw new Error(errorMessage);
    }

    return data;
}
