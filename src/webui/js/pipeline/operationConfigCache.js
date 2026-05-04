import { BACKEND_BASE_URL } from "../config.js";

/**
 * Caches and prefetches operation configuration data for the web UI pipeline.
 */

const _cache = new Map();

/**
 * Builds the internal cache key for an operation configuration.
 *
 * @param {string} name - Operation name.
 * @param {boolean} isSecondary - Whether the operation is secondary.
 * @returns {string} Cache key.
 */
function cacheKey(name, isSecondary) {
    return `${name}:${isSecondary ? 1 : 0}`;
}

/**
 * Checks whether a configuration is already cached.
 *
 * @param {string} name - Operation name.
 * @param {boolean} isSecondary - Whether the operation is secondary.
 * @returns {boolean} True if cached.
 */
export function hasCachedConfig(name, isSecondary) {
    return _cache.has(cacheKey(name, isSecondary));
}

/**
 * Retrieves a cached configuration.
 *
 * @param {string} name - Operation name.
 * @param {boolean} isSecondary - Whether the operation is secondary.
 * @returns {*} Cached configuration, or undefined if absent.
 */
export function getCachedConfig(name, isSecondary) {
    return _cache.get(cacheKey(name, isSecondary));
}

/**
 * Stores a configuration in the cache.
 *
 * @param {string} name - Operation name.
 * @param {boolean} isSecondary - Whether the operation is secondary.
 * @param {*} data - Configuration data to cache.
 * @returns {void}
 */
export function setCachedConfig(name, isSecondary, data) {
    _cache.set(cacheKey(name, isSecondary), data);
}

/**
 * Prefetches uncached operation configurations in a single batch request.
 *
 * @param {Array<{name: string, isSecondary: boolean}>} operations - Operations to prefetch.
 * @returns {Promise<void>}
 */
export async function prefetchConfigs(operations) {
    const uncached = operations.filter(
        ({ name, isSecondary }) => !hasCachedConfig(name, isSecondary),
    );

    if (uncached.length === 0) return;

    const body = uncached.map(({ name, isSecondary }) => ({
        name,
        is_secondary: isSecondary ? 1 : 0,
    }));

    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-operation-config-data-batch`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            },
        );

        if (!response.ok) return;

        const data = await response.json();
        for (const [key, config] of Object.entries(data)) {
            _cache.set(key, config);
        }
    } catch (_) {}
}
