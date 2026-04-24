import { BACKEND_BASE_URL } from "../config.js";

const _cache = new Map();

function cacheKey(name, isSecondary) {
    return `${name}:${isSecondary ? 1 : 0}`;
}

export function hasCachedConfig(name, isSecondary) {
    return _cache.has(cacheKey(name, isSecondary));
}

export function getCachedConfig(name, isSecondary) {
    return _cache.get(cacheKey(name, isSecondary));
}

export function setCachedConfig(name, isSecondary, data) {
    _cache.set(cacheKey(name, isSecondary), data);
}

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
