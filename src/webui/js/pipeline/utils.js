/**
 * Utility functions for the pipeline creator
 */

// --- Helpers

export function escapeHtml(s) {
    return String(s).replace(
        /[&<>"']/g,
        (m) =>
            ({
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#39;",
            })[m],
    );
}

export function uid(prefix = "") {
    return `${prefix}${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

export function getIconSVG(name) {
    if (name === "grip") {
        return '<svg class="w-5 h-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path d="M7 4a1 1 0 110-2 1 1 0 010 2zm6 0a1 1 0 110-2 1 1 0 010 2zM7 10a1 1 0 110-2 1 1 0 010 2zm6 0a1 1 0 110-2 1 1 0 010 2zM7 16a1 1 0 110-2 1 1 0 010 2zm6 0a1 1 0 110-2 1 1 0 010 2z"/></svg>';
    }
    return "";
}

export function parseDropPayload(dataTransfer) {
    try {
        const json =
            dataTransfer.getData("application/pipeline") ||
            dataTransfer.getData("text/plain");
        if (!json) return null;
        const parsed = JSON.parse(json);
        if (parsed && (parsed.id || parsed.instanceId)) return parsed;
        return null;
    } catch (error) {
        console.warn("Failed to parse drop payload:", error);
        return null;
    }
}

export function debounce(fn, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}
