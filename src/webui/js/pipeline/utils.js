/**
 * Shared utility helpers used by the pipeline UI.
 */

// --- Helpers

/**
 * Escapes HTML special characters in a string.
 *
 * @param {any} s - The value to escape.
 * @returns {string} The escaped string.
 */
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

/**
 * Generates a short unique identifier string.
 *
 * @param {string} [prefix=""] - Optional prefix for the identifier.
 * @returns {string} The generated identifier.
 */
export function uid(prefix = "") {
    return `${prefix}${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

/**
 * Creates a debounced version of the provided function.
 *
 * The debounced function delays invoking {@link fn} until after {@link delay}
 * milliseconds have elapsed since the last time the debounced function was called.
 * The debounced function preserves the original `this` context and arguments.
 *
 * @param {Function} fn - The function to debounce.
 * @param {number} delay - The delay in milliseconds to wait.
 * @returns {Function} A debounced function.
 *
 * @example
 * const debouncedResize = debounce(() => {
 *     console.log("Window resized");
 * }, 200);
 * window.addEventListener("resize", debouncedResize);
 */
export function debounce(fn, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}
