import {
    closeOnBackdropClick,
    closeOnEscape,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "./modal.js";

/**
 * Confirmation dialog UI helpers for rendering and controlling the modal.
 */

const OVERLAY_ID = "confirmationDialogOverlay";
const MODAL_ID = "confirmationDialogModal";

let initialized = false;
let activeResolve = null;

/**
 * Returns the shared overlay and modal elements used by the confirmation dialog.
 *
 * @returns {{overlay: HTMLElement, modal: HTMLElement}}
 */
function getDialogElements() {
    return getOrCreateModalElements({
        overlayId: OVERLAY_ID,
        modalId: MODAL_ID,
        modalClassName:
            "bg-[#1a1a1a] rounded-xl shadow-2xl w-auto min-w-[20rem] max-w-[min(92vw,34rem)] mx-4 border border-[#414141] overflow-hidden",
    });
}

/**
 * Resolves the active confirmation promise and hides the dialog.
 *
 * @param {boolean} value - The result to resolve with.
 */
function resolveDialog(value) {
    if (activeResolve) {
        activeResolve(value);
        activeResolve = null;
    }
    hideModal(getDialogElements().overlay);
}

/**
 * Attaches one-time dismissal handlers for the confirmation dialog.
 */
function initializeDialog() {
    if (initialized) {
        return;
    }
    const { overlay } = getDialogElements();
    closeOnBackdropClick(overlay, () => resolveDialog(false));
    closeOnEscape(overlay, () => resolveDialog(false));
    initialized = true;
}

/**
 * Normalizes a string or array input into a list of non-empty trimmed lines.
 *
 * @param {string|string[]} value - The input value to normalize.
 * @returns {string[]}
 */
function normalizeLines(value) {
    if (Array.isArray(value)) {
        return value.filter(Boolean);
    }
    return String(value || "")
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
}

/**
 * Displays a confirmation dialog and resolves with the user's choice.
 *
 * @param {Object} [options={}] - Dialog options.
 * @param {string} [options.title="Confirm Action"] - Dialog title.
 * @param {string|string[]} [options.message="Are you sure?"] - Main message content.
 * @param {string|string[]} [options.detail=""] - Additional detail content.
 * @param {string} [options.confirmText="Confirm"] - Confirm button label.
 * @param {string} [options.cancelText="Cancel"] - Cancel button label.
 * @param {string} [options.variant="danger"] - Visual variant.
 * @returns {Promise<boolean>} Resolves true when confirmed, false otherwise.
 */
export function confirmDialog({
    title = "Confirm Action",
    message = "Are you sure?",
    detail = "",
    confirmText = "Confirm",
    cancelText = "Cancel",
    variant = "danger",
} = {}) {
    initializeDialog();

    if (activeResolve) {
        resolveDialog(false);
    }

    const { overlay, modal } = getDialogElements();
    modal.innerHTML = "";

    const isDanger = variant === "danger";
    const accentClass = isDanger ? "text-red-300" : "text-yellow-400";
    const iconClass = isDanger
        ? "bg-red-900/50 text-red-200 border-red-700/70"
        : "bg-yellow-500/15 text-yellow-300 border-yellow-500/40";
    const confirmClass = isDanger
        ? "bg-red-800 text-white border-red-600 hover:bg-red-700"
        : "bg-[#f9c845] text-[#232323] border-[#d4a83a] hover:bg-[#d4a83a]";

    const messageNodes = normalizeLines(message).map((line) =>
        createElement("p", { className: "text-gray-200", text: line }),
    );
    const detailNodes = normalizeLines(detail).map((line) =>
        createElement("p", { className: "text-sm text-gray-400", text: line }),
    );

    modal.appendChild(
        createElement("div", { className: "flex flex-col" }, [
            createElement("div", { className: "p-5 pb-4" }, [
                createElement("div", { className: "flex items-start gap-3" }, [
                    createElement("div", {
                        className: `mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border text-base font-bold ${iconClass}`,
                        text: isDanger ? "!" : "?",
                    }),
                    createElement("div", { className: "min-w-0 flex-1" }, [
                        createElement("h3", {
                            className: `text-lg font-bold leading-tight ${accentClass} mb-2`,
                            text: title,
                        }),
                        createElement("div", {
                            className: "space-y-1.5 text-sm leading-relaxed",
                        }, [...messageNodes, ...detailNodes]),
                    ]),
                ]),
            ]),
            createElement("div", {
                className:
                    "flex items-center justify-between gap-3 border-t border-[#333333] bg-[#171717] px-5 py-3",
            }, [
                createElement("button", {
                    type: "button",
                    className:
                        "rounded-md border border-[#414141] bg-[#242424] px-3.5 py-2 text-sm font-semibold text-[#f9c845] transition-colors hover:bg-[#303030]",
                    text: cancelText,
                    onclick: () => resolveDialog(false),
                }),
                createElement("button", {
                    type: "button",
                    className: `rounded-md border px-3.5 py-2 text-sm font-semibold transition-colors ${confirmClass}`,
                    text: confirmText,
                    onclick: () => resolveDialog(true),
                }),
            ]),
        ]),
    );

    showModal(overlay);

    return new Promise((resolve) => {
        activeResolve = resolve;
    });
}
