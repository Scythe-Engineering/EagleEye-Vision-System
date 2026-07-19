// Provides direct editing, formatting, and linting for pipeline_config.json.
import {
    closeOnBackdropClick,
    closeOnEscape,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "../../ui/modal.js";
import { confirmDialog } from "../../ui/confirmationDialog.js";
import { showDanger, showSuccess } from "../../ui/notificationSystem.js";
import { fetchPipelineConfigJson, savePipelineConfigJson } from "./dataApi.js";

const OVERLAY_ID = "pipelineJsonEditorOverlay";
const MODAL_ID = "pipelineJsonEditorModal";

/**
 * Convert a JSON syntax error position into a human-readable line and column.
 *
 * @param {string} content - Editor content.
 * @param {Error} error - JSON parse error.
 * @returns {{line: number|null, column: number|null}}
 */
function locateJsonError(content, error) {
    const lineColumnMatch = error.message.match(
        /line\s+(\d+)\s+column\s+(\d+)/i,
    );
    if (lineColumnMatch) {
        return {
            line: Number(lineColumnMatch[1]),
            column: Number(lineColumnMatch[2]),
        };
    }

    const positionMatch = error.message.match(/position\s+(\d+)/i);
    if (!positionMatch) {
        return { line: null, column: null };
    }

    const position = Math.min(Number(positionMatch[1]), content.length);
    const beforeError = content.slice(0, position);
    const lines = beforeError.split("\n");
    return {
        line: lines.length,
        column: lines.at(-1).length + 1,
    };
}

/**
 * Parse and lint editor content.
 *
 * @param {string} content - Raw JSON text.
 * @returns {{valid: boolean, value?: object, message: string, line?: number|null, column?: number|null}}
 */
export function lintPipelineJson(content) {
    try {
        const value = JSON.parse(content);
        if (!value || Array.isArray(value) || typeof value !== "object") {
            return {
                valid: false,
                message: "The root value must be a JSON object.",
            };
        }
        return {
            valid: true,
            value,
            message: `${Object.keys(value).length} pipeline${Object.keys(value).length === 1 ? "" : "s"} • Valid JSON`,
        };
    } catch (error) {
        const location = locateJsonError(content, error);
        return {
            valid: false,
            message: error.message,
            ...location,
        };
    }
}

/**
 * Build and initialize the pipeline JSON editor.
 *
 * @param {{button: HTMLElement|null, onSaved?: Function}} options - Editor controls.
 */
export function initializePipelineJsonEditor({ button, onSaved }) {
    if (!button) return;

    const { overlay, modal } = getOrCreateModalElements({
        overlayId: OVERLAY_ID,
        modalId: MODAL_ID,
        modalClassName:
            "bg-[#181818] rounded-2xl shadow-2xl w-full max-w-5xl mx-4 h-[82vh] flex flex-col border border-[#414141] overflow-hidden",
    });

    let originalContent = "";
    let loadedRevision = "";
    let saving = false;

    const title = createElement("h2", {
        className: "text-xl font-semibold text-[#f9c845]",
        text: "Pipeline JSON Editor",
    });
    const subtitle = createElement("p", {
        className: "text-xs text-[#999] mt-1",
        text: "Directly edit src/config/pipeline_config.json",
    });
    const closeButton = createElement("button", {
        type: "button",
        className:
            "w-9 h-9 rounded-lg border border-[#414141] text-gray-400 hover:text-white hover:bg-[#2a2a2a] transition-colors text-xl",
        text: "×",
        title: "Close JSON editor",
    });
    const header = createElement(
        "div",
        {
            className:
                "flex items-center justify-between px-5 py-4 bg-[#1e1e1e] border-b border-[#414141] shrink-0",
        },
        [createElement("div", {}, [title, subtitle]), closeButton],
    );

    const textarea = createElement("textarea", {
        className:
            "eagle-scrollbar flex-1 min-h-0 w-full resize-none bg-[#101010] text-[#e6e6e6] font-mono text-sm leading-6 p-5 outline-none border-0 focus:ring-0",
        spellcheck: "false",
        wrap: "off",
        "aria-label": "Pipeline JSON content",
    });
    const editorFrame = createElement(
        "div",
        {
            className:
                "flex flex-col flex-1 min-h-0 m-4 mb-3 rounded-xl overflow-hidden border border-[#343434] bg-[#101010] focus-within:border-[#f9c845]/70 transition-colors",
        },
        [textarea],
    );

    const lintStatus = createElement("div", {
        className: "text-sm min-w-0 truncate",
        text: "Loading...",
    });
    const cursorStatus = createElement("div", {
        className: "text-xs text-[#777] whitespace-nowrap",
        text: "Line 1, Column 1",
    });
    const formatButton = createElement("button", {
        type: "button",
        className:
            "px-4 py-2 rounded-lg bg-[#292929] text-[#f9c845] border border-[#414141] hover:border-[#f9c845] transition-colors",
        text: "Format JSON",
        title: "Format with four-space indentation",
    });
    const saveButton = createElement("button", {
        type: "button",
        className:
            "px-5 py-2 rounded-lg bg-[#f9c845] text-black font-semibold hover:bg-[#e6b73c] disabled:opacity-40 disabled:cursor-not-allowed transition-colors",
        text: "Save JSON",
    });
    const footer = createElement(
        "div",
        {
            className:
                "flex items-center gap-4 px-5 py-4 bg-[#1e1e1e] border-t border-[#414141] shrink-0",
        },
        [
            createElement("div", { className: "flex-1 min-w-0" }, [lintStatus]),
            cursorStatus,
            formatButton,
            saveButton,
        ],
    );
    modal.replaceChildren(header, editorFrame, footer);

    const updateCursorStatus = () => {
        const beforeCursor = textarea.value.slice(0, textarea.selectionStart);
        const lines = beforeCursor.split("\n");
        cursorStatus.textContent = `Line ${lines.length}, Column ${lines.at(-1).length + 1}`;
    };

    const updateLintStatus = () => {
        const result = lintPipelineJson(textarea.value);
        lintStatus.className = result.valid
            ? "text-sm min-w-0 truncate text-emerald-400"
            : "text-sm min-w-0 truncate text-red-300";
        const location =
            result.line && result.column
                ? `Line ${result.line}, column ${result.column}: `
                : "";
        lintStatus.textContent = `${result.valid ? "✓" : "✕"} ${location}${result.message}`;
        lintStatus.title = lintStatus.textContent;
        saveButton.disabled = !result.valid || saving;
        return result;
    };

    const hasUnsavedChanges = () => textarea.value !== originalContent;

    const closeEditor = async () => {
        if (saving) return;
        if (hasUnsavedChanges()) {
            const discard = await confirmDialog({
                title: "Discard JSON changes?",
                message: "The pipeline JSON contains unsaved changes.",
                detail: "Closing now will discard everything edited in this modal.",
                confirmText: "Discard Changes",
                variant: "warning",
            });
            if (!discard) return;
        }
        hideModal(overlay);
    };

    const openEditor = async () => {
        showModal(overlay);
        textarea.disabled = true;
        saveButton.disabled = true;
        lintStatus.className = "text-sm text-[#aaa]";
        lintStatus.textContent = "Loading pipeline_config.json...";
        try {
            const loadedConfig = await fetchPipelineConfigJson();
            originalContent = loadedConfig.content;
            loadedRevision = loadedConfig.revision;
            textarea.value = originalContent;
            textarea.disabled = false;
            updateLintStatus();
            updateCursorStatus();
            textarea.focus();
        } catch (error) {
            lintStatus.className = "text-sm text-red-300";
            lintStatus.textContent = `✕ ${error.message}`;
            showDanger("Failed to load pipeline JSON.");
        }
    };

    const formatContent = () => {
        const result = updateLintStatus();
        if (!result.valid) return;
        textarea.value = `${JSON.stringify(result.value, null, 4)}\n`;
        updateLintStatus();
        updateCursorStatus();
        textarea.focus();
    };

    const saveContent = async () => {
        const result = updateLintStatus();
        if (!result.valid || saving) return;
        saving = true;
        saveButton.disabled = true;
        saveButton.textContent = "Saving...";
        try {
            const savedConfig = await savePipelineConfigJson(
                textarea.value,
                loadedRevision,
            );
            loadedRevision = savedConfig.revision;
            originalContent = `${textarea.value.replace(/\s+$/, "")}\n`;
            textarea.value = originalContent;
            updateLintStatus();
            showSuccess(
                "Pipeline JSON saved. Restart required to apply changes.",
            );
            hideModal(overlay);
            await onSaved?.();
        } catch (error) {
            showDanger(error.message || "Failed to save pipeline JSON.");
            lintStatus.className = "text-sm min-w-0 truncate text-red-300";
            lintStatus.textContent = `✕ ${error.message}`;
        } finally {
            saving = false;
            saveButton.textContent = "Save JSON";
            saveButton.disabled = !lintPipelineJson(textarea.value).valid;
        }
    };

    textarea.addEventListener("input", updateLintStatus);
    textarea.addEventListener("click", updateCursorStatus);
    textarea.addEventListener("keyup", updateCursorStatus);
    textarea.addEventListener("keydown", (event) => {
        if (event.key === "Tab") {
            event.preventDefault();
            const start = textarea.selectionStart;
            textarea.setRangeText("    ", start, textarea.selectionEnd, "end");
            updateLintStatus();
            updateCursorStatus();
        } else if ((event.ctrlKey || event.metaKey) && event.key === "s") {
            event.preventDefault();
            void saveContent();
        }
    });

    button.addEventListener("click", () => void openEditor());
    closeButton.addEventListener("click", () => void closeEditor());
    formatButton.addEventListener("click", formatContent);
    saveButton.addEventListener("click", () => void saveContent());
    closeOnBackdropClick(overlay, () => void closeEditor());
    closeOnEscape(overlay, () => void closeEditor());
}
