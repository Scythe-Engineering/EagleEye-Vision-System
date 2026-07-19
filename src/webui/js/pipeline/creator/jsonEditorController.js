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
    const detectedPosition = positionMatch
        ? Number(positionMatch[1])
        : findJsonSyntaxErrorPosition(content);
    if (detectedPosition === null) {
        return { line: null, column: null };
    }

    const position = Math.min(detectedPosition, content.length);
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

/** Find the first invalid character with a small JSON grammar parser. */
function findJsonSyntaxErrorPosition(content) {
    let index = 0;
    const fail = () => {
        throw index;
    };
    const skipWhitespace = () => {
        while (/[ \t\r\n]/.test(content[index] || "")) index += 1;
    };
    const parseString = () => {
        if (content[index] !== '"') fail();
        index += 1;
        while (index < content.length) {
            const character = content[index];
            if (character === '"') {
                index += 1;
                return;
            }
            if (character === "\\") {
                index += 1;
                const escape = content[index];
                if (!'"\\/bfnrtu'.includes(escape || "")) fail();
                if (escape === "u") {
                    const unicodeDigits = content.slice(index + 1, index + 5);
                    if (!/^[0-9a-fA-F]{4}$/.test(unicodeDigits)) fail();
                    index += 4;
                }
            } else if (character.charCodeAt(0) < 0x20) {
                fail();
            }
            index += 1;
        }
        fail();
    };
    const parseValue = () => {
        skipWhitespace();
        const character = content[index];
        if (character === '"') {
            parseString();
        } else if (character === "{") {
            parseObject();
        } else if (character === "[") {
            parseArray();
        } else if (character === "-" || /\d/.test(character || "")) {
            const numberMatch = content
                .slice(index)
                .match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/);
            if (!numberMatch) fail();
            index += numberMatch[0].length;
        } else {
            const literal = ["true", "false", "null"].find((value) =>
                content.startsWith(value, index),
            );
            if (!literal) fail();
            index += literal.length;
        }
        skipWhitespace();
    };
    const parseObject = () => {
        index += 1;
        skipWhitespace();
        if (content[index] === "}") {
            index += 1;
            return;
        }
        while (index < content.length) {
            parseString();
            skipWhitespace();
            if (content[index] !== ":") fail();
            index += 1;
            parseValue();
            if (content[index] === "}") {
                index += 1;
                return;
            }
            if (content[index] !== ",") fail();
            index += 1;
            skipWhitespace();
        }
        fail();
    };
    const parseArray = () => {
        index += 1;
        skipWhitespace();
        if (content[index] === "]") {
            index += 1;
            return;
        }
        while (index < content.length) {
            parseValue();
            if (content[index] === "]") {
                index += 1;
                return;
            }
            if (content[index] !== ",") fail();
            index += 1;
            skipWhitespace();
        }
        fail();
    };

    try {
        parseValue();
        skipWhitespace();
        if (index !== content.length) fail();
        return null;
    } catch (errorPosition) {
        return typeof errorPosition === "number" ? errorPosition : index;
    }
}

/** Convert a source column to its visible position with four-space tab stops. */
function getVisualColumn(line, sourceColumn) {
    let visualColumn = 0;
    for (const character of line.slice(0, Math.max(0, sourceColumn - 1))) {
        visualColumn =
            character === "\t"
                ? visualColumn + (4 - (visualColumn % 4))
                : visualColumn + 1;
    }
    return visualColumn;
}

/** Escape text before inserting it into the syntax-highlight layer. */
function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

const TOKEN_PATTERN =
    /("(?:\\.|[^"\\])*")(\s*:)?|("(?:\\.|[^"\\])*")|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b|[{}\[\]]|[:,]/g;
const BRACKET_COLORS = [
    "text-[#ffd866]",
    "text-[#78dce8]",
    "text-[#ab9df2]",
    "text-[#a9dc76]",
    "text-[#fc9867]",
];

/** Return the bracket under the cursor and its matching partner. */
function findMatchingBrackets(content, cursorPosition) {
    const pairs = new Map();
    const stack = [];
    let inString = false;
    let escaped = false;

    for (let index = 0; index < content.length; index += 1) {
        const character = content[index];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (character === "\\" && inString) {
            escaped = true;
            continue;
        }
        if (character === '"') {
            inString = !inString;
            continue;
        }
        if (inString) continue;

        if (character === "{" || character === "[") {
            stack.push({ character, index });
            continue;
        }
        if (character !== "}" && character !== "]") continue;

        const opening = stack.at(-1);
        const isPair =
            opening &&
            ((opening.character === "{" && character === "}") ||
                (opening.character === "[" && character === "]"));
        if (!isPair) continue;
        stack.pop();
        pairs.set(opening.index, index);
        pairs.set(index, opening.index);
    }

    const position = [cursorPosition, cursorPosition - 1].find((index) =>
        "{}[]".includes(content[index] || ""),
    );
    if (position === undefined) return new Set();
    const partner = pairs.get(position);
    return partner === undefined
        ? new Set([position])
        : new Set([position, partner]);
}

/** Highlight one JSON line while retaining bracket nesting across lines. */
function highlightJsonLine(line, state, lineOffset, matchingBrackets) {
    let html = "";
    let lastIndex = 0;
    TOKEN_PATTERN.lastIndex = 0;
    for (const match of line.matchAll(TOKEN_PATTERN)) {
        html += escapeHtml(line.slice(lastIndex, match.index));
        const token = match[0];
        const tokenPosition = lineOffset + match.index;
        let tokenClass = "text-[#c5c8c6]";

        if (match[1] && match[2] !== undefined) {
            tokenClass = "text-[#78dce8]";
        } else if (match[3]) {
            tokenClass = "text-[#a9dc76]";
        } else if (/^-?\d/.test(token)) {
            tokenClass = "text-[#fc9867]";
        } else if (/^(?:true|false|null)$/.test(token)) {
            tokenClass = "text-[#ab9df2]";
        } else if ("{[".includes(token)) {
            tokenClass = BRACKET_COLORS[state.depth % BRACKET_COLORS.length];
            state.depth += 1;
        } else if ("}]".includes(token)) {
            state.depth = Math.max(0, state.depth - 1);
            tokenClass = BRACKET_COLORS[state.depth % BRACKET_COLORS.length];
        } else if (token === ":" || token === ",") {
            tokenClass = "text-[#727072]";
        }

        const matchClass = matchingBrackets.has(tokenPosition)
            ? " bg-[#f9c845]/25 outline outline-1 outline-[#f9c845]/70 rounded-sm"
            : "";
        html += `<span class="${tokenClass}${matchClass}">${escapeHtml(token)}</span>`;
        lastIndex = match.index + token.length;
    }
    return html + escapeHtml(line.slice(lastIndex));
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

    const syntaxLayer = createElement("div", {
        className:
            "absolute inset-0 overflow-hidden pointer-events-none font-mono text-sm leading-6",
        "aria-hidden": "true",
    });
    const syntaxContent = createElement("div", {
        className: "min-w-full w-max py-5",
    });
    syntaxLayer.appendChild(syntaxContent);

    const textarea = createElement("textarea", {
        className:
            "eagle-scrollbar absolute inset-0 w-full h-full resize-none bg-transparent font-mono text-sm leading-6 py-5 pr-5 pl-[4.5rem] outline-none border-0 focus:ring-0",
        style: "color: transparent; caret-color: #f9c845; -webkit-text-fill-color: transparent; tab-size: 4;",
        spellcheck: "false",
        wrap: "off",
        "aria-label": "Pipeline JSON content",
    });
    const editorFrame = createElement(
        "div",
        {
            className:
                "relative flex-1 min-h-0 m-4 mb-3 rounded-xl overflow-hidden border border-[#343434] bg-[#101010] focus-within:border-[#f9c845]/70 transition-colors",
        },
        [syntaxLayer, textarea],
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
    const syntaxLegend = createElement("div", {
        className:
            "flex items-center gap-4 px-5 py-2 bg-[#191919] border-b border-[#2f2f2f] text-xs text-[#888] shrink-0",
        html: '<span class="text-[#78dce8]">Keys</span><span class="text-[#a9dc76]">Strings</span><span class="text-[#fc9867]">Numbers</span><span class="text-[#ab9df2]">Literals</span><span class="text-[#ffd866]">Rainbow bracket pairs</span><span class="ml-auto text-[#666]">Indent guides every 4 spaces</span>',
    });
    modal.replaceChildren(header, syntaxLegend, editorFrame, footer);

    const renderSyntax = (lintResult) => {
        const lines = textarea.value.split("\n");
        const state = { depth: 0 };
        const matchingBrackets = findMatchingBrackets(
            textarea.value,
            textarea.selectionStart,
        );
        const errorLine = lintResult.valid ? null : lintResult.line || 1;
        let lineOffset = 0;

        syntaxContent.innerHTML = lines
            .map((line, index) => {
                const lineNumber = index + 1;
                const highlighted = highlightJsonLine(
                    line,
                    state,
                    lineOffset,
                    matchingBrackets,
                );
                lineOffset += line.length + 1;
                const isErrorLine = lineNumber === errorLine;
                const inlineError = isErrorLine
                    ? `<span class="ml-4 px-2 py-0.5 rounded border border-red-500/40 bg-red-950/90 text-red-200 text-xs">▲ ${escapeHtml(lintResult.message)}</span>`
                    : "";
                const errorMarker =
                    isErrorLine && lintResult.column
                        ? `<span class="absolute bottom-0 h-0 border-b-2 border-red-400" style="left: calc(4.5rem + ${getVisualColumn(line, lintResult.column)}ch); width: 1ch;"></span>`
                        : "";
                return `<div class="relative flex min-h-6 whitespace-pre ${isErrorLine ? "bg-red-950/35" : ""}"><span class="sticky left-0 z-10 inline-block w-[4.5rem] shrink-0 pr-4 text-right select-none text-[#555] bg-[#101010] border-r border-[#252525]">${lineNumber}</span><code class="inline-block pr-5" style="min-width: calc(100% - 4.5rem); background-image: repeating-linear-gradient(to right, transparent 0, transparent calc(4ch - 1px), rgba(249, 200, 69, 0.11) calc(4ch - 1px), rgba(249, 200, 69, 0.11) 4ch);">${highlighted || "&#8203;"}${inlineError}</code>${errorMarker}</div>`;
            })
            .join("");
        syntaxLayer.scrollTop = textarea.scrollTop;
        syntaxLayer.scrollLeft = textarea.scrollLeft;
    };

    const updateCursorStatus = (refreshSyntax = true) => {
        const beforeCursor = textarea.value.slice(0, textarea.selectionStart);
        const lines = beforeCursor.split("\n");
        cursorStatus.textContent = `Line ${lines.length}, Column ${lines.at(-1).length + 1}`;
        if (refreshSyntax) renderSyntax(lintPipelineJson(textarea.value));
    };

    const updateLintStatus = () => {
        const result = lintPipelineJson(textarea.value);
        lintStatus.className = result.valid
            ? "text-sm min-w-0 truncate text-emerald-400"
            : "text-sm min-w-0 truncate text-red-300";
        lintStatus.textContent = result.valid
            ? `✓ ${result.message}`
            : "✕ Invalid JSON • See the inline error in the editor";
        lintStatus.title = result.message;
        saveButton.disabled = !result.valid || saving;
        renderSyntax(result);
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
            updateCursorStatus(false);
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
        updateCursorStatus(false);
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

    textarea.addEventListener("input", () => {
        updateLintStatus();
        updateCursorStatus(false);
    });
    textarea.addEventListener("scroll", () => {
        syntaxLayer.scrollTop = textarea.scrollTop;
        syntaxLayer.scrollLeft = textarea.scrollLeft;
    });
    textarea.addEventListener("click", () => updateCursorStatus());
    textarea.addEventListener("keyup", (event) => {
        if (
            event.key.startsWith("Arrow") ||
            ["Home", "End", "PageUp", "PageDown"].includes(event.key)
        ) {
            updateCursorStatus();
        }
    });
    textarea.addEventListener("keydown", (event) => {
        if (event.key === "Tab") {
            event.preventDefault();
            const start = textarea.selectionStart;
            textarea.setRangeText("    ", start, textarea.selectionEnd, "end");
            updateLintStatus();
            updateCursorStatus(false);
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
