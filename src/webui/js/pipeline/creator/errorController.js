import { escapeHtml } from "../utils.js";
import { pipelineStore } from "../PipelineStore.js";
import { creatorContext } from "./context.js";
import { getSelectedPipeline } from "./stateHelpers.js";

/**
 * Manages pipeline operation error state, highlighting, and tooltip UI in the creator.
 */

const operationErrorsByUuid = new Map();
const downstreamErrorUuids = new Set();
let pipelineErrorPopup;

/**
 * Extracts missing argument names from a Python-style error message.
 *
 * @param {string} message - The error message to inspect.
 * @returns {string[]|null} The parsed argument names, or null if none are found.
 */
function extractMissingArgumentNames(message) {
    if (!message) return null;
    const match = message.match(/missing\s+\d+\s+required positional arguments?:\s*(.+)$/i);
    if (!match) return null;
    const rawList = match[1].trim();
    const quotedMatches = Array.from(rawList.matchAll(/'([^']+)'/g)).map((item) => item[1]);
    if (quotedMatches.length > 0) return quotedMatches;
    const normalized = rawList.replaceAll(/\band\b/gi, ",").replaceAll(/\s+/g, " ").trim();
    const parts = normalized.split(",").map((item) => item.trim()).filter((item) => item.length > 0);
    return parts.length > 0 ? parts : null;
}

/**
 * Builds the HTML content for the pipeline error popup.
 *
 * @param {object} errorRecord - The error record to render.
 * @returns {string} HTML markup for the popup.
 */
function buildPipelineErrorPopupContent(errorRecord) {
    const message = errorRecord?.message || "Unknown error";
    const missingArgs = extractMissingArgumentNames(message);
    const displayMessage = missingArgs
        ? `Please fill out the following settings fields in this operation: ${missingArgs.join(", ")}`
        : message;
    const count = errorRecord?.count || 1;
    const name = errorRecord?.name || "Operation Error";

    return `
        <div class="text-red-200 font-semibold text-sm mb-2 border-b border-[#3a1d1d] pb-2">${escapeHtml(
            name,
        )}</div>
        <div class="text-red-100 text-xs whitespace-pre-wrap" style="word-break: break-word; overflow-wrap: anywhere;">${escapeHtml(
            displayMessage,
        )}</div>
        <div class="text-red-300 text-xs mt-2">Seen ${count} time${
            count === 1 ? "" : "s"
        }</div>
    `;
}

/**
 * Creates or returns the shared pipeline error popup element.
 *
 * @returns {HTMLDivElement} The popup element.
 */
function ensurePipelineErrorPopup() {
    if (pipelineErrorPopup) return pipelineErrorPopup;
    pipelineErrorPopup = document.createElement("div");
    pipelineErrorPopup.id = "pipeline-error-popup";
    pipelineErrorPopup.className =
        "fixed z-50 bg-[#2b1f1f] border-2 border-[#ff5c5c] rounded-lg p-3 shadow-lg max-w-sm pointer-events-none opacity-0 transition-opacity duration-200";
    pipelineErrorPopup.style.fontSize = "0.875rem";
    pipelineErrorPopup.style.lineHeight = "1.25rem";
    pipelineErrorPopup.style.width = "max-content";
    pipelineErrorPopup.style.maxWidth = "320px";
    pipelineErrorPopup.style.height = "auto";
    pipelineErrorPopup.style.boxShadow =
        "4px 4px 12px rgba(0,0,0,0.45), 8px 8px 20px rgba(0,0,0,0.25), 2px 2px 6px rgba(255,92,92,0.15)";
    document.body.appendChild(pipelineErrorPopup);
    return pipelineErrorPopup;
}

/**
 * Positions the error popup near the given screen coordinates.
 *
 * @param {HTMLElement} popup - The popup element to position.
 * @param {number} anchorX - The anchor X coordinate.
 * @param {number} anchorY - The anchor Y coordinate.
 */
function positionPipelineErrorPopup(popup, anchorX, anchorY) {
    const margin = 12;
    const offset = 12;
    popup.style.left = `${anchorX + offset}px`;
    popup.style.top = `${anchorY + offset}px`;
    const rect = popup.getBoundingClientRect();
    const maxLeft = window.innerWidth - rect.width - margin;
    const maxTop = window.innerHeight - rect.height - margin;
    const clampedLeft = Math.min(Math.max(anchorX + offset, margin), Math.max(maxLeft, margin));
    const clampedTop = Math.min(Math.max(anchorY + offset, margin), Math.max(maxTop, margin));
    popup.style.left = `${clampedLeft}px`;
    popup.style.top = `${clampedTop}px`;
}

/**
 * Displays the pipeline error popup for the given error record.
 *
 * @param {object} errorRecord - The error record to display.
 * @param {MouseEvent} event - The mouse event providing popup coordinates.
 */
function showPipelineErrorPopup(errorRecord, event) {
    const popup = ensurePipelineErrorPopup();
    popup.innerHTML = buildPipelineErrorPopupContent(errorRecord);
    positionPipelineErrorPopup(popup, event.clientX, event.clientY);
    popup.classList.remove("opacity-0");
    popup.classList.add("opacity-100");
}

/**
 * Hides the shared pipeline error popup.
 */
function hidePipelineErrorPopup() {
    if (!pipelineErrorPopup) return;
    pipelineErrorPopup.classList.remove("opacity-100");
    pipelineErrorPopup.classList.add("opacity-0");
}

/**
 * Computes the set of downstream operation UUIDs affected by current errors.
 */
function computeDownstreamErrorUuids() {
    downstreamErrorUuids.clear();
    const errorUuids = new Set(operationErrorsByUuid.keys());
    if (errorUuids.size === 0) return;
    const connections = pipelineStore.getConnections();
    const outgoing = new Map();
    for (const connection of connections) {
        if (!outgoing.has(connection.fromUuid)) outgoing.set(connection.fromUuid, []);
        outgoing.get(connection.fromUuid).push(connection.toUuid);
    }
    const queue = Array.from(errorUuids);
    const visited = new Set(errorUuids);
    while (queue.length > 0) {
        const current = queue.shift();
        const nextNodes = outgoing.get(current) || [];
        for (const next of nextNodes) {
            if (visited.has(next)) continue;
            visited.add(next);
            downstreamErrorUuids.add(next);
            queue.push(next);
        }
    }
}

/**
 * Updates the node error icon visibility and tooltip bindings.
 *
 * @param {object} node - The flowchart node to update.
 * @param {object|null} errorRecord - The error record for the node.
 */
function applyFlowchartNodeErrorIcon(node, errorRecord) {
    const element = node.element;
    if (!element) return;
    const icon = element.querySelector(".node-error-icon");
    if (!icon) return;
    if (errorRecord) {
        icon.style.display = "inline-flex";
        if (!icon.dataset.pipelineErrorBound) {
            icon.dataset.pipelineErrorBound = "true";
            icon.addEventListener("mouseenter", (event) => {
                const uuid = pipelineStore.instanceIdToUuid.get(node.instanceId);
                const currentError = uuid ? operationErrorsByUuid.get(uuid) : null;
                if (currentError) showPipelineErrorPopup(currentError, event);
            });
            icon.addEventListener("mousemove", (event) => {
                if (pipelineErrorPopup?.classList.contains("opacity-100")) {
                    positionPipelineErrorPopup(
                        pipelineErrorPopup,
                        event.clientX,
                        event.clientY,
                    );
                }
            });
            icon.addEventListener("mouseleave", () => {
                hidePipelineErrorPopup();
            });
        }
    } else {
        icon.style.display = "none";
    }
}

/**
 * Applies error styling directly to a node when custom node APIs are unavailable.
 *
 * @param {object} node - The flowchart node to update.
 * @param {object|null} errorRecord - The error record for the node.
 * @param {boolean} isDownstream - Whether the node is downstream of an error.
 */
function applyFlowchartNodeErrorFallback(node, errorRecord, isDownstream) {
    const element = node.element;
    if (!element) return;
    if (errorRecord) {
        element.style.borderColor = "#ff5c5c";
        element.style.boxShadow =
            "0 0 0 2px rgba(255,92,92,0.35), 4px 4px 12px rgba(0, 0, 0, 0.5)";
    } else if (!node.isDragging) {
        element.style.borderColor = "#404040";
        element.style.boxShadow = "4px 4px 12px rgba(0, 0, 0, 0.5)";
    }
    element.classList.toggle("pipeline-error-node", Boolean(errorRecord));
    element.classList.toggle("pipeline-downstream-disabled", Boolean(isDownstream));
    let infoIcon = element.querySelector(".error-info-icon");
    if (errorRecord && !infoIcon) {
        const header = element.querySelector(".node-header");
        if (!header) return;
        infoIcon = document.createElement("div");
        infoIcon.className = "error-info-icon";
        infoIcon.textContent = "i";
        infoIcon.style.width = "18px";
        infoIcon.style.height = "18px";
        infoIcon.style.borderRadius = "50%";
        infoIcon.style.backgroundColor = "#ff5c5c";
        infoIcon.style.color = "#1a1a1a";
        infoIcon.style.fontSize = "12px";
        infoIcon.style.fontWeight = "700";
        infoIcon.style.display = "inline-flex";
        infoIcon.style.alignItems = "center";
        infoIcon.style.justifyContent = "center";
        infoIcon.style.marginLeft = "8px";
        infoIcon.style.cursor = "default";
        infoIcon.addEventListener("mouseenter", (event) => {
            showPipelineErrorPopup(errorRecord, event);
        });
        infoIcon.addEventListener("mousemove", (event) => {
            if (pipelineErrorPopup?.classList.contains("opacity-100")) {
                positionPipelineErrorPopup(
                    pipelineErrorPopup,
                    event.clientX,
                    event.clientY,
                );
            }
        });
        infoIcon.addEventListener("mouseleave", () => {
            hidePipelineErrorPopup();
        });
        header.appendChild(infoIcon);
    } else if (!errorRecord && infoIcon) {
        infoIcon.remove();
    }
}

/**
 * Applies error and downstream highlighting across all flowchart nodes.
 */
function applyPipelineErrorHighlights() {
    computeDownstreamErrorUuids();
    const flowchartRenderer = creatorContext.flowchartRenderer;
    if (!flowchartRenderer) return;
    for (const node of flowchartRenderer.nodes.values()) {
        const uuid = pipelineStore.instanceIdToUuid.get(node.instanceId);
        const errorRecord = uuid ? operationErrorsByUuid.get(uuid) : null;
        const isDownstream = uuid ? downstreamErrorUuids.has(uuid) : false;
        if (node.setErrorState) {
            node.setErrorState(errorRecord, isDownstream);
            applyFlowchartNodeErrorIcon(node, errorRecord);
        } else {
            applyFlowchartNodeErrorFallback(node, errorRecord, isDownstream);
        }
    }
}

/**
 * Handles incoming operation error updates and refreshes pipeline highlighting.
 *
 * @param {object} payload - The error update payload.
 */
function handleOperationErrorUpdate(payload) {
    if (!payload) return;
    const selectedPipeline = getSelectedPipeline();
    if (payload.pipeline_name && selectedPipeline) {
        if (payload.pipeline_name !== selectedPipeline.name) return;
    }
    operationErrorsByUuid.clear();
    const errors = Array.isArray(payload.errors) ? payload.errors : [];
    errors.forEach((errorRecord) => {
        if (errorRecord?.uuid) operationErrorsByUuid.set(errorRecord.uuid, errorRecord);
    });
    pipelineStore.setOperationErrors(errors);
    applyPipelineErrorHighlights();
}

export {
    applyFlowchartNodeErrorFallback,
    applyFlowchartNodeErrorIcon,
    applyPipelineErrorHighlights,
    buildPipelineErrorPopupContent,
    computeDownstreamErrorUuids,
    handleOperationErrorUpdate,
    hidePipelineErrorPopup,
    operationErrorsByUuid,
    downstreamErrorUuids,
    showPipelineErrorPopup,
};
