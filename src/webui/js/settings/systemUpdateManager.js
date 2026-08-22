// Manages the system update modal, status checks, live terminal, and backend restart flow.
import { BACKEND_BASE_URL } from "../config.js";
import {
    closeOnBackdropClick,
    closeOnEscape,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "../ui/modal.js";
import { showDanger } from "../ui/notificationSystem.js";

const OVERLAY_ID = "systemUpdateOverlay";
const MODAL_ID = "systemUpdateModal";
const CONFIRM_OVERLAY_ID = "systemUpdateConfirmOverlay";
const CONFIRM_MODAL_ID = "systemUpdateConfirmModal";
const DEFAULT_UPDATE_BRANCH = "main";

const PHASE_LABELS = {
    starting: "Starting update...",
    git_pull: "Pulling latest changes...",
    apt_update: "Updating package lists...",
    apt_upgrade: "Installing package upgrades...",
    complete: "Update complete. Restarting...",
    error: "Update failed",
};

const MAX_TERMINAL_LINES = 500;
const CHECK_ICON_PATH =
    "M10 .5a9.5 9.5 0 1 0 9.5 9.5A9.51 9.51 0 0 0 10 .5Zm3.707 8.207-4 4a1 1 0 0 1-1.414 0l-2-2a1 1 0 0 1 1.414-1.414L9 10.586l3.293-3.293a1 1 0 0 1 1.414 1.414Z";
const UPDATE_AVAILABLE_ICON_PATH =
    "M10 .5a9.5 9.5 0 1 0 9.5 9.5A9.51 9.51 0 0 0 10 .5ZM9.293 13.707a1 1 0 0 0 1.414 0l3-3a1 1 0 0 0-1.414-1.414L11 10.586V6a1 1 0 1 0-2 0v4.586L7.707 9.293a1 1 0 0 0-1.414 1.414l3 3Z";

let initialized = false;
let statusTimer = null;
let updateAvailable = false;
let updating = false;
let statusReason = "Checking update availability...";
/** @type {string | null} */
let currentUpdateId = null;
/** @type {HTMLElement | null} */
let progressBarFill = null;
/** @type {HTMLElement | null} */
let progressLabel = null;
/** @type {HTMLElement | null} */
let terminalElement = null;
/** @type {Text[]} */
let terminalLineNodes = [];
/** @type {((result: { confirmed: boolean, branch: string | null }) => void) | null} */
let activeConfirmResolve = null;

/**
 * Gets or creates the modal overlay elements used by the system update UI.
 * @returns {{overlay: HTMLElement, modal: HTMLElement}}
 */
function getOverlayElements() {
    return getOrCreateModalElements({
        overlayId: OVERLAY_ID,
        modalId: MODAL_ID,
        modalClassName:
            "bg-[#1a1a1a] rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] flex flex-col border border-[#414141]",
    });
}

/**
 * Gets or creates the confirmation modal overlay elements.
 * @returns {{overlay: HTMLElement, modal: HTMLElement}}
 */
function getConfirmOverlayElements() {
    return getOrCreateModalElements({
        overlayId: CONFIRM_OVERLAY_ID,
        modalId: CONFIRM_MODAL_ID,
        modalClassName:
            "bg-[#1a1a1a] rounded-xl shadow-2xl w-auto min-w-[22rem] max-w-[min(92vw,36rem)] mx-4 border border-[#414141] overflow-hidden",
    });
}

/**
 * Fetches JSON from the backend and throws on non-OK responses.
 * @param {string} path
 * @param {RequestInit} [options={}]
 * @returns {Promise<any>}
 */
async function fetchJson(path, options = {}) {
    const response = await fetch(`${BACKEND_BASE_URL}${path}`, options);
    let payload = {};
    try {
        payload = await response.json();
    } catch {
        payload = {};
    }
    if (!response.ok) {
        const error = new Error(
            payload.error ||
                payload.message ||
                `Request failed: ${response.status}`,
        );
        error.payload = payload;
        error.status = response.status;
        throw error;
    }
    return payload;
}

/**
 * Closes the system update progress modal unless an update is currently running.
 */
function close() {
    if (updating) {
        return;
    }
    hideModal(getOverlayElements().overlay);
}

/**
 * Resolves and closes the confirmation dialog.
 * @param {{ confirmed: boolean, branch: string | null }} result
 */
function resolveConfirmDialog(result) {
    if (activeConfirmResolve) {
        activeConfirmResolve(result);
        activeConfirmResolve = null;
    }
    hideModal(getConfirmOverlayElements().overlay);
}

/**
 * Updates the system update button to reflect current availability and state.
 */
function setButtonState() {
    const button = document.getElementById("updateSystemBtn");
    if (!button) {
        return;
    }

    button.disabled = updating || !updateAvailable;
    button.title = updateAvailable
        ? "Pull code updates, install apt upgrades, and restart the backend"
        : statusReason;
    button.textContent = updating ? "Updating..." : "Update System";
}

/**
 * Refreshes the backend-reported system update availability and reason.
 */
async function refreshUpdateStatus() {
    if (updating) {
        return;
    }

    try {
        const payload = await fetchJson("/system-update/status");
        updateAvailable = payload.available === true;
        statusReason =
            payload.reason || "Update requires WiFi with internet access";

        if (
            payload.in_progress === true &&
            payload.latest_progress &&
            typeof payload.latest_progress === "object"
        ) {
            applyCachedUpdateProgress(
                payload.latest_progress,
                typeof payload.update_id === "string"
                    ? payload.update_id
                    : null,
            );
            return;
        }
    } catch (error) {
        updateAvailable = false;
        statusReason =
            error.payload?.error || "Unable to check WiFi internet access";
    }
    setButtonState();
}

/**
 * Creates a small status icon for the version indicator.
 * @param {"up_to_date" | "update_needed"} status
 * @returns {HTMLElement}
 */
function createVersionStatusIcon(status) {
    const isUpToDate = status === "up_to_date";
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 20 20");
    icon.setAttribute("fill", "currentColor");
    icon.setAttribute("aria-hidden", "true");
    icon.setAttribute(
        "class",
        `h-5 w-5 shrink-0 ${isUpToDate ? "text-green-400" : "text-yellow-400"}`,
    );
    const iconPath = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "path",
    );
    iconPath.setAttribute(
        "d",
        isUpToDate ? CHECK_ICON_PATH : UPDATE_AVAILABLE_ICON_PATH,
    );
    icon.appendChild(iconPath);
    return icon;
}

/**
 * Builds the version summary row for the confirmation modal.
 * @param {string} currentSha
 * @param {string | null} remoteSha
 * @param {boolean} updateNeeded
 * @returns {HTMLElement}
 */
function buildVersionSummary(currentSha, remoteSha, updateNeeded) {
    const statusIcon = createVersionStatusIcon(
        updateNeeded ? "update_needed" : "up_to_date",
    );
    const versionLines = [
        createElement("div", {
            className: "font-mono text-sm text-gray-200",
            text: `Current: ${currentSha}`,
        }),
    ];

    if (updateNeeded && remoteSha) {
        versionLines.push(
            createElement("div", {
                className: "font-mono text-sm text-gray-300",
                text: `Remote: ${remoteSha}`,
            }),
        );
        versionLines.push(
            createElement("div", {
                className: "text-xs text-yellow-300/90",
                text: "Update available",
            }),
        );
    } else if (!updateNeeded) {
        versionLines.push(
            createElement("div", {
                className: "text-xs text-green-300/90",
                text: "Up to date",
            }),
        );
    } else {
        versionLines.push(
            createElement("div", {
                className: "text-xs text-yellow-300/90",
                text: "Remote version unavailable for this branch",
            }),
        );
    }

    return createElement(
        "div",
        {
            className:
                "mt-3 flex items-start gap-2.5 rounded-lg border border-[#414141] bg-[#141414] px-3 py-2.5",
        },
        [
            statusIcon,
            createElement(
                "div",
                { className: "min-w-0 space-y-0.5" },
                versionLines,
            ),
        ],
    );
}

/**
 * Renders confirmation modal content from loaded update info.
 * @param {object} infoPayload
 * @returns {void}
 */
function renderConfirmContent(infoPayload) {
    const { overlay, modal } = getConfirmOverlayElements();
    modal.innerHTML = "";

    const remoteBranches = Array.isArray(infoPayload.remote_branches)
        ? infoPayload.remote_branches
        : [];
    const currentSha =
        typeof infoPayload.current_sha === "string"
            ? infoPayload.current_sha
            : "unknown";
    const branchShaByName = new Map(
        remoteBranches
            .filter(
                (branch) =>
                    branch &&
                    typeof branch.name === "string" &&
                    typeof branch.sha === "string",
            )
            .map((branch) => [branch.name, branch.sha]),
    );

    const defaultBranch =
        typeof infoPayload.default_branch === "string" &&
        infoPayload.default_branch
            ? infoPayload.default_branch
            : DEFAULT_UPDATE_BRANCH;
    let selectedBranch = defaultBranch;

    if (!branchShaByName.has(selectedBranch) && selectedBranch) {
        remoteBranches.unshift({
            name: selectedBranch,
            sha:
                typeof infoPayload.remote_sha === "string"
                    ? infoPayload.remote_sha
                    : "",
        });
        if (
            typeof infoPayload.remote_sha === "string" &&
            infoPayload.remote_sha
        ) {
            branchShaByName.set(selectedBranch, infoPayload.remote_sha);
        }
    }

    const versionSummaryHost = createElement("div", { className: "w-full" });

    /**
     * Refreshes the version summary for the selected branch.
     */
    function refreshVersionSummary() {
        const remoteSha = branchShaByName.get(selectedBranch) || null;
        const updateNeeded = !remoteSha || remoteSha !== currentSha;
        versionSummaryHost.replaceChildren(
            buildVersionSummary(currentSha, remoteSha, updateNeeded),
        );
    }

    const otherBranches = remoteBranches.filter(
        (branch) => branch.name !== defaultBranch,
    );
    const branchOptions =
        otherBranches.length > 0
            ? otherBranches.map((branch) =>
                  createElement("option", {
                      value: branch.name,
                      text: branch.name,
                  }),
              )
            : [
                  createElement("option", {
                      value: "",
                      text: "No other remote branches",
                      selected: "selected",
                      disabled: "disabled",
                  }),
              ];

    const branchSelect = createElement(
        "select",
        {
            className:
                "w-full rounded-md border border-[#414141] bg-[#242424] px-3 py-2 text-sm text-gray-100 focus:border-[#f9c845] focus:outline-none",
            ...(otherBranches.length === 0 ? { disabled: "disabled" } : {}),
            onchange: (event) => {
                selectedBranch = event.target.value;
                refreshVersionSummary();
                trackingBranchLabel.textContent = `Tracking branch: ${selectedBranch}`;
            },
        },
        branchOptions,
    );

    const branchPicker = createElement(
        "div",
        {
            className: "mt-2 hidden",
        },
        [
            createElement("label", {
                className:
                    "mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-400",
                text: "Other branch",
            }),
            branchSelect,
        ],
    );
    const trackingBranchLabel = createElement("span", {
        className: "font-mono text-sm text-gray-200",
        text: `Tracking branch: ${defaultBranch}`,
    });
    const selectOtherBranchButton = createElement("button", {
        type: "button",
        className:
            "rounded-md border border-[#414141] bg-[#242424] px-3 py-2 text-sm font-semibold text-[#f9c845] transition-colors hover:bg-[#303030] disabled:cursor-not-allowed disabled:opacity-50",
        text: "Select other branch",
        ...(otherBranches.length === 0 ? { disabled: "disabled" } : {}),
        onclick: () => {
            const isHidden = branchPicker.classList.contains("hidden");
            if (isHidden) {
                branchPicker.classList.remove("hidden");
                selectedBranch = branchSelect.value;
                selectOtherBranchButton.textContent = `Track ${defaultBranch}`;
            } else {
                branchPicker.classList.add("hidden");
                selectedBranch = defaultBranch;
                selectOtherBranchButton.textContent = "Select other branch";
            }
            trackingBranchLabel.textContent = `Tracking branch: ${selectedBranch}`;
            refreshVersionSummary();
        },
    });

    refreshVersionSummary();

    modal.appendChild(
        createElement("div", { className: "flex flex-col" }, [
            createElement("div", { className: "p-5 pb-4" }, [
                createElement("div", { className: "flex items-start gap-3" }, [
                    createElement("div", {
                        className:
                            "mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-yellow-500/40 bg-yellow-500/15 text-base font-bold text-yellow-300",
                        text: "?",
                    }),
                    createElement("div", { className: "min-w-0 flex-1" }, [
                        createElement("h3", {
                            className:
                                "mb-2 text-lg font-bold leading-tight text-yellow-400",
                            text: "Update System?",
                        }),
                        createElement("p", {
                            className: "text-sm leading-relaxed text-gray-200",
                            text: "This will restart the system. Are you sure?",
                        }),
                        createElement("p", {
                            className:
                                "mt-1.5 text-sm leading-relaxed text-gray-400",
                            text: "The backend will checkout the selected branch, run apt update, and non-interactive apt upgrade before restarting.",
                        }),
                        versionSummaryHost,
                        createElement(
                            "div",
                            {
                                className:
                                    "mt-3 flex items-center justify-between gap-3 rounded-lg border border-[#414141] bg-[#141414] px-3 py-2.5",
                            },
                            [trackingBranchLabel, selectOtherBranchButton],
                        ),
                        branchPicker,
                    ]),
                ]),
            ]),
            createElement(
                "div",
                {
                    className:
                        "flex items-center justify-between gap-3 border-t border-[#333333] bg-[#171717] px-5 py-3",
                },
                [
                    createElement("button", {
                        type: "button",
                        className:
                            "rounded-md border border-[#414141] bg-[#242424] px-3.5 py-2 text-sm font-semibold text-[#f9c845] transition-colors hover:bg-[#303030]",
                        text: "Cancel",
                        onclick: () =>
                            resolveConfirmDialog({
                                confirmed: false,
                                branch: null,
                            }),
                    }),
                    createElement("button", {
                        type: "button",
                        className:
                            "rounded-md border border-[#d4a83a] bg-[#f9c845] px-3.5 py-2 text-sm font-semibold text-[#232323] transition-colors hover:bg-[#d4a83a]",
                        text: "Update and Restart",
                        onclick: () =>
                            resolveConfirmDialog({
                                confirmed: true,
                                branch: selectedBranch || null,
                            }),
                    }),
                ],
            ),
        ]),
    );

    showModal(overlay);
}

/**
 * Shows a loading or error state inside the confirmation modal.
 * @param {"loading" | "error"} state
 * @param {string} [message=""]
 */
function renderConfirmPlaceholder(state, message = "") {
    const { overlay, modal } = getConfirmOverlayElements();
    modal.innerHTML = "";
    const isError = state === "error";
    modal.appendChild(
        createElement("div", { className: "p-5" }, [
            createElement("h3", {
                className: `mb-2 text-lg font-bold ${isError ? "text-red-300" : "text-yellow-400"}`,
                text: isError
                    ? "Unable to Load Update Info"
                    : "Checking versions...",
            }),
            createElement("p", {
                className: "text-sm text-gray-300",
                text: isError
                    ? message || "Failed to fetch remote branch information."
                    : "Fetching remote branches and comparing git SHAs.",
            }),
            ...(isError
                ? [
                      createElement(
                          "div",
                          { className: "mt-4 flex justify-end" },
                          [
                              createElement("button", {
                                  type: "button",
                                  className:
                                      "rounded-md border border-[#414141] bg-[#242424] px-3.5 py-2 text-sm font-semibold text-[#f9c845] transition-colors hover:bg-[#303030]",
                                  text: "Close",
                                  onclick: () =>
                                      resolveConfirmDialog({
                                          confirmed: false,
                                          branch: null,
                                      }),
                              }),
                          ],
                      ),
                  ]
                : []),
        ]),
    );
    showModal(overlay);
}

/**
 * Shows the confirmation dialog before starting the update flow.
 */
async function renderConfirm() {
    if (activeConfirmResolve) {
        resolveConfirmDialog({ confirmed: false, branch: null });
    }

    renderConfirmPlaceholder("loading");
    const confirmPromise = new Promise((resolve) => {
        activeConfirmResolve = resolve;
    });

    try {
        const infoPayload = await fetchJson("/system-update/info");
        if (!activeConfirmResolve) {
            return;
        }
        renderConfirmContent(infoPayload);
    } catch (error) {
        if (!activeConfirmResolve) {
            return;
        }
        renderConfirmPlaceholder(
            "error",
            error.payload?.error ||
                error.message ||
                "Failed to fetch update info",
        );
    }

    const result = await confirmPromise;
    if (result.confirmed && result.branch) {
        runUpdate(result.branch);
    }
}

/**
 * Appends a line to the live terminal and scrolls to the bottom.
 * @param {string} line
 */
function appendTerminalLine(line) {
    if (!terminalElement || !line) {
        return;
    }

    if (terminalLineNodes.length > 0) {
        terminalElement.appendChild(document.createTextNode("\n"));
    }
    const lineNode = document.createTextNode(line);
    terminalElement.appendChild(lineNode);
    terminalLineNodes.push(lineNode);

    while (terminalLineNodes.length > MAX_TERMINAL_LINES) {
        const oldestLineNode = terminalLineNodes.shift();
        if (!oldestLineNode) {
            break;
        }
        const followingNewline = oldestLineNode.nextSibling;
        oldestLineNode.remove();
        if (
            followingNewline &&
            followingNewline.nodeType === Node.TEXT_NODE &&
            followingNewline.textContent === "\n"
        ) {
            followingNewline.remove();
        }
    }

    terminalElement.scrollTop = terminalElement.scrollHeight;
}

/**
 * Updates the progress bar fill and phase label.
 * @param {number} percent
 * @param {string} [label]
 */
function setProgress(percent, label) {
    if (progressBarFill) {
        const clampedPercent = Math.max(0, Math.min(100, percent));
        progressBarFill.style.width = `${clampedPercent}%`;
    }
    if (progressLabel && label) {
        progressLabel.textContent = label;
    }
}

/**
 * Renders the live progress modal with terminal output.
 */
function renderLiveProgress() {
    const { overlay, modal } = getOverlayElements();
    modal.innerHTML = "";
    showModal(overlay);

    progressLabel = createElement("div", {
        className: "mb-3 text-gray-200",
        text: "Starting update...",
    });

    progressBarFill = createElement("div", {
        className:
            "h-full rounded-full bg-yellow-400 transition-[width] duration-300 ease-out",
    });
    progressBarFill.style.width = "0%";

    const progressTrack = createElement(
        "div",
        {
            className:
                "h-3 w-full overflow-hidden rounded-full bg-[#2a2a2a] border border-[#414141]",
        },
        [progressBarFill],
    );

    terminalElement = createElement("pre", {
        className:
            "eagle-scrollbar mt-4 h-64 overflow-y-auto whitespace-pre-wrap rounded bg-[#101010] p-3 text-xs text-gray-300 border border-[#414141] font-mono",
        text: "",
    });
    terminalLineNodes = [];

    modal.appendChild(
        createElement("div", { className: "p-6" }, [
            createElement("h3", {
                className: "text-xl font-bold text-yellow-400 mb-4",
                text: "Updating System",
            }),
            progressLabel,
            progressTrack,
            terminalElement,
        ]),
    );
}

/**
 * Renders the in-modal error state for failed update attempts.
 * @param {string} message
 */
function renderError(message) {
    updating = false;
    setButtonState();
    const { overlay, modal } = getOverlayElements();
    modal.innerHTML = "";
    showModal(overlay);
    progressBarFill = null;
    progressLabel = null;
    terminalElement = null;
    terminalLineNodes = [];
    currentUpdateId = null;
    modal.appendChild(
        createElement("div", { className: "p-6" }, [
            createElement("h3", {
                className: "text-xl font-bold text-red-300 mb-3",
                text: "Update Failed",
            }),
            createElement("pre", {
                className:
                    "eagle-scrollbar max-h-64 overflow-y-auto whitespace-pre-wrap rounded bg-[#101010] p-3 text-sm text-red-100 border border-red-700/60 mb-5 font-mono",
                text: message,
            }),
            createElement("div", { className: "flex justify-end" }, [
                createElement("button", {
                    type: "button",
                    className:
                        "px-4 py-2 bg-[#2a2a2a] text-[#f9c845] rounded-md border border-[#414141] hover:bg-[#3a3a3a]",
                    text: "Close",
                    onclick: close,
                }),
            ]),
        ]),
    );
}

/**
 * Restarts the backend and reloads the page after a successful update.
 */
async function restartAfterUpdate() {
    setProgress(100, PHASE_LABELS.complete);
    try {
        await fetchJson("/restart-backend", { method: "POST" });
    } catch (error) {
        if (typeof error.status === "number" && error.status > 0) {
            const message =
                error.payload?.error ||
                error.payload?.message ||
                error.message ||
                "Failed to restart backend";
            renderError(message);
            return;
        }
        console.warn("Restart request failed or connection closed:", error);
    }
    setTimeout(() => {
        globalThis.location.reload();
    }, 2500);
}

/**
 * Handles a system update progress SSE payload.
 * @param {object} data
 */
export function handleSystemUpdateProgress(data) {
    if (!data || typeof data !== "object") {
        return;
    }

    if (!updating) {
        return;
    }

    if (
        currentUpdateId &&
        typeof data.update_id === "string" &&
        data.update_id !== currentUpdateId
    ) {
        return;
    }

    if (!terminalElement) {
        renderLiveProgress();
    }

    if (typeof data.line === "string" && data.line.length > 0) {
        appendTerminalLine(data.line);
    }

    const phaseLabel =
        PHASE_LABELS[data.phase] ||
        (typeof data.phase === "string" ? data.phase : "Updating...");
    if (typeof data.percent === "number") {
        setProgress(data.percent, phaseLabel);
    } else if (phaseLabel) {
        setProgress(
            progressBarFill
                ? Number.parseFloat(progressBarFill.style.width) || 0
                : 0,
            phaseLabel,
        );
    }

    if (!data.done) {
        return;
    }

    if (data.error) {
        const errorMessage =
            typeof data.error === "string" ? data.error : "Update failed";
        if (terminalElement) {
            appendTerminalLine(errorMessage);
            const terminalSnapshot =
                terminalElement.textContent || errorMessage;
            renderError(terminalSnapshot);
        } else {
            renderError(errorMessage);
        }
        return;
    }

    void restartAfterUpdate();
}

/**
 * Applies a cached progress payload from status or SSE reconnect recovery.
 * @param {object} latestProgress
 * @param {string | null} [updateId=null]
 */
function applyCachedUpdateProgress(latestProgress, updateId = null) {
    if (!latestProgress || typeof latestProgress !== "object") {
        return;
    }
    if (updateId) {
        currentUpdateId = updateId;
    } else if (typeof latestProgress.update_id === "string") {
        currentUpdateId = latestProgress.update_id;
    }
    updating = true;
    setButtonState();
    handleSystemUpdateProgress(latestProgress);
}

/**
 * Runs the system update sequence and waits for SSE progress events.
 * @param {string} targetBranch
 */
async function runUpdate(targetBranch) {
    updating = true;
    currentUpdateId = null;
    setButtonState();
    renderLiveProgress();

    try {
        const status = await fetchJson("/system-update/status");
        if (status.available !== true) {
            throw new Error(
                status.reason || "WiFi internet access is required.",
            );
        }

        appendTerminalLine(`Target branch: ${targetBranch}`);
        appendTerminalLine("Checking WiFi internet access... OK");
        setProgress(2, "Starting update...");
        const startResult = await fetchJson("/system-update/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ branch: targetBranch }),
        });
        if (typeof startResult.update_id === "string") {
            currentUpdateId = startResult.update_id;
        }
    } catch (error) {
        const message =
            error.payload?.error ||
            error.payload?.output ||
            error.message ||
            "Update failed";
        renderError(message);
    }
}

/**
 * Initializes the system update manager UI and event handlers.
 */
export function initializeSystemUpdateManager() {
    if (initialized) {
        return;
    }
    initialized = true;

    const button = document.getElementById("updateSystemBtn");
    if (!button) {
        return;
    }

    button.addEventListener("click", () => {
        if (!button.disabled) {
            renderConfirm();
        } else if (statusReason) {
            showDanger(statusReason);
        }
    });

    const { overlay } = getOverlayElements();
    closeOnBackdropClick(overlay, close);
    closeOnEscape(overlay, close);

    const { overlay: confirmOverlay } = getConfirmOverlayElements();
    closeOnBackdropClick(confirmOverlay, () =>
        resolveConfirmDialog({ confirmed: false, branch: null }),
    );
    closeOnEscape(confirmOverlay, () =>
        resolveConfirmDialog({ confirmed: false, branch: null }),
    );

    setButtonState();
    refreshUpdateStatus();
    statusTimer = setInterval(refreshUpdateStatus, 30000);

    window.addEventListener("beforeunload", () => {
        if (statusTimer) {
            clearInterval(statusTimer);
        }
    });
}
