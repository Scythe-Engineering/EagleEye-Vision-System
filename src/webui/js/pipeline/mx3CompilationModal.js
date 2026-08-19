import { BACKEND_BASE_URL } from "../config.js";
import { INPUT_CLASS, responseError } from "./modalFormHelpers.js";
import {
    closeOnBackdropClick,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "../ui/modal.js";

/**
 * Registers the MX3 compilation dialog and its retained job-status controller.
 * @param {{onStatusChange?: (status: object|null) => void, onSucceeded?: (status: object) => void}} callbacks Lifecycle callbacks.
 * @returns {{open: Function, close: Function, isOpen: Function}} Dialog controls.
 */
export function registerMx3CompilationModal({
    onStatusChange,
    onSucceeded,
} = {}) {
    let compilation = null;
    let compilationModel = null;
    let compilationOverlay;
    let compilationModal;
    let compilationStatusEl;
    let compilationDraft = null;
    let lastCompletedCompilationId = null;
    let parentOverlay = null;
    let parentOverlayAccessibility = null;
    let returnFocusId = null;

    /**
     * Returns whether the retained compilation status represents a live job.
     * @returns {boolean} Whether a compiler process is active.
     */
    function isCompilationActive() {
        return [
            "running",
            "cancelling",
            "queued",
            "pending",
            "starting",
            "installing",
        ].includes(String(compilation?.state || "").toLowerCase());
    }

    /**
     * Returns whether a compilation status completed successfully.
     * @param {object|null} status Compiler status snapshot.
     * @returns {boolean} Whether the job succeeded.
     */
    function compilationSucceeded(status) {
        return ["success", "succeeded", "completed"].includes(
            String(status?.state || "").toLowerCase(),
        );
    }

    /**
     * Applies a compiler status snapshot from the API or live event stream.
     * @param {object|null} status Compiler status snapshot.
     */
    function applyCompilationStatus(status) {
        compilation = status && typeof status === "object" ? status : null;
        renderCompilationStatus();
        onStatusChange?.(compilation);
        if (
            compilationSucceeded(compilation) &&
            compilation.job_id &&
            compilation.job_id !== lastCompletedCompilationId
        ) {
            lastCompletedCompilationId = compilation.job_id;
            onSucceeded?.(compilation);
        }
    }

    /**
     * Fetches the retained global MX3 compilation status for reconnect recovery.
     * @returns {Promise<void>}
     */
    async function loadCompilationStatus() {
        try {
            const response = await fetch(
                `${BACKEND_BASE_URL}/model-library/mx3-compilation`,
            );
            if (!response.ok) throw new Error(await responseError(response));
            const data = await response.json();
            applyCompilationStatus(data.compilation);
        } catch (error) {
            if (compilationStatusEl) {
                compilationStatusEl.textContent = `Unable to load compilation status: ${error.message}`;
                compilationStatusEl.className = "text-sm text-red-300";
            }
        }
    }

    /**
     * Creates a labelled compact field for the compilation form.
     * @param {string} label Visible label text.
     * @param {HTMLElement} input Form control.
     * @returns {HTMLElement} Label and control wrapper.
     */
    function compilationField(label, input) {
        return createElement(
            "label",
            { className: "block space-y-1 text-sm text-[#f9c845]" },
            [createElement("span", { text: label }), input],
        );
    }

    /**
     * Renders the current compiler state without replacing the settings form.
     */
    function renderCompilationStatus() {
        if (!compilationStatusEl) return;
        const compilerLogOpen =
            compilationStatusEl.querySelector("details")?.open ?? false;
        const active = isCompilationActive();
        const startButton = compilationModal?.querySelector(
            "[data-compilation-start]",
        );
        const cancelButton = compilationModal?.querySelector(
            "[data-compilation-cancel]",
        );
        if (startButton) startButton.disabled = active;
        if (cancelButton) {
            const state = String(compilation?.state || "").toLowerCase();
            cancelButton.disabled = !active || state === "cancelling";
        }
        compilationStatusEl.innerHTML = "";
        const status = compilation;
        if (!status || String(status.state).toLowerCase() === "idle") {
            compilationStatusEl.appendChild(
                createElement("p", {
                    className: "text-sm text-gray-300",
                    text: "Ready to compile the selected ONNX model.",
                }),
            );
            return;
        }
        const state = String(status.state || "Unknown");
        compilationStatusEl.appendChild(
            createElement("p", {
                className: `text-sm font-medium ${status.error ? "text-red-300" : compilationSucceeded(status) ? "text-green-300" : "text-[#f9c845]"}`,
                text: `${state}: ${status.stage || "Waiting"}${status.model_id ? ` (${status.model_id})` : ""}`,
            }),
        );
        const percent =
            status.percent === null || status.percent === undefined
                ? Number.NaN
                : Number(status.percent);
        const track = createElement("div", {
            className: "mt-2 h-2 overflow-hidden rounded bg-[#414141]",
            role: "progressbar",
            "aria-label": "MX3 compilation progress",
        });
        const fill = createElement("div", {
            className: `h-full bg-[#f9c845] ${Number.isFinite(percent) ? "" : "w-1/2 animate-pulse"}`,
        });
        if (Number.isFinite(percent)) {
            const clamped = Math.max(0, Math.min(100, percent));
            fill.style.width = `${clamped}%`;
            track.setAttribute("aria-valuenow", String(Math.round(clamped)));
            track.setAttribute("aria-valuemin", "0");
            track.setAttribute("aria-valuemax", "100");
        } else {
            track.setAttribute(
                "aria-valuetext",
                "Progress is not currently available",
            );
        }
        track.appendChild(fill);
        compilationStatusEl.appendChild(track);
        if (Number.isFinite(percent)) {
            compilationStatusEl.appendChild(
                createElement("p", {
                    className: "mt-1 text-right text-xs text-gray-300",
                    text: `${Math.round(percent)}%`,
                }),
            );
        }
        if (status.error) {
            compilationStatusEl.appendChild(
                createElement("p", {
                    className: "mt-2 text-sm text-red-300",
                    text: status.error,
                }),
            );
        }
        const logs = Array.isArray(status.logs)
            ? status.logs.join("\n")
            : String(status.logs || "");
        if (logs) {
            const details = createElement("details", {
                className: "mt-2 text-xs",
            });
            details.open = compilerLogOpen;
            details.appendChild(
                createElement("summary", {
                    className: "cursor-pointer text-[#f9c845]",
                    text: "Compiler log",
                }),
            );
            details.appendChild(
                createElement("pre", {
                    className:
                        "eagle-scrollbar mt-1 max-h-36 overflow-auto whitespace-pre-wrap rounded bg-[#232323] p-2 text-gray-300",
                    text: logs,
                }),
            );
            compilationStatusEl.appendChild(details);
        }
    }

    /**
     * Opens the MX3 compilation dialog for an ONNX source model.
     * @param {object} model Model containing the ONNX source artifact.
     * @param {HTMLElement} trigger Button that invoked the dialog.
     * @param {HTMLElement} [overlay] Parent modal overlay, when one is known.
     */
    function open(model, trigger, overlay) {
        compilationModel = model;
        parentOverlay =
            overlay || document.getElementById("modelLibraryOverlay") || null;
        parentOverlayAccessibility = parentOverlay
            ? {
                  hadInert: parentOverlay.hasAttribute("inert"),
                  ariaHidden: parentOverlay.getAttribute("aria-hidden"),
              }
            : null;
        returnFocusId = trigger?.id || null;
        compilationDraft = { advancedEdited: false };
        renderCompilationDialog();
        parentOverlay?.setAttribute("inert", "");
        parentOverlay?.setAttribute("aria-hidden", "true");
        showModal(compilationOverlay);
        compilationModal.querySelector("h2")?.focus();
        void loadCompilationStatus();
    }

    /** Closes the compilation dialog and restores focus to its invoking button. */
    function close() {
        const focusId = returnFocusId;
        hideModal(compilationOverlay);
        if (parentOverlay && parentOverlayAccessibility) {
            if (!parentOverlayAccessibility.hadInert) {
                parentOverlay.removeAttribute("inert");
            }
            if (parentOverlayAccessibility.ariaHidden === null) {
                parentOverlay.removeAttribute("aria-hidden");
            } else {
                parentOverlay.setAttribute(
                    "aria-hidden",
                    parentOverlayAccessibility.ariaHidden,
                );
            }
        }
        parentOverlay = null;
        parentOverlayAccessibility = null;
        returnFocusId = null;
        if (focusId) document.getElementById(focusId)?.focus();
    }

    /**
     * Validates the compilation settings and creates the supported request payload.
     * @returns {{settings: object, profile: object|null, overwrite: boolean}} Request body.
     */
    function compilationPayload() {
        const draft = compilationDraft;
        if (
            (compilationModel?.artifacts?.mx3_dfp ||
                compilationModel?.artifacts?.mx3_postprocessor ||
                compilationModel?.mx3_profile) &&
            !draft.overwrite.checked
        ) {
            throw new Error(
                "Confirm overwrite before replacing existing MX3 artifacts.",
            );
        }
        const numChips = Number(draft.numChips.value);
        const targetText = draft.targetFps.value.trim();
        const targetFps = targetText === "" ? null : Number(targetText);
        if (!Number.isInteger(numChips) || numChips < 1 || numChips > 16) {
            throw new Error("Number of chips must be an integer from 1 to 16.");
        }
        if (
            targetText &&
            (!Number.isFinite(targetFps) || targetFps < 1 || targetFps > 1000)
        ) {
            throw new Error("Target FPS must be between 1 and 1000.");
        }
        const widthText = draft.width.value.trim();
        const heightText = draft.height.value.trim();
        if ((widthText === "") !== (heightText === "")) {
            throw new Error(
                "Input width and height must be supplied together.",
            );
        }
        const width = Number(widthText);
        const height = Number(heightText);
        if (
            widthText &&
            (!Number.isInteger(width) ||
                width < 1 ||
                !Number.isInteger(height) ||
                height < 1)
        ) {
            throw new Error(
                "Input width and height must be positive integers.",
            );
        }
        if (draft.advancedEdited && !widthText) {
            throw new Error(
                "Supply input width and height when using an advanced profile.",
            );
        }
        const profile = widthText
            ? {
                  input_width: width,
                  input_height: height,
                  color_order: draft.colorOrder.value,
                  layout: draft.layout.value,
                  normalization: "zero_to_one",
                  use_model_shape: [
                      draft.useModelWidth.checked,
                      draft.useModelHeight.checked,
                  ],
                  decoder: "yolo_nms_xyxy",
                  adjustable_controls: {
                      confidence: draft.confidence.checked,
                      max_detections: draft.maxDetections.checked,
                  },
                  max_inflight: Number(draft.maxInflight.value),
              }
            : null;
        if (
            profile &&
            (!Number.isInteger(profile.max_inflight) ||
                profile.max_inflight < 1)
        ) {
            throw new Error(
                "Maximum in-flight frames must be a positive integer.",
            );
        }
        return {
            settings: {
                autocrop: draft.autocrop.checked,
                num_chips: numChips,
                effort: draft.effort.value,
                target_fps: targetFps,
            },
            profile,
            overwrite: draft.overwrite.checked,
        };
    }

    /**
     * Starts the selected model's compilation job.
     * @param {HTMLButtonElement} startButton Start button to temporarily disable.
     */
    async function startCompilation(startButton) {
        try {
            const payload = compilationPayload();
            startButton.disabled = true;
            const response = await fetch(
                `${BACKEND_BASE_URL}/model-library/${encodeURIComponent(compilationModel.id)}/mx3-compilation`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                },
            );
            if (!response.ok) throw new Error(await responseError(response));
            const data = await response.json();
            applyCompilationStatus(data.compilation);
        } catch (error) {
            startButton.disabled = false;
            const errorEl = compilationModal.querySelector(
                "[data-compilation-error]",
            );
            errorEl.textContent = error.message;
            errorEl.classList.remove("hidden");
        }
    }

    /** Requests cancellation of the active MX3 compilation job. */
    async function cancelCompilation() {
        if (!compilation?.job_id) return;
        try {
            const response = await fetch(
                `${BACKEND_BASE_URL}/model-library/mx3-compilation/${encodeURIComponent(compilation.job_id)}`,
                { method: "DELETE" },
            );
            if (!response.ok) throw new Error(await responseError(response));
            const data = await response.json();
            applyCompilationStatus(data.compilation);
        } catch (error) {
            const errorEl = compilationModal.querySelector(
                "[data-compilation-error]",
            );
            errorEl.textContent = error.message;
            errorEl.classList.remove("hidden");
        }
    }

    /** Builds the nested compilation popup and its guided/advanced controls. */
    function renderCompilationDialog() {
        compilationModal.innerHTML = "";
        const hasExistingMx3 = Boolean(
            compilationModel?.artifacts?.mx3_dfp ||
                compilationModel?.artifacts?.mx3_postprocessor ||
                compilationModel?.mx3_profile,
        );
        const header = createElement(
            "div",
            {
                className:
                    "flex items-center justify-between border-b border-[#414141] p-4",
            },
            [
                createElement("h2", {
                    id: "mx3CompilationTitle",
                    className: "text-lg font-semibold text-[#f9c845]",
                    tabindex: "-1",
                    text: "Compile for MX3",
                }),
                createElement("button", {
                    type: "button",
                    className:
                        "rounded text-xl text-gray-300 hover:text-white focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
                    text: "×",
                    "aria-label": "Close MX3 compilation",
                    onClick: close,
                }),
            ],
        );
        const autocrop = createElement("input", {
            type: "checkbox",
            checked: "checked",
            "aria-label": "Enable autocrop",
        });
        const numChips = createElement("input", {
            className: INPUT_CLASS,
            type: "number",
            min: "1",
            max: "16",
            value: "4",
        });
        const effort = createElement(
            "select",
            { className: INPUT_CLASS },
            ["lazy", "normal", "hard"].map((value) =>
                createElement("option", {
                    value,
                    text: value,
                    selected: value === "normal" ? "selected" : null,
                }),
            ),
        );
        const targetFps = createElement("input", {
            className: INPUT_CLASS,
            type: "number",
            min: "1",
            max: "1000",
            step: "any",
            placeholder: "Optional",
        });
        const width = createElement("input", {
            className: INPUT_CLASS,
            type: "number",
            min: "1",
            step: "1",
            placeholder: "Detected by backend",
        });
        const height = createElement("input", {
            className: INPUT_CLASS,
            type: "number",
            min: "1",
            step: "1",
            placeholder: "Detected by backend",
        });
        const colorOrder = createElement(
            "select",
            { className: INPUT_CLASS },
            ["rgb", "bgr"].map((value) =>
                createElement("option", { value, text: value }),
            ),
        );
        const layout = createElement(
            "select",
            { className: INPUT_CLASS },
            ["hwzc", "nchw", "nhwc"].map((value) =>
                createElement("option", { value, text: value }),
            ),
        );
        const useModelWidth = createElement("input", {
            type: "checkbox",
            "aria-label": "Use model width",
            onChange: () => {
                compilationDraft.advancedEdited = true;
            },
        });
        const useModelHeight = createElement("input", {
            type: "checkbox",
            checked: "checked",
            "aria-label": "Use model height",
            onChange: () => {
                compilationDraft.advancedEdited = true;
            },
        });
        const confidence = createElement("input", {
            type: "checkbox",
            checked: "checked",
            "aria-label": "Allow confidence adjustment",
            onChange: () => {
                compilationDraft.advancedEdited = true;
            },
        });
        const maxDetections = createElement("input", {
            type: "checkbox",
            checked: "checked",
            "aria-label": "Allow maximum detections adjustment",
            onChange: () => {
                compilationDraft.advancedEdited = true;
            },
        });
        const maxInflight = createElement("input", {
            className: INPUT_CLASS,
            type: "number",
            min: "1",
            step: "1",
            value: "8",
        });
        [width, height, colorOrder, layout, maxInflight].forEach((input) =>
            input.addEventListener("input", () => {
                compilationDraft.advancedEdited = true;
            }),
        );
        const overwrite = createElement("input", {
            type: "checkbox",
            "aria-label": "Confirm replacement of existing MX3 artifacts",
        });
        compilationDraft = {
            autocrop,
            numChips,
            effort,
            targetFps,
            width,
            height,
            colorOrder,
            layout,
            useModelWidth,
            useModelHeight,
            confidence,
            maxDetections,
            maxInflight,
            overwrite,
            advancedEdited: false,
        };
        const errorEl = createElement("p", {
            className:
                "hidden rounded border border-red-500/60 bg-red-900/30 p-2 text-sm text-red-300",
            "data-compilation-error": "",
        });
        compilationStatusEl = createElement("div", {
            className: "rounded border border-[#414141] bg-[#232323] p-3",
            role: "status",
            "aria-live": "polite",
        });
        const advanced = createElement("details", {
            className: "rounded border border-[#414141] p-3",
        });
        advanced.appendChild(
            createElement("summary", {
                className: "cursor-pointer text-sm font-medium text-[#f9c845]",
                text: "Advanced profile",
            }),
        );
        advanced.appendChild(
            createElement("p", {
                className: "mt-2 text-xs text-gray-300",
                text: "Leave input size blank for backend-guided YOLO26 defaults. Editing advanced options requires both dimensions.",
            }),
        );
        advanced.appendChild(
            createElement(
                "div",
                { className: "mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2" },
                [
                    compilationField("Input width", width),
                    compilationField("Input height", height),
                    compilationField("Color order", colorOrder),
                    compilationField("Layout", layout),
                    compilationField("Max in-flight frames", maxInflight),
                    createElement(
                        "div",
                        { className: "text-sm text-[#f9c845]" },
                        [
                            createElement("span", {
                                text: "Normalization: zero_to_one",
                            }),
                        ],
                    ),
                    createElement(
                        "div",
                        { className: "text-sm text-[#f9c845]" },
                        [
                            createElement("span", {
                                text: "Decoder: yolo_nms_xyxy",
                            }),
                        ],
                    ),
                    createElement(
                        "label",
                        {
                            className:
                                "flex items-center gap-2 text-sm text-gray-200",
                        },
                        [
                            useModelWidth,
                            createElement("span", { text: "Use model width" }),
                        ],
                    ),
                    createElement(
                        "label",
                        {
                            className:
                                "flex items-center gap-2 text-sm text-gray-200",
                        },
                        [
                            useModelHeight,
                            createElement("span", { text: "Use model height" }),
                        ],
                    ),
                    createElement(
                        "label",
                        {
                            className:
                                "flex items-center gap-2 text-sm text-gray-200",
                        },
                        [
                            confidence,
                            createElement("span", {
                                text: "Confidence adjustable",
                            }),
                        ],
                    ),
                    createElement(
                        "label",
                        {
                            className:
                                "flex items-center gap-2 text-sm text-gray-200",
                        },
                        [
                            maxDetections,
                            createElement("span", {
                                text: "Max detections adjustable",
                            }),
                        ],
                    ),
                ],
            ),
        );
        const startButton = createElement("button", {
            type: "button",
            className:
                "px-3 py-2 bg-[#f9c845] text-[#232323] rounded hover:bg-[#d4a83a] disabled:opacity-50 text-sm font-medium",
            text: "Start compilation",
            "data-compilation-start": "",
            disabled: isCompilationActive() ? "disabled" : null,
            onClick: () => startCompilation(startButton),
        });
        const cancelButton = createElement("button", {
            type: "button",
            className:
                "px-3 py-2 bg-red-700 text-white rounded hover:bg-red-600 disabled:opacity-50 text-sm",
            text: "Cancel",
            "data-compilation-cancel": "",
            disabled:
                isCompilationActive() &&
                String(compilation?.state || "").toLowerCase() !== "cancelling"
                    ? null
                    : "disabled",
            onClick: cancelCompilation,
        });
        const body = createElement(
            "div",
            { className: "eagle-scrollbar space-y-3 overflow-y-auto p-4" },
            [
                createElement("p", {
                    className: "text-sm text-gray-300",
                    text: `Compile ${compilationModel.display_name || compilationModel.id}'s ONNX source. Guided mode detects its input size and uses the YOLO26 defaults.`,
                }),
                createElement(
                    "label",
                    {
                        className:
                            "flex items-center gap-2 text-sm text-gray-200",
                    },
                    [
                        autocrop,
                        createElement("span", { text: "Enable autocrop" }),
                    ],
                ),
                createElement(
                    "div",
                    { className: "grid grid-cols-1 gap-3 sm:grid-cols-3" },
                    [
                        compilationField("Number of chips", numChips),
                        compilationField("Effort", effort),
                        compilationField("Target FPS", targetFps),
                    ],
                ),
                hasExistingMx3
                    ? createElement(
                          "label",
                          {
                              className:
                                  "flex items-center gap-2 rounded border border-yellow-500/50 bg-yellow-900/20 p-2 text-sm text-yellow-100",
                          },
                          [
                              overwrite,
                              createElement("span", {
                                  text: "Overwrite existing MX3 artifacts and profile",
                              }),
                          ],
                      )
                    : createElement("span"),
                advanced,
                errorEl,
                compilationStatusEl,
                createElement("div", { className: "flex justify-end gap-2" }, [
                    cancelButton,
                    startButton,
                ]),
            ],
        );
        compilationModal.append(header, body);
        renderCompilationStatus();
    }

    ({ overlay: compilationOverlay, modal: compilationModal } =
        getOrCreateModalElements({
            overlayId: "mx3CompilationOverlay",
            modalId: "mx3CompilationModal",
            overlayClassName:
                "fixed inset-0 z-[70] hidden flex items-center justify-center",
            overlayStyle:
                "background-color: rgba(0, 0, 0, 0.25); backdrop-filter: blur(6px);",
            modalClassName:
                "bg-[#1a1a1a] rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] flex flex-col border border-[#414141]",
        }));
    compilationModal.setAttribute("role", "dialog");
    compilationModal.setAttribute("aria-modal", "true");
    compilationModal.setAttribute("aria-labelledby", "mx3CompilationTitle");
    closeOnBackdropClick(compilationOverlay, close);

    /**
     * Closes this dialog without allowing the parent modal's Escape handler to run.
     * @param {KeyboardEvent} event The document key event.
     */
    function closeOnCompilationEscape(event) {
        if (
            event.key === "Escape" &&
            !compilationOverlay.classList.contains("hidden")
        ) {
            event.stopImmediatePropagation();
            close();
        }
    }

    document.addEventListener("keydown", closeOnCompilationEscape, true);
    document.addEventListener("mx3-compilation-progress", (event) => {
        applyCompilationStatus(event.detail);
    });
    document.addEventListener("mx3-compilation-reconnected", () => {
        void loadCompilationStatus();
    });
    document.addEventListener("backend-disconnected", () => close());
    void loadCompilationStatus();

    /**
     * Returns whether the compilation dialog is currently visible.
     * @returns {boolean} Whether the dialog is open.
     */
    function isOpen() {
        return !compilationOverlay.classList.contains("hidden");
    }

    return { open, close, isOpen };
}
