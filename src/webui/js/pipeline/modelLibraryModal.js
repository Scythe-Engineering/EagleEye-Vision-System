import { BACKEND_BASE_URL } from "../config.js";
import {
    closeOnBackdropClick,
    closeOnEscape,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "../ui/modal.js";

const ARTIFACT_SLOTS = ["pt", "onnx", "engine", "mx3_dfp", "mx3_postprocessor"];
const ARTIFACT_LABELS = {
    pt: "PyTorch (.pt)",
    onnx: "ONNX (.onnx)",
    engine: "TensorRT (.engine)",
    mx3_dfp: "MX3 DFP (.dfp)",
    mx3_postprocessor: "MX3 postprocessor (.onnx)",
};
const INPUT_CLASS =
    "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]";

/**
 * Detects the artifact slot represented by a selected model filename.
 * @param {string} filename Selected file name.
 * @returns {string} Model-library artifact slot.
 */
export function detectArtifactSlot(filename) {
    const lowerName = filename.toLowerCase();
    if (lowerName.endsWith(".pt")) return "pt";
    if (lowerName.endsWith(".engine")) return "engine";
    if (lowerName.endsWith(".dfp")) return "mx3_dfp";
    if (lowerName.endsWith(".onnx")) {
        return lowerName.includes("postprocess") ||
            lowerName.endsWith("_post.onnx")
            ? "mx3_postprocessor"
            : "onnx";
    }
    throw new Error("Choose a .pt, .onnx, .engine, or .dfp model file.");
}

/**
 * Registers the reusable model-library management modal.
 * @returns {{open: Function, close: Function, refresh: Function}}
 */
export function registerModelLibraryModal() {
    if (globalThis.ModelLibraryModal) {
        return globalThis.ModelLibraryModal;
    }

    let models = [];
    let selectedModelId = null;
    let selectCallback = null;
    let overlay;
    let modal;
    let listEl;
    let editorEl;
    let messageEl;
    let draftValues = null;
    const pendingArtifacts = new Map();

    /**
     * Returns an error message from an API response without assuming its shape.
     * @param {Response} response Failed HTTP response.
     * @returns {Promise<string>} Backend error message.
     */
    async function responseError(response) {
        try {
            const data = await response.json();
            return (
                data.error ||
                data.detail ||
                data.message ||
                JSON.stringify(data)
            );
        } catch (_) {
            return response.statusText || `HTTP ${response.status}`;
        }
    }

    /**
     * Displays a modal-level success or error message.
     * @param {string} text Message text.
     * @param {boolean} isError Whether this is an error message.
     */
    function setMessage(text = "", isError = false) {
        if (!messageEl) return;
        messageEl.textContent = text;
        messageEl.className = `text-sm rounded p-2 ${isError ? "bg-red-900/30 border border-red-500/60 text-red-300" : "bg-green-900/20 border border-green-500/40 text-green-300"}`;
        messageEl.classList.toggle("hidden", !text);
    }

    /**
     * Fetches the complete model library from the backend.
     * @returns {Promise<void>}
     */
    async function refresh() {
        setMessage("");
        try {
            const response = await fetch(`${BACKEND_BASE_URL}/model-library`);
            if (!response.ok) throw new Error(await responseError(response));
            const data = await response.json();
            models = Array.isArray(data.models) ? data.models : [];
            if (!models.some((model) => model.id === selectedModelId)) {
                selectedModelId = models[0]?.id || null;
            }
            render();
        } catch (error) {
            models = [];
            selectedModelId = null;
            render();
            setMessage(`Unable to load model library: ${error.message}`, true);
        }
    }

    /**
     * Produces the payload for model create and update requests.
     * @param {HTMLInputElement} nameInput Display-name field.
     * @param {HTMLTextAreaElement} classesInput Newline class-name field.
     * @param {HTMLTextAreaElement} profileInput MX3 profile JSON field.
     * @returns {{display_name: string, class_names: string[], mx3_profile: object|null}}
     */
    function modelPayload(nameInput, classesInput, profileInput) {
        const displayName = nameInput.value.trim();
        if (!displayName) throw new Error("Display name is required.");
        const classNames = classesInput.value
            .split(/\r?\n/)
            .map((value) => value.trim())
            .filter(Boolean);
        const payload = {
            display_name: displayName,
            class_names: classNames.length ? classNames : null,
            mx3_profile: null,
        };
        const profileText = profileInput.value.trim();
        if (profileText) {
            try {
                const profile = JSON.parse(profileText);
                if (
                    !profile ||
                    typeof profile !== "object" ||
                    Array.isArray(profile)
                ) {
                    throw new Error("MX3 profile must be a JSON object.");
                }
                payload.mx3_profile = profile;
            } catch (error) {
                throw new Error(
                    error.message || "MX3 profile must contain valid JSON.",
                );
            }
        }
        return payload;
    }

    /**
     * Uploads an artifact to a model slot.
     * @param {string} modelId Stable model ID.
     * @param {string} slot Artifact slot name.
     * @param {File} file Selected artifact file.
     * @returns {Promise<object>} Upload response payload.
     */
    async function uploadArtifact(modelId, slot, file) {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch(
            `${BACKEND_BASE_URL}/model-library/${encodeURIComponent(modelId)}/artifacts/${encodeURIComponent(slot)}`,
            { method: "POST", body: formData },
        );
        if (!response.ok) throw new Error(await responseError(response));
        return response.json();
    }

    /**
     * Removes an artifact from a model slot.
     * @param {string} modelId Stable model ID.
     * @param {string} slot Artifact slot name.
     * @returns {Promise<void>}
     */
    async function removeArtifact(modelId, slot) {
        const response = await fetch(
            `${BACKEND_BASE_URL}/model-library/${encodeURIComponent(modelId)}/artifacts/${encodeURIComponent(slot)}`,
            { method: "DELETE" },
        );
        if (!response.ok) throw new Error(await responseError(response));
        const data = await response.json();
        await refresh();
        setMessage(
            data.restart_required
                ? `${slot} removed. Affected pipelines require backend restart.`
                : `${slot} removed.`,
        );
    }

    /**
     * Renders the list of models and the selected model editor.
     */
    function render() {
        if (!listEl || !editorEl) return;
        listEl.innerHTML = "";
        listEl.appendChild(
            createElement("button", {
                type: "button",
                className:
                    "m-3 px-3 py-2 bg-[#f9c845] text-[#232323] rounded hover:bg-[#d4a83a] text-sm font-medium",
                text: "New Model",
                onClick: () => {
                    selectedModelId = null;
                    draftValues = null;
                    pendingArtifacts.clear();
                    render();
                },
            }),
        );
        if (!models.length) {
            listEl.appendChild(
                createElement("p", {
                    className: "text-sm text-[#ac8a2f] p-3",
                    text: "No models in the library.",
                }),
            );
        }
        models.forEach((model) => {
            const active = model.id === selectedModelId;
            const button = createElement("button", {
                type: "button",
                className: `w-full text-left p-3 border-b border-[#414141] hover:bg-[#2a2a2a] ${active ? "bg-[#3a3218]" : ""}`,
                onClick: () => {
                    selectedModelId = model.id;
                    draftValues = null;
                    pendingArtifacts.clear();
                    render();
                },
            });
            button.appendChild(
                createElement("div", {
                    className: "text-sm text-white font-medium",
                    text: model.display_name || model.id,
                }),
            );
            button.appendChild(
                createElement("div", {
                    className: "text-xs text-[#ac8a2f] break-all",
                    text: model.id,
                }),
            );
            listEl.appendChild(button);
        });
        renderEditor(
            models.find((model) => model.id === selectedModelId) || null,
        );
    }

    /**
     * Renders an existing-model editor or new-model editor.
     * @param {object|null} model Model to edit, or null for a new model.
     */
    function renderEditor(model) {
        editorEl.innerHTML = "";
        const isNew = !model;
        const title = createElement("h3", {
            className: "text-lg font-semibold text-[#f9c845]",
            text: isNew ? "New Model" : "Edit Model",
        });
        const nameInput = createElement("input", {
            className: INPUT_CLASS,
            type: "text",
            value: model?.display_name || draftValues?.displayName || "",
            placeholder: "Display name",
        });
        const classesInput = createElement("textarea", {
            className: `${INPUT_CLASS} min-h-20`,
            placeholder: "One class name per line (optional)",
            text: Array.isArray(model?.class_names)
                ? model.class_names.join("\n")
                : draftValues?.classNames || "",
        });
        const profileInput = createElement("textarea", {
            className: `${INPUT_CLASS} min-h-24 font-mono text-xs`,
            placeholder: "Optional MX3 profile JSON",
            text: model?.mx3_profile
                ? JSON.stringify(model.mx3_profile, null, 2)
                : draftValues?.mx3Profile || "",
        });
        const preserveDraft = () => {
            draftValues = {
                displayName: nameInput.value,
                classNames: classesInput.value,
                mx3Profile: profileInput.value,
            };
        };
        const form = createElement("div", { className: "space-y-3" }, [
            title,
            createElement("label", {
                className: "text-sm text-[#f9c845]",
                text: "Display name (required)",
            }),
            nameInput,
            createElement("label", {
                className: "text-sm text-[#f9c845]",
                text: "Ordered class names",
            }),
            classesInput,
            createElement("label", {
                className: "text-sm text-[#f9c845]",
                text: "MX3 profile JSON",
            }),
            profileInput,
        ]);
        const saveButton = createElement("button", {
            type: "button",
            className:
                "px-3 py-2 bg-[#f9c845] text-[#232323] rounded hover:bg-[#d4a83a] disabled:opacity-50 text-sm font-medium",
            text: isNew ? "Create Model" : "Save Details",
            onClick: async () => {
                saveButton.disabled = true;
                saveButton.textContent = isNew ? "Creating…" : "Saving…";
                try {
                    const payload = modelPayload(
                        nameInput,
                        classesInput,
                        profileInput,
                    );
                    const url = model
                        ? `${BACKEND_BASE_URL}/model-library/${encodeURIComponent(model.id)}`
                        : `${BACKEND_BASE_URL}/model-library`;
                    const response = await fetch(url, {
                        method: model ? "PATCH" : "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                    });
                    if (!response.ok)
                        throw new Error(await responseError(response));
                    const created = !model;
                    const data = await response.json().catch(() => ({}));
                    selectedModelId =
                        data.id ||
                        data.model?.id ||
                        model?.id ||
                        selectedModelId;

                    const queuedArtifacts = created
                        ? [...pendingArtifacts.entries()]
                        : [];
                    pendingArtifacts.clear();
                    draftValues = null;
                    const uploadFailures = [];
                    let restartRequired = data.restart_required;
                    for (const [slot, file] of queuedArtifacts) {
                        try {
                            const upload = await uploadArtifact(
                                selectedModelId,
                                slot,
                                file,
                            );
                            restartRequired ||= upload.restart_required;
                        } catch (error) {
                            uploadFailures.push(
                                `${file.name}: ${error.message}`,
                            );
                        }
                    }

                    await refresh();
                    if (uploadFailures.length) {
                        setMessage(
                            `Model created, but some formats could not be uploaded: ${uploadFailures.join("; ")}`,
                            true,
                        );
                    } else {
                        setMessage(
                            created
                                ? queuedArtifacts.length
                                    ? "Model created and formats uploaded."
                                    : "Model created."
                                : restartRequired
                                  ? "Model details saved. Affected pipelines require backend restart."
                                  : "Model details saved.",
                        );
                    }
                } catch (error) {
                    saveButton.disabled = false;
                    saveButton.textContent = isNew
                        ? "Create Model"
                        : "Save Details";
                    setMessage(error.message, true);
                }
            },
        });
        form.appendChild(saveButton);

        const artifacts = model?.artifacts || {};
        form.appendChild(
            createElement("h4", {
                className: "text-sm font-semibold text-[#f9c845] pt-3",
                text: "Model formats",
            }),
        );
        const fileInput = createElement("input", {
            type: "file",
            accept: ".pt,.onnx,.engine,.dfp",
            className: "hidden",
            "aria-label": "Choose a model file",
            onChange: async () => {
                const file = fileInput.files?.[0];
                if (!file) return;
                try {
                    const slot = detectArtifactSlot(file.name);
                    if (model) {
                        uploadButton.disabled = true;
                        uploadButton.textContent = "Uploading…";
                        const data = await uploadArtifact(model.id, slot, file);
                        await refresh();
                        setMessage(
                            data.restart_required
                                ? `${ARTIFACT_LABELS[slot]} uploaded. Affected pipelines require backend restart.`
                                : `${ARTIFACT_LABELS[slot]} uploaded.`,
                        );
                    } else {
                        preserveDraft();
                        pendingArtifacts.set(slot, file);
                        renderEditor(null);
                        setMessage(
                            `${ARTIFACT_LABELS[slot]} is ready to upload after the model is created.`,
                        );
                    }
                } catch (error) {
                    fileInput.value = "";
                    setMessage(error.message, true);
                    uploadButton.disabled = false;
                    uploadButton.textContent = "Upload Model";
                }
            },
        });
        const uploadButton = createElement("button", {
            type: "button",
            className:
                "w-fit px-3 py-2 bg-[#333] text-white rounded hover:bg-[#444] disabled:opacity-50 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            text: "Upload Model",
            onClick: () => fileInput.click(),
        });
        form.appendChild(
            createElement("div", { className: "flex flex-col gap-2" }, [
                uploadButton,
                fileInput,
                createElement("p", {
                    className: "text-xs text-[#c7a84f]",
                    text: "Supported formats: .pt, .onnx, .engine, and .dfp",
                }),
            ]),
        );

        const listedArtifacts = ARTIFACT_SLOTS.flatMap((slot) => {
            const artifact = artifacts[slot];
            const pendingFile = isNew ? pendingArtifacts.get(slot) : null;
            if (!artifact && !pendingFile) return [];
            const row = createElement("div", {
                className:
                    "flex flex-wrap items-center justify-between gap-2 border border-[#414141] rounded p-2",
            });
            row.appendChild(
                createElement("div", { className: "min-w-0" }, [
                    createElement("div", {
                        className: "text-sm text-white",
                        text: ARTIFACT_LABELS[slot],
                    }),
                    createElement("div", {
                        className: "text-xs text-[#c7a84f] break-all",
                        text: pendingFile
                            ? `${pendingFile.name} · Pending`
                            : artifact.filename,
                    }),
                ]),
            );
            row.appendChild(
                createElement("button", {
                    type: "button",
                    className:
                        "px-2 py-1 bg-red-700 text-white rounded hover:bg-red-600 text-xs focus:outline-none focus:ring-2 focus:ring-red-300",
                    text: pendingFile ? "Clear" : "Remove",
                    onClick: async () => {
                        if (pendingFile) {
                            preserveDraft();
                            pendingArtifacts.delete(slot);
                            renderEditor(null);
                            return;
                        }
                        try {
                            await removeArtifact(model.id, slot);
                        } catch (error) {
                            setMessage(
                                `Could not remove ${ARTIFACT_LABELS[slot]}: ${error.message}`,
                                true,
                            );
                        }
                    },
                }),
            );
            return [row];
        });
        if (listedArtifacts.length) {
            form.appendChild(
                createElement(
                    "div",
                    { className: "space-y-2" },
                    listedArtifacts,
                ),
            );
        } else {
            form.appendChild(
                createElement("p", {
                    className:
                        "rounded border border-dashed border-[#414141] p-3 text-sm text-gray-300",
                    text: "No model formats uploaded yet.",
                }),
            );
        }

        if (model) {
            const references = Array.isArray(model.referenced_by)
                ? model.referenced_by
                : [];
            if (references.length)
                form.appendChild(
                    createElement("p", {
                        className: "text-xs text-yellow-200",
                        text: `Cannot delete: referenced by ${references.join(", ")}.`,
                    }),
                );
            const actionRow = createElement("div", {
                className: "flex gap-2 pt-2",
            });
            actionRow.appendChild(
                createElement("button", {
                    type: "button",
                    className:
                        "px-3 py-2 bg-[#f9c845] text-[#232323] rounded hover:bg-[#d4a83a] text-sm font-medium",
                    text: "Use This Model",
                    onClick: () => {
                        if (selectCallback) selectCallback(model);
                        close();
                    },
                }),
            );
            actionRow.appendChild(
                createElement("button", {
                    type: "button",
                    className:
                        "px-3 py-2 bg-red-700 text-white rounded hover:bg-red-600 disabled:opacity-50 text-sm",
                    text: "Delete Model",
                    disabled: references.length ? "disabled" : null,
                    title: references.length
                        ? "Remove pipeline references before deleting"
                        : "",
                    onClick: async () => {
                        try {
                            const response = await fetch(
                                `${BACKEND_BASE_URL}/model-library/${encodeURIComponent(model.id)}`,
                                { method: "DELETE" },
                            );
                            if (!response.ok)
                                throw new Error(await responseError(response));
                            selectedModelId = null;
                            await refresh();
                            setMessage("Model deleted.");
                        } catch (error) {
                            setMessage(
                                `Model was not deleted: ${error.message}`,
                                true,
                            );
                        }
                    },
                }),
            );
            form.appendChild(actionRow);
        }
        editorEl.appendChild(form);
    }

    /** Hides the model-library modal. */
    function close() {
        draftValues = null;
        pendingArtifacts.clear();
        hideModal(overlay);
    }

    /**
     * Opens the modal and optionally supplies a model-selection callback.
     * @param {{selectedModelId?: string, onSelect?: Function}} options Open options.
     */
    function open(options = {}) {
        selectedModelId = options.selectedModelId || selectedModelId;
        selectCallback = options.onSelect || null;
        showModal(overlay);
        void refresh();
    }

    ({ overlay, modal } = getOrCreateModalElements({
        overlayId: "modelLibraryOverlay",
        modalId: "modelLibraryModal",
        overlayClassName:
            "fixed inset-0 z-[60] hidden flex items-center justify-center",
        overlayStyle:
            "background-color: rgba(0, 0, 0, 0.25); backdrop-filter: blur(6px);",
        modalClassName:
            "bg-[#1a1a1a] rounded-lg shadow-xl max-w-5xl w-full mx-4 max-h-[90vh] flex flex-col border border-[#414141]",
    }));
    modal.innerHTML = "";
    const header = createElement("div", {
        className:
            "flex items-center justify-between p-4 border-b border-[#414141]",
    });
    header.appendChild(
        createElement("h2", {
            className: "text-xl font-semibold text-[#f9c845]",
            text: "Model Library",
        }),
    );
    header.appendChild(
        createElement("button", {
            type: "button",
            className:
                "text-gray-300 hover:text-white text-xl focus:outline-none focus:ring-2 focus:ring-[#f9c845] rounded",
            text: "×",
            "aria-label": "Close Model Library",
            onClick: close,
        }),
    );
    messageEl = createElement("div", { className: "hidden" });
    listEl = createElement("div", {
        className:
            "eagle-scrollbar w-full md:w-2/5 border-r border-[#414141] overflow-y-auto",
    });
    editorEl = createElement("div", {
        className: "eagle-scrollbar w-full md:w-3/5 p-4 overflow-y-auto",
    });
    modal.appendChild(header);
    modal.appendChild(
        createElement("div", { className: "px-4 pt-3" }, [messageEl]),
    );
    modal.appendChild(
        createElement(
            "div",
            {
                className:
                    "flex flex-col md:flex-row min-h-0 flex-1 overflow-hidden",
            },
            [listEl, editorEl],
        ),
    );
    closeOnBackdropClick(overlay, close);
    closeOnEscape(overlay, close);

    const popup = { open, close, refresh };
    globalThis.ModelLibraryModal = popup;
    return popup;
}
