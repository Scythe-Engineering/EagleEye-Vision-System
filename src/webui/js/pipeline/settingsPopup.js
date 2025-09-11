import { BACKEND_BASE_URL } from "../config.js";

(function () {
    const OVERLAY_ID = "operationSettingsOverlay";
    const MODAL_ID = "operationSettingsModal";
    // Visualization state
    let _visInterval = null;
    let _currentVisObjectUrl = null;
    let _currentVisCamera = null;
    let _currentVisPipeline = null;
    let _currentVisAction = null;

    function createElement(tag, attrs = {}, children = []) {
        const el = document.createElement(tag);
        Object.entries(attrs).forEach(([k, v]) => {
            if (k === "className") {
                el.className = v;
            } else if (k === "text") {
                el.textContent = v;
            } else if (k === "html") {
                el.innerHTML = v;
            } else if (k.startsWith("on") && typeof v === "function") {
                el.addEventListener(k.substring(2).toLowerCase(), v);
            } else if (v !== undefined && v !== null) {
                el.setAttribute(k, String(v));
            }
        });
        (children || []).forEach((c) => el.appendChild(c));
        return el;
    }

    function buildField(name, def, currentValues, originalValues) {
        const fieldId = `setting-${name}`;
        const label = createElement("label", {
            for: fieldId,
            className: "block text-sm font-medium text-[#f9c845] mb-1",
            text: def.description || name,
        });

        let input;
        const currentValue =
            currentValues && name in currentValues
                ? currentValues[name]
                : def.default;

        const originalValue =
            originalValues && name in originalValues
                ? originalValues[name]
                : def.default;

        // Check if field has been edited
        const isEdited =
            JSON.stringify(currentValue) !== JSON.stringify(originalValue);

        if (def.options && Array.isArray(def.options)) {
            input = createElement("select", {
                id: fieldId,
                className:
                    "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            });
            def.options.forEach((opt) => {
                const optEl = createElement("option", {
                    value: String(opt),
                    text: String(opt),
                });
                if (
                    currentValue !== undefined &&
                    String(currentValue) === String(opt)
                )
                    optEl.selected = true;
                input.appendChild(optEl);
            });
        } else {
            const type = def.type;
            let inputType = "text";
            if (type === "int" || type === "float") inputType = "number";
            if (type === "str") inputType = "text";
            if (type === "bool") inputType = "checkbox";

            const attrs = {
                id: fieldId,
                type: inputType,
                className:
                    inputType === "checkbox"
                        ? "h-4 w-4 text-[#f9c845] focus:ring-[#f9c845] border-[#414141] rounded bg-[#232323]"
                        : "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            };
            if (inputType === "number") {
                if (typeof def.min === "number") attrs.min = String(def.min);
                if (typeof def.max === "number") attrs.max = String(def.max);
                if (type === "int") attrs.step = "1";
                if (type === "float") attrs.step = "any";
            }
            input = createElement("input", attrs);
            if (inputType === "checkbox") {
                input.checked = Boolean(currentValue);
            } else if (currentValue !== undefined && currentValue !== null) {
                input.value = String(currentValue);
            }
        }

        const hint = createElement("div", {
            className: "text-xs text-[#ac8a2f] ml-auto",
            text: def.required ? "Required" : "Optional",
        });

        const labelRow = createElement(
            "div",
            {
                className: "flex items-center mb-2",
            },
            [label, hint],
        );

        // Create input container with relative positioning
        const inputContainer = createElement("div", {
            className: "relative",
        });

        // Add input to container
        inputContainer.appendChild(input);

        // Create edited indicator (yellow circle) positioned next to input
        const editedIndicator = createElement("div", {
            className:
                "absolute -left-1 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-yellow-400 rounded-full",
            title: "This field has been modified from its default value",
            style: isEdited ? "" : "display: none;",
        });
        inputContainer.appendChild(editedIndicator);

        // Add event listener to detect real-time changes
        const updateIndicator = () => {
            let currentVal;
            if (input.tagName.toLowerCase() === "select") {
                currentVal = input.value;
            } else if (input.type === "checkbox") {
                currentVal = input.checked;
            } else if (def.type === "int") {
                currentVal = parseInt(input.value, 10);
            } else if (def.type === "float") {
                currentVal = parseFloat(input.value);
            } else {
                currentVal = input.value;
            }

            const edited =
                JSON.stringify(currentVal) !== JSON.stringify(originalValue);
            editedIndicator.style.display = edited ? "block" : "none";
        };

        input.addEventListener("input", updateIndicator);
        input.addEventListener("change", updateIndicator);

        const wrapper = createElement("div", { className: "mb-4" }, [
            labelRow,
            inputContainer,
        ]);
        return {
            wrapper,
            getValue: () => {
                if (input.tagName.toLowerCase() === "select")
                    return input.value;
                if (input.type === "checkbox") return input.checked;
                if (def.type === "int") return parseInt(input.value, 10);
                if (def.type === "float") return parseFloat(input.value);
                return input.value;
            },
        };
    }

    function renderForm(
        modalBody,
        config,
        initialValues,
        originalValues,
        onSave,
    ) {
        modalBody.innerHTML = "";
        const fields = [];

        const params = config?.parameters || {};
        Object.keys(params).forEach((key) => {
            const field = buildField(
                key,
                params[key],
                initialValues || {},
                originalValues || {},
            );
            fields.push({ name: key, ...field });
            modalBody.appendChild(field.wrapper);
        });

        // Set up auto-save event listeners
        setupAutoSaveListeners(
            modalBody,
            fields,
            originalValues,
            onSave,
            config,
        );

        return () => {
            const result = {};
            fields.forEach((f) => {
                result[f.name] = f.getValue();
            });
            return result;
        };
    }

    function setupAutoSaveListeners(
        modalBody,
        fields,
        originalValues,
        onSave,
        config,
    ) {
        console.log(
            "Setting up auto-save listeners for",
            fields.length,
            "fields",
        );

        // Function to check if restart is required
        const checkIfRestartRequired = (
            currentValues,
            originalValues,
            config,
        ) => {
            if (!config?.parameters) return false;

            for (const field of fields) {
                const paramConfig = config.parameters?.[field.name];
                if (paramConfig?.restart_for_change) {
                    const currentValue = currentValues[field.name];
                    const originalValue = originalValues[field.name];
                    if (
                        JSON.stringify(currentValue) !==
                        JSON.stringify(originalValue)
                    ) {
                        console.log(
                            `Field ${field.name} requires restart (changed from ${originalValue} to ${currentValue})`,
                        );
                        return true;
                    }
                }
            }
            return false;
        };

        // Function to trigger auto-save
        const triggerAutoSave = () => {
            console.log("[SETTINGS] Auto-save triggered", {
                operationName: config?.operationName || "unknown",
                timestamp: new Date().toISOString(),
            });
            // Get current form values and call onSave callback if provided
            const currentValues = {};
            fields.forEach((field) => {
                currentValues[field.name] = field.getValue();
            });
            console.log(
                "[SETTINGS] Current form values during auto-save:",
                currentValues,
            );

            // Check if restart is required
            const requiresRestart = checkIfRestartRequired(
                currentValues,
                originalValues,
                config,
            );
            currentValues._requiresRestart = requiresRestart;

            if (typeof onSave === "function") {
                // Mark this as an auto-save call
                currentValues._isAutoSave = true;
                console.log("Calling onSave with _isAutoSave=true");
                console.log(
                    "window.pipelineCreator exists:",
                    !!window.pipelineCreator,
                );
                if (window.pipelineCreator) {
                    console.log(
                        "window.pipelineCreator.autoSavePipeline exists:",
                        !!window.pipelineCreator.autoSavePipeline,
                    );
                }
                onSave(currentValues);

                // Update restart indicator if available
                if (window.pipelineCreator?.updateRestartIndicator) {
                    const requiresRestart = checkIfRestartRequired(
                        currentValues,
                        originalValues,
                        config,
                    );
                    window.pipelineCreator.updateRestartIndicator(
                        requiresRestart,
                    );
                }
            } else {
                console.log("onSave function not provided");
            }
        };

        // Set up event listeners for all inputs
        const allInputs = modalBody.querySelectorAll("input, select");
        allInputs.forEach((input) => {
            if (
                input.tagName.toLowerCase() !== "select" &&
                input.type !== "checkbox"
            ) {
                // Text inputs and number inputs: Enter key and blur
                input.addEventListener("keydown", (e) => {
                    if (e.key === "Enter") {
                        e.preventDefault();
                        triggerAutoSave();
                    }
                });

                input.addEventListener("blur", () => {
                    triggerAutoSave();
                });
            } else {
                // Select elements and checkboxes: change event
                input.addEventListener("change", () => {
                    triggerAutoSave();
                });
            }
        });
    }

    function findOverlayElements() {
        const overlay = document.getElementById(OVERLAY_ID);
        const modal = document.getElementById(MODAL_ID);
        const liveViewPanel = document.getElementById("operationLiveViewPanel");
        return { overlay, modal, liveViewPanel };
    }

    function applyTitle(modal, title) {
        const titleEl = modal.querySelector("[data-role='modal-title']");
        if (titleEl) titleEl.textContent = title || "Operation Settings";
    }

    function updateLiveView(operationName, isSecondary = false) {
        const { liveViewPanel } = findOverlayElements();
        if (!liveViewPanel) return;

        const liveViewContainer = liveViewPanel.querySelector(
            "[data-role='live-view-container']",
        );
        if (!liveViewContainer) return;

        // Update the live view title to include operation name
        const titleEl = liveViewPanel.querySelector("h3");
        if (titleEl) {
            titleEl.textContent = `${operationName} - Live View`;
        }

        // For now, keep the placeholder content
        // Future enhancement: Add actual live visualization logic here
        const placeholderContent =
            liveViewContainer.querySelector(".text-center");
        if (placeholderContent) {
            const operationText = placeholderContent.querySelector("p.text-lg");
            if (operationText) {
                operationText.textContent = `${operationName} Live Preview`;
            }
        }
    }

    function showNoVisualizationMessage(operationName) {
        const { liveViewPanel } = findOverlayElements();
        if (!liveViewPanel) return;

        const liveViewContainer = liveViewPanel.querySelector(
            "[data-role='live-view-container']",
        );
        if (!liveViewContainer) return;

        // Hide the image if it's visible
        const imgEl = liveViewContainer.querySelector("#operationLiveImage");
        if (imgEl) {
            imgEl.classList.add("hidden");
        }

        // Hide default placeholder and text
        const placeholderEl = liveViewContainer.querySelector(
            "[data-role='live-view-placeholder']",
        );
        const textElements = liveViewContainer.querySelectorAll("p");

        if (placeholderEl) placeholderEl.style.display = "none";
        textElements.forEach((p) => (p.style.display = "none"));

        // Show the no visualization message
        const noVisMessage = liveViewContainer.querySelector(
            "#noVisualizationMessage",
        );
        if (noVisMessage) {
            noVisMessage.classList.remove("hidden");
        }
    }

    function init() {
        let { overlay } = findOverlayElements();

        const closeButtons = overlay.querySelectorAll("[data-action='close']");
        closeButtons.forEach((btn) =>
            btn.addEventListener("click", () => close()),
        );
        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) close();
        });
    }

    function open({
        title,
        operationName,
        isSecondary,
        initialValues,
        onSave,
    }) {
        console.log("[SETTINGS] Opening settings popup", {
            operationName,
            isSecondary,
            title,
            initialValuesKeys: Object.keys(initialValues || {}),
            timestamp: new Date().toISOString(),
        });

        const { overlay, modal } = findOverlayElements();
        if (!overlay || !modal) return;

        applyTitle(modal, title);
        updateLiveView(operationName, isSecondary);
        const body = modal.querySelector("[data-role='modal-body']");

        // Show loading state
        body.innerHTML =
            '<div class="text-center text-[#f9c845] py-8">Loading configuration...</div>';

        // Determine camera and pipeline from pipeline builder dropdowns
        let selectedCameraName = null;
        let selectedPipelineName = null;
        try {
            const cameraSelectEl = document.getElementById("cameraSelect");
            const pipelineSelectEl = document.getElementById("pipelineSelect");
            if (cameraSelectEl && cameraSelectEl.selectedIndex >= 0) {
                selectedCameraName =
                    cameraSelectEl.options[cameraSelectEl.selectedIndex]
                        .textContent;
            }
            if (pipelineSelectEl?.value) {
                selectedPipelineName = pipelineSelectEl.value;
            }
        } catch (err) {
            console.warn("Could not read camera/pipeline selection:", err);
        }

        // Compute action name for visualize API (normalize similar to backend expectations)
        const computeActionName = (name) =>
            String(name || "")
                .replace(/\.py$/i, "")
                .toLowerCase()
                .replace(/\s+/g, "_");
        const actionNameForApi = computeActionName(operationName || "");

        // Start visualization on backend if camera and pipeline are available
        const startVisIfReady = async () => {
            if (!selectedCameraName || !selectedPipelineName) {
                console.log(
                    "[SETTINGS] Skipping visualization - missing camera or pipeline",
                    {
                        selectedCameraName,
                        selectedPipelineName,
                    },
                );
                return;
            }
            try {
                console.log("[SETTINGS] Starting visualization", {
                    camera: selectedCameraName,
                    pipeline: selectedPipelineName,
                    action: actionNameForApi,
                    timestamp: new Date().toISOString(),
                });
                await fetch(
                    `${BACKEND_BASE_URL}/start-visualize/${encodeURIComponent(selectedCameraName)}/${encodeURIComponent(selectedPipelineName)}`,
                    { method: "POST" },
                );

                _currentVisCamera = selectedCameraName;
                _currentVisPipeline = selectedPipelineName;
                _currentVisAction = actionNameForApi;

                // Ensure an img element exists and hide placeholder
                const liveViewPanelEl = document.getElementById(
                    "operationLiveViewPanel",
                );
                const liveViewContainer = liveViewPanelEl.querySelector(
                    "[data-role='live-view-container']",
                );
                let imgEl = liveViewContainer.querySelector(
                    "#operationLiveImage",
                );
                if (!imgEl) {
                    imgEl = document.createElement("img");
                    imgEl.id = "operationLiveImage";
                    imgEl.alt = "Live visualization";
                    imgEl.className =
                        "mx-auto mt-4 rounded-lg max-w-full max-h-[60vh]";
                    liveViewContainer.appendChild(imgEl);
                }

                // Hide placeholder content when showing live image
                const placeholderEl = liveViewContainer.querySelector(
                    "[data-role='live-view-placeholder']",
                );
                const textElements = liveViewContainer.querySelectorAll("p");
                const noVisMessage = liveViewContainer.querySelector(
                    "#noVisualizationMessage",
                );

                if (placeholderEl) placeholderEl.style.display = "none";
                textElements.forEach((p) => (p.style.display = "none"));
                if (noVisMessage) noVisMessage.classList.add("hidden");

                imgEl.classList.remove("hidden");

                // Start polling at 10Hz (every 100ms)
                if (_visInterval) clearInterval(_visInterval);
                _visInterval = setInterval(async () => {
                    try {
                        const url = `${BACKEND_BASE_URL}/visualize/${encodeURIComponent(_currentVisCamera)}/${encodeURIComponent(_currentVisPipeline)}/${encodeURIComponent(_currentVisAction)}`;
                        const response = await fetch(url, {
                            cache: "no-store",
                        });

                        if (!response.ok) {
                            // Check if it's a "no visualization" response
                            if (response.status === 500) {
                                const errorText = await response.text();
                                if (
                                    errorText.includes(
                                        "Function has no visualization",
                                    )
                                ) {
                                    showNoVisualizationMessage(operationName);
                                    return;
                                }
                            }
                            return;
                        }

                        const blob = await response.blob();
                        const objectUrl = URL.createObjectURL(blob);
                        // Update image src and revoke previous object URL
                        if (_currentVisObjectUrl)
                            URL.revokeObjectURL(_currentVisObjectUrl);
                        _currentVisObjectUrl = objectUrl;
                        imgEl.src = objectUrl;
                    } catch (err) {
                        console.warn(
                            "Error fetching visualization image:",
                            err,
                        );
                    }
                }, 100);
            } catch (err) {
                console.warn("Failed to start visualization:", err);
            }
        };

        // Fetch config data from server
        fetchConfigData(operationName, isSecondary)
            .then((config) => {
                if (!config) {
                    body.innerHTML =
                        '<div class="text-center text-red-400 py-8">Failed to load configuration</div>';
                    // Try to start visualization anyway (best-effort)
                    startVisIfReady();
                    return;
                }

                // Use the loaded values as baseline for comparison, not defaults
                const originalValues = { ...initialValues };

                const getValues = renderForm(
                    body,
                    config,
                    initialValues,
                    originalValues,
                    onSave,
                );

                const saveBtn = modal.querySelector("[data-action='save']");
                const cancelBtn = modal.querySelector("[data-action='cancel']");

                if (saveBtn) {
                    saveBtn.onclick = () => {
                        const values = getValues();
                        console.log("[SETTINGS] Saving operation settings", {
                            operationName,
                            isSecondary,
                            savedValues: values,
                            timestamp: new Date().toISOString(),
                        });
                        if (typeof onSave === "function") onSave(values);
                        console.log(
                            "[SETTINGS] Settings saved, closing popup",
                            {
                                operationName,
                                timestamp: new Date().toISOString(),
                            },
                        );
                        close();
                    };
                }
                if (cancelBtn) cancelBtn.onclick = () => close();

                // Start visualization now that modal content is ready
                startVisIfReady();
            })
            .catch((error) => {
                console.error("Error loading config:", error);
                body.innerHTML =
                    '<div class="text-center text-red-400 py-8">Error loading configuration</div>';
                // Best-effort start
                startVisIfReady();
            });

        overlay.classList.remove("hidden");
    }

    function stopVisualizationIfActive() {
        if (!_currentVisCamera || !_currentVisPipeline) {
            console.log("[SETTINGS] No active visualization to stop");
            return;
        }
        console.log("[SETTINGS] Stopping active visualization", {
            camera: _currentVisCamera,
            pipeline: _currentVisPipeline,
            action: _currentVisAction,
            timestamp: new Date().toISOString(),
        });
        try {
            fetch(
                `${BACKEND_BASE_URL}/stop-visualize/${encodeURIComponent(_currentVisCamera)}/${encodeURIComponent(_currentVisPipeline)}`,
                { method: "POST" },
            ).catch((err) =>
                console.warn("Failed to stop visualization:", err),
            );
        } finally {
            _currentVisCamera = null;
            _currentVisPipeline = null;
            _currentVisAction = null;
            console.log("[SETTINGS] Visualization state cleared");
        }
    }

    function close() {
        console.log("[SETTINGS] Closing settings popup", {
            wasVisualizing: !!_visInterval,
            hadVisualization: !!_currentVisObjectUrl,
            timestamp: new Date().toISOString(),
        });

        const { overlay } = findOverlayElements();
        if (!overlay) return;

        // Stop polling and visualization
        if (_visInterval) {
            clearInterval(_visInterval);
            _visInterval = null;
            console.log("[SETTINGS] Stopped visualization polling");
        }
        if (_currentVisObjectUrl) {
            try {
                URL.revokeObjectURL(_currentVisObjectUrl);
                console.log("[SETTINGS] Revoked visualization object URL");
            } catch (e) {
                console.warn("Failed to revoke object URL:", e);
            }
            _currentVisObjectUrl = null;
        }
        stopVisualizationIfActive();

        // Show placeholder content again and hide image
        const liveViewPanelEl = document.getElementById(
            "operationLiveViewPanel",
        );
        if (liveViewPanelEl) {
            const liveViewContainer = liveViewPanelEl.querySelector(
                "[data-role='live-view-container']",
            );
            if (liveViewContainer) {
                const imgEl = liveViewContainer.querySelector(
                    "#operationLiveImage",
                );
                const placeholderEl = liveViewContainer.querySelector(
                    "[data-role='live-view-placeholder']",
                );
                const textElements = liveViewContainer.querySelectorAll("p");

                if (imgEl) imgEl.classList.add("hidden");
                if (placeholderEl) placeholderEl.style.display = "";
                textElements.forEach((p) => (p.style.display = ""));

                // Hide no visualization message
                const noVisMessage = liveViewContainer.querySelector(
                    "#noVisualizationMessage",
                );
                if (noVisMessage) {
                    noVisMessage.classList.add("hidden");
                }

                // Reset the text back to default
                const operationText =
                    liveViewContainer.querySelector("p.text-lg");
                const descriptionText =
                    liveViewContainer.querySelector("p.text-sm");

                if (operationText) {
                    operationText.textContent = "Live Preview";
                }
                if (descriptionText) {
                    descriptionText.textContent =
                        "Visualizer will appear here when operation is active";
                }

                // Reset the title
                const titleEl = liveViewPanelEl.querySelector("h3");
                if (titleEl) {
                    titleEl.textContent = "Live View";
                }
            }
        }

        overlay.classList.add("hidden");
    }

    function fetchConfigData(operationName, isSecondary = false) {
        if (!operationName) {
            return Promise.reject(new Error("Operation name is required"));
        }

        // Convert boolean to integer for URL (0 = false, 1 = true)
        const isSecondaryInt = isSecondary ? 1 : 0;

        return fetch(
            `${BACKEND_BASE_URL}/get-operation-config-data/${encodeURIComponent(operationName)}/${isSecondaryInt}`,
        )
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then((data) => {
                // Ensure the data has the expected structure
                if (!data || typeof data !== "object") {
                    throw new Error("Invalid config data structure");
                }
                return data;
            })
            .catch((error) => {
                console.error("Error fetching config data:", error);
                return null;
            });
    }

    window.SettingsPopup = {
        init,
        open,
        close,
    };

    // Auto-init when DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
