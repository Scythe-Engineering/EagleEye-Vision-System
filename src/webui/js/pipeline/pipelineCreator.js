import {
    createDescriptionPopup,
    renderOperations,
    renderPipeline,
} from "./rendering.js";
import {
    handleDragStart,
    handleDragEnd,
    handleDragEnterPipeline,
    handleDragOverPipeline,
    handleDragLeavePipeline,
    handleDropOnPipeline,
} from "./dragDrop.js";
import { BACKEND_BASE_URL } from "../config.js";

// --- Operation definitions (populated from server)
let operations = [];

// Pipeline state
let pipeline = [];
let isInitialized = false;

// Restart tracking state
let restartRequiredOperations = new Map(); // Map<instanceId, Set<paramNames>>

// Camera state
let cameras = [];
let selectedCamera = null;

// Pipeline state
let pipelines = [];
let selectedPipeline = null;

// Auto-save state
let isAutoSaving = false;
let pendingAutoSave = false;

// DOM refs (assigned at init time)
let pipelineArea;
let pipelineContainer;
let pipelinePlaceholder;
let operationsList;
let runButton;
let cameraSelect;
let pipelineSelect;
let newPipelineButton;
let deletePipelineButton;
let restartIndicator;

async function fetchAvailableOperations() {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-available-operations`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Transform server data to match UI expectations
        operations = data.operations.map((op) => ({
            id: op.name, // Use filename as unique ID
            name: op.name
                .replace(".py", "")
                .replace(/_/g, " ")
                .replace(/\b\w/g, (l) => l.toUpperCase()), // Convert filename to readable name
            type: op.category.toUpperCase(), // Use category as type, convert to uppercase for consistency
            description: op.description,
            path: op.path,
            configDataPath: op.config_data_path,
            isSecondary: op.is_secondary, // Store the secondary operation flag
        }));

        console.log("Loaded operations from server:", operations);
    } catch (error) {
        console.error("Failed to fetch operations:", error);
        // Fallback to empty array or could show error message
        operations = [];
    }
}

async function fetchAvailableCameras() {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-available-cameras`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Transform server data to match UI expectations
        cameras = Object.entries(data).map(([name, urlSafeName]) => ({
            name: name,
            urlSafeName: urlSafeName,
        }));

        console.log("Loaded cameras from server:", cameras);
    } catch (error) {
        console.error("Failed to fetch cameras:", error);
        // Fallback to empty array
        cameras = [];
    }
}

function populateCameraDropdown() {
    cameraSelect.innerHTML = "";

    if (!Array.isArray(cameras) || cameras.length === 0) {
        const option = document.createElement("option");
        option.disabled = true;
        option.selected = true;
        option.textContent = "No cameras available";
        cameraSelect.appendChild(option);
        selectedCamera = null;
        return;
    }

    cameras.forEach((camera, index) => {
        const option = document.createElement("option");
        option.value = camera.urlSafeName;
        option.textContent = camera.name;
        if (index === 0) {
            option.selected = true;
            selectedCamera = camera; // Set the first camera as selected
        }
        cameraSelect.appendChild(option);
    });
}

async function fetchPipelinesForCamera(cameraName) {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-pipeline-names-for-camera/${encodeURIComponent(cameraName)}`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const pipelineNames = await response.json();

        // Transform server data to match UI expectations
        pipelines = pipelineNames.map((name) => ({
            name: name,
            displayName: name
                .replace(/_/g, " ")
                .replace(/\b\w/g, (l) => l.toUpperCase()), // Convert to readable name
        }));

        console.log("Loaded pipelines from server:", pipelines);
    } catch (error) {
        console.error("Failed to fetch pipelines:", error);
        // Fallback to empty array
        pipelines = [];
    }
}

async function fetchPipelineConfig(cameraName, pipelineName) {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-pipeline-config/${encodeURIComponent(cameraName)}/${encodeURIComponent(pipelineName)}`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const pipelineConfig = await response.json();

        console.log("Loaded pipeline config from server:", pipelineConfig);
        return pipelineConfig;
    } catch (error) {
        console.error("Failed to fetch pipeline config:", error);
        return [];
    }
}

function populatePipelineDropdown(selectedPipelineName = null) {
    pipelineSelect.innerHTML = "";

    // Add default option
    const defaultOption = document.createElement("option");
    defaultOption.disabled = true;
    defaultOption.textContent = "Select Pipeline";
    pipelineSelect.appendChild(defaultOption);

    let foundSelectedPipeline = false;

    // Add pipeline options
    pipelines.forEach((pipeline, index) => {
        const option = document.createElement("option");
        option.value = pipeline.name;
        option.textContent = pipeline.displayName;

        // Select the specified pipeline or the first one if none specified
        if (
            selectedPipelineName === pipeline.name ||
            (selectedPipelineName === null && index === 0)
        ) {
            option.selected = true;
            selectedPipeline = pipeline;
            foundSelectedPipeline = true;
        }

        pipelineSelect.appendChild(option);
    });

    // If we were looking for a specific pipeline but didn't find it, select the first one
    if (
        selectedPipelineName &&
        !foundSelectedPipeline &&
        pipelines.length > 0
    ) {
        console.warn(
            `Pipeline "${selectedPipelineName}" not found in pipelines list, selecting first pipeline`,
        );
        const firstOption = pipelineSelect.querySelector(
            "option:not([disabled])",
        );
        if (firstOption) {
            firstOption.selected = true;
            selectedPipeline = pipelines[0];
        }
    }

    // If no pipelines exist, ensure selectedPipeline is null
    if (pipelines.length === 0) {
        selectedPipeline = null;
    }
}

async function handleCameraSelection() {
    const selectedValue = cameraSelect.value;
    selectedCamera = cameras.find(
        (camera) => camera.urlSafeName === selectedValue,
    );
    console.log("Selected camera:", selectedCamera);

    // Fetch pipelines for the selected camera
    if (selectedCamera) {
        await fetchPipelinesForCamera(selectedCamera.name);
        populatePipelineDropdown(); // No specific selection, will default to first pipeline

        // After populating pipelines, check if we should auto-fill
        await checkAndTriggerAutoFill();

        // Update delete button visibility
        updateDeleteButtonVisibility();
    }
}

async function handlePipelineSelection() {
    const selectedValue = pipelineSelect.value;
    selectedPipeline = pipelines.find(
        (pipeline) => pipeline.name === selectedValue,
    );
    console.log("Selected pipeline:", selectedPipeline);

    // Auto-fill pipeline if a pipeline is selected and we have a selected camera
    if (selectedPipeline && selectedCamera) {
        await loadPipelineIntoBuilder(
            selectedCamera.name,
            selectedPipeline.name,
        );
    }

    // Update delete button visibility
    updateDeleteButtonVisibility();

    // Don't check restart requirements when selecting existing pipeline
    // Restart requirements should only be checked when user makes changes
}

async function loadPipelineIntoBuilder(cameraName, pipelineName) {
    try {
        // Check if operations are loaded
        if (operations.length === 0) {
            console.warn("Operations not loaded yet, cannot load pipeline");
            return;
        }

        // Clear current pipeline
        pipeline = [];

        // Fetch pipeline configuration
        const pipelineConfig = await fetchPipelineConfig(
            cameraName,
            pipelineName,
        );

        // Transform config into pipeline operations
        pipelineConfig.forEach((configItem, index) => {
            // Find the corresponding operation from available operations
            // Try different matching strategies
            let operation = operations.find(
                (op) => op.id === configItem.action_name + ".py",
            );

            // If not found, try without .py extension
            if (!operation) {
                operation = operations.find(
                    (op) => op.id === configItem.action_name,
                );
            }

            // If still not found, try matching by name
            if (!operation) {
                operation = operations.find(
                    (op) =>
                        op.name.toLowerCase().replace(/\s+/g, "_") ===
                        configItem.action_name,
                );
            }

            if (operation) {
                // Create pipeline item with operation data and config
                const pipelineItem = {
                    ...operation,
                    instanceId: `${operation.id}_${Date.now()}_${index}`,
                    config: configItem.action_params || {},
                    originalConfig: { ...(configItem.action_params || {}) }, // Store original config for restart comparison
                    name: operation.name,
                    requiresRestart: false, // Initialize restart flag (will be updated if needed)
                };
                pipeline.push(pipelineItem);
            } else {
                console.warn(
                    `Operation ${configItem.action_name} not found in available operations. Available operations:`,
                    operations.map((op) => op.id),
                );
            }
        });

        // Re-render the pipeline
        renderPipeline(
            pipeline,
            pipelineContainer,
            pipelinePlaceholder,
            openOperationSettings,
            updateRunButton,
            removeFromPipeline,
            handleDragStart,
            handleDragEnd,
        );

        // Update the run button state
        updateRunButton();

        // Don't check restart requirements when loading existing pipeline
        // Restart requirements should only be checked when user makes changes
        console.log("Pipeline loaded:", pipeline);
    } catch (error) {
        console.error("Failed to load pipeline:", error);
    }
}

async function checkAndTriggerAutoFill() {
    try {
        // Check if camera dropdown has a selected value and update selectedCamera if needed
        if (cameraSelect?.value) {
            const selectedCameraValue = cameraSelect.value;
            const cameraObj = cameras.find(
                (c) => c.urlSafeName === selectedCameraValue,
            );
            if (cameraObj) {
                selectedCamera = cameraObj;
            }
        }

        // Check if we have a selected camera
        if (!selectedCamera) {
            console.log("No camera selected, skipping auto-fill");
            return;
        }

        // Check if pipeline dropdown has a selected value
        if (!pipelineSelect?.value) {
            console.log("No pipeline selected, skipping auto-fill");
            return;
        }

        // Get the selected pipeline name from the dropdown
        const selectedPipelineName = pipelineSelect.value;

        // Find the pipeline object
        const pipelineObj = pipelines.find(
            (p) => p.name === selectedPipelineName,
        );

        if (!pipelineObj) {
            console.log("Selected pipeline not found in pipelines list");
            return;
        }

        // Update the selectedPipeline variable to match the dropdown
        selectedPipeline = pipelineObj;

        console.log(
            "Both camera and pipeline are pre-selected, triggering auto-fill",
        );
        await loadPipelineIntoBuilder(selectedCamera.name, pipelineObj.name);
    } catch (error) {
        console.error("Error during auto-fill check:", error);
    }
}

// --- Pipeline actions

async function removeFromPipeline(instanceId) {
    pipeline = pipeline.filter((item) => item.instanceId !== instanceId);
    renderPipeline(
        pipeline,
        pipelineContainer,
        pipelinePlaceholder,
        {
            openOperationSettings,
            updateRunButton,
            removeFromPipeline,
            handleDragStart,
            handleDragEnd,
        },
    );

    // Auto-save when removing items
    autoSavePipeline();

    // Pipeline structure changed (operation removed) - always require restart
    console.log("Operation removed from pipeline - requiring backend restart");
    await updateRestartIndicator(true);
    // Clear any existing parameter-level restart tracking since structure change overrides it
    restartRequiredOperations.clear();
}

function runPipeline() {
    console.log("Running pipeline:", pipeline);
    alert("Pipeline run! Check console for details.");
}

function openOperationSettings(opOrItem) {
    const title = `${opOrItem.name || opOrItem.id || "Operation"} Settings`;
    const operationName = opOrItem.name || opOrItem.id;
    const isSecondary = opOrItem.isSecondary || false; // Get the secondary flag
    const initialValues = opOrItem.config || {}; // Use actual config values from pipeline

    // Store original config for comparison
    if (!opOrItem.originalConfig) {
        opOrItem.originalConfig = { ...initialValues };
    }

    const onSave = (values) => {
        console.log("Saved settings for", opOrItem, values);
        // Update the config in the pipeline item
        const isAutoSave = values._isAutoSave;
        const requiresRestart = values._requiresRestart;
        console.log("isAutoSave flag:", isAutoSave);
        console.log("requiresRestart flag:", requiresRestart);

        delete values._isAutoSave; // Remove the flag before storing
        delete values._requiresRestart; // Remove the flag before storing

        // Store the previous config for comparison
        const previousConfig = { ...opOrItem.config };

        opOrItem.config = values;
        opOrItem.requiresRestart = requiresRestart || false;
        console.log("Updated opOrItem.config:", opOrItem.config);
        console.log(
            "Updated opOrItem.requiresRestart:",
            opOrItem.requiresRestart,
        );

        // Auto-save when settings are modified (always save, regardless of trigger)
        console.log("Calling autoSavePipeline...");
        autoSavePipeline();

        // Check for restart requirements only for changed parameters
        const changedParams = [];
        for (const [key, value] of Object.entries(values)) {
            if (JSON.stringify(previousConfig[key]) !== JSON.stringify(value)) {
                changedParams.push({ paramName: key, value: value });
            }
        }

        // Check restart requirements for each changed parameter
        if (changedParams.length > 0) {
            changedParams.forEach(({ paramName, value }) => {
                checkPipelineRestartRequirements(opOrItem, paramName, value);
            });
        } else {
            // If no parameters changed but function was called, still update indicator
            checkPipelineRestartRequirements();
        }
    };

    const doOpen = () => {
        try {
            window.SettingsPopup.open({
                title,
                operationName,
                isSecondary,
                initialValues,
                onSave,
            });
        } catch (err) {
            console.error("Failed to open SettingsPopup:", err);
        }
    };

    if (window.SettingsPopup) {
        doOpen();
        return;
    }

    const scriptUrl = "../../js/pipeline/settingsPopup.js";
    const already = document.querySelector(`script[src="${scriptUrl}"]`);
    if (already) {
        already.addEventListener("load", doOpen);
        return;
    }

    const s = document.createElement("script");
    s.src = scriptUrl;
    s.onload = () => {
        if (!window.SettingsPopup) {
            console.warn("SettingsPopup loaded but did not register on window");
            return;
        }
        doOpen();
    };
    s.onerror = () =>
        console.error("Failed to load settings popup script at", scriptUrl);
    document.head.appendChild(s);
}

function updateRunButton() {
    if (runButton) {
        runButton.disabled = pipeline.length === 0;
    }
}

async function autoSavePipeline() {
    if (!selectedCamera || !selectedPipeline) {
        console.log("No camera or pipeline selected, skipping auto-save");
        return;
    }

    if (isAutoSaving) {
        pendingAutoSave = true;
        return;
    }
    isAutoSaving = true;
    try {
        const pipelineConfig = pipeline.map((item) => {
            const configParams = {};
            if (item.config) {
                Object.keys(item.config).forEach((key) => {
                    const value = item.config[key];
                    if (value !== undefined && value !== null) {
                        configParams[key] = value;
                    }
                });
            }
            return {
                action_name: item.id.replace(".py", ""),
                action_params: configParams,
            };
        });

        const response = await fetch(
            `${BACKEND_BASE_URL}/save-pipeline-config/${encodeURIComponent(selectedCamera.name)}/${encodeURIComponent(selectedPipeline.name)}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(pipelineConfig),
            }
        );
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        await response.json();
    } catch (error) {
        console.error("Failed to auto-save pipeline:", error);
    } finally {
        isAutoSaving = false;
        if (pendingAutoSave) {
            pendingAutoSave = false;
            // Trigger another save with the latest state
            autoSavePipeline();
        }
    }
}

async function createNewPipeline() {
    if (!selectedCamera) {
        alert("Please select a camera first.");
        return;
    }

    // Prompt for new pipeline name
    const newPipelineName = prompt("Enter a name for the new pipeline:");
    if (!newPipelineName || newPipelineName.trim() === "") {
        return; // User cancelled or entered empty name
    }

    const pipelineFileName = newPipelineName.trim().replace(/\s+/g, "_");

    // Check if pipeline already exists
    const existingPipeline = pipelines.find((p) => p.name === pipelineFileName);
    if (existingPipeline) {
        if (
            !confirm(
                `Pipeline "${newPipelineName}" already exists. Do you want to overwrite it?`,
            )
        ) {
            return;
        }
    }

    try {
        // Clear current pipeline completely
        pipeline.length = 0; // More efficient way to clear array
        pipeline = []; // Ensure it's a fresh empty array

        // Create new pipeline object
        const newPipelineObj = {
            name: pipelineFileName,
            displayName: newPipelineName.trim(),
        };

        // Update selected pipeline
        selectedPipeline = newPipelineObj;

        // Add to pipelines list if not already there
        const existingIndex = pipelines.findIndex(
            (p) => p.name === pipelineFileName,
        );
        if (existingIndex >= 0) {
            pipelines[existingIndex] = newPipelineObj;
        } else {
            pipelines.push(newPipelineObj);
        }

        // Update pipeline dropdown and select the new pipeline
        populatePipelineDropdown(pipelineFileName);

        // Ensure the dropdown shows the correct selection
        setTimeout(() => {
            if (pipelineSelect && selectedPipeline) {
                pipelineSelect.value = selectedPipeline.name;
                console.log("Dropdown value set to:", selectedPipeline.name);
            }
        }, 10);

        // Re-render the empty pipeline
        renderPipeline(
            pipeline,
            pipelineContainer,
            pipelinePlaceholder,
            openOperationSettings,
            updateRunButton,
            removeFromPipeline,
            handleDragStart,
            handleDragEnd,
        );

        // Update the run button state
        updateRunButton();

        // Update delete button visibility
        updateDeleteButtonVisibility();

        // Clear restart requirements for new pipeline
        restartRequiredOperations.clear();
        await updateRestartIndicator(false);

        console.log("New pipeline created:", newPipelineName);
        console.log("Pipeline state:", pipeline);
        console.log("Selected pipeline:", selectedPipeline);
        console.log("Pipelines list:", pipelines);
    } catch (error) {
        console.error("Failed to create new pipeline:", error);
        alert(
            "Failed to create new pipeline. Please check the console for details.",
        );
    }
}

async function deleteCurrentPipeline() {
    if (!selectedPipeline) {
        alert("No pipeline selected to delete.");
        return;
    }

    if (!selectedCamera) {
        alert("No camera selected.");
        return;
    }

    // Show confirmation dialog
    const confirmed = confirm(
        `Are you sure you want to delete the pipeline "${selectedPipeline.displayName}"?\n\nThis action cannot be undone.`,
    );

    if (!confirmed) {
        return;
    }

    try {
        // Call the backend delete endpoint
        const response = await fetch(
            `${BACKEND_BASE_URL}/delete-pipeline/${encodeURIComponent(selectedCamera.name)}/${encodeURIComponent(selectedPipeline.name)}`,
            {
                method: "DELETE",
                headers: {
                    "Content-Type": "application/json",
                },
            },
        );

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Pipeline deleted from backend:", result);

        // Remove the pipeline from the pipelines array
        const pipelineIndex = pipelines.findIndex(
            (p) => p.name === selectedPipeline.name,
        );

        if (pipelineIndex === -1) {
            console.error("Pipeline not found in pipelines array");
            alert("Failed to delete pipeline. Pipeline not found.");
            return;
        }

        // Remove from array
        pipelines.splice(pipelineIndex, 1);

        console.log("Deleted pipeline:", selectedPipeline.name);
        console.log("Remaining pipelines:", pipelines);

        // Clear current pipeline
        pipeline.length = 0;
        pipeline = [];

        // Update selected pipeline
        selectedPipeline = null;

        // Update pipeline dropdown
        populatePipelineDropdown();

        // Re-render the empty pipeline
        renderPipeline(
            pipeline,
            pipelineContainer,
            pipelinePlaceholder,
            openOperationSettings,
            updateRunButton,
            removeFromPipeline,
            handleDragStart,
            handleDragEnd,
        );

        // Update the run button state
        updateRunButton();

        // Update delete button visibility
        updateDeleteButtonVisibility();

        // Clear restart requirements for deleted pipeline
        restartRequiredOperations.clear();
        await updateRestartIndicator(false);

        console.log("Pipeline deleted successfully");
    } catch (error) {
        console.error("Failed to delete pipeline:", error);
        alert(
            "Failed to delete pipeline. Please check the console for details.",
        );
    }
}

function updateDeleteButtonVisibility() {
    if (deletePipelineButton) {
        if (selectedPipeline && selectedCamera) {
            deletePipelineButton.classList.remove("hidden");
        } else {
            deletePipelineButton.classList.add("hidden");
        }
    }
}

// --- Restart Indicator Functions

async function updateRestartIndicator(show = false) {
    try {
        await fetch(`${BACKEND_BASE_URL}/set_restart_required`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ required: show }),
        });
        console.log(`Backend notified: restart ${show ? "required" : "not required"}`);
    } catch (error) {
        console.error("Failed to notify backend about restart requirement:", error);
    }

    if (!restartIndicator) return;

    const restartMessage =
        restartIndicator.querySelector(".text-red-100") ||
        restartIndicator.querySelector("span");

    if (show) {
        restartIndicator.classList.remove("hidden");
        if (restartMessage) restartMessage.textContent = "Backend restart required";
        restartIndicator.classList.add("backend-state-warning");
    } else {
        restartIndicator.classList.add("hidden");
        // Clear warning styling so future checks can show it correctly
        restartIndicator.classList.remove("backend-state-warning");
    }
}

async function handleRestartBackend() {
    try {
        // Get the restart button from inside the restartIndicator
        const restartButton = restartIndicator?.querySelector(
            "#restartBackendButton",
        );

        // Disable button during restart
        if (restartButton) {
            restartButton.disabled = true;
            restartButton.textContent = "Restarting...";
        }

        try {
            await fetch(`${BACKEND_BASE_URL}/restart-backend`, {
                method: "POST",
            });
        } catch (error) {
            console.warn("Failed to send restart request:", error);
        }

        console.log("Backend restarted successfully");

        // Clear restart tracking state before reload
        restartRequiredOperations.clear();

        // Reload page to refresh all state
        window.location.reload();
    } catch (error) {
        console.error("Failed to restart backend:", error);
    }
}

async function checkBackendRestartStatus() {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get_restart_required`,
        );

        if (response.ok) {
            const data = await response.json();
            const restartRequired = data.restart_required || false;

            if (restartRequired) {
                console.log(
                    "Backend indicates restart is required - showing indicator",
                );
                await updateRestartIndicator(true);
            } else {
                console.log("Backend indicates no restart required");
            }
        } else {
            console.warn(
                "Failed to get restart status from backend:",
                response.status,
            );
        }
    } catch (error) {
        console.error("Error checking backend restart status:", error);
    }
}

async function checkPipelineRestartRequirements(
    operationItem = null,
    changedParamName = null,
    changedValue = null,
) {
    // Don't update restart indicator if backend restart is already shown
    const restartIndicator = document.getElementById("restartIndicator");
    if (
        restartIndicator &&
        !restartIndicator.classList.contains("hidden") &&
        restartIndicator.classList.contains("backend-state-warning")
    ) {
        return; // Backend restart indicator is already shown, don't override
    }

    // If checking a specific parameter change
    if (operationItem && changedParamName !== null && changedValue !== null) {
        await checkSpecificParameterRestart(
            operationItem,
            changedParamName,
            changedValue,
        );
    } else if (operationItem) {
        // If just checking an operation (e.g., when operation is added/removed)
        await checkOperationRestartRequirements(operationItem);
    }

    // Update restart indicator based on stored restart requirements
    const hasRestartRequirements = restartRequiredOperations.size > 0;
    await updateRestartIndicator(hasRestartRequirements);
}

async function checkSpecificParameterRestart(
    operationItem,
    paramName,
    currentValue,
) {
    try {
        // Fetch the operation's config definition
        const isSecondary = operationItem.isSecondary || false;
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-operation-config-data/${encodeURIComponent(operationItem.id)}/${isSecondary ? 1 : 0}`,
        );

        if (response.ok) {
            const configData = await response.json();
            const params = configData.parameters || {};
            const paramDef = params[paramName];

            if (paramDef?.restart_for_change) {
                // Get the original value from when the pipeline was loaded
                const originalValue = operationItem.originalConfig[paramName];

                // Check if current value differs from original value (not default)
                const requiresRestart =
                    currentValue !== undefined &&
                    currentValue !== null &&
                    originalValue !== undefined &&
                    originalValue !== null &&
                    JSON.stringify(currentValue) !==
                        JSON.stringify(originalValue);

                // Update restart tracking
                const instanceId = operationItem.instanceId;
                if (!restartRequiredOperations.has(instanceId)) {
                    restartRequiredOperations.set(instanceId, new Set());
                }

                const paramSet = restartRequiredOperations.get(instanceId);
                if (requiresRestart) {
                    paramSet.add(paramName);
                    console.log(
                        `Operation ${operationItem.name} parameter ${paramName} requires restart (current: ${JSON.stringify(currentValue)}, original: ${JSON.stringify(originalValue)})`,
                    );
                } else {
                    paramSet.delete(paramName);
                    // If no parameters require restart for this operation, remove it from the map
                    if (paramSet.size === 0) {
                        restartRequiredOperations.delete(instanceId);
                    }
                }

                // Update the operation's restart flag for backward compatibility
                operationItem.requiresRestart = paramSet.size > 0;
            }
        }
    } catch (error) {
        console.warn(
            `Failed to check restart requirements for ${operationItem.name} parameter ${paramName}:`,
            error,
        );
    }
}

async function checkOperationRestartRequirements(operationItem) {
    // Check all parameters of the operation for restart requirements
    for (const [paramName, value] of Object.entries(
        operationItem.config || {},
    )) {
        await checkSpecificParameterRestart(operationItem, paramName, value);
    }
}

export async function initPipelineCreator() {
    if (isInitialized) return;

    pipelineArea = document.getElementById("pipelineArea");
    pipelineContainer = document.getElementById("pipelineContainer");
    pipelinePlaceholder = document.getElementById("pipelinePlaceholder");
    operationsList = document.getElementById("operationsList");
    runButton = document.getElementById("runButton");
    cameraSelect = document.getElementById("cameraSelect");
    pipelineSelect = document.getElementById("pipelineSelect");
    newPipelineButton = document.getElementById("newPipelineButton");
    deletePipelineButton = document.getElementById("deletePipelineButton");
    restartIndicator = document.getElementById("restartIndicator");

    // Initialize description popup
    createDescriptionPopup();

    // Inject component-specific styles once: hide action buttons outside the
    // pipeline builder and apply grayscale by default, removing it on hover.
    const styleElementId = "pipeline-creator-styles";
    if (!document.getElementById(styleElementId)) {
        const styleEl = document.createElement("style");
        styleEl.id = styleElementId;
        styleEl.textContent = `
.op-settings-btn, .remove-btn { display: none !important; }
#pipelineArea .op-settings-btn, #pipelineArea .remove-btn { display: inline-flex !important; }
.icon-grayscale { filter: grayscale(100%); transition: filter .15s ease-in-out; }
#pipelineArea .group:hover .icon-grayscale, #pipelineArea .group:focus-within .icon-grayscale { filter: none; }
`;
        document.head.appendChild(styleEl);
    }

    // Fetch operations from server before rendering
    await fetchAvailableOperations();

    // Fetch cameras from server and populate dropdown
    await fetchAvailableCameras();
    populateCameraDropdown();

    // Add camera selection event listener
    if (cameraSelect) {
        cameraSelect.addEventListener("change", handleCameraSelection);
    }

    // Add pipeline selection event listener
    if (pipelineSelect) {
        pipelineSelect.addEventListener("change", handlePipelineSelection);
    }

    // Fetch pipelines for initially selected camera
    if (selectedCamera) {
        await fetchPipelinesForCamera(selectedCamera.name);
        populatePipelineDropdown();
    }

    // Check for pre-selected values and trigger auto-fill if both are selected
    await checkAndTriggerAutoFill();

    // Update delete button visibility
    updateDeleteButtonVisibility();

    // Setup drag and drop handlers
    const setupDragDrop = (element) => {
        element.addEventListener("dragenter", (e) =>
            handleDragEnterPipeline(e),
        );
        element.addEventListener("dragover", (e) =>
            handleDragOverPipeline(e, pipeline, pipelineContainer),
        );
        element.addEventListener("dragleave", (e) =>
            handleDragLeavePipeline(e, pipeline, pipelinePlaceholder),
        );
        element.addEventListener("drop", async (e) => {
            const pipelineLengthBefore = pipeline.length;
            const pipelineOrderBefore = pipeline
                .map((item) => item.instanceId)
                .join(",");

            await handleDropOnPipeline(
                e,
                pipeline,
                operations,
                pipelineContainer,
                pipelinePlaceholder,
                {
                    renderPipeline: () =>
                        renderPipeline(
                            pipeline,
                            pipelineContainer,
                            pipelinePlaceholder,
                            openOperationSettings,
                            updateRunButton,
                            removeFromPipeline,
                            handleDragStart,
                            handleDragEnd,
                        ),
                    updateRunButton,
                    openOperationSettings,
                },
            );

            // Check if pipeline structure changed (added, removed, or reordered)
            const pipelineOrderAfter = pipeline
                .map((item) => item.instanceId)
                .join(",");
            const structureChanged =
                pipeline.length !== pipelineLengthBefore ||
                pipelineOrderBefore !== pipelineOrderAfter;

            if (structureChanged) {
                // Pipeline structure changed - always require restart
                console.log(
                    "Pipeline structure changed - requiring backend restart",
                );
                await updateRestartIndicator(true);
                // Clear any existing parameter-level restart tracking since structure change overrides it
                restartRequiredOperations.clear();

                // Auto-save when pipeline structure changes (add, remove, reorder)
                autoSavePipeline();
            } else if (pipeline.length > pipelineLengthBefore) {
                // If operations were added, check restart requirements for all operations
                pipeline.forEach((item) => {
                    checkPipelineRestartRequirements(item);
                });

                // Auto-save when operations are added
                autoSavePipeline();
            }
        });
    };

    // Add drag/drop event listeners
    setupDragDrop(pipelineArea);
    setupDragDrop(pipelineContainer);
    setupDragDrop(pipelinePlaceholder);

    if (runButton) {
        runButton.addEventListener("click", runPipeline);
    }

    if (newPipelineButton) {
        newPipelineButton.addEventListener("click", createNewPipeline);
    }

    if (deletePipelineButton) {
        deletePipelineButton.addEventListener("click", deleteCurrentPipeline);
    }

    // Set up restart button event listener (button is inside restartIndicator)
    if (restartIndicator) {
        const restartButton = restartIndicator.querySelector(
            "#restartBackendButton",
        );
        if (restartButton) {
            restartButton.addEventListener("click", handleRestartBackend);
        }
    }

    // Initial render
    renderOperations(
        operations,
        operationsList,
        openOperationSettings,
        handleDragStart,
    );
    renderPipeline(
        pipeline,
        pipelineContainer,
        pipelinePlaceholder,
        {
            openOperationSettings,
            updateRunButton,
            removeFromPipeline,
            handleDragStart,
            handleDragEnd,
        },
    );

    // Check backend restart status on initialization
    await checkBackendRestartStatus();

    isInitialized = true;

    // Check if backend restart indicator should be shown (legacy support)
    if (window.showBackendRestartIndicator) {
        window.showBackendRestartIndicator();
    }

    // Expose functions for external modules (like dragDrop.js)
    window.pipelineCreator = {
        autoSavePipeline: autoSavePipeline,
        updateRestartIndicator: updateRestartIndicator,
        checkPipelineRestartRequirements: checkPipelineRestartRequirements,
        checkBackendRestartStatus: checkBackendRestartStatus,
        restartIndicator: restartIndicator,
    };
}
