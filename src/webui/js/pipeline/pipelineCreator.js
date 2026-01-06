import {
    createDescriptionPopup,
    renderOperations,
    renderPipeline,
    FlowchartRenderer,
} from "./rendering.js";
import {
    handleDragStart,
    handleDragEnd,
    handleDragEnterPipeline,
    handleDragOverPipeline,
    handleDragLeavePipeline,
    handleDropOnPipeline,
} from "./dragDrop.js";
import { uid } from "./utils.js";
import { BACKEND_BASE_URL } from "../config.js";

function handleDragStartWithLogging(
    event,
    item,
    fromIndex = null,
    collection = null,
) {
    console.log("[PIPELINE] Drag start initiated", {
        draggedElement: event.target,
        itemInstanceId: item?.instanceId || null,
        fromIndex: fromIndex,
        timestamp: new Date().toISOString(),
    });
    return handleDragStart(event, item, collection, fromIndex);
}

function handleDragEndWithLogging(
    event,
    pipelineContainer,
    pipelinePlaceholder,
    pipeline,
) {
    console.log("[PIPELINE] Drag end", {
        draggedElement: event.target,
        timestamp: new Date().toISOString(),
    });
    return handleDragEnd(
        event,
        pipelineContainer,
        pipelinePlaceholder,
        pipeline,
    );
}

function handleDropOnPipelineWithLogging(
    event,
    pipeline,
    operations,
    pipelineContainer,
    pipelinePlaceholder,
    callbacks,
) {
    const pipelineOrderBefore = pipeline.map((item) => ({
        id: item.id,
        name: item.name,
        instanceId: item.instanceId,
    }));

    console.log("[PIPELINE] Drop operation started", {
        pipelineLengthBefore: pipeline.length,
        pipelineOrderBefore,
        dropTarget: event.target,
        timestamp: new Date().toISOString(),
    });

    const result = handleDropOnPipeline(
        event,
        pipeline,
        operations,
        pipelineContainer,
        pipelinePlaceholder,
        callbacks,
    );

    const pipelineOrderAfter = pipeline.map((item) => ({
        id: item.id,
        name: item.name,
        instanceId: item.instanceId,
    }));

    console.log("[PIPELINE] Drop operation completed", {
        pipelineLengthAfter: pipeline.length,
        pipelineOrderAfter,
        orderChanged:
            JSON.stringify(pipelineOrderBefore.map((p) => p.instanceId)) !==
            JSON.stringify(pipelineOrderAfter.map((p) => p.instanceId)),
        lengthChanged: pipelineOrderBefore.length !== pipelineOrderAfter.length,
        timestamp: new Date().toISOString(),
    });

    return result;
}

let operations = [];
let pipeline = [];
let isInitialized = false;
let restartRequiredOperations = new Map();
let cameras = [];
let selectedCamera = null;
let pipelines = [];
let selectedPipeline = null;
let isAutoSaving = false;
let pendingAutoSave = false;

let flowchartRenderer = null;
let useFlowchartMode = true;

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
let flowchartCanvas;

async function fetchAvailableOperations() {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-available-operations`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        operations = data.operations.map((op) => ({
            id: op.name,
            name: op.name
                .replaceAll(".py", "")
                .replaceAll("_", " ")
                .replaceAll(/\b\w/g, (l) => l.toUpperCase()),
            type: op.category.toUpperCase(),
            description: op.description,
            path: op.path,
            configDataPath: op.config_data_path,
            isSecondary: op.is_secondary,
        }));

        console.log("Loaded operations from server:", operations);
    } catch (error) {
        console.error("Failed to fetch operations:", error);
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

        cameras = Object.entries(data).map(([name, urlSafeName]) => ({
            name: name,
            urlSafeName: urlSafeName,
        }));

        console.log("Loaded cameras from server:", cameras);
    } catch (error) {
        console.error("Failed to fetch cameras:", error);
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

    for (let index = 0; index < cameras.length; index++) {
        const camera = cameras[index];
        const option = document.createElement("option");
        option.value = camera.urlSafeName;
        option.textContent = camera.name;
        if (index === 0) {
            option.selected = true;
            selectedCamera = camera;
        }
        cameraSelect.appendChild(option);
    }
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

        pipelines = pipelineNames.map((name) => ({
            name: name,
            displayName: name
                .replaceAll("_", " ")
                .replaceAll(/\b\w/g, (l) => l.toUpperCase()),
        }));

        console.log("Loaded pipelines from server:", pipelines);
    } catch (error) {
        console.error("Failed to fetch pipelines:", error);
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

    const defaultOption = document.createElement("option");
    defaultOption.disabled = true;
    defaultOption.textContent = "Select Pipeline";
    pipelineSelect.appendChild(defaultOption);

    let foundSelectedPipeline = false;

    for (let index = 0; index < pipelines.length; index++) {
        const pipelineItem = pipelines[index];
        const option = document.createElement("option");
        option.value = pipelineItem.name;
        option.textContent = pipelineItem.displayName;

        if (
            selectedPipelineName === pipelineItem.name ||
            (selectedPipelineName === null && index === 0)
        ) {
            option.selected = true;
            selectedPipeline = pipelineItem;
            foundSelectedPipeline = true;
        }

        pipelineSelect.appendChild(option);
    }

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

    if (selectedCamera) {
        await fetchPipelinesForCamera(selectedCamera.name);
        populatePipelineDropdown();
        await checkAndTriggerAutoFill();
        updateDeleteButtonVisibility();
    }
}

async function handlePipelineSelection() {
    const selectedValue = pipelineSelect.value;
    selectedPipeline = pipelines.find(
        (pipelineItem) => pipelineItem.name === selectedValue,
    );
    console.log("Selected pipeline:", selectedPipeline);

    if (selectedPipeline && selectedCamera) {
        await loadPipelineIntoBuilder(
            selectedCamera.name,
            selectedPipeline.name,
        );
    }

    updateDeleteButtonVisibility();
}

async function loadPipelineIntoBuilder(cameraName, pipelineName) {
    try {
        if (operations.length === 0) {
            console.warn("Operations not loaded yet, cannot load pipeline");
            return;
        }

        pipeline = [];

        const pipelineConfig = await fetchPipelineConfig(
            cameraName,
            pipelineName,
        );

        for (let index = 0; index < pipelineConfig.length; index++) {
            const configItem = pipelineConfig[index];
            let operation = operations.find(
                (op) => op.id === configItem.action_name + ".py",
            );

            if (!operation) {
                operation = operations.find(
                    (op) => op.id === configItem.action_name,
                );
            }

            if (!operation) {
                operation = operations.find(
                    (op) =>
                        op.name.toLowerCase().replaceAll(/\s+/g, "_") ===
                        configItem.action_name,
                );
            }

            if (operation) {
                const pipelineItem = {
                    ...operation,
                    instanceId: `${operation.id}_${Date.now()}_${index}`,
                    config: configItem.action_params || {},
                    originalConfig: configItem.action_params || {},
                    name: operation.name,
                    requiresRestart: false,
                    position: configItem.position || null,
                };
                pipeline.push(pipelineItem);
            } else {
                console.warn(
                    `Operation ${configItem.action_name} not found in available operations. Available operations:`,
                    operations.map((op) => op.id),
                );
            }
        }

        await renderCurrentPipeline();
        updateRunButton();
    } catch (error) {
        console.error("Failed to load pipeline:", error);
    }
}

async function renderCurrentPipeline() {
    if (useFlowchartMode && flowchartRenderer) {
        await flowchartRenderer.renderPipeline(pipeline);
    } else {
        renderPipeline(pipeline, pipelineContainer, pipelinePlaceholder, {
            openOperationSettings,
            updateRunButton,
            removeFromPipeline,
            handleDragStart: handleDragStartWithLogging,
            handleDragEnd: handleDragEndWithLogging,
        });
    }
}

async function checkAndTriggerAutoFill() {
    try {
        if (cameraSelect?.value) {
            const selectedCameraValue = cameraSelect.value;
            const cameraObj = cameras.find(
                (c) => c.urlSafeName === selectedCameraValue,
            );
            if (cameraObj) {
                selectedCamera = cameraObj;
            }
        }

        if (!selectedCamera) {
            console.log("No camera selected, skipping auto-fill");
            return;
        }

        if (!pipelineSelect?.value) {
            console.log("No pipeline selected, skipping auto-fill");
            return;
        }

        const selectedPipelineName = pipelineSelect.value;

        const pipelineObj = pipelines.find(
            (p) => p.name === selectedPipelineName,
        );

        if (!pipelineObj) {
            console.log("Selected pipeline not found in pipelines list");
            return;
        }

        selectedPipeline = pipelineObj;

        console.log(
            "Both camera and pipeline are pre-selected, triggering auto-fill",
        );
        await loadPipelineIntoBuilder(selectedCamera.name, pipelineObj.name);
    } catch (error) {
        console.error("Error during auto-fill check:", error);
    }
}

async function removeFromPipeline(instanceId) {
    const removedOperation = pipeline.find(
        (item) => item.instanceId === instanceId,
    );
    console.log("[PIPELINE] Removing operation from pipeline", {
        removedOperation: removedOperation
            ? {
                  id: removedOperation.id,
                  name: removedOperation.name,
                  instanceId: removedOperation.instanceId,
              }
            : null,
        pipelineLengthBefore: pipeline.length,
        timestamp: new Date().toISOString(),
    });

    pipeline = pipeline.filter((item) => item.instanceId !== instanceId);

    if (useFlowchartMode && flowchartRenderer) {
        flowchartRenderer.removeNode(instanceId);
    }

    console.log("[PIPELINE] Pipeline after removal", {
        pipelineLengthAfter: pipeline.length,
        remainingOperations: pipeline.map((op) => ({
            id: op.id,
            name: op.name,
            instanceId: op.instanceId,
        })),
        timestamp: new Date().toISOString(),
    });

    await renderCurrentPipeline();
    autoSavePipeline();

    console.log("Operation removed from pipeline - requiring backend restart");
    await updateRestartIndicator(true);
    restartRequiredOperations.clear();
}

function runPipeline() {
    console.log("Running pipeline:", pipeline);
    alert("Pipeline run! Check console for details.");
}

function openOperationSettings(opOrItem) {
    const title = `${opOrItem.name || opOrItem.id || "Operation"} Settings`;
    const operationName = opOrItem.name || opOrItem.id;
    const isSecondary = opOrItem.isSecondary || false;
    const initialValues = opOrItem.config || {};

    if (!opOrItem.originalConfig) {
        opOrItem.originalConfig = { ...initialValues };
    }

    const onSave = (values) => {
        console.log("Saved settings for", opOrItem, values);
        const isAutoSaveFlag = values._isAutoSave;
        const requiresRestart = values._requiresRestart;
        console.log("isAutoSave flag:", isAutoSaveFlag);
        console.log("requiresRestart flag:", requiresRestart);

        delete values._isAutoSave;
        delete values._requiresRestart;

        const previousConfig = { ...opOrItem.config };

        opOrItem.config = values;
        opOrItem.requiresRestart = requiresRestart || false;
        console.log("Updated opOrItem.config:", opOrItem.config);
        console.log(
            "Updated opOrItem.requiresRestart:",
            opOrItem.requiresRestart,
        );

        console.log("Calling autoSavePipeline...");
        autoSavePipeline();

        const changedParams = [];
        for (const [key, value] of Object.entries(values)) {
            if (JSON.stringify(previousConfig[key]) !== JSON.stringify(value)) {
                changedParams.push({ paramName: key, value: value });
            }
        }

        if (changedParams.length > 0) {
            for (const { paramName, value } of changedParams) {
                checkPipelineRestartRequirements(opOrItem, paramName, value);
            }
        } else {
            checkPipelineRestartRequirements();
        }
    };

    const doOpen = () => {
        try {
            globalThis.SettingsPopup.open({
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

    const loadFileManager = () => {
        if (globalThis.FileManagerPopup) {
            return Promise.resolve();
        }

        const fileManagerUrl = "../../js/pipeline/fileManager.js";
        const fileManagerAlready = document.querySelector(
            `script[src="${fileManagerUrl}"]`,
        );
        if (fileManagerAlready) {
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            const s = document.createElement("script");
            s.type = "module";
            s.src = fileManagerUrl;
            s.onload = () => {
                if (!globalThis.FileManagerPopup) {
                    console.warn(
                        "FileManagerPopup loaded but did not register on globalThis",
                    );
                }
                resolve();
            };
            s.onerror = () => {
                console.error(
                    "Failed to load file manager script at",
                    fileManagerUrl,
                );
                reject(new Error("Failed to load file manager"));
            };
            document.head.appendChild(s);
        });
    };

    if (globalThis.SettingsPopup) {
        void loadFileManager().then(doOpen).catch(console.error);
        return;
    }

    const scriptUrl = "../../js/pipeline/settingsPopup.js";
    const already = document.querySelector(`script[src="${scriptUrl}"]`);
    if (already) {
        already.addEventListener("load", () => {
            void loadFileManager().then(doOpen).catch(console.error);
        });
        return;
    }

    const s = document.createElement("script");
    s.type = "module";
    s.src = scriptUrl;
    s.onload = () => {
        if (!globalThis.SettingsPopup) {
            console.warn(
                "SettingsPopup loaded but did not register on globalThis",
            );
            return;
        }
        void loadFileManager().then(doOpen).catch(console.error);
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
                for (const key of Object.keys(item.config)) {
                    const value = item.config[key];
                    if (value !== undefined && value !== null) {
                        configParams[key] = value;
                    }
                }
            }
            return {
                action_name: item.id.replaceAll(".py", ""),
                action_params: configParams,
                position: item.position || null,
            };
        });

        const response = await fetch(
            `${BACKEND_BASE_URL}/save-pipeline-config/${encodeURIComponent(selectedCamera.name)}/${encodeURIComponent(selectedPipeline.name)}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(pipelineConfig),
            },
        );
        if (!response.ok)
            throw new Error(`HTTP error! status: ${response.status}`);
        await response.json();
    } catch (error) {
        console.error("Failed to auto-save pipeline:", error);
    } finally {
        isAutoSaving = false;
        if (pendingAutoSave) {
            pendingAutoSave = false;
            autoSavePipeline();
        }
    }
}

async function createNewPipeline() {
    if (!selectedCamera) {
        alert("Please select a camera first.");
        return;
    }

    const newPipelineName = prompt("Enter a name for the new pipeline:");
    if (!newPipelineName || newPipelineName.trim() === "") {
        return;
    }

    const pipelineFileName = newPipelineName.trim().replaceAll(/\s+/g, "_");

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
        pipeline.length = 0;
        pipeline = [];

        const newPipelineObj = {
            name: pipelineFileName,
            displayName: newPipelineName.trim(),
        };

        selectedPipeline = newPipelineObj;

        const existingIndex = pipelines.findIndex(
            (p) => p.name === pipelineFileName,
        );
        if (existingIndex >= 0) {
            pipelines[existingIndex] = newPipelineObj;
        } else {
            pipelines.push(newPipelineObj);
        }

        populatePipelineDropdown(pipelineFileName);

        setTimeout(() => {
            if (pipelineSelect && selectedPipeline) {
                pipelineSelect.value = selectedPipeline.name;
                console.log("Dropdown value set to:", selectedPipeline.name);
            }
        }, 10);

        console.log(
            "[PIPELINE] Re-rendering empty pipeline for new pipeline creation",
            {
                pipelineName: newPipelineName,
                timestamp: new Date().toISOString(),
            },
        );

        await renderCurrentPipeline();
        updateRunButton();
        updateDeleteButtonVisibility();

        // Save the empty pipeline to backend so it persists
        await autoSavePipeline();

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

    const pipelineToDelete = selectedPipeline;

    const confirmed = confirm(
        `Are you sure you want to delete the pipeline "${pipelineToDelete.displayName}"?\n\nThis action cannot be undone.`,
    );

    if (!confirmed) {
        return;
    }

    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/delete-pipeline/${encodeURIComponent(selectedCamera.name)}/${encodeURIComponent(pipelineToDelete.name)}`,
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

        const pipelineIndex = pipelines.findIndex(
            (p) => p.name === pipelineToDelete.name,
        );

        if (pipelineIndex === -1) {
            console.error("Pipeline not found in pipelines array");
            alert("Failed to delete pipeline. Pipeline not found.");
            return;
        }

        pipelines.splice(pipelineIndex, 1);

        console.log("Deleted pipeline:", pipelineToDelete.name);
        console.log("Remaining pipelines:", pipelines);

        pipeline.length = 0;
        pipeline = [];

        selectedPipeline = null;

        populatePipelineDropdown();

        await renderCurrentPipeline();
        updateRunButton();
        updateDeleteButtonVisibility();

        restartRequiredOperations.clear();
        await updateRestartIndicator(false);
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

async function updateRestartIndicator(show = false) {
    try {
        await fetch(`${BACKEND_BASE_URL}/set_restart_required`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ required: show }),
        });
        console.log(
            `Backend notified: restart ${show ? "required" : "not required"}`,
        );
    } catch (error) {
        console.error(
            "Failed to notify backend about restart requirement:",
            error,
        );
    }

    if (!restartIndicator) return;

    const restartMessage =
        restartIndicator.querySelector(".text-red-100") ||
        restartIndicator.querySelector("span");

    if (show) {
        restartIndicator.classList.remove("hidden");
        if (restartMessage)
            restartMessage.textContent = "Backend restart required";
        restartIndicator.classList.add("backend-state-warning");
    } else {
        restartIndicator.classList.add("hidden");
        restartIndicator.classList.remove("backend-state-warning");
    }
}

async function handleRestartBackend() {
    try {
        const restartButton = restartIndicator?.querySelector(
            "#restartBackendButton",
        );

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

        restartRequiredOperations.clear();

        globalThis.location.reload();
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
    const restartIndicatorEl = document.getElementById("restartIndicator");
    if (
        restartIndicatorEl &&
        !restartIndicatorEl.classList.contains("hidden") &&
        restartIndicatorEl.classList.contains("backend-state-warning")
    ) {
        return;
    }

    if (operationItem && changedParamName !== null && changedValue !== null) {
        await checkSpecificParameterRestart(
            operationItem,
            changedParamName,
            changedValue,
        );
    } else if (operationItem) {
        await checkOperationRestartRequirements(operationItem);
    }

    const hasRestartRequirements = restartRequiredOperations.size > 0;
    await updateRestartIndicator(hasRestartRequirements);
}

async function checkSpecificParameterRestart(
    operationItem,
    paramName,
    currentValue,
) {
    try {
        const isSecondary = operationItem.isSecondary || false;
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-operation-config-data/${encodeURIComponent(operationItem.id)}/${isSecondary ? 1 : 0}`,
        );

        if (response.ok) {
            const configData = await response.json();
            const params = configData.parameters || {};
            const paramDef = params[paramName];

            if (paramDef?.restart_for_change) {
                const originalValue = operationItem.originalConfig[paramName];

                const requiresRestart =
                    currentValue !== undefined &&
                    currentValue !== null &&
                    originalValue !== undefined &&
                    originalValue !== null &&
                    JSON.stringify(currentValue) !==
                        JSON.stringify(originalValue);

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
                    if (paramSet.size === 0) {
                        restartRequiredOperations.delete(instanceId);
                    }
                }

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
    for (const [paramName, value] of Object.entries(
        operationItem.config || {},
    )) {
        await checkSpecificParameterRestart(operationItem, paramName, value);
    }
}

async function refreshPipelineCreator() {
    try {
        console.log(
            "[PIPELINE] Refreshing pipeline creator after reconnection",
        );

        await fetchAvailableOperations();

        if (operationsList && operations.length > 0) {
            renderOperations(
                operations,
                operationsList,
                openOperationSettings,
                handleDragStartWithLogging,
            );
        }

        await fetchAvailableCameras();
        populateCameraDropdown();

        if (selectedCamera) {
            await fetchPipelinesForCamera(selectedCamera.name);
            populatePipelineDropdown();

            if (selectedPipeline) {
                await loadPipelineIntoBuilder(
                    selectedCamera.name,
                    selectedPipeline.name,
                );
            }
        }

        updateDeleteButtonVisibility();

        await checkBackendRestartStatus();
    } catch (error) {
        console.error("[PIPELINE] Error refreshing pipeline creator:", error);
    }
}

async function handleFlowchartPipelineChange(changeEvent) {
    if (!selectedPipeline) {
        if (!selectedCamera) {
            alert("Please select a camera first, then create a new pipeline.");
            return;
        }

        const shouldCreate = confirm(
            "You need to create a pipeline before adding operations. Would you like to create a new pipeline now?",
        );
        if (!shouldCreate) {
            return;
        }

        await createNewPipeline();

        if (!selectedPipeline) {
            return;
        }
    }

    if (changeEvent.type === "add") {
        const operation = operations.find(
            (op) => op.id === changeEvent.operationId,
        );
        if (!operation) {
            console.warn(`Operation ${changeEvent.operationId} not found`);
            return;
        }

        const newItem = {
            ...operation,
            instanceId: uid(operation.id + "-"),
            config: {},
            originalConfig: {},
            position: changeEvent.position,
            requiresRestart: false,
        };

        pipeline.push(newItem);

        await renderCurrentPipeline();
        autoSavePipeline();
        updateRestartIndicator(true);
        restartRequiredOperations.clear();
    }
}

function initFlowchartRenderer() {
    flowchartCanvas = document.getElementById("flowchartCanvas");

    if (!flowchartCanvas) {
        console.warn("Flowchart canvas not found, falling back to list mode");
        useFlowchartMode = false;
        return;
    }

    flowchartRenderer = new FlowchartRenderer(flowchartCanvas, {
        gridSpacing: 20,
        nodeSpacingX: 300,
        nodeSpacingY: 150,
        openOperationSettings,
        updateRunButton,
        removeFromPipeline,
        onPipelineChange: handleFlowchartPipelineChange,
        autoSavePipeline,
    });

    window.flowchartRenderer = flowchartRenderer;
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

    createDescriptionPopup();

    const styleElementId = "pipeline-creator-styles";
    if (!document.getElementById(styleElementId)) {
        const styleEl = document.createElement("style");
        styleEl.id = styleElementId;
        styleEl.textContent = `
.op-settings-btn, .remove-btn { display: none !important; }
#pipelineArea .op-settings-btn, #pipelineArea .remove-btn { display: inline-flex !important; }
.icon-grayscale { filter: grayscale(100%); transition: filter .15s ease-in-out; }
#pipelineArea .group:hover .icon-grayscale, #pipelineArea .group:focus-within .icon-grayscale { filter: none; }
#flowchartCanvas { background-color: #1a1a1a; }
.flowchart-node .node-settings-btn:hover img,
.flowchart-node .node-remove-btn:hover img { filter: none !important; }
`;
        document.head.appendChild(styleEl);
    }

    initFlowchartRenderer();

    await fetchAvailableOperations();

    await fetchAvailableCameras();
    populateCameraDropdown();

    if (cameraSelect) {
        cameraSelect.addEventListener("change", handleCameraSelection);
    }

    if (pipelineSelect) {
        pipelineSelect.addEventListener("change", handlePipelineSelection);
    }

    if (selectedCamera) {
        await fetchPipelinesForCamera(selectedCamera.name);
        populatePipelineDropdown();
    }

    await checkAndTriggerAutoFill();

    updateDeleteButtonVisibility();

    if (!useFlowchartMode) {
        const setupDragDrop = (element) => {
            if (!element) return;

            element.addEventListener("dragenter", (e) =>
                handleDragEnterPipeline(e),
            );
            element.addEventListener("dragover", (e) => {
                if (!selectedPipeline) {
                    e.preventDefault();
                    return;
                }
                handleDragOverPipeline(e, pipeline, pipelineContainer);
            });
            element.addEventListener("dragleave", (e) =>
                handleDragLeavePipeline(e, pipeline, pipelinePlaceholder),
            );
            element.addEventListener("drop", async (e) => {
                if (!selectedPipeline) {
                    console.log(
                        "[PIPELINE] Cannot drop operations: no pipeline selected",
                    );
                    e.preventDefault();
                    return;
                }

                const pipelineLengthBefore = pipeline.length;
                const pipelineOrderBefore = pipeline
                    .map((item) => item.instanceId)
                    .join(",");

                handleDropOnPipelineWithLogging(
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
                                {
                                    updateRunButton,
                                    handleDragStart: handleDragStartWithLogging,
                                    handleDragEnd: handleDragEndWithLogging,
                                    removeFromPipeline,
                                    openOperationSettings,
                                },
                            ),
                        updateRunButton,
                        openOperationSettings,
                    },
                );

                const pipelineOrderAfter = pipeline
                    .map((item) => item.instanceId)
                    .join(",");
                const structureChanged =
                    pipeline.length !== pipelineLengthBefore ||
                    pipelineOrderBefore !== pipelineOrderAfter;

                if (structureChanged) {
                    console.log(
                        "[PIPELINE] Pipeline structure changed - requiring backend restart",
                    );
                    await updateRestartIndicator(true);
                    restartRequiredOperations.clear();
                    autoSavePipeline();
                }
            });
        };

        setupDragDrop(pipelineArea);
        setupDragDrop(pipelineContainer);
        setupDragDrop(pipelinePlaceholder);
    }

    if (runButton) {
        runButton.addEventListener("click", runPipeline);
    }

    if (newPipelineButton) {
        newPipelineButton.addEventListener("click", createNewPipeline);
    }

    if (deletePipelineButton) {
        deletePipelineButton.addEventListener("click", deleteCurrentPipeline);
    }

    if (restartIndicator) {
        const restartButton = restartIndicator.querySelector(
            "#restartBackendButton",
        );
        if (restartButton) {
            restartButton.addEventListener("click", handleRestartBackend);
        }
    }

    renderOperations(
        operations,
        operationsList,
        openOperationSettings,
        handleDragStartWithLogging,
    );

    await renderCurrentPipeline();

    await checkBackendRestartStatus();

    isInitialized = true;

    if (globalThis.showBackendRestartIndicator) {
        globalThis.showBackendRestartIndicator();
    }

    globalThis.pipelineCreator = {
        autoSavePipeline: autoSavePipeline,
        updateRestartIndicator: updateRestartIndicator,
        checkPipelineRestartRequirements: checkPipelineRestartRequirements,
        checkBackendRestartStatus: checkBackendRestartStatus,
        restartIndicator: restartIndicator,
        refreshPipelineCreator: refreshPipelineCreator,
        flowchartRenderer: flowchartRenderer,
        selectedPipeline: null,
    };

    Object.defineProperty(globalThis.pipelineCreator, "selectedPipeline", {
        get: () => selectedPipeline,
        enumerable: true,
    });
}
