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
import { debounce } from "./utils.js";
import { BACKEND_BASE_URL } from "../config.js";
import { pipelineStore } from "./PipelineStore.js";
import { showDanger, showWarning } from "../ui/notificationSystem.js";

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
    const pipelineNodes = pipelineStore.getNodes();
    const pipelineOrderBefore = pipelineNodes.map((node) => ({
        id: node.operationId,
        name: node.name,
        instanceId: node.instanceId,
    }));

    console.log("[PIPELINE] Drop operation started", {
        pipelineLengthBefore: pipelineNodes.length,
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

    const pipelineNodesAfter = pipelineStore.getNodes();
    const pipelineOrderAfter = pipelineNodesAfter.map((node) => ({
        id: node.operationId,
        name: node.name,
        instanceId: node.instanceId,
    }));

    console.log("[PIPELINE] Drop operation completed", {
        pipelineLengthAfter: pipelineNodesAfter.length,
        pipelineOrderAfter,
        orderChanged:
            JSON.stringify(pipelineOrderBefore.map((p) => p.instanceId)) !==
            JSON.stringify(pipelineOrderAfter.map((p) => p.instanceId)),
        lengthChanged: pipelineOrderBefore.length !== pipelineOrderAfter.length,
        timestamp: new Date().toISOString(),
    });

    return result;
}

let isInitialized = false;

let flowchartRenderer = null;
let useFlowchartMode = true;

const restartRequiredOperations = new Map();

let pipelineArea;
let pipelineContainer;
let pipelinePlaceholder;
let operationsList;
let runButton;
let pipelineSelect;
let pipelineCameraNote;
let newPipelineButton;
let deletePipelineButton;
let restartIndicator;
let flowchartCanvas;

const autoSavePipeline = debounce(autoSavePipelineImpl, 500);

function getOperations() {
    return pipelineStore.state.operations;
}

function getPipeline() {
    return pipelineStore.getNodesForRenderer();
}

function getPipelines() {
    return pipelineStore.state.pipelines;
}

function getSelectedPipeline() {
    const pipelineName = pipelineStore.state.currentPipeline.pipelineName;
    return pipelineStore.state.pipelines.find((p) => p.name === pipelineName);
}

function getDeviceInputNodes() {
    return pipelineStore.getNodes().filter((node) => {
        return pipelineStore.normalizeOperationId(node.operationId) ===
            "device_input";
    });
}

function getDeviceInputCameraNames() {
    const names = new Set();
    pipelineStore.getNodes().forEach((node) => {
        const operationId = pipelineStore.normalizeOperationId(node.operationId);
        if (operationId === "device_input") {
            const cameraName = node.config?.camera_name;
            if (cameraName) {
                names.add(cameraName);
            }
        }
    });
    return Array.from(names);
}

function formatPipelineCameraNote(cameraNames) {
    if (cameraNames.length === 0) {
        return { text: "No cameras configured", title: "" };
    }
    const sortedNames = [...cameraNames].sort();
    if (sortedNames.length <= 2) {
        return {
            text: `Cameras: ${sortedNames.join(", ")}`,
            title: sortedNames.join(", "),
        };
    }
    const visibleNames = sortedNames.slice(0, 2).join(", ");
    return {
        text: `Cameras: ${visibleNames} (+${sortedNames.length - 2} more)`,
        title: sortedNames.join(", "),
    };
}

function updatePipelineCameraNote() {
    if (!pipelineCameraNote) {
        return;
    }
    const cameraNames = getDeviceInputCameraNames();
    const note = formatPipelineCameraNote(cameraNames);
    pipelineCameraNote.textContent = note.text;
    pipelineCameraNote.title = note.title;
}

async function fetchAvailableOperations() {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-available-operations`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        const operations = data.operations.map((op) => ({
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

        pipelineStore.setOperations(operations);
        console.log("Loaded operations from server:", operations);
    } catch (error) {
        showDanger("Failed to fetch operations");
        console.error("Failed to fetch operations:", error);
        pipelineStore.setOperations([]);
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

        const cameras = Object.entries(data).map(([name, urlSafeName]) => ({
            name: name,
            urlSafeName: urlSafeName,
        }));

        pipelineStore.setCameras(cameras);
        console.log("Loaded cameras from server:", cameras);
    } catch (error) {
        showDanger("Failed to fetch cameras");
        console.error("Failed to fetch cameras:", error);
        pipelineStore.setCameras([]);
    }
}

function populateCameraDropdown() {
    cameraSelect.innerHTML = "";

    const cameras = pipelineStore.state.cameras;
    if (!Array.isArray(cameras) || cameras.length === 0) {
        const option = document.createElement("option");
        option.disabled = true;
        option.selected = true;
        option.textContent = "No cameras available";
        cameraSelect.appendChild(option);
        pipelineStore.setCurrentCamera(null);
        return;
    }

    for (let index = 0; index < cameras.length; index++) {
        const camera = cameras[index];
        const option = document.createElement("option");
        option.value = camera.urlSafeName;
        option.textContent = camera.name;
        if (index === 0) {
            option.selected = true;
            pipelineStore.setCurrentCamera(camera.name);
        }
        cameraSelect.appendChild(option);
    }
}

async function fetchPipelines() {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-pipeline-names`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const pipelineNames = await response.json();

        const pipelines = pipelineNames.map((name) => ({
            name: name,
            displayName: name
                .replaceAll("_", " ")
                .replaceAll(/\b\w/g, (l) => l.toUpperCase()),
        }));

        pipelineStore.setPipelines(pipelines);
        console.log("Loaded pipelines from server:", pipelines);
    } catch (error) {
        showDanger("Failed to fetch pipelines");
        console.error("Failed to fetch pipelines:", error);
        pipelineStore.setPipelines([]);
    }
}

async function fetchPipelineConfig(pipelineName) {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-pipeline-config/${encodeURIComponent(pipelineName)}`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const pipelineConfig = await response.json();

        console.log("Loaded pipeline config from server:", pipelineConfig);
        return pipelineConfig;
    } catch (error) {
        showDanger("Failed to fetch pipeline config");
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
    const pipelines = getPipelines();

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
            pipelineStore.setCurrentPipeline(pipelineItem.name);
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
            pipelineStore.setCurrentPipeline(pipelines[0].name);
        }
    }

    if (pipelines.length === 0) {
        pipelineStore.setCurrentPipeline(null);
    }
}

async function handlePipelineSelection() {
    const selectedValue = pipelineSelect.value;
    const pipelines = getPipelines();
    const selectedPipeline = pipelines.find(
        (pipelineItem) => pipelineItem.name === selectedValue,
    );
    console.log("Selected pipeline:", selectedPipeline);

    if (selectedPipeline) {
        pipelineStore.setCurrentPipeline(selectedPipeline.name);
        await loadPipelineIntoBuilder(selectedPipeline.name);
    }

    updateDeleteButtonVisibility();
}

async function loadPipelineIntoBuilder(pipelineName) {
    try {
        const operations = getOperations();
        if (operations.length === 0) {
            console.warn("Operations not loaded yet, cannot load pipeline");
            return;
        }

        const pipelineConfig = await fetchPipelineConfig(pipelineName);

        const allConnections = [];
        pipelineConfig.forEach((configItem) => {
            if (
                configItem.connections &&
                Array.isArray(configItem.connections)
            ) {
                allConnections.push(...configItem.connections);
            }
        });

        pipelineStore.loadPipelineData(pipelineConfig, allConnections);

        await renderCurrentPipeline();

        updateRunButton();
        updatePipelineCameraNote();
    } catch (error) {
        showDanger("Failed to load pipeline");
        console.error("Failed to load pipeline:", error);
    }
}

async function renderCurrentPipeline() {
    const pipeline = getPipeline();
    const connections = pipelineStore.getConnectionsForRenderer();

    if (useFlowchartMode && flowchartRenderer) {
        const options = { connections };
        await flowchartRenderer.renderPipeline(pipeline, options);
        await fetchAndUpdateThreadInfo();
    } else {
        renderPipeline(pipeline, pipelineContainer, pipelinePlaceholder, {
            openOperationSettings,
            updateRunButton,
            removeFromPipeline,
            handleDragStart: handleDragStartWithLogging,
            handleDragEnd: handleDragEndWithLogging,
        });
    }
    updatePipelineCameraNote();
}

async function fetchAndUpdateThreadInfo() {
    const selectedPipeline = getSelectedPipeline();

    if (!selectedPipeline || pipelineStore.isRestartRequired()) {
        hideAllThreadBadges();
        return;
    }

    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-pipeline-thread-info/${encodeURIComponent(selectedPipeline.name)}`,
        );

        if (!response.ok) {
            console.warn("Failed to fetch thread info:", response.status);
            hideAllThreadBadges();
            return;
        }

        const data = await response.json();

        if (flowchartRenderer) {
            const nodes = flowchartRenderer.nodes;

            for (const [instanceId, node] of nodes) {
                const uuid = pipelineStore.instanceIdToUuid.get(instanceId);
                if (uuid && data.operations) {
                    const threadInfo = data.operations[uuid];
                    node.updateThreadInfo(threadInfo);
                } else {
                    node.hideThreadBadge();
                }
            }
        }
    } catch (error) {
        console.error("Error fetching thread info:", error);
        hideAllThreadBadges();
    }
}

function hideAllThreadBadges() {
    if (flowchartRenderer) {
        for (const node of flowchartRenderer.nodes.values()) {
            node.hideThreadBadge();
        }
    }
}

async function checkAndTriggerAutoFill() {
    try {
        const pipelines = getPipelines();

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

        pipelineStore.setCurrentPipeline(pipelineObj.name);

        console.log(
            "Pipeline pre-selected, triggering auto-fill",
        );
        await loadPipelineIntoBuilder(pipelineObj.name);
    } catch (error) {
        console.error("Error during auto-fill check:", error);
    }
}

async function removeFromPipeline(instanceId) {
    const removedNode = pipelineStore.getNode(instanceId);
    const deviceInputCountBefore = getDeviceInputNodes().length;

    console.log("[PIPELINE] Removing operation from pipeline", {
        removedOperation: removedNode
            ? {
                  id: removedNode.operationId,
                  name: removedNode.name,
                  instanceId: removedNode.instanceId,
              }
            : null,
        pipelineLengthBefore: pipelineStore.getNodes().length,
        timestamp: new Date().toISOString(),
    });

    pipelineStore.removeNode(instanceId);

    if (useFlowchartMode && flowchartRenderer) {
        flowchartRenderer.removeNode(instanceId);
    }

    console.log("[PIPELINE] Pipeline after removal", {
        pipelineLengthAfter: pipelineStore.getNodes().length,
        remainingOperations: pipelineStore.getNodes().map((node) => ({
            id: node.operationId,
            name: node.name,
            instanceId: node.instanceId,
        })),
        timestamp: new Date().toISOString(),
    });

    await renderCurrentPipeline();
    autoSavePipeline();

    const deviceInputCountAfter = getDeviceInputNodes().length;
    if (deviceInputCountBefore > 0 && deviceInputCountAfter === 0) {
        showWarning(
            "No device_input nodes configured; camera_name required for camera input.",
        );
    }

    console.log("Operation removed from pipeline - requiring backend restart");
    await updateRestartIndicator(true);
    pipelineStore.clearRestartRequired();
}

function runPipeline() {
    console.log("Running pipeline:", pipeline);
    alert("Pipeline run! Check console for details.");
}

function openOperationSettings(opOrItem) {
    const title = `${opOrItem.name || opOrItem.id || "Operation"} Settings`;
    const operationName = opOrItem.name || opOrItem.id;
    const operationUuid = opOrItem.uuid || opOrItem.instanceId;
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

        // Update the actual node in PipelineStore, not just the copy
        const node = pipelineStore.getNode(opOrItem.instanceId);
        if (node) {
            pipelineStore.updateNodeConfig(opOrItem.instanceId, values);
            node.requiresRestart = requiresRestart || false;
            console.log("Updated node.config:", node.config);
            console.log("Updated node.requiresRestart:", node.requiresRestart);
            updatePipelineCameraNote();
        } else {
            // Fallback to updating the copy if node not found (shouldn't happen)
            opOrItem.config = values;
            opOrItem.requiresRestart = requiresRestart || false;
            console.log("Updated opOrItem.config:", opOrItem.config);
            console.log(
                "Updated opOrItem.requiresRestart:",
                opOrItem.requiresRestart,
            );
            updatePipelineCameraNote();
        }

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
                operationUuid,
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
        runButton.disabled = pipelineStore.getNodes().length === 0;
    }
}

async function autoSavePipelineImpl() {
    const selectedPipeline = getSelectedPipeline();

    if (!selectedPipeline) {
        console.log("No pipeline selected, skipping auto-save");
        return;
    }

    try {
        const pipelineConfig = pipelineStore.exportToConfig();

        const response = await fetch(
            `${BACKEND_BASE_URL}/save-pipeline-config/${encodeURIComponent(selectedPipeline.name)}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(pipelineConfig),
            },
        );
        if (!response.ok)
            throw new Error(`HTTP error! status: ${response.status}`);
        await response.json();
        console.log("Pipeline auto-saved successfully");
    } catch (error) {
        console.error("Failed to auto-save pipeline:", error);
    }
}

async function createNewPipeline() {
    const newPipelineName = prompt("Enter a name for the new pipeline:");
    if (!newPipelineName || newPipelineName.trim() === "") {
        return;
    }

    const pipelineFileName = newPipelineName.trim().replaceAll(/\s+/g, "_");

    const pipelines = getPipelines();
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
        pipelineStore.clearPipeline();

        const newPipelineObj = {
            name: pipelineFileName,
            displayName: newPipelineName.trim(),
        };

        const currentPipelines = pipelineStore.state.pipelines;
        const existingIndex = currentPipelines.findIndex(
            (p) => p.name === pipelineFileName,
        );
        if (existingIndex >= 0) {
            currentPipelines[existingIndex] = newPipelineObj;
        } else {
            currentPipelines.push(newPipelineObj);
        }

        pipelineStore.setCurrentPipeline(pipelineFileName);
        populatePipelineDropdown(pipelineFileName);

        setTimeout(() => {
            if (pipelineSelect) {
                pipelineSelect.value = pipelineFileName;
                console.log("Dropdown value set to:", pipelineFileName);
            }
        }, 10);

        // Automatically add device_input operation
        const operations = getOperations();
        const deviceInputOp = operations.find(
            (op) => op.id === "device_input.py",
        );
        if (deviceInputOp) {
            pipelineStore.addNode(
                { id: deviceInputOp.id, config: {} },
                { x: 100, y: 100 },
            );
        }

        console.log(
            "[PIPELINE] Re-rendering pipeline with device_input for new pipeline creation",
            {
                pipelineName: newPipelineName,
                timestamp: new Date().toISOString(),
            },
        );

        await renderCurrentPipeline();
        updateRunButton();
        updateDeleteButtonVisibility();

        // Save the empty pipeline to backend so it persists
        await autoSavePipelineImpl();

        pipelineStore.clearRestartRequired();
        await updateRestartIndicator(false);

        console.log("New pipeline created:", newPipelineName);
        console.log("Pipeline state:", pipelineStore.getNodes());
        console.log("Selected pipeline:", getSelectedPipeline());
        console.log("Pipelines list:", getPipelines());
    } catch (error) {
        console.error("Failed to create new pipeline:", error);
        alert(
            "Failed to create new pipeline. Please check the console for details.",
        );
    }
}

async function deleteCurrentPipeline() {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline) {
        alert("No pipeline selected to delete.");
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
            `${BACKEND_BASE_URL}/delete-pipeline/${encodeURIComponent(pipelineToDelete.name)}`,
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

        const currentPipelines = pipelineStore.state.pipelines;
        const pipelineIndex = currentPipelines.findIndex(
            (p) => p.name === pipelineToDelete.name,
        );

        if (pipelineIndex === -1) {
            console.error("Pipeline not found in pipelines array");
            alert("Failed to delete pipeline. Pipeline not found.");
            return;
        }

        currentPipelines.splice(pipelineIndex, 1);

        console.log("Deleted pipeline:", pipelineToDelete.name);
        console.log("Remaining pipelines:", currentPipelines);

        pipelineStore.clearPipeline();
        pipelineStore.setCurrentPipeline(null);

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
        const selectedPipeline = getSelectedPipeline();
        if (selectedPipeline) {
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
        showDanger("Failed to notify backend about restart requirement");
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

    if (show) {
        hideAllThreadBadges();
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

        const operations = getOperations();
        if (operationsList && operations.length > 0) {
            renderOperations(
                operations,
                operationsList,
                openOperationSettings,
                handleDragStartWithLogging,
            );
        }

        await fetchAvailableCameras();

        await fetchPipelines();
        populatePipelineDropdown();

        const selectedPipeline = getSelectedPipeline();
        if (selectedPipeline) {
            await loadPipelineIntoBuilder(selectedPipeline.name);
        }

        updateDeleteButtonVisibility();

        await checkBackendRestartStatus();
    } catch (error) {
        console.error("[PIPELINE] Error refreshing pipeline creator:", error);
    }
}

async function handleFlowchartPipelineChange(changeEvent) {
    const selectedPipeline = getSelectedPipeline();

    if (!selectedPipeline) {
        const shouldCreate = confirm(
            "You need to create a pipeline before adding operations. Would you like to create a new pipeline now?",
        );
        if (!shouldCreate) {
            return;
        }

        await createNewPipeline();

        if (!getSelectedPipeline()) {
            return;
        }
    }

    if (changeEvent.type === "add") {
        const node = pipelineStore.addNode(
            { id: changeEvent.operationId },
            changeEvent.position,
        );
        if (!node) {
            console.warn(`Operation ${changeEvent.operationId} not found`);
            return;
        }

        await renderCurrentPipeline();
        autoSavePipeline();
        updateRestartIndicator(true);
        pipelineStore.clearRestartRequired();
        hideAllThreadBadges();
        updatePipelineCameraNote();
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
    pipelineSelect = document.getElementById("pipelineSelect");
    pipelineCameraNote = document.getElementById("pipelineCameraNote");
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

    if (pipelineSelect) {
        pipelineSelect.addEventListener("change", handlePipelineSelection);
    }

    await fetchPipelines();
    populatePipelineDropdown();

    await checkAndTriggerAutoFill();

    updateDeleteButtonVisibility();

    if (!useFlowchartMode) {
        const setupDragDrop = (element) => {
            if (!element) return;

            element.addEventListener("dragenter", (e) =>
                handleDragEnterPipeline(e),
            );
            element.addEventListener("dragover", (e) => {
                if (!getSelectedPipeline()) {
                    e.preventDefault();
                    return;
                }
                handleDragOverPipeline(e, getPipeline(), pipelineContainer);
            });
            element.addEventListener("dragleave", (e) =>
                handleDragLeavePipeline(e, getPipeline(), pipelinePlaceholder),
            );
            element.addEventListener("drop", async (e) => {
                if (!getSelectedPipeline()) {
                    console.log(
                        "[PIPELINE] Cannot drop operations: no pipeline selected",
                    );
                    e.preventDefault();
                    return;
                }

                const pipelineNodes = getPipeline();
                const pipelineLengthBefore = pipelineNodes.length;
                const pipelineOrderBefore = pipelineNodes
                    .map((item) => item.instanceId)
                    .join(",");

                handleDropOnPipelineWithLogging(
                    e,
                    pipelineNodes,
                    getOperations(),
                    pipelineContainer,
                    pipelinePlaceholder,
                    {
                        renderPipeline: () =>
                            renderPipeline(
                                getPipeline(),
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

                const pipelineNodesAfter = getPipeline();
                const pipelineOrderAfter = pipelineNodesAfter
                    .map((item) => item.instanceId)
                    .join(",");
                const structureChanged =
                    pipelineNodesAfter.length !== pipelineLengthBefore ||
                    pipelineOrderBefore !== pipelineOrderAfter;

                if (structureChanged) {
                    console.log(
                        "[PIPELINE] Pipeline structure changed - requiring backend restart",
                    );
                    await updateRestartIndicator(true);
                    pipelineStore.clearRestartRequired();
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
        getOperations(),
        operationsList,
        openOperationSettings,
        handleDragStartWithLogging,
    );

    // Initialize globalThis.pipelineCreator BEFORE rendering so it's available during placeholder checks
    globalThis.pipelineCreator = {
        autoSavePipeline: autoSavePipeline,
        updateRestartIndicator: updateRestartIndicator,
        checkPipelineRestartRequirements: checkPipelineRestartRequirements,
        checkBackendRestartStatus: checkBackendRestartStatus,
        restartIndicator: restartIndicator,
        refreshPipelineCreator: refreshPipelineCreator,
        flowchartRenderer: flowchartRenderer,
        selectedPipeline: null,
        getAvailableCameras: () => pipelineStore.state.cameras,
        refreshAvailableCameras: () => fetchAvailableCameras(),
    };

    Object.defineProperty(globalThis.pipelineCreator, "selectedPipeline", {
        get: () => getSelectedPipeline(),
        enumerable: true,
    });

    await renderCurrentPipeline();

    await checkBackendRestartStatus();

    isInitialized = true;

    if (globalThis.showBackendRestartIndicator) {
        globalThis.showBackendRestartIndicator();
    }
}
