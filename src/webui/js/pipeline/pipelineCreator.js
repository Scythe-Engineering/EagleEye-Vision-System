import { createDescriptionPopup } from "./rendering.js";
import { pipelineStore } from "./PipelineStore.js";
import { creatorContext } from "./creator/context.js";
import { cachePipelineCreatorElements } from "./creator/selectors.js";
import { ensurePipelineCreatorStyles } from "./creator/styles.js";
import { getSelectedPipeline } from "./creator/stateHelpers.js";
import { createOperationSettingsController } from "./creator/settingsController.js";
import {
    applyBackendRestartState,
    checkBackendRestartStatus,
    checkPipelineRestartRequirements,
    handleRestartBackend,
    updateRestartIndicator,
} from "./creator/restartController.js";
import {
    bindHistoryButtons,
    attachHistoryKeyboardShortcuts,
    initializePipelineHistory,
} from "./creator/historyController.js";
import { registerPipelineCreatorGlobalApi } from "./creator/globalApi.js";
import {
    autoSavePipeline,
    checkAndTriggerAutoFill,
    createNewPipeline,
    deleteCurrentPipeline,
    handlePipelineSelection,
    loadPipelineIntoBuilder,
    populatePipelineDropdown,
    refreshPipelineCreator,
    renderCurrentPipeline,
    runPipeline,
    updateDeleteButtonVisibility,
    updateRunButton,
} from "./creator/pipelineActions.js";
import {
    handleDragStartWithLogging,
    initFlowchartRenderer,
    removeFromPipeline,
    postFlowchartStructureRefresh,
    handleFlowchartPipelineChange,
} from "./creator/flowchartController.js";
import { handleOperationErrorUpdate } from "./creator/errorController.js";
import {
    handleProfilingUpdate,
    checkAndClearStaleProfiling,
    scheduleProfilingUiApply,
    applySelectedPipelineProfiling,
    clearProfilingAverageState,
    refreshProfilingDetailsPopupIfVisible,
    openProfilingDetailsPopup,
    closeProfilingDetailsPopup,
} from "./creator/profilingController.js";
import { fetchAvailableCameras as fetchAvailableCamerasApi } from "./creator/dataApi.js";
import { registerSettingsPopup } from "./settingsPopup.js";

registerSettingsPopup();

let isInitialized = false;

function bindProfilingDetailsControls() {
    const { elements } = creatorContext;

    document.addEventListener("backend-disconnected", closeProfilingDetailsPopup);

    elements.profilingDetailsBackdrop?.addEventListener("click", () => {
        closeProfilingDetailsPopup();
    });
    elements.profilingDetailsCloseButton?.addEventListener("click", () => {
        closeProfilingDetailsPopup();
    });
    elements.profilingDetailsInfoButton?.addEventListener("click", () => {
        openProfilingDetailsPopup();
    });
    elements.profilingDetailsAverageCheckbox?.addEventListener("change", () => {
        clearProfilingAverageState();
        refreshProfilingDetailsPopupIfVisible(undefined);
    });
}

export async function initPipelineCreator() {
    if (isInitialized) return;

    cachePipelineCreatorElements();
    bindProfilingDetailsControls();
    createDescriptionPopup();
    ensurePipelineCreatorStyles();

    const openOperationSettings = createOperationSettingsController({
        pipelineStore,
        updatePipelineCameraNote: () => {},
        autoSavePipeline,
    });

    let renderPipelineView = null;
    let createPipelineView = null;

    renderPipelineView = () =>
        renderCurrentPipeline({
            openOperationSettings,
            handleFlowchartPipelineChange: (changeEvent) =>
                handleFlowchartPipelineChange(changeEvent, {
                    autoSavePipeline,
                    renderCurrentPipeline: renderPipelineView,
                    updateRunButton,
                    onCreatePipeline: createPipelineView,
                }),
            autoSavePipeline,
            updateRunButton,
            removeFromPipeline,
        });

    createPipelineView = () =>
        createNewPipeline({
            renderCurrentPipeline: renderPipelineView,
            updateDeleteButtonVisibility,
            autoSavePipeline,
        });

    const flowchartRenderer = initFlowchartRenderer({
        openOperationSettings,
        updateRunButton,
        removeFromPipeline,
        autoSavePipeline,
        onPipelineChange: (changeEvent) =>
            handleFlowchartPipelineChange(changeEvent, {
                autoSavePipeline,
                renderCurrentPipeline: renderPipelineView,
                updateRunButton,
                onCreatePipeline: createPipelineView,
            }),
    });

    initializePipelineHistory(pipelineStore, {
        renderCallback: async () => {
            await renderPipelineView();
        },
        autoSaveCallback: () => autoSavePipeline(),
        postRefreshCallback: () => postFlowchartStructureRefresh(),
    });

    pipelineStore.subscribe("profiling:updated", ({ snapshot, pipelineName }) => {
        const selectedPipeline = getSelectedPipeline();
        if (!selectedPipeline || selectedPipeline.name !== pipelineName) {
            return;
        }
        scheduleProfilingUiApply(snapshot);
    });

    const loadPipelineIntoBuilderWithRender = (pipelineName) =>
        loadPipelineIntoBuilder(pipelineName, {
            renderCurrentPipeline: renderPipelineView,
        });

    const refreshCallbacks = {
        openOperationSettings,
        handleDragStartWithLogging,
        checkBackendRestartStatus,
        loadPipelineIntoBuilder: loadPipelineIntoBuilderWithRender,
    };

    const createPipelineCallbacks = {
        renderCurrentPipeline: renderPipelineView,
        updateDeleteButtonVisibility,
        autoSavePipeline,
    };

    const deletePipelineCallbacks = {
        renderCurrentPipeline: renderPipelineView,
        updateDeleteButtonVisibility,
        updateRunButton,
    };

    creatorContext.elements.pipelineSelect?.addEventListener("change", () => {
        handlePipelineSelection({
            loadPipelineIntoBuilder: loadPipelineIntoBuilderWithRender,
        });
    });
    creatorContext.elements.runButton?.addEventListener("click", runPipeline);
    creatorContext.elements.newPipelineButton?.addEventListener("click", () => {
        createNewPipeline(createPipelineCallbacks);
    });
    creatorContext.elements.deletePipelineButton?.addEventListener("click", () => {
        deleteCurrentPipeline(deletePipelineCallbacks);
    });
    bindHistoryButtons(
        creatorContext.elements.undoButton,
        creatorContext.elements.redoButton,
    );
    attachHistoryKeyboardShortcuts();

    creatorContext.elements.restartIndicator
        ?.querySelector("#restartBackendButton")
        ?.addEventListener("click", handleRestartBackend);

    registerPipelineCreatorGlobalApi({
        autoSavePipeline,
        updateRestartIndicator,
        checkPipelineRestartRequirements,
        checkBackendRestartStatus,
        restartIndicator: creatorContext.elements.restartIndicator,
        refreshPipelineCreator: () => refreshPipelineCreator(refreshCallbacks),
        flowchartRenderer,
        getAvailableCameras: () => pipelineStore.state.cameras,
        refreshAvailableCameras: () => fetchAvailableCamerasApi(pipelineStore),
        handleOperationErrorUpdate,
        handleProfilingUpdate,
        getOperations: () => pipelineStore.state.operations,
        getSelectedPipeline,
    });

    await refreshPipelineCreator(refreshCallbacks);
    await checkBackendRestartStatus();

    isInitialized = true;

    if (globalThis.showBackendRestartIndicator) {
        globalThis.showBackendRestartIndicator();
    }
}
