// Exposes the pipeline creator API on globalThis for legacy UI consumers.

/**
 * Registers the pipeline creator globals on `globalThis.pipelineCreator`.
 *
 * @param {Object} dependencies - Pipeline creator functions and state accessors.
 * @returns {void}
 */
export function registerPipelineCreatorGlobalApi({
    autoSavePipeline,
    updateRestartIndicator,
    checkPipelineRestartRequirements,
    checkBackendRestartStatus,
    restartIndicator,
    refreshPipelineCreator,
    flowchartRenderer,
    getAvailableCameras,
    refreshAvailableCameras,
    handleOperationErrorUpdate,
    handleProfilingUpdate,
    getOperations,
    getSelectedPipeline,
}) {
    globalThis.pipelineCreator = {
        autoSavePipeline,
        updateRestartIndicator,
        checkPipelineRestartRequirements,
        checkBackendRestartStatus,
        restartIndicator,
        refreshPipelineCreator,
        flowchartRenderer,
        selectedPipeline: null,
        getAvailableCameras,
        refreshAvailableCameras,
        handleOperationErrorUpdate,
        handleProfilingUpdate,
        getOperations,
    };

    Object.defineProperty(globalThis.pipelineCreator, "selectedPipeline", {
        get: () => getSelectedPipeline(),
        enumerable: true,
    });
}
