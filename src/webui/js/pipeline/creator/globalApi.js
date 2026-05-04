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
