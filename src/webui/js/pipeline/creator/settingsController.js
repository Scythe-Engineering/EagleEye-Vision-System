export function createOperationSettingsController({
    pipelineStore,
    updatePipelineCameraNote,
    autoSavePipeline,
}) {
    return function openOperationSettings(opOrItem) {
        const latestNode =
            pipelineStore.getNode(opOrItem.instanceId) ||
            pipelineStore.getNode(opOrItem.uuid) ||
            null;
        const settingsItem = latestNode || opOrItem;
        const title = `${settingsItem.name || settingsItem.id || "Operation"} Settings`;
        const operationName = settingsItem.name || settingsItem.id;
        const operationId =
            settingsItem.operationId || settingsItem.id || settingsItem.name;
        const operationUuid = settingsItem.uuid || settingsItem.instanceId;
        const isSecondary = settingsItem.isSecondary || false;
        const initialValues = { ...(settingsItem.config || {}) };

        if (!settingsItem.originalConfig) {
            settingsItem.originalConfig = { ...initialValues };
        }

        const onSave = (values) => {
            console.log("Saved settings for", settingsItem, values);
            const isAutoSaveFlag = values._isAutoSave;
            const requiresRestart = values._requiresRestart;
            console.log("isAutoSave flag:", isAutoSaveFlag);
            console.log("requiresRestart flag:", requiresRestart);

            delete values._isAutoSave;
            delete values._requiresRestart;

            const previousConfig = { ...(settingsItem.config || {}) };
            if (JSON.stringify(previousConfig) === JSON.stringify(values)) {
                console.log("Settings unchanged; skipping save notification");
                return;
            }

            const node =
                pipelineStore.getNode(settingsItem.instanceId) ||
                pipelineStore.getNode(settingsItem.uuid);
            if (node) {
                pipelineStore.updateNodeConfig(node.instanceId, values);
                node.requiresRestart = requiresRestart || false;
                if (opOrItem && opOrItem !== node) {
                    opOrItem.config = { ...values };
                    opOrItem.requiresRestart = node.requiresRestart;
                }
                console.log("Updated node.config:", node.config);
                console.log(
                    "Updated node.requiresRestart:",
                    node.requiresRestart,
                );
                updatePipelineCameraNote();
            } else {
                settingsItem.config = values;
                opOrItem.requiresRestart = requiresRestart || false;
                console.log("Updated opOrItem.config:", opOrItem.config);
                console.log(
                    "Updated opOrItem.requiresRestart:",
                    opOrItem.requiresRestart,
                );
                updatePipelineCameraNote();
            }

            console.log("Saving pipeline config after settings change...");
            void autoSavePipeline({
                showNotification: true,
                requiresRestart,
            });
        };

        const doOpen = () => {
            try {
                globalThis.SettingsPopup.open({
                    title,
                    operationName,
                    operationId,
                    operationUuid,
                    isSecondary,
                    initialValues,
                    onSave,
                });
            } catch (err) {
                console.error("Failed to open SettingsPopup:", err);
            }
        };

        if (!globalThis.FileManagerPopup) {
            console.error("FileManagerPopup not available");
            return;
        }

        if (globalThis.SettingsPopup) {
            doOpen();
            return;
        }

        console.error("SettingsPopup not available");
    };
}
