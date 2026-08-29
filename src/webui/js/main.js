/*
 * Main Web UI bootstrap: wires page setup, live backend events, and shared UI state.
 */
import { populateFieldDropdown } from "./dropdown/fieldDropdown.js";
import { setupSidebar } from "./ui/sidebar.js";
import { initializeFirstBootWizard } from "./setup/firstBootWizard.js";
import { setupCameraFeedHandlers } from "./feeds/cameraFeedHandlers.js";
import {
    saveSettings,
    loadSettings,
    renderNetworkTableStatus,
} from "./settings/settingsHandler.js";
import {
    initializeTerminalHandlers,
    handleLogUpdate,
    refreshLogMessages,
} from "./settings/terminalHandler.js";
import { initializeTestVideoManager } from "./settings/testVideoManager.js";
import { initializeAssetFileManager } from "./settings/assetFileManager.js";
import { initializeNetworkManager } from "./settings/networkManager.js";
import {
    handleSystemUpdateProgress,
    initializeSystemUpdateManager,
} from "./settings/systemUpdateManager.js";
import {
    updateRobotTransform,
    updateDetectedObjects,
    updateCameraPose,
} from "./init3DView.js";
import { BACKEND_BASE_URL } from "./config.js";
import {
    showSuccess,
    showWarning,
    showDanger,
    clearAll,
} from "./ui/notificationSystem.js";
import { createSystemStatusModule } from "./system/systemStatus.js";
import { createStatusIconController } from "./ui/statusIcon.js";
import { initializeTooltips } from "./ui/tooltip.js";
import {
    setBackendConnected,
    setNetworkTablesConnected,
    subscribeConnectionStatus,
} from "./ui/connectionStatus.js";
import { cameraPoseToFieldSpaceMatrix } from "./utils/fieldSpaceTransforms.js";
import "../style.css";

/**
 * Returns the currently active view id from the URL or visible DOM state.
 *
 * @returns {string} The active view element id.
 */
function getCurrentViewId() {
    // First check URL parameter
    const url = new URL(globalThis.location.href);
    const tabParam = url.searchParams.get("tab");
    if (tabParam && document.getElementById(tabParam)) {
        return tabParam;
    }

    // Fallback: find the view that doesn't have the 'hidden' class
    const views = document.querySelectorAll("[id^='view-']");
    for (const view of views) {
        if (!view.classList.contains("hidden")) {
            return view.id;
        }
    }

    // Default fallback
    return "view-views";
}

// Function to refresh views when connection is re-established
/**
 * Refreshes view-specific content after the backend reconnects.
 *
 * @param {string} currentViewId - The current view element id.
 * @returns {Promise<void>}
 */
async function refreshViewsOnReconnection(currentViewId) {
    console.log("Refreshing views after reconnection");

    try {
        if (currentViewId === "view-pipeline") {
            console.log("Refreshing pipeline builder");
            await refreshPipelineCreator();
        }

        console.log("Views refreshed successfully after reconnection");
    } catch (error) {
        console.error("Error refreshing views after reconnection:", error);
    }
}

// Function to refresh pipeline creator specifically
/**
 * Refreshes the pipeline creator module if it is available globally.
 *
 * @returns {Promise<void>}
 */
async function refreshPipelineCreator() {
    // Use the refresh function from the pipeline creator module if available
    if (globalThis.pipelineCreator?.refreshPipelineCreator) {
        await globalThis.pipelineCreator.refreshPipelineCreator();
    } else {
        console.warn("Pipeline creator refresh function not available");
    }
}

/**
 * Initializes the web UI after the page loads.
 *
 * @returns {Promise<void>}
 */
window.onload = async () => {
    const destroyTooltips = initializeTooltips();

    await populateFieldDropdown();
    setupSidebar();
    await initializeFirstBootWizard();
    setupCameraFeedHandlers();
    initializeTerminalHandlers();
    initializeTestVideoManager();
    initializeAssetFileManager();
    initializeNetworkManager();
    initializeSystemUpdateManager();
    saveSettings();

    const systemStatusModule = createSystemStatusModule();
    const faviconLink = document.getElementById("favicon");
    const faviconController = createStatusIconController({
        targetLink: faviconLink,
        baseIconUrl: faviconLink?.getAttribute("href") ?? "/favicon.ico",
    });
    /**
     * Updates the favicon status when backend connection state changes.
     *
     * @param {{status: string}} state - The connection status state.
     */
    const unsubscribeConnectionStatus = subscribeConnectionStatus((state) => {
        faviconController.setStatus(state.status);
    });

    /**
     * Cleans up UI resources before the page unloads.
     */
    window.addEventListener("beforeunload", () => {
        destroyTooltips();
        unsubscribeConnectionStatus();
        faviconController.destroy();
    });

    const clearAllButton = document.getElementById("clearAllNotificationsBtn");
    if (clearAllButton) {
        /**
         * Clears all notification messages.
         */
        clearAllButton.addEventListener("click", () => {
            clearAll();
        });
    }

    const testNotificationsBtn = document.getElementById(
        "testNotificationsBtn",
    );
    if (testNotificationsBtn) {
        /**
         * Triggers sample notification messages for testing.
         */
        testNotificationsBtn.addEventListener("click", () => {
            showSuccess("This is a success notification!");
            /**
             * Shows the warning test notification after a short delay.
             */
            setTimeout(() => {
                showWarning("This is a warning notification!");
            }, 300);
            /**
             * Shows the danger test notification after a short delay.
             */
            setTimeout(() => {
                showDanger("This is a danger notification!");
            }, 600);
        });
    }

    /**
     * Shows the connection-lost overlay and notifies the app of backend disconnect.
     */
    const showConnectionLostOverlay = () => {
        document.dispatchEvent(new CustomEvent("backend-disconnected"));
        const overlay = document.getElementById("connection-lost-overlay");
        if (overlay) {
            overlay.classList.remove("hidden");
        }
        if (globalThis.SettingsPopup?.close) {
            globalThis.SettingsPopup.close();
        }
    };

    /**
     * Hides the connection-lost overlay.
     */
    const hideConnectionLostOverlay = () => {
        const overlay = document.getElementById("connection-lost-overlay");
        if (overlay) {
            overlay.classList.add("hidden");
        }
    };

    // Use EventSource for Server-Sent Events (SSE)
    const es = new EventSource(`${BACKEND_BASE_URL}/sse/stream`);
    let lastHeartbeat = Date.now();
    let wasDisconnected = false;
    const HEARTBEAT_TIMEOUT_MS = 15000; // consider connection lost if no heartbeat

    /**
     * Handles SSE connection open events.
     */
    es.addEventListener("open", () => {
        console.log(
            `SSE connection established at ${new Date().toISOString()}`,
        );
        setBackendConnected(true);
        hideConnectionLostOverlay();
        document.dispatchEvent(new CustomEvent("mx3-compilation-reconnected"));
        if (wasDisconnected) {
            console.log("SSE reconnected after disconnection");
            // Brief delay to allow connection to stabilize before refreshing views
            /**
             * Refreshes views and settings after reconnection stabilizes.
             */
            setTimeout(async () => {
                wasDisconnected = false;
                const currentViewId = getCurrentViewId();
                await refreshViewsOnReconnection(currentViewId);
                // Refresh logs after reconnection
                refreshLogMessages();
                // Load settings after reconnection
                await loadSettings();
            }, 500);
        }
    });

    /**
     * Updates the last heartbeat timestamp.
     */
    es.addEventListener("heartbeat", () => {
        lastHeartbeat = Date.now();
        setBackendConnected(true);
        hideConnectionLostOverlay();
    });

    /**
     * Processes robot transform updates from SSE.
     *
     * @param {MessageEvent} e - The SSE message event.
     */
    es.addEventListener("update_robot_transform", (e) => {
        try {
            const data = JSON.parse(e.data);
            if (
                data?.transform_matrix &&
                Array.isArray(data.transform_matrix) &&
                data.transform_matrix.length === 4
            ) {
                const isValid4x4Matrix = data.transform_matrix.every(
                    (row) => Array.isArray(row) && row.length === 4,
                );

                if (isValid4x4Matrix) {
                    const fieldSpaceTransform = cameraPoseToFieldSpaceMatrix(
                        data.transform_matrix,
                    );
                    if (fieldSpaceTransform) {
                        updateRobotTransform(fieldSpaceTransform);
                    }
                } else {
                    console.warn(
                        "Invalid transformation matrix format received:",
                        data,
                    );
                }
            } else {
                console.warn(
                    "Invalid camera transformation data received:",
                    data,
                );
            }
        } catch (err) {
            console.warn(
                "Failed to parse SSE update_robot_transform event",
                err,
            );
        }
    });

    /**
     * Processes camera pose updates from SSE.
     *
     * @param {MessageEvent} e - The SSE message event.
     */
    es.addEventListener("update_camera_pose", (e) => {
        try {
            const data = JSON.parse(e.data);
            const fieldSpaceTransform = cameraPoseToFieldSpaceMatrix(
                data?.transform_matrix,
            );

            if (
                typeof data?.camera_bus_id === "string" &&
                fieldSpaceTransform !== null
            ) {
                updateCameraPose({
                    cameraBusId: data.camera_bus_id,
                    cameraName:
                        typeof data?.camera_name === "string"
                            ? data.camera_name
                            : data.camera_bus_id,
                    transformMatrix: fieldSpaceTransform,
                    timestampMs: Number.isFinite(data?.timestamp_ms)
                        ? data.timestamp_ms
                        : Date.now(),
                });
            } else {
                console.warn("Invalid camera pose SSE payload received:", data);
            }
        } catch (err) {
            console.warn("Failed to parse SSE update_camera_pose event", err);
        }
    });

    /**
     * Processes detected object updates from SSE.
     *
     * @param {MessageEvent} e - The SSE message event.
     */
    es.addEventListener("update_detected_objects", (e) => {
        try {
            const data = JSON.parse(e.data);
            if (data?.detections && Array.isArray(data.detections)) {
                updateDetectedObjects(data.detections);
            } else {
                updateDetectedObjects([]);
            }
        } catch (err) {
            console.warn(
                "Failed to parse SSE update_detected_objects event",
                err,
            );
        }
    });

    /**
     * Processes log updates from SSE.
     *
     * @param {MessageEvent} e - The SSE message event.
     */
    es.addEventListener("log_update", (e) => {
        try {
            const data = JSON.parse(e.data);
            handleLogUpdate(data);
        } catch (err) {
            console.warn("Failed to parse SSE log_update event", err);
        }
    });

    /**
     * Processes system update progress updates from SSE.
     *
     * @param {MessageEvent} e - The SSE message event.
     */
    es.addEventListener("system_update_progress", (e) => {
        try {
            const data = JSON.parse(e.data);
            handleSystemUpdateProgress(data);
        } catch (err) {
            console.warn(
                "Failed to parse SSE system_update_progress event",
                err,
            );
        }
    });

    /**
     * Forwards MX3 compiler status updates to consumers without importing UI modules.
     *
     * @param {MessageEvent} e - The SSE message event.
     */
    const handleMx3CompilationProgress = (e) => {
        try {
            document.dispatchEvent(
                new CustomEvent("mx3-compilation-progress", {
                    detail: JSON.parse(e.data),
                }),
            );
        } catch (err) {
            console.warn(
                "Failed to parse SSE mx3_compilation_progress event",
                err,
            );
        }
    };
    if (typeof globalThis.mx3CompilationSseHandler === "function") {
        es.removeEventListener(
            "mx3_compilation_progress",
            globalThis.mx3CompilationSseHandler,
        );
    }
    es.addEventListener(
        "mx3_compilation_progress",
        handleMx3CompilationProgress,
    );
    globalThis.mx3CompilationSseHandler = handleMx3CompilationProgress;

    /**
     * Processes pipeline operation error updates from SSE.
     *
     * @param {MessageEvent} e - The SSE message event.
     */
    es.addEventListener("pipeline_operation_errors", (e) => {
        try {
            const data = JSON.parse(e.data);
            const handler =
                globalThis.pipelineCreator?.handleOperationErrorUpdate;
            if (typeof handler === "function") {
                handler(data);
            } else {
                globalThis.pendingPipelineOperationErrors =
                    globalThis.pendingPipelineOperationErrors || [];
                globalThis.pendingPipelineOperationErrors.push(data);
            }
        } catch (err) {
            console.warn(
                "Failed to parse SSE pipeline_operation_errors event",
                err,
            );
        }
    });

    if (typeof globalThis.profilingUpdateSseHandler === "function") {
        es.removeEventListener(
            "profiling_update",
            globalThis.profilingUpdateSseHandler,
        );
        delete globalThis.profilingUpdateSseHandler;
    }

    const socket = globalThis.socket;
    const recentProfilingUpdateKeys = new Map();
    const PROFILING_DEDUPE_WINDOW_MS = 1000;

    /**
     * Validates, deduplicates, and forwards profiling update payloads.
     *
     * @param {string|object} data - Raw profiling update payload.
     */
    const handleProfilingUpdatePayload = (data) => {
        try {
            const parsedData =
                typeof data === "string" ? JSON.parse(data) : data;
            const hasRequiredFields =
                typeof parsedData?.pipeline_name === "string" &&
                Number.isFinite(parsedData?.frame_seq) &&
                Number.isFinite(parsedData?.frame_time_ms) &&
                Number.isFinite(parsedData?.timestamp_ms) &&
                parsedData?.operations !== null &&
                typeof parsedData?.operations === "object" &&
                Array.isArray(parsedData?.timesteps);
            if (!hasRequiredFields) {
                return;
            }

            const updateKey = `${parsedData.pipeline_name}:${parsedData.frame_seq}:${parsedData.timestamp_ms}`;
            const now = Date.now();
            const lastSeen = recentProfilingUpdateKeys.get(updateKey);
            if (lastSeen && now - lastSeen < PROFILING_DEDUPE_WINDOW_MS) {
                return;
            }
            recentProfilingUpdateKeys.set(updateKey, now);
            if (recentProfilingUpdateKeys.size > 100) {
                for (const [key, seenAt] of recentProfilingUpdateKeys) {
                    if (now - seenAt > PROFILING_DEDUPE_WINDOW_MS) {
                        recentProfilingUpdateKeys.delete(key);
                    }
                }
            }

            globalThis.pipelineCreator?.handleProfilingUpdate?.(parsedData);
        } catch (err) {
            console.warn("Failed to parse profiling_update event payload", err);
        }
    };

    /**
     * Handles profiling updates received via SSE.
     *
     * @param {MessageEvent} e - The SSE message event.
     */
    const handleProfilingUpdateFromSse = (e) => {
        handleProfilingUpdatePayload(e?.data);
    };

    es.addEventListener("profiling_update", handleProfilingUpdateFromSse);
    globalThis.profilingUpdateSseHandler = handleProfilingUpdateFromSse;

    /**
     * Handles profiling updates received via Socket.IO.
     *
     * @param {unknown} data - The socket payload.
     */
    const handleProfilingUpdateFromSocket = (data) => {
        handleProfilingUpdatePayload(data);
    };

    if (socket?.on) {
        if (typeof globalThis.profilingUpdateSocketHandler === "function") {
            socket.off?.(
                "profiling_update",
                globalThis.profilingUpdateSocketHandler,
            );
        }
        globalThis.profilingUpdateSocketHandler =
            handleProfilingUpdateFromSocket;
        socket.on("profiling_update", globalThis.profilingUpdateSocketHandler);
    } else {
        console.warn(
            "Socket.IO client unavailable for profiling_update events",
        );
    }

    /**
     * Processes system status updates from SSE.
     *
     * @param {MessageEvent} e - The SSE message event.
     */
    es.addEventListener("system_status", (e) => {
        try {
            const data = JSON.parse(e.data);
            systemStatusModule.render(data);
            renderNetworkTableStatus(data.network_table);
            setNetworkTablesConnected(data?.network_table?.connected === true);
        } catch (err) {
            console.warn("Failed to parse SSE system_status event", err);
        }
    });

    /**
     * Handles SSE errors by marking the backend disconnected.
     */
    es.onerror = () => {
        console.warn("SSE connection error or lost");
        setBackendConnected(false);
        showConnectionLostOverlay();
        wasDisconnected = true;
    };

    // watchdog to detect missed heartbeats
    /**
     * Periodically checks for missed SSE heartbeats.
     */
    setInterval(() => {
        if (Date.now() - lastHeartbeat > HEARTBEAT_TIMEOUT_MS) {
            console.warn("SSE connection lost - heartbeat timeout");
            setBackendConnected(false);
            wasDisconnected = true;
            showConnectionLostOverlay();
        }
    }, 2000);
};
