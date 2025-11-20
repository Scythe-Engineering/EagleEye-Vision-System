import { populateFieldDropdown } from "./dropdown/fieldDropdown.js";
import { setupSidebar } from "./ui/sidebar.js";
import { setupCameraFeedHandlers } from "./feeds/cameraFeedHandlers.js";
import { saveSettings } from "./settings/saveSettings.js";
import { initializeTerminalHandlers, handleLogUpdate, refreshLogMessages } from "./settings/terminalHandler.js";
import { updateRobotTransform, updateDetectedObjects } from "./init3DView.js";
import { BACKEND_BASE_URL } from "./config.js";
import "../style.css";
import { Matrix4 } from "three";

const mmToM = 1000;

// Function to get the currently active view ID
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
async function refreshPipelineCreator() {
    // Use the refresh function from the pipeline creator module if available
    if (globalThis.pipelineCreator?.refreshPipelineCreator) {
        await globalThis.pipelineCreator.refreshPipelineCreator();
    } else {
        console.warn("Pipeline creator refresh function not available");
    }
}

const convertDataToFieldSpace = (data) => {
    const transform = data.transform_matrix;
    const resultMatrix = new Matrix4();

    resultMatrix.set(
        transform[0][0],
        transform[0][2],
        transform[0][1],
        (transform[0][3] - 8.774125) * mmToM,
        transform[2][0],
        transform[2][2],
        transform[2][1],
        transform[2][3] * mmToM,
        -transform[1][0],
        -transform[1][2],
        -transform[1][1],
        (-transform[1][3] + 4.025901) * mmToM,
        transform[3][0],
        transform[3][1],
        transform[3][2],
        transform[3][3],
    );

    return resultMatrix;
};

window.onload = async () => {
    populateFieldDropdown();
    setupSidebar();
    setupCameraFeedHandlers();
    initializeTerminalHandlers();
    saveSettings();

    const showConnectionLostOverlay = () => {
        const overlay = document.getElementById("connection-lost-overlay");
        if (overlay) {
            overlay.classList.remove("hidden");
        }
        if (globalThis.SettingsPopup?.close) {
            globalThis.SettingsPopup.close();
        }
    };

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

    es.addEventListener("open", () => {
        console.log(
            `SSE connection established at ${new Date().toISOString()}`,
        );
        hideConnectionLostOverlay();
        if (wasDisconnected) {
            console.log("SSE reconnected after disconnection");
            // Brief delay to allow connection to stabilize before refreshing views
            setTimeout(async () => {
                wasDisconnected = false;
                const currentViewId = getCurrentViewId();
                await refreshViewsOnReconnection(currentViewId);
                // Refresh logs after reconnection
                refreshLogMessages();
            }, 500);
        }
    });

    es.addEventListener("heartbeat", (e) => {
        lastHeartbeat = Date.now();
        hideConnectionLostOverlay();
    });

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
                    const fieldSpaceTransform = convertDataToFieldSpace(data);
                    updateRobotTransform(fieldSpaceTransform);
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

    es.addEventListener("log_update", (e) => {
        try {
            const data = JSON.parse(e.data);
            handleLogUpdate(data);
        } catch (err) {
            console.warn("Failed to parse SSE log_update event", err);
        }
    });

    es.onerror = () => {
        console.warn("SSE connection error or lost");
        showConnectionLostOverlay();
        wasDisconnected = true;
    };

    // watchdog to detect missed heartbeats
    setInterval(() => {
        if (Date.now() - lastHeartbeat > HEARTBEAT_TIMEOUT_MS) {
            console.warn("SSE connection lost - heartbeat timeout");
            wasDisconnected = true;
            showConnectionLostOverlay();
        }
    }, 2000);
};
