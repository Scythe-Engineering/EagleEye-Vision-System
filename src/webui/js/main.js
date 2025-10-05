import { populateFieldDropdown } from "./dropdown/fieldDropdown.js";
import { setupSidebar } from "./ui/sidebar.js";
import { setupCameraFeedHandlers } from "./feeds/cameraFeedHandlers.js";
import { saveSettings } from "./settings/saveSettings.js";
import { updateRobotTransform } from "./init3DView.js";
import { BACKEND_BASE_URL } from "./config.js";
import "../style.css";
import { Matrix4 } from "three";

const mmToM = 1000;

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
    saveSettings();

    const showConnectionLostOverlay = () => {
        const overlay = document.getElementById("connection-lost-overlay");
        if (overlay) {
            overlay.classList.remove("hidden");
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
        console.log("SSE open");
        hideConnectionLostOverlay();
        if (wasDisconnected) setTimeout(() => window.location.reload(), 1000);
        wasDisconnected = false;
    });

    es.addEventListener("heartbeat", (e) => {
        lastHeartbeat = Date.now();
        hideConnectionLostOverlay();

        // Reload page after reconnection to refresh state
        setTimeout(() => {
            window.location.reload();
        }, 1000);
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

    es.onerror = () => {
        console.warn("SSE error");
        showConnectionLostOverlay();
        wasDisconnected = true;
    };

    // watchdog to detect missed heartbeats
    setInterval(() => {
        if (Date.now() - lastHeartbeat > HEARTBEAT_TIMEOUT_MS) {
            showConnectionLostOverlay();
        }
    }, 2000);
};
