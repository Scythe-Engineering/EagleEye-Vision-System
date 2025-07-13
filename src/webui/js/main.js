import { populateFieldDropdown } from "./dropdown/fieldDropdown.js";
import { setupSidebar } from "./ui/sidebar.js";
import { setupCameraFeedHandlers } from "./feeds/cameraFeedHandlers.js";
import { saveSettings } from "./settings/saveSettings.js";
import { updateRobotTransform } from "./init3DView.js";
import "../style.css";
import io from "socket.io-client";
import { Matrix4, Vector3, Quaternion, Matrix3 } from "three";

const mmToM = 1000;

const convertDataToFieldSpace = (data) => {
    const transform = data.transform_matrix;
    const resultMatrix = new Matrix4();
    
    resultMatrix.set(
        transform[0][0], transform[0][2], transform[0][1], (transform[0][3] - 8.774125) * mmToM,
        transform[2][0], transform[2][2], transform[2][1], transform[2][3] * mmToM,
        -transform[1][0], -transform[1][2], -transform[1][1], (-transform[1][3] + 4.025901) * mmToM,
        transform[3][0], transform[3][1], transform[3][2], transform[3][3]
    );
    
    return resultMatrix;
};

window.onload = () => {
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

    // Socket.IO client for camera position updates
    const socket = io({
        transports: ["websocket"],
        upgrade: false,
        rememberUpgrade: false,
        timeout: 5000,
        forceNew: true,
    });

    socket.on("connect", () => {
        console.log("Socket connected");
        hideConnectionLostOverlay();
    });

    socket.on("disconnect", () => {
        console.log("Socket disconnected");
        showConnectionLostOverlay();
    });

    socket.on("connect_error", (error) => {
        console.error("Connection error:", error);
        showConnectionLostOverlay();
    });

    socket.on("reconnect", () => {
        console.log("Socket reconnected");
        hideConnectionLostOverlay();
    });

    socket.on("reconnect_error", (error) => {
        console.error("Reconnection error:", error);
        showConnectionLostOverlay();
    });

    socket.on("update_robot_transform", (data) => {
        if (
            data?.transform_matrix &&
            Array.isArray(data.transform_matrix) &&
            data.transform_matrix.length === 4
        ) {
            // Validate that it's a proper 4x4 matrix
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
            console.warn("Invalid camera transformation data received:", data);
        }
    });
};
