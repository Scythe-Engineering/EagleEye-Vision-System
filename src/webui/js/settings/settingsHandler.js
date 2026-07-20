import { BACKEND_BASE_URL } from "../config.js";
import { isDemoMode } from "../demoMode.js";
import { setNetworkTablesConnected } from "../ui/connectionStatus.js";
import { showDanger, showSuccess, showWarning } from "../ui/notificationSystem.js";

// Handles loading, rendering, and saving settings for the web UI.

/**
 * Render the NetworkTables connection status badge in the settings UI.
 *
 * @param {object|null} status - NetworkTables status payload.
 */
export function renderNetworkTableStatus(status) {
    const badge = document.getElementById("networkTableStatusBadge");
    const dot = document.getElementById("networkTableStatusDot");
    const text = document.getElementById("networkTableStatusText");

    if (!badge || !dot || !text) {
        return;
    }

    badge.className =
        "inline-flex items-center gap-2 rounded-full border px-2 py-1 text-xs font-semibold";
    dot.className = "h-2 w-2 rounded-full";

    if (status?.connected === true) {
        badge.classList.add(
            "border-emerald-500/40",
            "bg-emerald-900/30",
            "text-emerald-200",
        );
        dot.classList.add("bg-emerald-400");
        text.textContent = "Connected";
        return;
    }

    if (status?.status === "ok" || status?.status === "unavailable") {
        badge.classList.add(
            "border-red-500/40",
            "bg-red-950/40",
            "text-red-200",
        );
        dot.classList.add("bg-red-400");
        text.textContent = "Disconnected";
        return;
    }

    badge.classList.add(
        "border-gray-500/40",
        "bg-gray-800/40",
        "text-gray-200",
    );
    dot.classList.add("bg-gray-400");
    text.textContent = "Unknown";
}

/**
 * Load the current NetworkTables status from the backend and update the UI.
 */
async function loadNetworkTableStatus() {
    try {
        const response = await fetch(`${BACKEND_BASE_URL}/get-system-status`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const status = await response.json();
        renderNetworkTableStatus(status.network_table);
        setNetworkTablesConnected(status?.network_table?.connected === true);
    } catch (error) {
        console.error("Error loading NetworkTables status:", error);
        renderNetworkTableStatus(null);
        setNetworkTablesConnected(false);
    }
}

/**
 * Load general settings from the backend and populate the settings form.
 */
export async function loadSettings() {
    try {
        const response = await fetch(`${BACKEND_BASE_URL}/get-general-conf`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });

        if (!response.ok) {
            throw new Error("Network response was not ok");
        }

        const settings = await response.json();
        const robotAddressInput = document.getElementById("robotAddressInput");
        const viewStreamDownscaleInput = document.getElementById(
            "viewStreamDownscaleInput",
        );

        if (robotAddressInput && settings.network_table_address) {
            robotAddressInput.value = settings.network_table_address;
        }
        if (
            viewStreamDownscaleInput &&
            settings.view_stream_downscale !== undefined
        ) {
            viewStreamDownscaleInput.value = settings.view_stream_downscale;
        }
    } catch (error) {
        console.error("Error loading settings:", error);
    }

    await loadNetworkTableStatus();
}

/**
 * Wire up the save button to persist settings changes.
 */
export function saveSettings() {
    const saveSettingsBtn = document.getElementById("saveSettingsBtn");
    if (!saveSettingsBtn) {
        return;
    }
    if (isDemoMode()) {
        saveSettingsBtn.setAttribute("disabled", "true");
        saveSettingsBtn.classList.add("hidden");
        return;
    }

    /**
     * Validate and submit the current settings form values.
     */
    const handleSaveClick = async () => {
        const robotAddressInput = document.getElementById("robotAddressInput");
        const viewStreamDownscaleInput = document.getElementById(
            "viewStreamDownscaleInput",
        );
        if (!robotAddressInput || !viewStreamDownscaleInput) {
            return;
        }

        const viewStreamDownscale = Number.parseFloat(
            viewStreamDownscaleInput.value,
        );
        if (
            Number.isNaN(viewStreamDownscale) ||
            viewStreamDownscale < 0.1 ||
            viewStreamDownscale > 1
        ) {
            showWarning("Stream downscale must be between 0.1 and 1.");
            return;
        }

        const settings = {
            network_table_address: robotAddressInput.value,
            view_stream_downscale: viewStreamDownscale,
        };

        try {
            const response = await fetch(
                `${BACKEND_BASE_URL}/save-general-conf`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(settings),
                },
            );

            if (response.ok) {
                showSuccess("Settings have been saved!");
            } else {
                showDanger("Failed to save settings on the server.");
            }
        } catch (error) {
            console.error("Error saving settings:", error);
            showDanger("An error occurred while saving settings.");
        }
    };

    saveSettingsBtn.addEventListener("click", handleSaveClick);
}
