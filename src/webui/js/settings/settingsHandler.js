import { BACKEND_BASE_URL } from "../config.js";
import { setNetworkTablesConnected } from "../ui/connectionStatus.js";

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
        
        const updateSystemBtn = document.getElementById("updateSystemBtn");
        if (updateSystemBtn) {
            if (status.internet_connected === true) {
                updateSystemBtn.disabled = false;
                updateSystemBtn.title = "Update System";
            } else {
                updateSystemBtn.disabled = true;
                updateSystemBtn.title = "No internet access available";
            }
        }
    } catch (error) {
        console.error("Error loading NetworkTables status:", error);
        renderNetworkTableStatus(null);
        setNetworkTablesConnected(false);
    }
}

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

export function saveSettings() {
    const saveSettingsBtn = document.getElementById("saveSettingsBtn");
    if (!saveSettingsBtn) {
        return;
    }

    saveSettingsBtn.addEventListener("click", async () => {
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
            alert("Stream downscale must be between 0.1 and 1.");
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
                alert("Settings have been saved!");
            } else {
                alert("Failed to save settings on the server.");
            }
        } catch (error) {
            console.error("Error saving settings:", error);
            alert("An error occurred while saving settings.");
        }
    });
}

window.showUpdateConfirmation = function() {
    const modal = document.getElementById('updateModal');
    if (modal) {
        modal.classList.remove('hidden');
    }
};

window.closeUpdateModal = function() {
    const modal = document.getElementById('updateModal');
    if (modal) {
        modal.classList.add('hidden');
    }
};

window.startSystemUpdate = async function() {
    const content = document.getElementById('updateModalContent');
    const progress = document.getElementById('updateModalProgress');
    const actions = document.getElementById('updateModalActions');
    
    if (content) content.classList.add('hidden');
    if (progress) progress.classList.remove('hidden');
    if (actions) actions.classList.add('hidden');
    
    try {
        const response = await fetch(`${BACKEND_BASE_URL}/update-system`, {
            method: 'POST',
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Polling loop to wait for backend to come back up
        const MAX_RETRIES = 400; // Up to 400 seconds
        const checkBackend = async (attempt) => {
            if (attempt >= MAX_RETRIES) {
                console.error("Backend failed to come back online after update.");
                alert("Update completed, but the system is taking too long to restart. Please refresh manually.");
                window.closeUpdateModal();
                if (content) content.classList.remove('hidden');
                if (progress) progress.classList.add('hidden');
                if (actions) actions.classList.remove('hidden');
                return;
            }
            try {
                const checkResponse = await fetch(`${BACKEND_BASE_URL}/get-system-status`, { method: "GET" });
                if (checkResponse.ok) {
                    window.location.reload();
                    return;
                }
            } catch (e) {
                // Network error, backend is likely down. We will retry.
            }
            setTimeout(() => checkBackend(attempt + 1), 1000);
        };
        setTimeout(() => checkBackend(0), 3000); // Give it some time to go down
    } catch (error) {
        console.error("Error updating system:", error);
        alert("Failed to initiate system update.");
        window.closeUpdateModal();
        if (content) content.classList.remove('hidden');
        if (progress) progress.classList.add('hidden');
        if (actions) actions.classList.remove('hidden');
    }
};
