import { BACKEND_BASE_URL } from "../config.js";

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
        
        if (robotAddressInput && settings.network_table_address) {
            robotAddressInput.value = settings.network_table_address;
        }
    } catch (error) {
        console.error("Error loading settings:", error);
    }
}

export function saveSettings() {
    const saveSettingsBtn = document.getElementById("saveSettingsBtn");
    if (!saveSettingsBtn) {
        return;
    }

    saveSettingsBtn.addEventListener("click", async () => {
        const robotAddressInput = document.getElementById("robotAddressInput");
        if (!robotAddressInput) {
            return;
        }

        const settings = {
            network_table_address: robotAddressInput.value,
        };

        try {
            const response = await fetch(`${BACKEND_BASE_URL}/save-general-conf`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(settings),
            });

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

