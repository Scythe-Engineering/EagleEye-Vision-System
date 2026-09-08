import { BACKEND_BASE_URL } from "../config.js";
import { mountCameraPreviewGrid } from "../ui/cameraPreviewGrid.js";

let dialog = null;
let cameraNameGrid = null;
let loadToken = 0;

/** Wire the camera-name editor once, loading previews only while open. */
export function initializeCameraNamesModal() {
    const opener = document.getElementById("manageCameraNamesBtn");
    if (!opener || dialog) return;
    dialog = document.createElement("dialog");
    dialog.id = "cameraNamesModal";
    dialog.className = "m-auto w-[calc(100%-2rem)] max-w-4xl max-h-[90vh] overflow-y-auto rounded-lg border border-[#414141] bg-[#1a1a1a] p-6 text-gray-200 shadow-xl backdrop:bg-black/60 backdrop:backdrop-blur-sm";
    dialog.setAttribute("aria-labelledby", "cameraNamesTitle");
    dialog.setAttribute("aria-describedby", "cameraNamesDescription");
    dialog.innerHTML = `
        <div class="mb-4 flex items-center justify-between gap-4">
            <h2 id="cameraNamesTitle" class="text-xl font-bold text-[#f9c845]">Camera Names</h2>
            <button type="button" id="closeCameraNamesBtn" autofocus class="rounded-md border border-[#414141] px-3 py-2 hover:border-[#f9c845]">Close</button>
        </div>
        <p id="cameraNamesDescription" class="mb-4 text-sm text-gray-300">Name cameras by where they are mounted. Hardware IDs and feed URLs stay unchanged.</p>
        <div id="settingsCameraNameGrid" aria-live="polite"></div>`;
    document.body.appendChild(dialog);
    dialog.querySelector("#closeCameraNamesBtn").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
        const bounds = dialog.getBoundingClientRect();
        if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
    });
    dialog.addEventListener("close", () => {
        ++loadToken;
        cameraNameGrid?.destroy();
        cameraNameGrid = null;
        dialog.querySelector("#settingsCameraNameGrid").replaceChildren();
        opener.focus();
    });
    document.addEventListener("backend-disconnected", () => dialog.close());
    opener.addEventListener("click", () => {
        dialog.querySelector("#settingsCameraNameGrid").textContent = "Loading cameras…";
        dialog.showModal();
        void loadCameraNameGrid();
    });
}

/**
 * Fetch active cameras and render their placement-name editor in the camera-name modal.
 *
 * @returns {Promise<void>} Resolves after the grid has been updated.
 */
async function loadCameraNameGrid() {
    const token = ++loadToken;
    const container = document.getElementById("settingsCameraNameGrid");
    if (!container) return;

    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/camera-config/cameras`,
        );
        const payload = await response.json();
        if (token !== loadToken || !dialog.open) return;
        if (!response.ok) {
            throw new Error(payload?.error || "Unable to load cameras");
        }
        cameraNameGrid?.destroy();
        cameraNameGrid = null;
        container.className = "text-sm text-gray-300";
        if (!payload.cameras?.length) {
            container.textContent =
                "No active cameras were found. Connect a camera and restart the backend.";
            return;
        }
        cameraNameGrid = mountCameraPreviewGrid(container, {
            cameras: Array.isArray(payload?.cameras) ? payload.cameras : [],
            onRename: async (camera, displayName) => {
                const saveResponse = await fetch(
                    `${BACKEND_BASE_URL}/camera-config/${encodeURIComponent(camera.bus_id)}/display-name`,
                    {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ display_name: displayName }),
                    },
                );
                const saved = await saveResponse.json();
                if (!saveResponse.ok) {
                    throw new Error(
                        saved?.error || "Unable to save camera name",
                    );
                }
                return saved;
            },
        });
    } catch (error) {
        if (token !== loadToken || !dialog.open) return;
        cameraNameGrid?.destroy();
        cameraNameGrid = null;
        container.textContent = `Unable to load cameras: ${error.message}`;
        container.className = "text-sm text-red-300";
    }
}

