import { BACKEND_BASE_URL } from "../config.js";

const EMPTY_IMAGE_SRC =
    "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=";

/**
 * Return the human-friendly name for a camera without changing its identity.
 *
 * @param {{name?: string, display_name?: string | null, bus_id?: string}} camera - Camera record.
 * @returns {string} Operator-facing camera name.
 */
export function cameraDisplayName(camera) {
    return (
        camera.display_name?.trim() || camera.name || camera.bus_id || "Camera"
    );
}

/**
 * Build a thumbnail URL without changing the camera's stream identity.
 *
 * @param {{name?: string, stream_name?: string}} camera - Camera record.
 * @returns {string} Single-frame preview URL.
 */
export function cameraPreviewUrl(camera) {
    const streamName = camera.stream_name || camera.name || "";
    return `${BACKEND_BASE_URL}/feed/${encodeURIComponent(streamName)}?snapshot=1`;
}

/**
 * Render a responsive camera preview grid with rename and configure actions.
 *
 * Finite snapshots avoid tying up browser connections with one stream per camera.
 *
 * @param {HTMLElement} container - Element that receives the grid.
 * @param {{cameras: Array<object>, selectedBusId?: string, showRefresh?: boolean, onSelect?: (camera: object) => void, onRename: (camera: object, displayName: string) => Promise<{display_name?: string} | void>}} options - Grid behavior.
 * @returns {{destroy: () => void}} Lifecycle controller.
 */
export function mountCameraPreviewGrid(container, options) {
    const cameras = Array.isArray(options.cameras) ? options.cameras : [];
    const images = [];
    /** Release any thumbnail requests when the grid is replaced.
     * @returns {void}
     */
    function destroy() {
        for (const image of images) image.src = EMPTY_IMAGE_SRC;
    }

    container.replaceChildren();
    const grid = document.createElement("div");
    grid.className = "grid gap-3";
    // Settings lives in a narrow panel even on wide screens.
    grid.style.gridTemplateColumns =
        "repeat(auto-fit, minmax(min(100%, 220px), 1fr))";
    grid.setAttribute("aria-label", "Camera setup");

    for (const camera of cameras) {
        const name = cameraDisplayName(camera);
        const card = document.createElement("article");
        card.className =
            "overflow-hidden rounded-md border bg-[#171717] p-2 " +
            (String(camera.bus_id) === String(options.selectedBusId)
                ? "border-[#f9c845]"
                : "border-[#414141]");

        const image = document.createElement("img");
        image.className = "h-24 w-full rounded object-contain bg-black";
        image.alt = `Preview of ${name}`;
        image.dataset.previewSrc = cameraPreviewUrl(camera);
        image.src = image.dataset.previewSrc;
        images.push(image);

        const title = document.createElement("p");
        title.className = "mt-2 truncate text-sm font-semibold text-white";
        title.textContent = name;
        title.title = name;

        const technical = document.createElement("p");
        technical.className = "truncate text-xs text-[#ac8a2f]";
        technical.textContent = `bus_id: ${camera.bus_id}`;
        technical.title = `Technical camera: ${camera.name}`;

        const renameLabel = document.createElement("label");
        renameLabel.className = "mt-2 block! text-xs text-gray-300";
        renameLabel.textContent = "Placement description";
        const renameInput = document.createElement("input");
        renameInput.type = "text";
        renameInput.value = camera.display_name || "";
        renameInput.placeholder = "e.g. Front bumper";
        renameInput.maxLength = 80;
        renameInput.className =
            "mt-1 mr-0! w-full! rounded border border-[#414141] bg-[#232323] px-2 py-1 text-sm text-white focus:outline-none focus:ring-1 focus:ring-[#f9c845]";
        renameInput.setAttribute(
            "aria-label",
            `Placement description for ${name}`,
        );
        renameLabel.appendChild(renameInput);

        const actions = document.createElement("div");
        actions.className = "mt-2 flex gap-2";
        const renameButton = document.createElement("button");
        renameButton.type = "button";
        renameButton.className =
            "rounded border border-[#414141] px-2 py-1 text-xs text-gray-200 hover:border-[#f9c845] disabled:opacity-50";
        renameButton.textContent = "Save name";
        actions.appendChild(renameButton);
        let selectButton = null;
        if (options.onSelect) {
            selectButton = document.createElement("button");
            selectButton.type = "button";
            selectButton.className =
                "rounded bg-[#f9c845] px-2 py-1 text-xs font-semibold text-black hover:bg-[#d4a83a]";
            selectButton.textContent = "Configure";
            actions.appendChild(selectButton);
        }

        const status = document.createElement("p");
        status.className = "mt-1 min-h-4 text-xs text-red-300";
        status.setAttribute("aria-live", "polite");

        /**
         * Save a placement description before leaving this camera's card.
         * @returns {Promise<boolean>} Whether the save succeeded.
         */
        async function saveName() {
            const displayName = renameInput.value.trim();
            if (!displayName) {
                status.className = "mt-1 min-h-4 text-xs text-red-300";
                status.textContent = "Enter a placement description.";
                renameInput.focus();
                return false;
            }
            renameButton.disabled = true;
            renameInput.disabled = true;
            if (selectButton) selectButton.disabled = true;
            status.textContent = "";
            try {
                const saved = await options.onRename(camera, displayName);
                camera.display_name = saved?.display_name || displayName;
                title.textContent = cameraDisplayName(camera);
                title.title = cameraDisplayName(camera);
                image.alt = `Preview of ${cameraDisplayName(camera)}`;
                renameInput.value = camera.display_name;
                status.className = "mt-1 min-h-4 text-xs text-emerald-300";
                status.textContent = "Name saved.";
                return true;
            } catch (error) {
                status.className = "mt-1 min-h-4 text-xs text-red-300";
                status.textContent =
                    error.message || "Unable to save camera name.";
                return false;
            } finally {
                renameButton.disabled = false;
                renameInput.disabled = false;
                if (selectButton) selectButton.disabled = false;
            }
        }
        renameButton.addEventListener("click", () => void saveName());
        renameInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                event.preventDefault();
                void saveName();
            }
        });
        selectButton?.addEventListener("click", async () => {
            const draft = renameInput.value.trim();
            if (draft !== (camera.display_name || "") && !(await saveName())) {
                return;
            }
            options.onSelect(camera);
        });

        card.append(image, title, technical, renameLabel, actions, status);
        grid.appendChild(card);
    }
    container.appendChild(grid);

    if (options.showRefresh === false) return { destroy };

    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.textContent = "Refresh previews";
    refresh.className =
        "mt-3 rounded-md border border-[#414141] px-3 py-2 text-sm text-gray-200 hover:border-[#f9c845]";
    refresh.addEventListener("click", () => {
        for (const image of images) {
            image.src = `${image.dataset.previewSrc}&t=${Date.now()}`;
        }
    });
    container.appendChild(refresh);
    return { destroy };
}
