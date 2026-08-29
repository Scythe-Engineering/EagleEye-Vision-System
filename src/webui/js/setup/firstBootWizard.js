import { BACKEND_BASE_URL } from "../config.js";
import { registerModelLibraryModal } from "../pipeline/modelLibraryModal.js";
import { activateAppView } from "../ui/sidebar.js";
import { showDanger, showSuccess } from "../ui/notificationSystem.js";
import { openCameraCalibration } from "../utils/cameraConfigUtils.js";
import { buildGenerationPayload, upsertCameraSetup } from "./wizardState.js";

const VERIFICATION_SESSION_KEY = "eagleeye-first-boot-verification";
const EXTRINSICS_FIELDS = [
    ["pitch", "Pitch (degrees)"],
    ["yaw", "Yaw (degrees)"],
    ["roll", "Roll (degrees)"],
    ["x_offset", "Forward offset (m)"],
    ["y_offset", "Left offset (m)"],
    ["z_offset", "Height offset (m)"],
];

const state = {
    status: null,
    cameras: [],
    setups: [],
    currentCamera: null,
    selectedModel: null,
    verificationTimer: null,
};

/**
 * Fetch JSON and translate backend errors into readable exceptions.
 *
 * @param {string} path - Backend path.
 * @param {RequestInit} [options] - Fetch options.
 * @returns {Promise<object>} Parsed response payload.
 */
async function fetchJson(path, options = {}) {
    const response = await fetch(`${BACKEND_BASE_URL}${path}`, options);
    let payload = null;
    try {
        payload = await response.json();
    } catch {
        payload = null;
    }
    if (!response.ok) {
        throw new Error(
            payload?.error ||
                payload?.message ||
                `Request failed: ${response.status}`,
        );
    }
    return payload || {};
}

/**
 * Create a DOM element with optional text and classes.
 *
 * @param {string} tagName - Element tag name.
 * @param {object} [options] - Element properties.
 * @returns {HTMLElement} New element.
 */
function element(tagName, options = {}) {
    const node = document.createElement(tagName);
    if (options.text !== undefined) node.textContent = options.text;
    if (options.className) node.className = options.className;
    if (options.type) node.type = options.type;
    if (options.id) node.id = options.id;
    return node;
}

/**
 * Return the wizard content mount point.
 *
 * @returns {HTMLElement} Wizard content element.
 */
function contentElement() {
    return document.getElementById("firstBootContent");
}

/**
 * Replace wizard content and update the progress label.
 *
 * @param {string} progress - Human-readable step label.
 * @returns {HTMLElement} Empty content element.
 */
function beginStep(progress) {
    const content = contentElement();
    content.replaceChildren();
    content.tabIndex = -1;
    document.getElementById("firstBootProgress").textContent = progress;
    queueMicrotask(() => content.focus());
    return content;
}

/**
 * Build a standard wizard action button.
 *
 * @param {string} label - Visible label.
 * @param {Function} onClick - Click handler.
 * @param {boolean} [primary=false] - Whether to use primary styling.
 * @returns {HTMLButtonElement} Configured button.
 */
function actionButton(label, onClick, primary = false) {
    const button = element("button", {
        text: label,
        type: "button",
        className: primary
            ? "rounded-md bg-[#f9c845] px-4 py-2 font-semibold text-black hover:bg-[#d4a83a] focus:outline-none focus:ring-2 focus:ring-white disabled:cursor-not-allowed disabled:opacity-50"
            : "rounded-md border border-[#414141] px-4 py-2 text-gray-200 hover:border-[#f9c845] focus:outline-none focus:ring-2 focus:ring-[#f9c845] disabled:cursor-not-allowed disabled:opacity-50",
    });
    button.addEventListener("click", onClick);
    return button;
}

/**
 * Render the first wizard screen.
 */
function renderWelcome() {
    const content = beginStep("Welcome");
    content.append(
        element("h2", {
            text: "Set up real camera pipelines",
            className: "mb-3 text-xl font-bold text-white",
        }),
        element("p", {
            text: "The wizard repeats calibration and pipeline choices for each camera, then restarts EagleEye once to run them.",
            className: "mb-5 max-w-2xl text-gray-300",
        }),
    );
    const actions = element("div", { className: "flex flex-wrap gap-3" });
    actions.append(
        actionButton("Start setup", renderCameraSelection, true),
        actionButton("Skip for now", () => void skipWizard()),
    );
    content.append(actions);
}

/**
 * Return cameras not already saved in this wizard run.
 *
 * @returns {Array<object>} Remaining active cameras.
 */
function remainingCameras() {
    const configured = new Set(state.setups.map((setup) => setup.bus_id));
    return state.cameras.filter((camera) => !configured.has(camera.bus_id));
}

/**
 * Render the active camera picker.
 */
function renderCameraSelection() {
    const content = beginStep(
        `Camera ${state.setups.length + 1}: choose camera`,
    );
    content.append(
        element("h2", {
            text: "Choose a camera",
            className: "mb-3 text-xl font-bold text-white",
        }),
    );
    const cameras = remainingCameras();
    if (!cameras.length) {
        content.append(
            element("p", {
                text: state.cameras.length
                    ? "Every active camera is configured."
                    : "No active cameras were found. Connect a camera, then refresh.",
                className: "mb-4 text-gray-300",
            }),
        );
        const actions = element("div", { className: "flex gap-3" });
        actions.append(
            actionButton(
                "Refresh cameras",
                () => void refreshCameraSelection(),
            ),
        );
        if (state.setups.length) {
            actions.append(actionButton("Continue", renderNetworkTables, true));
        }
        content.append(actions);
        return;
    }

    const label = element("label", {
        text: "Active camera",
        className: "mb-2 block font-semibold text-[#f9c845]",
    });
    const select = element("select", {
        className:
            "mb-5 w-full max-w-xl rounded-md border border-[#414141] bg-[#232323] px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
    });
    select.setAttribute("aria-label", "Active camera");
    cameras.forEach((camera) => {
        const option = document.createElement("option");
        option.value = camera.bus_id;
        option.textContent = `${camera.name} (${camera.bus_id})`;
        select.appendChild(option);
    });
    content.append(label, select);
    content.append(
        actionButton(
            "Continue to calibration",
            () => {
                state.currentCamera = cameras.find(
                    (camera) => camera.bus_id === select.value,
                );
                state.selectedModel = null;
                void renderCalibration();
            },
            true,
        ),
    );
}

/**
 * Render and refresh the intrinsics calibration step.
 */
async function renderCalibration() {
    const content = beginStep(
        `Camera ${state.setups.length + 1}: intrinsics calibration`,
    );
    content.append(
        element("h2", {
            text: `${state.currentCamera.name}: camera calibration`,
            className: "mb-3 text-xl font-bold text-white",
        }),
        element("p", {
            text: "Capture a ChArUco board across the whole image. EagleEye needs saved intrinsics before it can generate this pipeline.",
            className: "mb-4 max-w-2xl text-gray-300",
        }),
    );
    const status = element("p", {
        text: "Checking calibration...",
        className: "mb-4 text-[#ac8a2f]",
    });
    const continueButton = actionButton(
        "Continue to camera position",
        renderExtrinsics,
        true,
    );
    continueButton.disabled = true;
    const refresh = async () => {
        try {
            const cameraConfig = await fetchJson(
                `/camera-config/${encodeURIComponent(state.currentCamera.bus_id)}`,
            );
            continueButton.disabled = !cameraConfig.intrinsics_exists;
            status.textContent = cameraConfig.intrinsics_exists
                ? "Intrinsics saved."
                : "Calibration is still required.";
        } catch (error) {
            status.textContent = error.message;
        }
    };
    const actions = element("div", { className: "flex flex-wrap gap-3" });
    actions.append(
        actionButton("Open calibration", async () => {
            try {
                await openCameraCalibration(state.currentCamera.bus_id);
            } catch (error) {
                showDanger(error.message);
            }
        }),
        actionButton("Refresh status", () => void refresh()),
        continueButton,
    );
    content.append(status, actions);
    await refresh();
}

/**
 * Render and save camera mounting extrinsics.
 */
async function renderExtrinsics() {
    const content = beginStep(
        `Camera ${state.setups.length + 1}: camera position`,
    );
    content.append(
        element("h2", {
            text: `${state.currentCamera.name}: position on robot`,
            className: "mb-3 text-xl font-bold text-white",
        }),
        element("p", {
            text: "Measure from the robot origin to the camera lens. Rotations are in degrees and offsets are in meters.",
            className: "mb-4 max-w-2xl text-gray-300",
        }),
    );
    const form = element("form", {
        className: "grid max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2",
    });
    const inputs = {};
    EXTRINSICS_FIELDS.forEach(([name, labelText]) => {
        const wrapper = element("label", {
            text: labelText,
            className: "block text-sm font-semibold text-[#f9c845]",
        });
        const input = element("input", {
            className:
                "mt-1 w-full rounded-md border border-[#414141] bg-[#232323] px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
        });
        input.type = "number";
        input.step = "0.01";
        input.required = true;
        wrapper.appendChild(input);
        form.appendChild(wrapper);
        inputs[name] = input;
    });
    content.append(form);

    try {
        const cameraConfig = await fetchJson(
            `/camera-config/${encodeURIComponent(state.currentCamera.bus_id)}`,
        );
        EXTRINSICS_FIELDS.forEach(([name]) => {
            inputs[name].value = String(cameraConfig.extrinsics?.[name] ?? 0);
        });
    } catch (error) {
        showDanger(error.message);
        return;
    }

    const saveButton = actionButton(
        "Save camera position",
        async () => {
            if (!form.reportValidity()) return;
            const extrinsics = Object.fromEntries(
                EXTRINSICS_FIELDS.map(([name]) => [
                    name,
                    Number.parseFloat(inputs[name].value),
                ]),
            );
            saveButton.disabled = true;
            try {
                await fetchJson(
                    `/camera-config/${encodeURIComponent(state.currentCamera.bus_id)}/extrinsics`,
                    {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(extrinsics),
                    },
                );
                renderPurpose();
            } catch (error) {
                showDanger(error.message);
                saveButton.disabled = false;
            }
        },
        true,
    );
    saveButton.classList.add("mt-5");
    content.append(saveButton);
}

/**
 * Render localization/detection choices and the reusable model-library prompt.
 */
function renderPurpose() {
    const content = beginStep(
        `Camera ${state.setups.length + 1}: pipeline purpose`,
    );
    content.append(
        element("h2", {
            text: `${state.currentCamera.name}: choose what it does`,
            className: "mb-3 text-xl font-bold text-white",
        }),
    );
    const fieldset = element("fieldset", { className: "mb-5 space-y-3" });
    fieldset.appendChild(
        element("legend", {
            text: "Pipeline purpose",
            className: "mb-2 font-semibold text-[#f9c845]",
        }),
    );
    const choices = [
        [
            "localize",
            "Localize",
            "AprilTags, robot pose, and NetworkTables pose",
        ],
        ["detect", "Detect", "Game pieces in field space"],
        ["both", "Both", "Robot pose and game-piece detection together"],
    ];
    choices.forEach(([value, title, description], index) => {
        const label = element("label", {
            className:
                "flex cursor-pointer items-start gap-3 rounded-md border border-[#414141] p-3 hover:border-[#f9c845]",
        });
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = "firstBootPurpose";
        radio.value = value;
        radio.checked = index === 2;
        radio.className = "mt-1 accent-[#f9c845]";
        const copy = element("span");
        copy.append(
            element("strong", { text: title, className: "block text-white" }),
            element("span", {
                text: description,
                className: "text-sm text-gray-400",
            }),
        );
        label.append(radio, copy);
        fieldset.appendChild(label);
    });
    content.append(fieldset);

    const modelPanel = element("div", {
        className: "mb-5 rounded-md border border-[#414141] bg-[#171717] p-4",
    });
    const modelStatus = element("p", {
        text: "No model selected. A Both pipeline idles until a compatible model reaches the library.",
        className: "mb-3 text-sm text-[#ac8a2f]",
    });
    const manageModelButton = actionButton("Upload or choose a model", () => {
        registerModelLibraryModal().open({
            selectedModelId: state.selectedModel?.id,
            onSelect: async (model) => {
                modelStatus.textContent = "Checking CPU compatibility...";
                try {
                    await fetchJson(
                        `/model-library/${encodeURIComponent(model.id)}/resolve?device_id=cpu`,
                    );
                    state.selectedModel = model;
                    modelStatus.textContent = `Selected model: ${model.display_name || model.id}`;
                } catch (error) {
                    state.selectedModel = null;
                    modelStatus.textContent = `Model cannot run on CPU: ${error.message}`;
                }
            },
        });
    });
    modelPanel.append(
        element("h3", {
            text: "Detection model",
            className: "mb-2 font-semibold text-[#f9c845]",
        }),
        modelStatus,
        manageModelButton,
    );
    content.append(modelPanel);

    const purposeStatus = element("p", {
        className: "mb-3 text-sm text-red-300",
    });
    const updateModelVisibility = () => {
        const mode = fieldset.querySelector("input:checked")?.value;
        modelPanel.classList.toggle("hidden", mode === "localize");
        purposeStatus.textContent = "";
        if (!state.selectedModel) {
            modelStatus.textContent =
                mode === "detect"
                    ? "Detect-only requires a compatible CPU model."
                    : "No model selected. A Both pipeline idles until a compatible model reaches the library.";
        }
    };
    fieldset.addEventListener("change", updateModelVisibility);
    updateModelVisibility();

    content.append(
        purposeStatus,
        actionButton(
            "Save this camera",
            () => {
                const mode = fieldset.querySelector("input:checked")?.value;
                if (mode === "detect" && !state.selectedModel) {
                    purposeStatus.textContent =
                        "Choose a compatible model for detect-only mode.";
                    manageModelButton.focus();
                    return;
                }
                state.setups = upsertCameraSetup(state.setups, {
                    bus_id: state.currentCamera.bus_id,
                    name: state.currentCamera.name,
                    mode,
                    model_id:
                        mode === "localize"
                            ? ""
                            : state.selectedModel?.id || "",
                });
                renderCameraSummary();
            },
            true,
        ),
    );
}

/**
 * Render completed cameras and choose whether to repeat the camera steps.
 */
function renderCameraSummary() {
    const content = beginStep(`${state.setups.length} camera setup(s) ready`);
    content.append(
        element("h2", {
            text: "Configured cameras",
            className: "mb-3 text-xl font-bold text-white",
        }),
    );
    const list = element("ul", { className: "mb-5 space-y-2" });
    state.setups.forEach((setup) => {
        list.appendChild(
            element("li", {
                text: `${setup.name}: ${setup.mode}${setup.model_id ? " with model" : setup.mode === "localize" ? "" : " (model slot empty)"}`,
                className: "rounded-md border border-[#414141] px-3 py-2",
            }),
        );
    });
    const actions = element("div", { className: "flex flex-wrap gap-3" });
    if (remainingCameras().length) {
        actions.append(
            actionButton("Add another camera", renderCameraSelection),
        );
    }
    actions.append(
        actionButton("Continue to NetworkTables", renderNetworkTables, true),
    );
    content.append(list, actions);
}

/**
 * Render the global NetworkTables address and final generation action.
 */
async function renderNetworkTables() {
    const content = beginStep("NetworkTables and generation");
    content.append(
        element("h2", {
            text: "Connect to the robot",
            className: "mb-3 text-xl font-bold text-white",
        }),
        element("p", {
            text: "Enter the roboRIO or simulation server address. EagleEye restarts once after it saves the generated pipelines.",
            className: "mb-4 max-w-2xl text-gray-300",
        }),
    );
    const label = element("label", {
        text: "NetworkTables address",
        className: "block max-w-xl font-semibold text-[#f9c845]",
    });
    const input = element("input", {
        className:
            "mt-1 w-full rounded-md border border-[#414141] bg-[#232323] px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
    });
    input.type = "text";
    input.required = true;
    input.autocomplete = "off";
    try {
        const config = await fetchJson("/get-general-conf");
        input.value = config.network_table_address || "";
    } catch {
        input.value = "";
    }
    label.appendChild(input);
    content.append(label);
    const status = element("p", {
        className: "mt-4 text-sm text-[#ac8a2f]",
    });
    const generateButton = actionButton(
        "Generate and start pipelines",
        async () => {
            let requestBody;
            try {
                requestBody = buildGenerationPayload(state.setups, input.value);
            } catch (error) {
                status.textContent = error.message;
                return;
            }
            generateButton.disabled = true;
            status.textContent = "Generating pipelines...";
            try {
                const generated = await fetchJson("/first-boot/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(requestBody),
                });
                sessionStorage.setItem(VERIFICATION_SESSION_KEY, "1");
                status.textContent =
                    "Restarting EagleEye to start pipelines...";
                await restartAndReload(generated.runtime_id);
            } catch (error) {
                showDanger(error.message);
                status.textContent = error.message;
                generateButton.disabled = false;
            }
        },
        true,
    );
    generateButton.classList.add("mt-5");
    content.append(generateButton, status);
}

/**
 * Ask the existing backend restart endpoint to apply generated pipelines.
 *
 * @param {string} previousRuntimeId - Runtime identifier before restart.
 */
async function restartAndReload(previousRuntimeId) {
    try {
        await fetchJson("/restart-backend", { method: "POST" });
    } catch {
        // The successful restart normally closes this request mid-response.
    }
    const deadline = Date.now() + 45000;
    while (Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 750));
        try {
            const runtime = await fetchJson("/get_restart_required");
            if (
                !previousRuntimeId ||
                runtime.runtime_id !== previousRuntimeId
            ) {
                const url = new URL(globalThis.location.href);
                url.searchParams.set("tab", "view-3d");
                globalThis.location.assign(url.toString());
                return;
            }
        } catch {
            // Backend is still restarting.
        }
    }
    throw new Error("Backend restart timed out. Check Settings > System Logs.");
}

/**
 * Persist first-boot skip and return to the camera view.
 */
async function skipWizard() {
    try {
        await fetchJson("/first-boot/skip", { method: "POST" });
        state.status.required = false;
        activateAppView("view-views");
        showSuccess("First-boot setup skipped. Reopen it from Settings.");
    } catch (error) {
        showDanger(error.message);
    }
}

/**
 * Refresh active cameras without discarding completed camera steps.
 */
async function refreshCameraSelection() {
    try {
        const status = await fetchJson("/first-boot/status");
        state.status = status;
        state.cameras = status.cameras || [];
        renderCameraSelection();
    } catch (error) {
        showDanger(`Unable to refresh cameras: ${error.message}`);
    }
}

/**
 * Fetch active cameras and open a fresh wizard run.
 */
async function openWizard() {
    try {
        state.status = await fetchJson("/first-boot/status");
        state.cameras = state.status.cameras || [];
        state.setups = [];
        state.currentCamera = null;
        state.selectedModel = null;
        activateAppView("view-first-boot");
        renderWelcome();
    } catch (error) {
        showDanger(`Unable to open setup: ${error.message}`);
    }
}

/**
 * Render live verification state in the existing 3D field view.
 *
 * @param {object} status - First-boot status payload.
 */
function renderVerification(status) {
    const networkElement = document.getElementById(
        "firstBootVerifyNetworkTables",
    );
    const pipelinesElement = document.getElementById(
        "firstBootVerifyPipelines",
    );
    const keysElement = document.getElementById("firstBootVerifyKeys");
    const hintElement = document.getElementById("firstBootVerifyHint");
    const pipelines = status.pipelines || [];
    const keys = status.networktable_keys || [];
    const activePipelines = pipelines.filter(
        (pipeline) => pipeline.active,
    ).length;
    networkElement.textContent = status.network_table?.connected
        ? "Connected"
        : "Disconnected";
    networkElement.className = status.network_table?.connected
        ? "text-emerald-300"
        : "text-red-300";
    pipelinesElement.textContent = `${activePipelines}/${pipelines.length} active`;
    pipelinesElement.className =
        pipelines.length && activePipelines === pipelines.length
            ? "text-emerald-300"
            : "text-red-300";
    keysElement.replaceChildren();
    keys.forEach((key) => {
        const item = element("li", {
            text: key.present
                ? `Present: EagleEye/${key.key}`
                : key.required
                  ? `Waiting: EagleEye/${key.key}`
                  : `Idle model slot: EagleEye/${key.key}`,
            className: key.present
                ? "text-emerald-300"
                : key.required
                  ? "text-red-300"
                  : "text-[#ac8a2f]",
        });
        keysElement.appendChild(item);
    });
    const allRequiredKeysPresent = keys
        .filter((key) => key.required)
        .every((key) => key.present);
    const anyKeyPresent = keys.some((key) => key.present);
    const checksPass = Boolean(
        status.network_table?.connected &&
        pipelines.length > 0 &&
        activePipelines === pipelines.length &&
        allRequiredKeysPresent &&
        anyKeyPresent,
    );
    document.getElementById("firstBootVerifyFinishBtn").disabled = !checksPass;
    hintElement.textContent = checksPass
        ? "Live checks pass. Confirm the robot pose is correct on the field."
        : "Keep an AprilTag in view and check the address, camera calibration, and pipeline logs.";
}

/**
 * Refresh the final live verification checks.
 */
async function refreshVerification() {
    try {
        const status = await fetchJson("/first-boot/status");
        renderVerification(status);
    } catch (error) {
        document.getElementById("firstBootVerifyHint").textContent =
            error.message;
    }
}

/**
 * Open the final verification panel over the existing live 3D view.
 */
function openVerification() {
    clearInterval(state.verificationTimer);
    activateAppView("view-3d");
    const panel = document.getElementById("firstBootVerificationPanel");
    panel.classList.remove("hidden");
    panel.focus();
    void refreshVerification();
    state.verificationTimer = setInterval(refreshVerification, 2000);
}

/**
 * Wire first-boot routing, Settings reopening, and final verification.
 */
export async function initializeFirstBootWizard() {
    document
        .getElementById("openFirstBootWizardBtn")
        ?.addEventListener("click", () => void openWizard());
    document
        .getElementById("firstBootCloseBtn")
        ?.addEventListener("click", () => {
            activateAppView("view-settings");
        });
    document
        .getElementById("firstBootVerifyRetryBtn")
        ?.addEventListener("click", () => void refreshVerification());
    document
        .getElementById("firstBootVerifySetupBtn")
        ?.addEventListener("click", () => {
            clearInterval(state.verificationTimer);
            document
                .getElementById("firstBootVerificationPanel")
                .classList.add("hidden");
            void openWizard();
        });
    document
        .getElementById("firstBootVerifyFinishBtn")
        ?.addEventListener("click", async () => {
            try {
                await fetchJson("/first-boot/finish", { method: "POST" });
                clearInterval(state.verificationTimer);
                sessionStorage.removeItem(VERIFICATION_SESSION_KEY);
                document
                    .getElementById("firstBootVerificationPanel")
                    .classList.add("hidden");
                activateAppView("view-views");
            } catch (error) {
                showDanger(`Unable to finish setup: ${error.message}`);
            }
        });

    try {
        state.status = await fetchJson("/first-boot/status");
        state.cameras = state.status.cameras || [];
        if (
            state.status.verification_pending ||
            (sessionStorage.getItem(VERIFICATION_SESSION_KEY) === "1" &&
                state.status.completed)
        ) {
            openVerification();
        } else if (
            state.status.required ||
            new URL(globalThis.location.href).searchParams.get("tab") ===
                "view-first-boot"
        ) {
            activateAppView("view-first-boot");
            renderWelcome();
        }
    } catch (error) {
        console.error("Unable to load first-boot state:", error);
    }
}
