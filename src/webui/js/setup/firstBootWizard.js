import { BACKEND_BASE_URL } from "../config.js";
import { registerModelLibraryModal } from "../pipeline/modelLibraryModal.js";
import { activateAppView } from "../ui/sidebar.js";
import { showDanger, showSuccess } from "../ui/notificationSystem.js";
import {
    saveExtrinsics,
    selectCameraConfig,
} from "../utils/cameraConfigUtils.js";
import {
    WIZARD_STEP,
    buildGenerationPayload,
    guidedStepCopy,
    isGuidedStep,
    nextWizardStep,
    previousWizardStep,
    upsertCameraSetup,
    wizardStepView,
} from "./wizardState.js";

const VERIFICATION_SESSION_KEY = "eagleeye-first-boot-verification";
const WIZARD_SESSION_KEY = "eagleeye-first-boot-wizard";

const state = {
    status: null,
    cameras: [],
    setups: [],
    currentCamera: null,
    selectedModel: null,
    step: WIZARD_STEP.WELCOME,
    verificationTimer: null,
    verificationInFlight: false,
    guideToken: 0,
    modelResolveToken: 0,
    guideTarget: null,
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
 * Persist in-progress wizard state for this browser tab.
 */
function persistSession() {
    sessionStorage.setItem(
        WIZARD_SESSION_KEY,
        JSON.stringify({
            setups: state.setups,
            currentCamera: state.currentCamera,
            selectedModel: state.selectedModel,
            step: state.step,
        }),
    );
}

/**
 * Clear persisted wizard and verification session keys.
 */
function clearWizardSession() {
    sessionStorage.removeItem(WIZARD_SESSION_KEY);
}

/**
 * Restore in-progress wizard state if this tab still has a run.
 *
 * @returns {string | null} Saved step, if any.
 */
function restoreSession() {
    try {
        const raw = sessionStorage.getItem(WIZARD_SESSION_KEY);
        if (!raw) return null;
        const saved = JSON.parse(raw);
        if (!saved?.step) return null;
        state.setups = Array.isArray(saved.setups) ? saved.setups : [];
        state.currentCamera = saved.currentCamera || null;
        state.selectedModel = saved.selectedModel || null;
        return saved.step;
    } catch {
        return null;
    }
}

/**
 * Remove the visual highlight from the last guided target.
 */
function clearGuideTarget() {
    if (state.guideTarget) {
        state.guideTarget.classList.remove("first-boot-guide-target");
        state.guideTarget.removeAttribute("data-wizard-hint");
        state.guideTarget = null;
    }
}

/**
 * Highlight the control the user should use on the current app page.
 *
 * @param {HTMLElement | null} target - Element to emphasize.
 */
function highlightGuideTarget(target) {
    clearGuideTarget();
    if (!target) return;
    state.guideTarget = target;
    target.classList.add("first-boot-guide-target");
    target.dataset.wizardHint = "Press me!";
    target.scrollIntoView({ block: "center", behavior: "smooth" });
}

/**
 * Hide the persistent first-boot guide overlay.
 */
function hideGuide() {
    clearGuideTarget();
    const panel = document.getElementById("firstBootGuidePanel");
    panel?.classList.add("hidden");
    const continueButton = document.getElementById("firstBootGuideContinueBtn");
    if (continueButton) continueButton.disabled = false;
}

/**
 * Show a status line inside the guide overlay.
 *
 * @param {string} message - Status text.
 * @param {boolean} [isError=false] - Whether to style the message as an error.
 */
function setGuideStatus(message, isError = false) {
    const status = document.getElementById("firstBootGuideStatus");
    if (!status) return;
    status.textContent = message;
    status.classList.toggle("hidden", !message);
    status.classList.toggle("text-red-300", isError);
    status.classList.toggle("text-[#ac8a2f]", !isError);
}

/**
 * Fill and reveal the persistent guide overlay.
 *
 * @param {{progress: string, title: string, instructions: string}} copy - Overlay copy.
 */
function showGuide(copy) {
    document.getElementById("firstBootGuideProgress").textContent =
        copy.progress;
    document.getElementById("firstBootGuideTitle").textContent = copy.title;
    document.getElementById("firstBootGuideInstructions").textContent =
        copy.instructions;
    setGuideStatus("");
    const panel = document.getElementById("firstBootGuidePanel");
    panel.classList.remove("hidden");
    panel.focus();
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
            text: "Choose each camera here, then the wizard opens the existing Camera Config and Settings pages for calibration, mounting, and NetworkTables. A guide stays on screen with Cancel and Continue.",
            className: "mb-5 max-w-2xl text-gray-300",
        }),
    );
    const actions = element("div", { className: "flex flex-wrap gap-3" });
    actions.append(
        actionButton(
            "Start setup",
            () => goToStep(WIZARD_STEP.CAMERA_SELECT),
            true,
        ),
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
            actions.append(
                actionButton(
                    "Continue",
                    () => goToStep(WIZARD_STEP.CAMERA_SUMMARY),
                    true,
                ),
            );
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
    if (
        state.currentCamera &&
        cameras.some((camera) => camera.bus_id === state.currentCamera.bus_id)
    ) {
        select.value = state.currentCamera.bus_id;
    }
    content.append(label, select);
    content.append(
        actionButton(
            "Continue to calibration",
            () => {
                state.currentCamera = cameras.find(
                    (camera) => camera.bus_id === select.value,
                );
                state.selectedModel = null;
                goToStep(WIZARD_STEP.CALIBRATION);
            },
            true,
        ),
    );
}

/**
 * Ask what the selected camera should do, including an optional detection model.
 */
function renderPurpose() {
    const cameraName = state.currentCamera?.name || "this camera";
    const content = beginStep(
        `Camera ${state.setups.length + 1}: pipeline purpose`,
    );
    content.append(
        element("h2", {
            text: `What should ${cameraName} do?`,
            className: "mb-3 text-xl font-bold text-white",
        }),
        element("p", {
            text: "Calibration and mounting are done on Camera Config Utils. Choose the pipeline this camera should run.",
            className: "mb-4 max-w-2xl text-gray-300",
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
    const existing = state.setups.find(
        (setup) => setup.bus_id === state.currentCamera?.bus_id,
    );
    const selectedMode = existing?.mode || "both";
    choices.forEach(([value, title, description]) => {
        const label = element("label", {
            className:
                "flex cursor-pointer items-start gap-3 rounded-md border border-[#414141] p-3 hover:border-[#f9c845]",
        });
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = "firstBootPurpose";
        radio.value = value;
        radio.checked = value === selectedMode;
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
                const requestId = ++state.modelResolveToken;
                modelStatus.textContent = "Checking CPU compatibility...";
                try {
                    await fetchJson(
                        `/model-library/${encodeURIComponent(model.id)}/resolve?device_id=cpu`,
                    );
                    if (requestId !== state.modelResolveToken) return;
                    state.selectedModel = model;
                    persistSession();
                    modelStatus.textContent = `Selected model: ${model.display_name || model.id}`;
                } catch (error) {
                    if (requestId !== state.modelResolveToken) return;
                    state.selectedModel = null;
                    persistSession();
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
        if (state.selectedModel) {
            modelStatus.textContent = `Selected model: ${state.selectedModel.display_name || state.selectedModel.id}`;
            return;
        }
        modelStatus.textContent =
            mode === "detect"
                ? "Detect-only requires a compatible CPU model."
                : "No model selected. A Both pipeline idles until a compatible model reaches the library.";
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
                goToStep(WIZARD_STEP.CAMERA_SUMMARY);
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
        element("p", {
            text: "Add another camera, or continue to Settings to set the NetworkTables address.",
            className: "mb-4 max-w-2xl text-gray-300",
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
            actionButton("Add another camera", () => {
                state.currentCamera = null;
                state.selectedModel = null;
                goToStep(WIZARD_STEP.CAMERA_SELECT);
            }),
        );
    }
    actions.append(
        actionButton(
            "Continue to NetworkTables",
            () => goToStep(WIZARD_STEP.NETWORK_TABLES),
            true,
        ),
    );
    content.append(list, actions);
}

/**
 * Show a guided overlay on an existing configuration page.
 *
 * @param {string} step - Guided wizard step.
 * @param {number} token - Generation token for the current goToStep call.
 */
async function showGuidedStep(step, token) {
    const cameraName = state.currentCamera?.name || "this camera";
    const copy = guidedStepCopy(step, cameraName, state.setups.length + 1);
    activateAppView(wizardStepView(step));
    showGuide(copy);

    if (step === WIZARD_STEP.CALIBRATION || step === WIZARD_STEP.EXTRINSICS) {
        if (!state.currentCamera?.bus_id) {
            goToStep(WIZARD_STEP.CAMERA_SELECT);
            return;
        }
        try {
            await selectCameraConfig(state.currentCamera.bus_id);
            if (state.guideToken !== token || state.step !== step) return;
            highlightGuideTarget(
                document.getElementById(
                    step === WIZARD_STEP.CALIBRATION
                        ? "utilsCalibrateIntrinsicsBtn"
                        : "utilsSaveExtrinsicsBtn",
                ),
            );
        } catch (error) {
            if (state.guideToken !== token || state.step !== step) return;
            setGuideStatus(error.message, true);
        }
        return;
    }

    if (state.guideToken !== token || state.step !== step) return;
    highlightGuideTarget(document.getElementById("saveSettingsBtn"));
}

/**
 * Route the wizard to a dedicated page or a guided app view.
 *
 * @param {string} step - Wizard step identifier.
 */
function goToStep(step) {
    state.step = step;
    const token = ++state.guideToken;
    persistSession();
    if (isGuidedStep(step)) {
        void showGuidedStep(step, token);
        return;
    }
    hideGuide();
    activateAppView("view-first-boot");
    switch (step) {
        case WIZARD_STEP.CAMERA_SELECT:
            renderCameraSelection();
            break;
        case WIZARD_STEP.PURPOSE:
            renderPurpose();
            break;
        case WIZARD_STEP.CAMERA_SUMMARY:
            renderCameraSummary();
            break;
        default:
            renderWelcome();
    }
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
 * Generate pipelines from the Settings page address and restart EagleEye.
 */
async function generatePipelines() {
    const continueButton = document.getElementById("firstBootGuideContinueBtn");
    const address = document.getElementById("robotAddressInput")?.value || "";
    let requestBody;
    try {
        requestBody = buildGenerationPayload(state.setups, address);
    } catch (error) {
        setGuideStatus(error.message, true);
        return;
    }
    if (continueButton) continueButton.disabled = true;
    setGuideStatus("Generating pipelines...");
    try {
        const generated = await fetchJson("/first-boot/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
        });
        sessionStorage.setItem(VERIFICATION_SESSION_KEY, "1");
        clearWizardSession();
        setGuideStatus("Restarting EagleEye to start pipelines...");
        await restartAndReload(generated.runtime_id);
    } catch (error) {
        showDanger(error.message);
        setGuideStatus(error.message, true);
        if (continueButton) continueButton.disabled = false;
    }
}

/**
 * Advance from a guided overlay, validating the current page when needed.
 */
async function handleGuideContinue() {
    if (state.step === WIZARD_STEP.CALIBRATION) {
        try {
            const cameraConfig = await fetchJson(
                `/camera-config/${encodeURIComponent(state.currentCamera.bus_id)}`,
            );
            if (!cameraConfig.intrinsics_exists) {
                setGuideStatus(
                    "Calibration is still required. Capture frames and click Calibrate & Save, or upload an intrinsics file.",
                );
                return;
            }
        } catch (error) {
            setGuideStatus(error.message, true);
            return;
        }
        goToStep(nextWizardStep(state.step));
        return;
    }
    if (state.step === WIZARD_STEP.EXTRINSICS) {
        if (!state.currentCamera?.bus_id) {
            setGuideStatus("Choose a camera before saving extrinsics.", true);
            return;
        }
        try {
            const selectedBusId =
                document.getElementById("utilsCameraSelect")?.value;
            if (selectedBusId !== state.currentCamera.bus_id) {
                await selectCameraConfig(state.currentCamera.bus_id);
                setGuideStatus(
                    "The selected camera changed. Enter this camera's extrinsics before continuing.",
                );
                return;
            }
            if (!(await saveExtrinsics())) {
                setGuideStatus("Save extrinsics before continuing.");
                return;
            }
        } catch (error) {
            setGuideStatus(error.message, true);
            return;
        }
        goToStep(nextWizardStep(state.step));
        return;
    }
    if (state.step === WIZARD_STEP.NETWORK_TABLES) {
        await generatePipelines();
    }
}

/**
 * Leave a guided page and return to the previous wizard-owned screen.
 */
function handleGuideCancel() {
    goToStep(previousWizardStep(state.step));
}

/**
 * Persist first-boot skip and return to the camera view.
 */
async function skipWizard() {
    try {
        await fetchJson("/first-boot/skip", { method: "POST" });
        state.status.required = false;
        clearWizardSession();
        hideGuide();
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
        clearWizardSession();
        goToStep(WIZARD_STEP.WELCOME);
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
 * Return whether the verification panel and its 3D host view are visible.
 *
 * @param {HTMLElement | null} panel - Verification panel element.
 * @returns {boolean} Whether polling may continue.
 */
function isVerificationPanelVisible(panel) {
    return Boolean(
        panel &&
        !panel.classList.contains("hidden") &&
        !document.getElementById("view-3d")?.classList.contains("hidden"),
    );
}

/**
 * Stop verification polling without interrupting an in-flight refresh.
 */
function stopVerificationPolling() {
    clearInterval(state.verificationTimer);
    state.verificationTimer = null;
}

/**
 * Refresh the final live verification checks.
 */
async function refreshVerification() {
    const panel = document.getElementById("firstBootVerificationPanel");
    if (!isVerificationPanelVisible(panel)) {
        stopVerificationPolling();
        return;
    }
    if (state.verificationInFlight) return;
    state.verificationInFlight = true;
    try {
        const status = await fetchJson("/first-boot/status");
        if (!isVerificationPanelVisible(panel)) return;
        renderVerification(status);
    } catch (error) {
        document.getElementById("firstBootVerifyHint").textContent =
            error.message;
    } finally {
        state.verificationInFlight = false;
    }
}

/**
 * Open the final verification panel over the existing live 3D view.
 */
function openVerification() {
    stopVerificationPolling();
    hideGuide();
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
            hideGuide();
            activateAppView("view-settings");
        });
    document
        .getElementById("firstBootGuideCancelBtn")
        ?.addEventListener("click", handleGuideCancel);
    document
        .getElementById("firstBootGuideContinueBtn")
        ?.addEventListener("click", () => void handleGuideContinue());
    document
        .getElementById("firstBootVerifyRetryBtn")
        ?.addEventListener("click", () => void refreshVerification());
    document.querySelector(".sidebar")?.addEventListener("click", () => {
        const panel = document.getElementById("firstBootVerificationPanel");
        if (!isVerificationPanelVisible(panel)) stopVerificationPolling();
    });
    document
        .getElementById("firstBootVerifySetupBtn")
        ?.addEventListener("click", () => {
            stopVerificationPolling();
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
                stopVerificationPolling();
                sessionStorage.removeItem(VERIFICATION_SESSION_KEY);
                clearWizardSession();
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
            const savedStep = restoreSession();
            goToStep(savedStep || WIZARD_STEP.WELCOME);
        }
    } catch (error) {
        console.error("Unable to load first-boot state:", error);
    }
}
