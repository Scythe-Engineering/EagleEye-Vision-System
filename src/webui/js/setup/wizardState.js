const CAMERA_MODES = new Set(["localize", "detect", "both"]);

export const WIZARD_STEP = {
    WELCOME: "welcome",
    CAMERA_SELECT: "camera-select",
    CALIBRATION: "calibration",
    EXTRINSICS: "extrinsics",
    PURPOSE: "purpose",
    CAMERA_SUMMARY: "camera-summary",
    NETWORK_TABLES: "network-tables",
};

const GUIDED_STEPS = new Set([
    WIZARD_STEP.CALIBRATION,
    WIZARD_STEP.EXTRINSICS,
    WIZARD_STEP.NETWORK_TABLES,
]);

/**
 * Add or replace one camera setup while preserving camera order.
 *
 * @param {Array<object>} setups - Current per-camera setup records.
 * @param {object} nextSetup - Completed setup for one camera.
 * @returns {Array<object>} Updated setup records.
 */
export function upsertCameraSetup(setups, nextSetup) {
    const index = setups.findIndex(
        (setup) => setup.bus_id === nextSetup.bus_id,
    );
    if (index < 0) return [...setups, nextSetup];
    return setups.map((setup, currentIndex) =>
        currentIndex === index ? nextSetup : setup,
    );
}

/**
 * Build and validate the backend generation payload.
 *
 * @param {Array<object>} setups - Completed per-camera setup records.
 * @param {string} networkTableAddress - NetworkTables server address.
 * @returns {{network_table_address: string, cameras: Array<object>}} Request body.
 */
export function buildGenerationPayload(setups, networkTableAddress) {
    const address = networkTableAddress.trim();
    if (!address) throw new Error("NetworkTables address is required.");
    if (!setups.length) throw new Error("Configure at least one camera.");

    const seen = new Set();
    const cameras = setups.map((setup) => {
        if (!setup.bus_id || seen.has(setup.bus_id)) {
            throw new Error("Each camera can be configured once.");
        }
        if (!CAMERA_MODES.has(setup.mode)) {
            throw new Error(`Unsupported camera mode: ${setup.mode}`);
        }
        seen.add(setup.bus_id);
        return {
            bus_id: setup.bus_id,
            mode: setup.mode,
            model_id: setup.mode === "localize" ? "" : setup.model_id || "",
        };
    });
    return { network_table_address: address, cameras };
}

/**
 * Return whether a wizard step overlays an existing app page.
 *
 * @param {string} step - Wizard step identifier.
 * @returns {boolean} True when the step uses a persistent guide overlay.
 */
export function isGuidedStep(step) {
    return GUIDED_STEPS.has(step);
}

/**
 * Return the application view that hosts a wizard step.
 *
 * @param {string} step - Wizard step identifier.
 * @returns {string} Mounted view element ID.
 */
export function wizardStepView(step) {
    if (step === WIZARD_STEP.CALIBRATION || step === WIZARD_STEP.EXTRINSICS) {
        return "view-utils";
    }
    if (step === WIZARD_STEP.NETWORK_TABLES) {
        return "view-settings";
    }
    return "view-first-boot";
}

/**
 * Return the next wizard step after the user continues.
 *
 * @param {string} step - Current wizard step.
 * @returns {string | null} Next step, or null when generation should run.
 */
export function nextWizardStep(step) {
    switch (step) {
        case WIZARD_STEP.WELCOME:
            return WIZARD_STEP.CAMERA_SELECT;
        case WIZARD_STEP.CAMERA_SELECT:
            return WIZARD_STEP.CALIBRATION;
        case WIZARD_STEP.CALIBRATION:
            return WIZARD_STEP.EXTRINSICS;
        case WIZARD_STEP.EXTRINSICS:
            return WIZARD_STEP.PURPOSE;
        case WIZARD_STEP.PURPOSE:
            return WIZARD_STEP.CAMERA_SUMMARY;
        case WIZARD_STEP.CAMERA_SUMMARY:
            return WIZARD_STEP.NETWORK_TABLES;
        default:
            return null;
    }
}

/**
 * Return the previous wizard step after the user cancels a guided page.
 *
 * @param {string} step - Current wizard step.
 * @returns {string} Previous step.
 */
export function previousWizardStep(step) {
    switch (step) {
        case WIZARD_STEP.CALIBRATION:
            return WIZARD_STEP.CAMERA_SELECT;
        case WIZARD_STEP.EXTRINSICS:
            return WIZARD_STEP.CALIBRATION;
        case WIZARD_STEP.PURPOSE:
            return WIZARD_STEP.EXTRINSICS;
        case WIZARD_STEP.CAMERA_SUMMARY:
            return WIZARD_STEP.PURPOSE;
        case WIZARD_STEP.NETWORK_TABLES:
            return WIZARD_STEP.CAMERA_SUMMARY;
        default:
            return WIZARD_STEP.WELCOME;
    }
}

/**
 * Return overlay copy for a guided configuration step.
 *
 * @param {string} step - Guided wizard step.
 * @param {string} cameraName - Human-readable camera name.
 * @param {number} cameraNumber - 1-based camera index in this wizard run.
 * @returns {{progress: string, title: string, instructions: string}} Overlay copy.
 */
export function guidedStepCopy(step, cameraName, cameraNumber) {
    if (step === WIZARD_STEP.CALIBRATION) {
        return {
            progress: `Camera ${cameraNumber}: calibration`,
            title: `Calibrate ${cameraName}`,
            instructions:
                "This camera is selected on Camera Config Utils. Click Calibrate Camera, capture a ChArUco board across the whole image, then Calibrate & Save. You can also upload an existing intrinsics JSON file. Click Continue when this camera has saved intrinsics.",
        };
    }
    if (step === WIZARD_STEP.EXTRINSICS) {
        return {
            progress: `Camera ${cameraNumber}: camera position`,
            title: `Set ${cameraName} on the robot`,
            instructions:
                "Enter pitch, yaw, roll, and the offsets from the robot origin to the camera lens, then click Save Extrinsics. Rotations are in degrees and offsets are in meters. Click Continue when the position is saved.",
        };
    }
    return {
        progress: "NetworkTables",
        title: "Connect to the robot",
        instructions:
            "Enter the roboRIO or simulation IP under Network Table, then click Save Settings. Click Continue when the address is set. EagleEye will generate pipelines and restart once.",
    };
}
