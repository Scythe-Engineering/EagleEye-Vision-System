import assert from "node:assert/strict";
import test from "node:test";

import {
    WIZARD_STEP,
    buildGenerationPayload,
    guidedStepCopy,
    isGuidedStep,
    nextWizardStep,
    previousWizardStep,
    upsertCameraSetup,
    wizardStepView,
} from "../../src/webui/js/setup/wizardState.js";

test("camera setup repeats without duplicating a camera", () => {
    const first = upsertCameraSetup([], {
        bus_id: "0-1",
        mode: "localize",
        model_id: "",
    });
    const updated = upsertCameraSetup(first, {
        bus_id: "0-1",
        mode: "both",
        model_id: "model-1",
    });

    assert.deepEqual(updated, [
        { bus_id: "0-1", mode: "both", model_id: "model-1" },
    ]);
});

test("generation payload keeps only the backend camera contract", () => {
    const payload = buildGenerationPayload(
        [
            {
                bus_id: "0-1",
                name: "Front camera",
                mode: "both",
                model_id: "",
            },
            {
                bus_id: "0-2",
                name: "Back camera",
                mode: "localize",
                model_id: "ignored",
            },
        ],
        " 10.33.22.2 ",
    );

    assert.deepEqual(payload, {
        network_table_address: "10.33.22.2",
        cameras: [
            { bus_id: "0-1", mode: "both", model_id: "" },
            { bus_id: "0-2", mode: "localize", model_id: "" },
        ],
    });
});

test("generation requires unique cameras and a NetworkTables address", () => {
    assert.throws(() => buildGenerationPayload([], "10.0.0.2"));
    assert.throws(() =>
        buildGenerationPayload(
            [
                { bus_id: "1", mode: "both" },
                { bus_id: "1", mode: "detect" },
            ],
            "10.0.0.2",
        ),
    );
    assert.throws(() =>
        buildGenerationPayload([{ bus_id: "1", mode: "both" }], "   "),
    );
});

test("guided steps overlay existing configuration views", () => {
    assert.equal(isGuidedStep(WIZARD_STEP.CALIBRATION), true);
    assert.equal(isGuidedStep(WIZARD_STEP.EXTRINSICS), true);
    assert.equal(isGuidedStep(WIZARD_STEP.NETWORK_TABLES), true);
    assert.equal(isGuidedStep(WIZARD_STEP.CAMERA_SELECT), false);
    assert.equal(isGuidedStep(WIZARD_STEP.PURPOSE), false);
    assert.equal(wizardStepView(WIZARD_STEP.CALIBRATION), "view-utils");
    assert.equal(wizardStepView(WIZARD_STEP.EXTRINSICS), "view-utils");
    assert.equal(wizardStepView(WIZARD_STEP.NETWORK_TABLES), "view-settings");
    assert.equal(wizardStepView(WIZARD_STEP.PURPOSE), "view-first-boot");
});

test("continue and cancel move between wizard pages and app workflows", () => {
    assert.equal(
        nextWizardStep(WIZARD_STEP.WELCOME),
        WIZARD_STEP.CAMERA_SELECT,
    );
    assert.equal(
        nextWizardStep(WIZARD_STEP.CAMERA_SELECT),
        WIZARD_STEP.CALIBRATION,
    );
    assert.equal(
        nextWizardStep(WIZARD_STEP.CALIBRATION),
        WIZARD_STEP.EXTRINSICS,
    );
    assert.equal(nextWizardStep(WIZARD_STEP.EXTRINSICS), WIZARD_STEP.PURPOSE);
    assert.equal(
        nextWizardStep(WIZARD_STEP.PURPOSE),
        WIZARD_STEP.CAMERA_SUMMARY,
    );
    assert.equal(
        nextWizardStep(WIZARD_STEP.CAMERA_SUMMARY),
        WIZARD_STEP.NETWORK_TABLES,
    );
    assert.equal(nextWizardStep(WIZARD_STEP.NETWORK_TABLES), null);
    assert.equal(
        previousWizardStep(WIZARD_STEP.CALIBRATION),
        WIZARD_STEP.CAMERA_SELECT,
    );
    assert.equal(
        previousWizardStep(WIZARD_STEP.NETWORK_TABLES),
        WIZARD_STEP.CAMERA_SUMMARY,
    );
});

test("guided overlay copy names the current camera and page action", () => {
    const calibration = guidedStepCopy(WIZARD_STEP.CALIBRATION, "Front", 1);
    assert.equal(calibration.progress, "Camera 1: calibration");
    assert.match(calibration.instructions, /Calibrate Camera/);
    const extrinsics = guidedStepCopy(WIZARD_STEP.EXTRINSICS, "Front", 1);
    assert.match(extrinsics.instructions, /Save Extrinsics/);
    const networkTables = guidedStepCopy(
        WIZARD_STEP.NETWORK_TABLES,
        "Front",
        1,
    );
    assert.match(networkTables.instructions, /Save Settings/);
});
