import assert from "node:assert/strict";
import test from "node:test";

import {
    buildGenerationPayload,
    upsertCameraSetup,
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
