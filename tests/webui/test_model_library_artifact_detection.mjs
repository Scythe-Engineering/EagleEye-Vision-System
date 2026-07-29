import assert from "node:assert/strict";

import { detectArtifactSlot } from "../../src/webui/js/pipeline/modelLibraryModal.js";

assert.equal(detectArtifactSlot("robot.pt"), "pt");
assert.equal(detectArtifactSlot("robot.onnx"), "onnx");
assert.equal(
    detectArtifactSlot("robot_postprocessor.onnx"),
    "mx3_postprocessor",
);
assert.equal(detectArtifactSlot("robot.engine"), "engine");
assert.equal(detectArtifactSlot("robot.dfp"), "mx3_dfp");
assert.throws(() => detectArtifactSlot("robot.zip"), /Choose a/);
