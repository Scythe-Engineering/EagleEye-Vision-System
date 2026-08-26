import assert from "node:assert/strict";

import { getPipelineTemplate } from "../../src/webui/js/pipeline/creator/newPipelineDialog.js";

for (const templateId of [
    "apriltag_localization",
    "object_detection_cpu",
    "object_detection_mx3",
]) {
    const first = getPipelineTemplate(templateId);
    const second = getPipelineTemplate(templateId);
    const firstIds = new Set(first.map((node) => node.uuid));
    const secondIds = new Set(second.map((node) => node.uuid));

    assert.equal(firstIds.size, first.length);
    assert.equal(secondIds.size, second.length);
    assert.equal(
        [...firstIds].some((uuid) => secondIds.has(uuid)),
        false,
        `${templateId} reused a node UUID`,
    );

    for (const node of first) {
        for (const connection of node.connections || []) {
            assert(firstIds.has(connection.from_uuid));
            assert(firstIds.has(connection.to_uuid));
        }
    }
}
