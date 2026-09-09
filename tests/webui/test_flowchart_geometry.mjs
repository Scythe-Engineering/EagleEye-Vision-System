import assert from "node:assert/strict";
import test from "node:test";
import { FlowchartNode } from "../../src/webui/js/pipeline/flowchartNode.js";
import { FlowchartConnections } from "../../src/webui/js/pipeline/flowchartConnections.js";

function element() {
    const attributes = new Map();
    return {
        setAttribute: (name, value) => attributes.set(name, value),
        getAttribute: (name) => attributes.get(name),
    };
}

function connectionFixture() {
    const manager = Object.create(FlowchartConnections.prototype);
    const connection = {
        path: {
            ...element(),
            getTotalLength: () => 200,
            getPointAtLength: () => ({ x: 150, y: 60 }),
        },
        hitArea: element(),
        labelGroup: element(),
        labelText: element(),
        labelBackground: element(),
    };
    manager.connections = new Map([["edge", connection]]);
    const from = { getOutputPortPosition: () => ({ x: 100, y: 60 }) };
    const to = { getInputPortPosition: () => ({ x: 300, y: 60 }) };
    const update = (skip = false) =>
        manager.updateConnection("edge", from, "pose", to, "pose", skip);
    return { manager, connection, from, to, update };
}

test("hidden ports do not become endpoints at node corners", () => {
    const node = new FlowchartNode({
        instanceId: "node",
        position: { x: 80, y: 150 },
    });
    const rect = { left: 0, top: 0, width: 0, height: 0 };
    node.element = { closest: () => null, getBoundingClientRect: () => rect };
    assert.equal(
        node.getPortCenterPosition({ getBoundingClientRect: () => rect }),
        null,
    );
});

test("port centers stay in world coordinates across canvas zoom and pan", () => {
    const node = new FlowchartNode({
        instanceId: "node",
        position: { x: 80, y: 150 },
    });
    for (const scale of [0.2, 0.5, 1, 2.5]) {
        const viewport = {
            style: { transform: `translate(47px, -25px) scale(${scale})` },
        };
        const left = 47 + 80 * scale,
            top = -25 + 150 * scale;
        node.element = {
            closest: () => viewport,
            getBoundingClientRect: () => ({
                left,
                top,
                width: 200 * scale,
                height: 100 * scale,
            }),
        };
        const port = {
            getBoundingClientRect: () => ({
                left: left + 195 * scale,
                top: top + 60 * scale,
                width: 10 * scale,
                height: 10 * scale,
            }),
        };
        const actual = node.getPortCenterPosition(port);
        assert.ok(Math.abs(actual.x - 280) < 1e-8);
        assert.ok(Math.abs(actual.y - 215) < 1e-8);
    }
});

test("hidden text measurements are retried even when endpoints have not moved", () => {
    const { connection, update } = connectionFixture();
    connection.labelText.getBBox = () => ({ x: 0, y: 0, width: 0, height: 0 });
    update();
    assert.equal(connection.labelBackground.getAttribute("width"), undefined);
    connection.labelText.getBBox = () => ({
        x: -35,
        y: -6,
        width: 70,
        height: 12,
    });
    update();
    assert.equal(connection.labelBackground.getAttribute("width"), "82");
    assert.equal(connection.labelBackground.getAttribute("height"), "18");
    assert.equal(connection.labelBackground.getAttribute("x"), "-41");
    assert.equal(connection.labelBackground.getAttribute("y"), "-9");
    assert.equal(connection.labelDirty, false);
});

test("drag completion updates a deferred label without another endpoint change", () => {
    const { connection, from, update } = connectionFixture();
    connection.labelText.getBBox = () => ({
        x: -30,
        y: -6,
        width: 60,
        height: 12,
    });
    update();
    from.getOutputPortPosition = () => ({ x: 110, y: 80 });
    connection.path.getPointAtLength = () => ({ x: 190, y: 70 });
    update(true);
    assert.equal(
        connection.labelGroup.getAttribute("transform"),
        "translate(150, 60)",
    );
    update();
    assert.equal(
        connection.labelGroup.getAttribute("transform"),
        "translate(190, 70)",
    );
});

test("label backgrounds remeasure changed text metrics in SVG units", () => {
    const { manager, connection } = connectionFixture();
    connection.labelText.getBBox = () => ({
        x: -20,
        y: -5,
        width: 40,
        height: 10,
    });
    manager.updateLabel(connection);
    connection.labelText.getBBox = () => ({
        x: -50,
        y: -10,
        width: 100,
        height: 20,
    });
    manager.updateLabel(connection);
    assert.equal(connection.labelBackground.getAttribute("width"), "112");
    assert.equal(connection.labelBackground.getAttribute("height"), "26");
    assert.equal(connection.labelBackground.getAttribute("x"), "-56");
});

test("unmeasurable endpoints do not poison cached connection geometry", () => {
    const { connection, from, update } = connectionFixture();
    from.getOutputPortPosition = () => null;
    update();
    assert.equal(connection.path.getAttribute("d"), undefined);
    assert.equal(connection.lastPosKey, undefined);
});
