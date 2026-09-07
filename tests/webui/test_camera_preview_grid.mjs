import assert from "node:assert/strict";
import test from "node:test";
import { mountCameraPreviewGrid } from "../../src/webui/js/ui/cameraPreviewGrid.js";

// Only the DOM operations used by the grid, with awaitable event handlers.
class Element {
    children = [];
    dataset = {};
    style = {};
    listeners = {};
    append(...children) {
        this.children.push(...children);
    }
    appendChild(child) {
        this.append(child);
    }
    replaceChildren(...children) {
        this.children = children;
    }
    setAttribute(name, value) {
        this[name] = value;
    }
    addEventListener(name, listener) {
        this.listeners[name] = listener;
    }
    focus() {}
}

test("configure saves a draft and previews retain the camera identity", async (t) => {
    const originalDocument = globalThis.document;
    globalThis.document = { createElement: () => new Element() };
    t.after(() => {
        if (originalDocument === undefined) delete globalThis.document;
        else globalThis.document = originalDocument;
    });

    const host = new Element();
    const camera = {
        name: "USB Camera",
        stream_name: "USB_Camera",
        bus_id: "1-2",
    };
    let selected = null;
    let failSave = true;
    const grid = mountCameraPreviewGrid(host, {
        cameras: [camera],
        onRename: async (record, displayName) => {
            assert.equal(record.bus_id, "1-2");
            if (failSave) throw new Error("Storage unavailable");
            return { display_name: displayName };
        },
        onSelect: (record) => {
            selected = record;
        },
    });
    const [image, title, , label, actions, status] =
        host.children[0].children[0].children;
    const input = label.children[0];
    const configure = actions.children[1];
    assert.match(image.src, /\/feed\/USB_Camera\?snapshot=1$/);
    input.value = "Front bumper";
    await configure.listeners.click();
    assert.equal(selected, null);
    assert.equal(status.textContent, "Storage unavailable");
    assert.equal(input.disabled, false);

    failSave = false;
    await configure.listeners.click();
    assert.equal(selected, camera);
    assert.equal(title.textContent, "Front bumper");
    assert.equal(camera.name, "USB Camera");
    assert.equal(camera.display_name, "Front bumper");
    assert.match(image.src, /\/feed\/USB_Camera\?snapshot=1$/);
    host.children[1].listeners.click();
    assert.match(image.src, /\?snapshot=1&t=\d+$/);
    grid.destroy();
    assert.match(image.src, /^data:/);
});
