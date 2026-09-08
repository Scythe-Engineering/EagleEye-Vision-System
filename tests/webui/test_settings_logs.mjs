import assert from "node:assert/strict";
import test from "node:test";
import {
    VirtualLogView,
    LOG_ROW_HEIGHT,
    LOG_PAGE_SIZE,
} from "../../src/webui/js/settings/virtualLogView.js";

class Element {
    children = [];
    style = {};
    dataset = {};
    clientHeight = 400;
    scrollTop = 0;
    handlers = {};
    appendChild(node) {
        if (node.fragment) this.children.push(...node.children);
        else this.children.push(node);
    }
    replaceChildren(node) {
        this.children = [];
        this.appendChild(node);
    }
    addEventListener(name, fn) {
        this.handlers[name] = fn;
    }
}
let frames;
globalThis.document = {
    createElement: () => new Element(),
    createDocumentFragment: () =>
        Object.assign(new Element(), { fragment: true }),
};
globalThis.ResizeObserver = class {
    observe() {}
};
globalThis.requestAnimationFrame = (fn) => {
    frames.push(fn);
    return frames.length;
};

async function settle() {
    for (let i = 0; i < 20; i++) {
        await Promise.resolve();
        const ready = frames.splice(0);
        for (const frame of ready) frame();
    }
}
function fixture(count = 15000) {
    frames = [];
    const output = new Element();
    const requests = [];
    const status = new Element();
    let lines = Array.from({ length: count }, (_, i) => `line ${i}`);
    const snapshots = new Map();
    let token = 0;
    const fetchPage = async ({ snapshot, offset, limit }) => {
        requests.push({ snapshot, offset, limit });
        if (!snapshot) {
            snapshot = String(++token);
            snapshots.set(snapshot, [...lines]);
        }
        const saved = snapshots.get(snapshot);
        if (!saved) throw new Error("History expired");
        offset ??= Math.max(0, Math.floor((saved.length - 1) / limit) * limit);
        return {
            snapshot,
            offset,
            total_count: saved.length,
            messages: saved.slice(offset, offset + limit),
        };
    };
    const view = new VirtualLogView(output, {
        fetchPage,
        status,
        latestButton: new Element(),
        createRow: (text) =>
            Object.assign(new Element(), { textContent: text }),
    });
    return {
        view,
        output,
        status,
        requests,
        snapshots,
        add: (line) => lines.push(line),
    };
}
async function scroll(output, row) {
    output.scrollTop = row * LOG_ROW_HEIGHT;
    output.handlers.scroll();
    await settle();
}

test("fetches pages on demand and renders only viewport plus overscan", async () => {
    const { view, output, requests } = fixture();
    assert.equal(requests.length, 0);
    view.activate();
    await settle();
    assert.equal(requests.length, 1);
    assert.equal(requests[0].limit, LOG_PAGE_SIZE);
    assert.equal(view.total, 15000);
    assert.ok(view.content.children.length <= 36);
    assert.equal(view.content.children.at(-1).textContent, "line 14999");
    await scroll(output, 6000);
    assert.ok(requests.some((r) => r.offset === 6000));
    assert.ok(view.content.children.length <= 36);
    assert.equal(
        view.content.children.find((r) => r.dataset.logIndex === "6000")
            .textContent,
        "line 6000",
    );
    const before = requests.length;
    await scroll(output, 6001);
    assert.equal(requests.length, before);
    assert.equal(output.scrollTop, 6001 * LOG_ROW_HEIGHT);
});

test("live updates and tab revisits preserve a reader, Latest resumes live tail", async () => {
    const { view, output, requests, add } = fixture();
    view.activate();
    await settle();
    await scroll(output, 6000);
    const before = requests.length;
    add("newest");
    view.invalidate();
    view.activate();
    await settle();
    assert.equal(requests.length, before);
    assert.equal(output.scrollTop, 6000 * LOG_ROW_HEIGHT);
    assert.equal(view.total, 15000);
    await view.jumpToLatest();
    await settle();
    assert.equal(view.total, 15001);
    assert.equal(view.content.children.at(-1).textContent, "newest");
    output.clientHeight = 0;
    output.handlers.scroll();
    assert.equal(view.followTail, true);
    add("hidden update");
    view.invalidate();
    await settle();
    assert.equal(view.total, 15001);
    output.clientHeight = 400;
    view.activate();
    await settle();
    assert.equal(view.content.children.at(-1).textContent, "hidden update");
});

test("page cache remains bounded when traversing a large history", async () => {
    const { view, output } = fixture();
    view.activate();
    await settle();
    for (let row = 0; row < 10000; row += 400) await scroll(output, row);
    assert.ok(view.pages.size <= 12);
    assert.ok(view.content.children.length <= 36);
    await scroll(output, 0);
    assert.equal(view.content.children[0].textContent, "line 0");
});

test("expired history shows a recoverable error without a retry loop", async () => {
    const { view, output, status, snapshots, requests } = fixture();
    view.activate();
    await settle();
    snapshots.clear();
    await scroll(output, 4000);
    assert.match(status.textContent, /expired/);
    const count = requests.length;
    await settle();
    assert.equal(requests.length, count);
    await view.jumpToLatest();
    await settle();
    assert.equal(view.error, "");
    assert.equal(view.content.children.at(-1).textContent, "line 14999");
});

test("late tail response does not replace history after user scrolls up", async () => {
    const { view, output } = fixture();
    view.activate();
    await settle();
    let finish;
    view.fetchPage = () =>
        new Promise((resolve) => {
            finish = resolve;
        });
    view.invalidate();
    output.scrollTop -= 100;
    output.handlers.scroll();
    finish({ snapshot: "late", offset: 0, total_count: 1, messages: ["late"] });
    await settle();
    assert.equal(view.total, 15000);
    assert.equal(view.dirty, true);
});

test("clear rejects in-flight pages, empty history and local notices render safely", async () => {
    const { view } = fixture(0);
    view.activate();
    await settle();
    assert.equal(view.content.children.length, 0);
    let finish;
    view.fetchPage = () =>
        new Promise((resolve) => {
            finish = resolve;
        });
    view.invalidate();
    view.clear();
    view.append("display cleared");
    finish({ snapshot: "late", offset: 0, total_count: 1, messages: ["late"] });
    await settle();
    assert.equal(view.total, 0);
    assert.equal(view.content.children[0].textContent, "display cleared");
});
