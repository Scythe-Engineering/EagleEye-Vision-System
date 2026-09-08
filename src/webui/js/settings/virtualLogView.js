// Fixed-height physical lines keep scrolling predictable without measuring history.
export const LOG_ROW_HEIGHT = 20;
export const LOG_PAGE_SIZE = 200;
const OVERSCAN = 8;
const MAX_CACHED_PAGES = 12;

export class VirtualLogView {
    constructor(output, { fetchPage, createRow, status, latestButton }) {
        this.output = output;
        this.fetchPage = fetchPage;
        this.createRow = createRow;
        this.status = status;
        this.latestButton = latestButton;
        this.pages = new Map();
        this.pending = new Map();
        this.snapshot = null;
        this.total = 0;
        this.generation = 0;
        this.dirty = true;
        this.followTail = true;
        this.loading = null;
        this.error = "";
        this.notes = [];
        this.frame = null;
        this.content = document.createElement("div");
        this.content.style.position = "relative";
        this.content.style.minWidth = "100%";
        this.output.replaceChildren(this.content);
        output.addEventListener(
            "scroll",
            () => {
                if (!output.clientHeight) return;
                this.followTail =
                    output.scrollTop + output.clientHeight >=
                    this.rowCount * LOG_ROW_HEIGHT - 24;
                this.scheduleRender();
                if (this.followTail && this.dirty) this.refresh();
            },
            { passive: true },
        );
        latestButton?.addEventListener("click", () => this.jumpToLatest());
        this.observer = new ResizeObserver(() => this.activate());
        this.observer.observe(output);
    }

    get rowCount() {
        return this.total + this.notes.length;
    }

    activate() {
        if (!this.output.clientHeight) return;
        if (this.dirty && this.followTail) this.refresh();
        else this.render();
    }

    invalidate() {
        this.dirty = true;
        if (this.output.clientHeight && this.followTail) this.refresh();
        this.updateStatus();
    }

    async refresh() {
        if (this.loading) return this.loading;
        const generation = this.generation;
        this.dirty = false;
        this.error = "";
        this.loading = (async () => {
            try {
                const data = await this.fetchPage({ limit: LOG_PAGE_SIZE });
                if (generation !== this.generation) return;
                // A user may start reading history while the tail request runs.
                if (this.snapshot && !this.followTail) {
                    this.dirty = true;
                    return;
                }
                this.snapshot = data.snapshot;
                this.total = data.total_count;
                this.pages.clear();
                this.pages.set(data.offset, data.messages);
                this.pending.clear();
                this.generation++;
                this.render();
            } catch (error) {
                if (generation === this.generation) this.error = error.message;
            } finally {
                this.loading = null;
                this.updateStatus();
            }
        })();
        this.updateStatus();
        return this.loading;
    }

    jumpToLatest() {
        this.followTail = true;
        this.error = "";
        this.dirty = true;
        return this.refresh();
    }

    scheduleRender() {
        if (this.frame !== null) return;
        this.frame = requestAnimationFrame(() => {
            this.frame = null;
            this.render();
        });
    }

    render() {
        const height = this.output.clientHeight;
        if (!height) return;
        this.content.style.height = `${this.rowCount * LOG_ROW_HEIGHT}px`;
        if (this.followTail) {
            this.output.scrollTop = Math.max(
                0,
                this.rowCount * LOG_ROW_HEIGHT - height,
            );
        }
        const first = Math.max(
            0,
            Math.floor(this.output.scrollTop / LOG_ROW_HEIGHT) - OVERSCAN,
        );
        const last = Math.min(
            this.rowCount,
            Math.ceil((this.output.scrollTop + height) / LOG_ROW_HEIGHT) +
                OVERSCAN,
        );
        const fragment = document.createDocumentFragment();
        const needed = new Set();
        for (let index = first; index < last; index++) {
            const offset = Math.floor(index / LOG_PAGE_SIZE) * LOG_PAGE_SIZE;
            const page = this.pages.get(offset);
            const message =
                index >= this.total
                    ? this.notes[index - this.total]
                    : page?.[index - offset];
            if (index < this.total) needed.add(offset);
            const row = this.createRow(message ?? "Loading…");
            row.dataset.logIndex = String(index);
            Object.assign(row.style, {
                position: "absolute",
                top: `${index * LOG_ROW_HEIGHT}px`,
                height: `${LOG_ROW_HEIGHT}px`,
                lineHeight: `${LOG_ROW_HEIGHT}px`,
                whiteSpace: "pre",
                padding: "0 16px",
                minWidth: "100%",
                width: "max-content",
            });
            fragment.appendChild(row);
        }
        this.content.replaceChildren(fragment);
        for (const offset of needed) {
            if (this.pages.has(offset)) {
                const page = this.pages.get(offset);
                this.pages.delete(offset);
                this.pages.set(offset, page);
            } else if (!this.error) this.loadPage(offset);
        }
        while (this.pages.size > MAX_CACHED_PAGES)
            this.pages.delete(this.pages.keys().next().value);
        this.updateStatus();
    }

    async loadPage(offset) {
        if (
            !this.snapshot ||
            this.pending.has(offset) ||
            this.pending.size >= 2
        )
            return;
        const generation = this.generation;
        const snapshot = this.snapshot;
        this.pending.set(offset, true);
        try {
            const data = await this.fetchPage({
                snapshot,
                offset,
                limit: LOG_PAGE_SIZE,
            });
            if (generation !== this.generation || snapshot !== this.snapshot)
                return;
            this.pages.set(offset, data.messages);
            this.scheduleRender();
        } catch (error) {
            if (generation === this.generation) this.error = error.message;
        } finally {
            if (generation === this.generation) {
                this.pending.delete(offset);
                if (!this.error) this.scheduleRender();
            }
            this.updateStatus();
        }
    }

    append(message) {
        this.notes.push(...message.split("\n"));
        this.notes = this.notes.slice(-50);
        this.render();
    }

    clear() {
        this.generation++;
        this.snapshot = null;
        this.total = 0;
        this.pages.clear();
        this.pending.clear();
        this.notes = [];
        this.dirty = false;
        this.error = "";
        this.followTail = true;
        this.render();
    }

    updateStatus() {
        if (this.status)
            this.status.textContent =
                this.error ||
                (this.loading || this.pending.size
                    ? "Loading logs…"
                    : `${this.total.toLocaleString()} retained lines${this.dirty ? " · New logs available" : ""}`);
        if (this.latestButton)
            this.latestButton.textContent = this.error
                ? "Retry / Latest"
                : "Latest";
    }
}
