import { BACKEND_BASE_URL } from "../config.js";

export class OperationBrowser {
    constructor({ listEl, searchEl, newBtn, onOpenCode, onOpenConfig, onDelete, onCreated }) {
        this._listEl = listEl;
        this._searchEl = searchEl;
        this._onOpenCode = onOpenCode;
        this._onOpenConfig = onOpenConfig;
        this._onDelete = onDelete;
        this._onCreated = onCreated;
        this._ops = [];

        searchEl.addEventListener("input", () => this._renderList());
        newBtn.addEventListener("click", () => this._promptCreate());
    }

    async refresh() {
        this._listEl.innerHTML = `<div class="text-[#666] text-sm p-2">Loading...</div>`;
        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            this._ops = data.operations || [];
            this._renderList();
        } catch (e) {
            this._listEl.innerHTML = `<div class="text-red-400 text-sm p-2">Failed to load: ${e.message}</div>`;
        }
    }

    _renderList() {
        const query = (this._searchEl.value || "").toLowerCase();
        const filtered = this._ops.filter((op) => op.name.toLowerCase().includes(query));
        this._listEl.innerHTML = "";

        if (filtered.length === 0) {
            this._listEl.innerHTML = `<div class="text-[#666] text-sm p-2">No operations found</div>`;
            return;
        }

        for (const op of filtered) {
            const card = document.createElement("div");
            card.className =
                "mb-2 p-3 bg-[#2a2a2a] rounded-lg border border-[#3a3a3a] hover:border-[#555] transition-colors";
            card.innerHTML = `
                <div class="font-mono text-sm text-[#f9c845] mb-1 truncate" title="${op.name}">${op.name}</div>
                <div class="text-xs text-[#888] mb-2 truncate" title="${op.description || ''}">${op.description || "No description"}</div>
                <div class="flex gap-1.5 flex-wrap">
                    <button class="btn-code px-2 py-0.5 bg-[#1f1f1f] text-[#ccc] text-xs rounded border border-[#414141] hover:border-[#f9c845] hover:text-[#f9c845] transition-colors">Code</button>
                    <button class="btn-config px-2 py-0.5 bg-[#1f1f1f] text-[#ccc] text-xs rounded border border-[#414141] hover:border-[#f9c845] hover:text-[#f9c845] transition-colors ${op.has_config ? "" : "opacity-50"}">Config</button>
                    <button class="btn-delete px-2 py-0.5 bg-[#1f1f1f] text-red-400 text-xs rounded border border-[#414141] hover:border-red-500 transition-colors ml-auto">Delete</button>
                </div>
            `;
            if (typeof this._onOpenCode === "function") {
                card.querySelector(".btn-code").addEventListener("click", () => this._onOpenCode(op.name));
            }
            if (typeof this._onOpenConfig === "function") {
                card.querySelector(".btn-config").addEventListener("click", () => this._onOpenConfig(op.name));
            }
            if (typeof this._onDelete === "function") {
                card.querySelector(".btn-delete").addEventListener("click", () => this._confirmDelete(op.name));
            }
            this._listEl.appendChild(card);
        }
    }

    _confirmDelete(name) {
        if (!confirm(`Delete operation "${name}"? This cannot be undone.`)) return;
        this._onDelete(name);
    }

    async _promptCreate() {
        const name = prompt("Enter operation name (snake_case, e.g. my_operation):");
        if (!name) return;
        if (!/^[a-z][a-z0-9_]*$/.test(name)) {
            alert("Name must be snake_case: lowercase letters, digits, underscores, starting with a letter.");
            return;
        }
        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });
            if (!res.ok) {
                const body = await res.text();
                let msg = body;
                try { msg = JSON.parse(body).error || body; } catch (_) {}
                alert(`Failed to create operation: ${msg || res.status}`);
                return;
            }
            await this.refresh();
            if (this._onCreated) this._onCreated(name);
        } catch (e) {
            alert(`Error: ${e.message}`);
        }
    }
}
