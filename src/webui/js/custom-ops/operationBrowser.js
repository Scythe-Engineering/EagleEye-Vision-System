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
        this._listEl.innerHTML = `<div class="text-text-dim text-sm p-2">Loading...</div>`;
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
            this._listEl.innerHTML = `<div class="text-text-dim text-sm p-2">No operations found</div>`;
            return;
        }

        for (const op of filtered) {
            const card = document.createElement("div");
            card.className = "custom-op-card group mb-2 p-3 bg-surface-200 rounded-lg border border-border-default hover:border-white/30 transition-colors";
            card.innerHTML = `
                <div class="font-mono text-sm text-brand-primary mb-1 truncate" title="${op.name}">${op.name}</div>
                <div class="text-xs text-text-subtle mb-2 truncate" title="${op.description || ""}">${op.description || "No description"}</div>
                <div class="custom-op-actions flex gap-1 items-center opacity-0 group-hover:opacity-100 transition-opacity">
                    <button class="btn-code p-1 rounded text-text-subtle hover:text-brand-primary hover:bg-surface-300 transition-colors" title="Open Code">${this._codeIcon()}</button>
                    <button class="btn-config p-1 rounded text-text-subtle hover:text-brand-primary hover:bg-surface-300 transition-colors ${op.has_config ? "" : "opacity-50 cursor-not-allowed"}" title="Open Config">${this._configIcon()}</button>
                    <button class="btn-delete p-1 rounded text-text-subtle hover:text-red-400 hover:bg-surface-300 transition-colors ml-auto" title="Delete">${this._deleteIcon()}</button>
                </div>
            `;
            if (typeof this._onOpenCode === "function") {
                card.querySelector(".btn-code").addEventListener("click", () => this._onOpenCode(op.name));
            }
            if (typeof this._onOpenConfig === "function") {
                const configBtn = card.querySelector(".btn-config");
                configBtn.addEventListener("click", () => {
                    if (op.has_config) this._onOpenConfig(op.name);
                });
            }
            if (typeof this._onDelete === "function") {
                card.querySelector(".btn-delete").addEventListener("click", () => this._confirmDelete(op.name));
            }
            this._listEl.appendChild(card);
        }
    }

    _codeIcon() {
        return `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>`;
    }

    _configIcon() {
        return `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>`;
    }

    _deleteIcon() {
        return `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>`;
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
