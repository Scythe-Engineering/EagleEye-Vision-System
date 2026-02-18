export class TabManager {
    constructor({ tabBarEl, onSelect, onClose, emptyTabEl = null }) {
        this._tabBarEl = tabBarEl;
        this._onSelect = onSelect;
        this._onClose = onClose;
        this._emptyEl = emptyTabEl;
        this._tabs = new Map(); // tabId -> {operationName, fileType, content, isDirty, diagnostics}
        this._activeTabId = null;
    }

    _tabId(operationName, fileType) {
        return `${operationName}::${fileType}`;
    }

    has(operationName, fileType) {
        return this._tabs.has(this._tabId(operationName, fileType));
    }

    open(operationName, fileType, content) {
        const id = this._tabId(operationName, fileType);
        if (!this._tabs.has(id)) {
            this._tabs.set(id, {
                operationName,
                fileType,
                content,
                originalContent: content,
                isDirty: false,
                diagnostics: [],
            });
        }
        this.select(operationName, fileType);
    }

    getContent(operationName, fileType) {
        const tab = this._tabs.get(this._tabId(operationName, fileType));
        return tab ? tab.content : null;
    }

    select(operationName, fileType) {
        const id = this._tabId(operationName, fileType);
        if (!this._tabs.has(id)) return;
        this._saveCurrentContent();
        this._activeTabId = id;
        this._render();
        if (this._onSelect) {
            const tab = this._tabs.get(id);
            this._onSelect(tab);
        }
    }

    getActive() {
        if (!this._activeTabId) return null;
        return this._tabs.get(this._activeTabId) || null;
    }

    getActiveId() {
        return this._activeTabId;
    }

    updateContent(content) {
        if (!this._activeTabId) return;
        const tab = this._tabs.get(this._activeTabId);
        if (!tab) return;
        tab.content = content;
        tab.isDirty = content !== tab.originalContent;
        this._renderDirtyIndicator(this._activeTabId, tab.isDirty);
    }

    markClean() {
        if (!this._activeTabId) return;
        const tab = this._tabs.get(this._activeTabId);
        if (!tab) return;
        tab.originalContent = tab.content;
        tab.isDirty = false;
        this._renderDirtyIndicator(this._activeTabId, false);
    }

    markCleanFor(operationName, fileType) {
        const id = this._tabId(operationName, fileType);
        const tab = this._tabs.get(id);
        if (!tab) return;
        tab.originalContent = tab.content;
        tab.isDirty = false;
        this._renderDirtyIndicator(id, false);
    }

    setDiagnostics(diagnostics) {
        if (!this._activeTabId) return;
        const tab = this._tabs.get(this._activeTabId);
        if (tab) tab.diagnostics = diagnostics;
    }

    _saveCurrentContent() {
        if (!this._activeTabId) return;
        // Content is kept current via updateContent() calls from the editor change listener.
    }

    close(tabId, force = false) {
        const tab = this._tabs.get(tabId);
        if (!tab) return;
        if (tab.isDirty && !force) {
            if (!confirm(`"${tab.operationName} (${tab.fileType})" has unsaved changes. Close anyway?`)) return;
        }
        this._tabs.delete(tabId);
        if (this._activeTabId === tabId) {
            const remaining = [...this._tabs.keys()];
            this._activeTabId = remaining.length > 0 ? remaining[remaining.length - 1] : null;
        }
        this._render();
        if (this._activeTabId) {
            const activeTab = this._tabs.get(this._activeTabId);
            if (activeTab && this._onSelect) this._onSelect(activeTab);
        } else if (this._onClose) {
            this._onClose(tabId, null);
        }
    }

    closeAllForOperation(operationName) {
        const toRemove = [];
        for (const [id, tab] of this._tabs) {
            if (tab.operationName === operationName) toRemove.push(id);
        }
        for (const id of toRemove) {
            this._tabs.delete(id);
            if (this._activeTabId === id) this._activeTabId = null;
        }
        if (!this._activeTabId && this._tabs.size > 0) {
            const remaining = [...this._tabs.keys()];
            this._activeTabId = remaining[remaining.length - 1];
        }
        this._render();
        if (this._activeTabId && this._onSelect) {
            this._onSelect(this._tabs.get(this._activeTabId));
        } else if (!this._activeTabId && this._onClose) {
            this._onClose(null, null);
        }
    }

    _renderDirtyIndicator(tabId, isDirty) {
        const el = this._tabBarEl.querySelector(`[data-tab-id="${CSS.escape(tabId)}"] .dirty-dot`);
        if (el) el.classList.toggle("hidden", !isDirty);
    }

    _render() {
        this._tabBarEl.innerHTML = "";

        if (this._tabs.size === 0) {
            if (this._emptyEl) this._tabBarEl.appendChild(this._emptyEl);
            return;
        }

        for (const [id, tab] of this._tabs) {
            const isActive = id === this._activeTabId;
            const label = `${tab.operationName} · ${tab.fileType}`;
            const tabEl = document.createElement("div");
            tabEl.dataset.tabId = id;
            tabEl.className = [
                "flex items-center gap-1.5 px-3 py-1.5 rounded-t-md text-xs font-medium cursor-pointer select-none transition-colors shrink-0",
                isActive
                    ? "bg-[#1f1f1f] text-[#f9c845] border border-b-0 border-[#414141]"
                    : "bg-[#2a2a2a] text-[#888] hover:text-[#ccc]",
            ].join(" ");

            tabEl.innerHTML = `
                <span class="dirty-dot w-2 h-2 rounded-full bg-[#f9c845] ${tab.isDirty ? "" : "hidden"}"></span>
                <span class="max-w-[160px] truncate" title="${label}">${label}</span>
                <button class="close-btn ml-1 text-[#666] hover:text-[#f9c845]" title="Close">×</button>
            `;

            tabEl.addEventListener("click", (e) => {
                if (e.target.classList.contains("close-btn")) return;
                this.select(tab.operationName, tab.fileType);
            });
            tabEl.querySelector(".close-btn").addEventListener("click", (e) => {
                e.stopPropagation();
                this.close(id);
            });

            this._tabBarEl.appendChild(tabEl);
        }
    }
}
