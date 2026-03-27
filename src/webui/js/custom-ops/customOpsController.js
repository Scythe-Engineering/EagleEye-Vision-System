import { BACKEND_BASE_URL } from "../config.js";
import { CodeEditor } from "./codeEditor.js";
import { TabManager } from "./tabManager.js";
import { OperationBrowser } from "./operationBrowser.js";

const LINT_DEBOUNCE_MS = 600;

export class CustomOpsController {
    constructor() {
        this._editor = null;
        this._tabManager = null;
        this._browser = null;
        this._lintTimer = null;
        this._initialized = false;
        this._onKeydown = null;
        // Cache keyed by "operationName::fileType" to avoid redundant fetches.
        this._contentCache = new Map();
    }

    init() {
        if (this._initialized) {
            this._browser?.refresh();
            return;
        }
        this._initialized = true;

        const editorContainer = document.getElementById("customOpsEditorContainer");
        const editorPlaceholder = document.getElementById("customOpsEditorPlaceholder");
        const tabBarEl = document.getElementById("customOpsTabBar");
        const saveBtn = document.getElementById("customOpsSaveBtn");
        const lintStatus = document.getElementById("customOpsLintStatus");
        const saveStatus = document.getElementById("customOpsSaveStatus");
        const restartBanner = document.getElementById("customOpsRestartBanner");
        const restartBtn = document.getElementById("customOpsRestartBtn");
        const listEl = document.getElementById("customOpsList");
        const searchEl = document.getElementById("customOpsSearch");
        const newBtn = document.getElementById("customOpsNewBtn");

        this._editor = new CodeEditor(editorContainer, (content) => {
            this._tabManager.updateContent(content);
            this._scheduleLint();
        });

        this._tabManager = new TabManager({
            tabBarEl,
            emptyTabEl: document.getElementById("customOpsTabBarEmpty"),
            onSelect: (tab) => {
                editorContainer.classList.remove("hidden");
                editorPlaceholder.classList.add("hidden");
                saveBtn.classList.remove("hidden");
                saveStatus.classList.remove("hidden");
                this._editor.setLanguage(tab.fileType === "config" ? "json" : "python");
                this._editor.setContent(tab.content);
                this._editor.setDiagnostics(tab.diagnostics);
                this._updateLintStatus(tab.diagnostics);
            },
            onClose: (closedId, newActiveId) => {
                if (!newActiveId) {
                    editorContainer.classList.add("hidden");
                    editorPlaceholder.classList.remove("hidden");
                    saveBtn.classList.add("hidden");
                    saveStatus.classList.add("hidden");
                    saveStatus.textContent = "";
                    lintStatus.textContent = "No file open";
                }
            },
        });

        saveBtn.addEventListener("click", () => this._save(saveStatus, restartBanner));
        this._bindKeyboardShortcuts(saveStatus, restartBanner);
        this._bindRestartBanner(restartBtn, saveStatus, restartBanner);
        this._initBrowser(listEl, searchEl, newBtn, restartBanner);
    }

    _bindKeyboardShortcuts(saveStatus, restartBanner) {
        this._onKeydown = (e) => {
            const utilsView = document.getElementById("view-utils");
            const customOpsPanel = document.getElementById("utilsSubtabCustomOps");
            if (!utilsView || utilsView.classList.contains("hidden")) return;
            if (!customOpsPanel || customOpsPanel.classList.contains("hidden")) return;
            if ((e.ctrlKey || e.metaKey) && e.key === "s") {
                e.preventDefault();
                this._save(saveStatus, restartBanner);
            }
        };
        document.addEventListener("keydown", this._onKeydown);
    }

    _bindRestartBanner(restartBtn, saveStatus, restartBanner) {
        restartBtn?.addEventListener("click", async () => {
            try {
                const res = await fetch(`${BACKEND_BASE_URL}/restart-backend`, { method: "POST" });
                if (res.ok) {
                    restartBanner?.classList.add("hidden");
                    saveStatus.textContent = "Backend restarting...";
                    saveStatus.className = "text-green-400 text-xs ml-auto";
                } else {
                    alert(`Restart failed: HTTP ${res.status}`);
                }
            } catch (e) {
                alert(`Restart error: ${e.message}`);
            }
        });
    }

    _initBrowser(listEl, searchEl, newBtn, restartBanner) {
        this._browser = new OperationBrowser({
            listEl,
            searchEl,
            newBtn,
            onOpenCode: (name) => this._openFile(name, "code"),
            onOpenConfig: (name) => this._openFile(name, "config"),
            onDelete: (name) => this._deleteOp(name, restartBanner),
            onCreated: (name) => this._openFile(name, "code"),
        });
        this._browser.refresh();
    }

    destroy() {
        if (this._onKeydown) {
            document.removeEventListener("keydown", this._onKeydown);
            this._onKeydown = null;
        }
    }

    async _openFile(operationName, fileType) {
        if (this._tabManager.has(operationName, fileType)) {
            this._tabManager.select(operationName, fileType);
            return;
        }
        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations/${operationName}/${fileType}`);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                alert(`Failed to open file: ${err.error || res.status}`);
                return;
            }
            const data = await res.json();
            this._contentCache.set(`${operationName}::${fileType}`, data.content);
            this._tabManager.open(operationName, fileType, data.content);
        } catch (e) {
            alert(`Error loading file: ${e.message}`);
        }
    }

    async _deleteOp(name, restartBanner) {
        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations/${name}`, { method: "DELETE" });
            if (!res.ok) {
                const body = await res.text();
                let msg = body;
                try { msg = JSON.parse(body).error || body; } catch (_) {}
                alert(`Failed to delete: ${msg || res.status}`);
                return;
            }
            const data = await res.json();
            this._tabManager.closeAllForOperation(name);
            this._contentCache.delete(`${name}::code`);
            this._contentCache.delete(`${name}::config`);
            this._browser.refresh();
            if (data.restart_required) restartBanner.classList.remove("hidden");
        } catch (e) {
            alert(`Error: ${e.message}`);
        }
    }

    _scheduleLint() {
        clearTimeout(this._lintTimer);
        this._lintTimer = setTimeout(() => this._runLint(), LINT_DEBOUNCE_MS);
    }

    async _runLint() {
        const tab = this._tabManager.getActive();
        if (!tab) return;
        const lintStatus = document.getElementById("customOpsLintStatus");

        const code = tab.fileType === "code"
            ? this._editor.getContent()
            : await this._getFileContent(tab.operationName, "code");
        const config = tab.fileType === "config"
            ? this._editor.getContent()
            : await this._getFileContent(tab.operationName, "config");

        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations/${tab.operationName}/lint`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code: code || "", config: config || "" }),
            });
            if (!res.ok) {
                console.warn(`customOpsController: lint endpoint returned ${res.status} for "${tab.operationName}"`);
                return;
            }
            const data = await res.json();
            const diags = data.diagnostics || [];
            this._tabManager.setDiagnostics(diags);
            this._editor.setDiagnostics(diags);
            this._updateLintStatus(diags, lintStatus);
        } catch (err) {
            console.warn(`customOpsController: lint request failed for "${tab.operationName}":`, err);
        }
    }

    _updateLintStatus(diagnostics, el) {
        const statusEl = el || document.getElementById("customOpsLintStatus");
        if (!statusEl) return;
        if (!diagnostics || diagnostics.length === 0) {
            statusEl.textContent = "No issues";
            statusEl.className = "text-green-400 text-xs";
            return;
        }
        const errors = diagnostics.filter((d) => d.severity === "error").length;
        const warnings = diagnostics.filter((d) => d.severity === "warning").length;
        const parts = [];
        if (errors) parts.push(`${errors} error${errors > 1 ? "s" : ""}`);
        if (warnings) parts.push(`${warnings} warning${warnings > 1 ? "s" : ""}`);
        statusEl.textContent = parts.join(", ");
        statusEl.className = errors ? "text-red-400 text-xs" : "text-yellow-400 text-xs";
    }

    async _getFileContent(operationName, fileType) {
        // Prefer in-memory content from an open tab (may be unsaved edits).
        const tabContent = this._tabManager.getContent(operationName, fileType);
        if (tabContent !== null) return tabContent;
        // Fall back to cache, then network.
        const cacheKey = `${operationName}::${fileType}`;
        if (this._contentCache.has(cacheKey)) return this._contentCache.get(cacheKey);
        return this._fetchFile(operationName, fileType);
    }

    async _fetchFile(operationName, fileType) {
        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations/${operationName}/${fileType}`);
            if (!res.ok) return "";
            const data = await res.json();
            const content = data.content || "";
            this._contentCache.set(`${operationName}::${fileType}`, content);
            return content;
        } catch (_) {
            return "";
        }
    }

    async _save(saveStatus, restartBanner) {
        const tab = this._tabManager.getActive();
        if (!tab) return;

        const code = tab.fileType === "code"
            ? this._editor.getContent()
            : await this._getFileContent(tab.operationName, "code");
        const config = tab.fileType === "config"
            ? this._editor.getContent()
            : await this._getFileContent(tab.operationName, "config");

        saveStatus.textContent = "Saving...";
        saveStatus.className = "text-[#888] text-xs ml-auto";

        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations/${tab.operationName}/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code, config }),
            });

            if (!res.ok) {
                const body = await res.text();
                let errData = {};
                try { errData = JSON.parse(body); } catch (_) {}
                saveStatus.textContent = `Save failed: ${errData.error || res.status}`;
                saveStatus.className = "text-red-400 text-xs ml-auto";
                if (errData.diagnostics) {
                    this._tabManager.setDiagnostics(errData.diagnostics);
                    this._editor.setDiagnostics(errData.diagnostics);
                    this._updateLintStatus(errData.diagnostics);
                }
                return;
            }

            const data = await res.json();

            // Update cache with the freshly saved content.
            this._contentCache.set(`${tab.operationName}::code`, code);
            this._contentCache.set(`${tab.operationName}::config`, config);

            // Mark both tabs clean since both files were saved together.
            this._tabManager.markCleanFor(tab.operationName, "code");
            this._tabManager.markCleanFor(tab.operationName, "config");
            saveStatus.textContent = "Saved";
            saveStatus.className = "text-green-400 text-xs ml-auto";
            setTimeout(() => { saveStatus.textContent = ""; }, 3000);

            if (data.restart_required) restartBanner.classList.remove("hidden");
        } catch (e) {
            saveStatus.textContent = `Error: ${e.message}`;
            saveStatus.className = "text-red-400 text-xs ml-auto";
        }
    }
}
