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
        const toolbar = document.getElementById("customOpsToolbar");
        const saveBtn = document.getElementById("customOpsSaveBtn");
        const lintStatus = document.getElementById("customOpsLintStatus");
        const saveStatus = document.getElementById("customOpsSaveStatus");
        const currentFileLabel = document.getElementById("customOpsCurrentFile");
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
            onSelect: (tab) => {
                editorContainer.classList.remove("hidden");
                editorPlaceholder.classList.add("hidden");
                toolbar.classList.remove("hidden");
                this._editor.setLanguage(tab.fileType === "config" ? "json" : "python");
                this._editor.setContent(tab.content);
                this._editor.setDiagnostics(tab.diagnostics);
                currentFileLabel.textContent = `${tab.operationName} — ${tab.fileType === "config" ? "config_def.json" : tab.operationName + ".py"}`;
                this._updateLintStatus(tab.diagnostics);
            },
            onClose: (closedId, newActiveId) => {
                if (!newActiveId) {
                    editorContainer.classList.add("hidden");
                    editorPlaceholder.classList.remove("hidden");
                    toolbar.classList.add("hidden");
                    currentFileLabel.textContent = "";
                    lintStatus.textContent = "No file open";
                    saveStatus.textContent = "";
                }
            },
        });

        this._browser = new OperationBrowser({
            listEl,
            searchEl,
            newBtn,
            onOpenCode: (name) => this._openFile(name, "code"),
            onOpenConfig: (name) => this._openFile(name, "config"),
            onDelete: (name) => this._deleteOp(name, restartBanner),
            onCreated: (name) => this._openFile(name, "code"),
        });

        saveBtn.addEventListener("click", () => this._save(saveStatus, restartBanner));

        document.addEventListener("keydown", (e) => {
            const active = document.getElementById("view-custom-ops");
            if (!active || active.classList.contains("hidden")) return;
            if ((e.ctrlKey || e.metaKey) && e.key === "s") {
                e.preventDefault();
                this._save(saveStatus, restartBanner);
            }
        });

        restartBtn?.addEventListener("click", async () => {
            try {
                const res = await fetch(`${BACKEND_BASE_URL}/restart-backend`, { method: "POST" });
                if (res.ok) {
                    const saveStatus = document.getElementById("customOpsSaveStatus");
                    if (saveStatus) {
                        saveStatus.textContent = "Backend restarting...";
                        saveStatus.className = "text-green-400 text-xs ml-auto";
                    }
                } else {
                    alert(`Restart failed: HTTP ${res.status}`);
                }
            } catch (e) {
                alert(`Restart error: ${e.message}`);
            }
        });

        this._browser.refresh();
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
            this._tabManager.open(operationName, fileType, data.content);
        } catch (e) {
            alert(`Error loading file: ${e.message}`);
        }
    }

    async _deleteOp(name, restartBanner) {
        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations/${name}`, { method: "DELETE" });
            const data = await res.json();
            if (!res.ok) {
                alert(`Failed to delete: ${data.error || res.status}`);
                return;
            }
            this._tabManager.closeAllForOperation(name);
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
            : await this._fetchFile(tab.operationName, "code");
        const config = tab.fileType === "config"
            ? this._editor.getContent()
            : await this._fetchFile(tab.operationName, "config");

        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations/${tab.operationName}/lint`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code: code || "", config: config || "" }),
            });
            const data = await res.json();
            const diags = data.diagnostics || [];
            this._tabManager.setDiagnostics(diags);
            this._editor.setDiagnostics(diags);
            this._updateLintStatus(diags, lintStatus);
        } catch (_) {}
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

    async _fetchFile(operationName, fileType) {
        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations/${operationName}/${fileType}`);
            if (!res.ok) return "";
            const data = await res.json();
            return data.content || "";
        } catch (_) {
            return "";
        }
    }

    async _save(saveStatus, restartBanner) {
        const tab = this._tabManager.getActive();
        if (!tab) return;

        const code = tab.fileType === "code"
            ? this._editor.getContent()
            : await this._fetchFile(tab.operationName, "code");
        const config = tab.fileType === "config"
            ? this._editor.getContent()
            : await this._fetchFile(tab.operationName, "config");

        saveStatus.textContent = "Saving...";
        saveStatus.className = "text-[#888] text-xs ml-auto";

        try {
            const res = await fetch(`${BACKEND_BASE_URL}/custom-operations/${tab.operationName}/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code, config }),
            });
            const data = await res.json();

            if (!res.ok) {
                saveStatus.textContent = `Save failed: ${data.error || res.status}`;
                saveStatus.className = "text-red-400 text-xs ml-auto";
                if (data.diagnostics) {
                    this._tabManager.setDiagnostics(data.diagnostics);
                    this._editor.setDiagnostics(data.diagnostics);
                    this._updateLintStatus(data.diagnostics);
                }
                return;
            }

            this._tabManager.markClean();
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
