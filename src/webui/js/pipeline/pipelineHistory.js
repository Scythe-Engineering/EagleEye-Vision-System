const MAX_HISTORY_SIZE = 20;
const HISTORY_DEBOUNCE_MS = 300;

class PipelineHistory {
    constructor() {
        this.undoStack = [];
        this.redoStack = [];
        this.isApplyingHistory = false;

        this.currentSnapshot = null;
        this.commitTimer = null;

        this.pipelineStore = null;
        this.renderCallback = null;
        this.autoSaveCallback = null;
        this.postRefreshCallback = null;

        this.undoButton = null;
        this.redoButton = null;
    }

    /**
     * Wire the history system into the pipeline store.
     *
     * @param {import('./PipelineStore.js').PipelineStore} pipelineStore
     * @param {{
     *   renderCallback: () => Promise<void>,
     *   autoSaveCallback: () => void,
     *   postRefreshCallback: () => Promise<void>,
     * }} callbacks
     */
    init(pipelineStore, { renderCallback, autoSaveCallback, postRefreshCallback }) {
        this.pipelineStore = pipelineStore;
        this.renderCallback = renderCallback;
        this.autoSaveCallback = autoSaveCallback;
        this.postRefreshCallback = postRefreshCallback;
        this.currentSnapshot = this._captureState();

        const scheduleCommit = () => this._scheduleCommit();
        pipelineStore.subscribe("pipeline:changed", scheduleCommit);
        pipelineStore.subscribe("node:position:changed", scheduleCommit);

        // Reset when the user switches pipelines or clears the canvas.
        pipelineStore.subscribe("pipeline:loaded", () => {
            if (!this.isApplyingHistory) this._resetHistory();
        });

        pipelineStore.subscribe("pipeline:cleared", () => {
            if (!this.isApplyingHistory) this._resetHistory();
        });
    }

    _resetHistory() {
        this.undoStack = [];
        this.redoStack = [];
        this._cancelPendingCommit();
        this.currentSnapshot = this._captureState();
        this._updateButtons();
    }

    _captureState() {
        if (!this.pipelineStore) return null;

        const nodes = this.pipelineStore.getNodes().map((node) => ({
            action_name: node.operationId,
            action_params: { ...node.config },
            position: { ...node.position },
            uuid: node.uuid,
        }));

        const connections = [];
        for (const conn of this.pipelineStore.state.currentPipeline.connections.values()) {
            connections.push({
                from_uuid: conn.fromUuid,
                from_port: conn.fromPort,
                to_uuid: conn.toUuid,
                to_port: conn.toPort,
                data_type: conn.dataType,
                is_default: conn.isDefault,
                // customWaypoints is an array of {x, y}; shallow-copy each point.
                custom_waypoints: conn.customWaypoints
                    ? conn.customWaypoints.map((pt) => ({ ...pt }))
                    : null,
            });
        }

        return { nodes, connections };
    }

    _snapshotsEqual(a, b) {
        return JSON.stringify(a) === JSON.stringify(b);
    }

    _scheduleCommit() {
        if (this.isApplyingHistory) return;

        clearTimeout(this.commitTimer);
        this.commitTimer = setTimeout(() => this._commitCurrentChange(), HISTORY_DEBOUNCE_MS);
    }

    _commitCurrentChange() {
        this.commitTimer = null;
        if (this.isApplyingHistory) return;

        const nextSnapshot = this._captureState();
        if (!this._snapshotsEqual(this.currentSnapshot, nextSnapshot)) {
            this._pushToUndo(this.currentSnapshot);
            this.currentSnapshot = nextSnapshot;
            this._updateButtons();
        }
    }

    _pushToUndo(snapshot) {
        if (!snapshot) return;
        this.undoStack.push(snapshot);
        if (this.undoStack.length > MAX_HISTORY_SIZE) {
            this.undoStack.shift();
        }
        // Any new user action invalidates the redo branch.
        this.redoStack = [];
    }

    _cancelPendingCommit() {
        clearTimeout(this.commitTimer);
        this.commitTimer = null;
    }

    _flushPendingCommit() {
        if (!this.commitTimer) return;
        clearTimeout(this.commitTimer);
        this._commitCurrentChange();
    }

    async _applySnapshot(snapshot) {
        this._cancelPendingCommit();
        this.isApplyingHistory = true;
        try {
            this.pipelineStore.loadPipelineData(snapshot.nodes, snapshot.connections);
            if (this.renderCallback) await this.renderCallback();
            if (this.autoSaveCallback) this.autoSaveCallback();
            if (this.postRefreshCallback) await this.postRefreshCallback();
        } finally {
            this.isApplyingHistory = false;
        }
        // NOTE: originalConfig is not stored in snapshots; after undo/redo it will
        // equal action_params. This means the restart indicator may not reflect the
        // true "last-saved-to-backend" original. Acceptable limitation.
        this.currentSnapshot = this._captureState();
    }

    canUndo() {
        return this.undoStack.length > 0;
    }

    canRedo() {
        return this.redoStack.length > 0;
    }

    async undo() {
        this._flushPendingCommit();
        if (!this.canUndo() || this.isApplyingHistory) return;

        const previousSnapshot = this.undoStack.pop();
        this.redoStack.push(this.currentSnapshot);

        await this._applySnapshot(previousSnapshot);
        this._updateButtons();
    }

    async redo() {
        this._flushPendingCommit();
        if (!this.canRedo() || this.isApplyingHistory) return;

        const nextSnapshot = this.redoStack.pop();
        this.undoStack.push(this.currentSnapshot);

        await this._applySnapshot(nextSnapshot);
        this._updateButtons();
    }

    setButtons(undoButton, redoButton) {
        this.undoButton = undoButton;
        this.redoButton = redoButton;
        this._updateButtons();
    }

    _updateButtons() {
        const canUndo = this.canUndo();
        const canRedo = this.canRedo();

        if (this.undoButton) {
            this.undoButton.disabled = !canUndo;
            this.undoButton.classList.toggle("opacity-40", !canUndo);
            this.undoButton.classList.toggle("cursor-not-allowed", !canUndo);
            this.undoButton.classList.toggle("cursor-pointer", canUndo);
        }
        if (this.redoButton) {
            this.redoButton.disabled = !canRedo;
            this.redoButton.classList.toggle("opacity-40", !canRedo);
            this.redoButton.classList.toggle("cursor-not-allowed", !canRedo);
            this.redoButton.classList.toggle("cursor-pointer", canRedo);
        }
    }
}

export const pipelineHistory = new PipelineHistory();
