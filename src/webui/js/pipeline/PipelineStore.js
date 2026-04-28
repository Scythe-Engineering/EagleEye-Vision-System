import { uid } from "./utils.js";

class PipelineNode {
    constructor(operationData, existingUuid = null, existingPosition = null) {
        this.uuid = existingUuid || uid("op-");
        this.operationId = operationData.id;
        this.name = operationData.name;
        this.type = operationData.type;
        this.description = operationData.description;
        this.path = operationData.path;
        this.configDataPath = operationData.configDataPath;
        this.isSecondary = operationData.isSecondary;
        this.isDataSource = Boolean(operationData.isDataSource);
        this.hasVisualization = Boolean(operationData.hasVisualization);
        this.config = operationData.config || {};
        this.originalConfig = operationData.originalConfig || {};
        this.position = existingPosition || { x: 100, y: 100 };
        this.requiresRestart = operationData.requiresRestart || false;

        this.instanceId = `${this.operationId}_${Date.now()}_${Math.random()
            .toString(36)
            .slice(2, 11)}`;
    }

    toJSON() {
        return {
            action_name: this.operationId,
            action_params: this.config,
            position: this.position,
            uuid: this.uuid,
        };
    }
}

class PipelineStore {
    constructor() {
        this.state = {
            cameras: [],
            pipelines: [],
            operations: [],
            currentPipeline: {
                cameraName: null,
                pipelineName: null,
                nodes: new Map(),
                connections: new Map(),
            },
            ui: {
                restartRequired: false,
                operationErrors: new Map(),
                downstreamDisabledNodes: new Set(),
                profilingByPipeline: new Map(),
                profilingLastUpdateMsByPipeline: new Map(),
            },
        };

        this.listeners = new Map();
        this.uuidToInstanceId = new Map();
        this.instanceIdToUuid = new Map();
        this.operationLookup = new Map();

        this.isAutoSaving = false;
        this.pendingAutoSave = false;
    }

    subscribe(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);

        return () => {
            const callbacks = this.listeners.get(event);
            if (callbacks) {
                callbacks.delete(callback);
            }
        };
    }

    emit(event, data) {
        const callbacks = this.listeners.get(event);
        if (callbacks) {
            callbacks.forEach((callback) => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(
                        `Error in event listener for ${event}:`,
                        error,
                    );
                }
            });
        }
    }

    setOperations(operations) {
        this.state.operations = operations;
        this.operationLookup.clear();

        operations.forEach((op) => {
            const normalized = this.normalizeOperationId(op.id);
            this.operationLookup.set(normalized, op);
            this.operationLookup.set(op.id, op);
        });

        this.emit("operations:loaded", operations);
    }

    setCameras(cameras) {
        this.state.cameras = cameras;
        this.emit("cameras:loaded", cameras);
    }

    setPipelines(pipelines) {
        this.state.pipelines = pipelines;
        this.emit("pipelines:loaded", pipelines);
    }

    setCurrentCamera(cameraName) {
        this.state.currentPipeline.cameraName = cameraName;
        this.emit("camera:selected", cameraName);
    }

    setCurrentPipeline(pipelineName) {
        this.state.currentPipeline.pipelineName = pipelineName;
        this.emit("pipeline:selected", pipelineName);
    }

    normalizeOperationId(id) {
        if (!id) return "";
        return id.replace(/\.py$/, "").toLowerCase().replace(/\s+/g, "_");
    }

    findOperation(actionName) {
        const normalized = this.normalizeOperationId(actionName);
        return (
            this.operationLookup.get(actionName) ||
            this.operationLookup.get(normalized) ||
            null
        );
    }

    addNode(operationData, position = null, existingUuid = null) {
        const operation = this.findOperation(operationData.id);
        if (!operation) {
            console.error(`Operation not found: ${operationData.id}`);
            return null;
        }

        const node = new PipelineNode(
            { ...operation, config: operationData.config || {} },
            existingUuid,
            position,
        );

        this.state.currentPipeline.nodes.set(node.uuid, node);
        this.uuidToInstanceId.set(node.uuid, node.instanceId);
        this.instanceIdToUuid.set(node.instanceId, node.uuid);

        this.emit("node:added", { node });
        this.emit("pipeline:changed", {
            type: "node:added",
            uuid: node.uuid,
        });

        return node;
    }

    removeNode(identifier) {
        const uuid = this.resolveToUuid(identifier);
        if (!uuid) {
            console.error(`Node not found: ${identifier}`);
            return false;
        }

        const node = this.state.currentPipeline.nodes.get(uuid);
        if (!node) return false;

        const connectionsToRemove = [];
        for (const [key, conn] of this.state.currentPipeline.connections) {
            if (conn.fromUuid === uuid || conn.toUuid === uuid) {
                connectionsToRemove.push(key);
            }
        }

        connectionsToRemove.forEach((key) => {
            this.state.currentPipeline.connections.delete(key);
        });

        this.state.currentPipeline.nodes.delete(uuid);
        this.uuidToInstanceId.delete(uuid);
        this.instanceIdToUuid.delete(node.instanceId);

        if (connectionsToRemove.length > 0) {
            this.emit("connections:changed", {
                type: "bulk:removed",
                keys: connectionsToRemove,
            });
        }

        this.emit("node:removed", { uuid, instanceId: node.instanceId });
        this.emit("pipeline:changed", { type: "node:removed", uuid });

        return true;
    }

    updateNodePosition(identifier, position) {
        const uuid = this.resolveToUuid(identifier);
        if (!uuid) return false;

        const node = this.state.currentPipeline.nodes.get(uuid);
        if (!node) return false;

        node.position = position;
        this.emit("node:position:changed", { uuid, position });

        return true;
    }

    updateNodeConfig(identifier, config) {
        const uuid = this.resolveToUuid(identifier);
        if (!uuid) return false;

        const node = this.state.currentPipeline.nodes.get(uuid);
        if (!node) return false;

        node.config = { ...config };
        this.emit("node:config:changed", { uuid, config });
        this.emit("pipeline:changed", {
            type: "node:config:changed",
            uuid,
        });

        return true;
    }

    addConnection(
        fromId,
        fromPort,
        toId,
        toPort,
        dataType = null,
        isDefault = false,
        customWaypoints = null,
    ) {
        const fromUuid = this.resolveToUuid(fromId);
        const toUuid = this.resolveToUuid(toId);

        if (!fromUuid || !toUuid) {
            console.error("Invalid node IDs for connection", {
                fromId,
                toId,
            });
            return null;
        }

        const connectionKey = `${fromUuid}-${fromPort}-${toUuid}-${toPort}`;

        if (this.state.currentPipeline.connections.has(connectionKey)) {
            console.warn("Connection already exists:", connectionKey);
            return connectionKey;
        }

        const existingToConnection = Array.from(
            this.state.currentPipeline.connections.values(),
        ).find((conn) => conn.toUuid === toUuid && conn.toPort === toPort);

        if (existingToConnection) {
            const existingKey = `${existingToConnection.fromUuid}-${existingToConnection.fromPort}-${existingToConnection.toUuid}-${existingToConnection.toPort}`;
            this.removeConnection(existingKey);
        }

        const connection = {
            fromUuid,
            fromPort,
            toUuid,
            toPort,
            dataType,
            isDefault: isDefault || false,
            customWaypoints: customWaypoints || null,
        };

        this.state.currentPipeline.connections.set(connectionKey, connection);

        this.emit("connection:added", { key: connectionKey, connection });
        this.emit("connections:changed", {
            type: "added",
            key: connectionKey,
        });
        this.emit("pipeline:changed", {
            type: "connection:added",
            key: connectionKey,
        });

        return connectionKey;
    }

    updateConnectionWaypoints(connectionKey, customWaypoints) {
        const connection =
            this.state.currentPipeline.connections.get(connectionKey);
        if (!connection) return false;

        connection.customWaypoints = customWaypoints;

        this.emit("connection:waypoints:changed", {
            key: connectionKey,
            customWaypoints,
        });
        this.emit("pipeline:changed", {
            type: "connection:waypoints:changed",
            key: connectionKey,
        });

        return true;
    }

    removeConnection(connectionKey) {
        if (!this.state.currentPipeline.connections.has(connectionKey)) {
            return false;
        }

        this.state.currentPipeline.connections.delete(connectionKey);

        this.emit("connection:removed", { key: connectionKey });
        this.emit("connections:changed", {
            type: "removed",
            key: connectionKey,
        });
        this.emit("pipeline:changed", {
            type: "connection:removed",
            key: connectionKey,
        });

        return true;
    }

    toggleConnectionDefault(connectionKey) {
        const connection =
            this.state.currentPipeline.connections.get(connectionKey);
        if (!connection) return false;

        connection.isDefault = !connection.isDefault;

        this.emit("connection:default:toggled", {
            key: connectionKey,
            isDefault: connection.isDefault,
        });
        this.emit("connections:changed", {
            type: "default:toggled",
            key: connectionKey,
        });
        this.emit("pipeline:changed", {
            type: "connection:default:toggled",
            key: connectionKey,
        });

        return true;
    }

    resolveToUuid(identifier) {
        if (this.state.currentPipeline.nodes.has(identifier)) {
            return identifier;
        }

        return this.instanceIdToUuid.get(identifier) || null;
    }

    resolveToInstanceId(uuid) {
        return this.uuidToInstanceId.get(uuid) || null;
    }

    getNode(identifier) {
        const uuid = this.resolveToUuid(identifier);
        return uuid ? this.state.currentPipeline.nodes.get(uuid) : null;
    }

    getNodes() {
        return Array.from(this.state.currentPipeline.nodes.values());
    }

    getNodesForRenderer() {
        return this.getNodes().map((node) => ({
            ...node,
            id: node.operationId,
            instanceId: node.instanceId,
        }));
    }

    getConnections() {
        return Array.from(this.state.currentPipeline.connections.values());
    }

    getConnectionsForRenderer() {
        return this.getConnections()
            .map((conn) => {
                const fromInstanceId = this.resolveToInstanceId(conn.fromUuid);
                const toInstanceId = this.resolveToInstanceId(conn.toUuid);

                if (!fromInstanceId || !toInstanceId) {
                    console.warn(
                        "Could not resolve instanceIds for connection",
                        conn,
                    );
                    return null;
                }

                return {
                    id: `${fromInstanceId}-${conn.fromPort}-${toInstanceId}-${conn.toPort}`,
                    fromNodeId: fromInstanceId,
                    toNodeId: toInstanceId,
                    fromPortName: conn.fromPort,
                    toPortName: conn.toPort,
                    dataType: conn.dataType,
                    isDefault: conn.isDefault,
                    customWaypoints: conn.customWaypoints || null,
                };
            })
            .filter(Boolean);
    }

    setRestartRequired(required) {
        this.state.ui.restartRequired = Boolean(required);
        this.emit("restart:changed", {
            restartRequired: this.state.ui.restartRequired,
        });
    }

    isRestartRequired() {
        return this.state.ui.restartRequired;
    }

    clearRestartRequired() {
        this.setRestartRequired(false);
    }

    clearPipeline() {
        this.state.currentPipeline.nodes.clear();
        this.state.currentPipeline.connections.clear();
        this.uuidToInstanceId.clear();
        this.instanceIdToUuid.clear();
        this.clearRestartRequired();
        this.clearOperationErrors();
        this.clearProfilingSnapshots();

        this.emit("pipeline:cleared");
    }

    setProfilingSnapshot(snapshot) {
        const pipelineName = snapshot?.pipeline_name;
        if (!pipelineName || typeof pipelineName !== "string") {
            return;
        }

        const copiedSnapshot = {
            ...snapshot,
            operations: { ...(snapshot.operations || {}) },
            timesteps: Array.isArray(snapshot.timesteps)
                ? snapshot.timesteps.map((row) => ({ ...row }))
                : [],
        };

        this.state.ui.profilingByPipeline.set(pipelineName, copiedSnapshot);
        this.state.ui.profilingLastUpdateMsByPipeline.set(
            pipelineName,
            Date.now(),
        );
        this.emit("profiling:updated", {
            pipelineName,
            snapshot: copiedSnapshot,
        });
    }

    getProfilingSnapshot(pipelineName) {
        const snapshot = this.state.ui.profilingByPipeline.get(pipelineName);
        if (!snapshot) {
            return null;
        }

        return {
            ...snapshot,
            operations: { ...(snapshot.operations || {}) },
            timesteps: Array.isArray(snapshot.timesteps)
                ? snapshot.timesteps.map((row) => ({ ...row }))
                : [],
        };
    }

    getProfilingLastUpdateMs(pipelineName) {
        return (
            this.state.ui.profilingLastUpdateMsByPipeline.get(pipelineName) || 0
        );
    }

    clearProfilingSnapshots() {
        this.state.ui.profilingByPipeline.clear();
        this.state.ui.profilingLastUpdateMsByPipeline.clear();
        this.emit("profiling:cleared", {});
    }

    setOperationErrors(errors) {
        const errorMap = new Map();
        (errors || []).forEach((errorRecord) => {
            if (errorRecord?.uuid) {
                errorMap.set(errorRecord.uuid, { ...errorRecord });
            }
        });

        this.state.ui.operationErrors = errorMap;
        this.updateDownstreamDisabledNodes();
        this.emit("operation-errors:changed", {
            errors: this.getOperationErrors(),
        });
    }

    clearOperationErrors() {
        this.state.ui.operationErrors = new Map();
        this.state.ui.downstreamDisabledNodes.clear();
        this.emit("operation-errors:changed", { errors: [] });
    }

    getOperationErrors() {
        return Array.from(this.state.ui.operationErrors.values());
    }

    getDownstreamDisabledNodes() {
        return new Set(this.state.ui.downstreamDisabledNodes);
    }

    updateDownstreamDisabledNodes() {
        const disabledNodes = new Set();
        const errorUuids = new Set(this.state.ui.operationErrors.keys());

        if (errorUuids.size === 0) {
            this.state.ui.downstreamDisabledNodes.clear();
            return;
        }

        const outgoingConnections = new Map();
        for (const connection of this.state.currentPipeline.connections.values()) {
            if (!outgoingConnections.has(connection.fromUuid)) {
                outgoingConnections.set(connection.fromUuid, []);
            }
            outgoingConnections
                .get(connection.fromUuid)
                .push(connection.toUuid);
        }

        const queue = Array.from(errorUuids);
        const visited = new Set(errorUuids);

        while (queue.length > 0) {
            const current = queue.shift();
            const downstream = outgoingConnections.get(current) || [];
            for (const next of downstream) {
                if (visited.has(next)) {
                    continue;
                }
                visited.add(next);
                disabledNodes.add(next);
                queue.push(next);
            }
        }

        this.state.ui.downstreamDisabledNodes = disabledNodes;
    }

    loadPipelineData(configItems, connectionsData = []) {
        this.clearPipeline();

        const uuidToNode = new Map();

        configItems.forEach((item) => {
            const node = this.addNode(
                {
                    id: item.action_name,
                    config: item.action_params || {},
                },
                item.position,
                item.uuid,
            );

            if (node) {
                uuidToNode.set(node.uuid, node);
                if (item.action_params) {
                    node.originalConfig = { ...item.action_params };
                }
            }
        });

        connectionsData.forEach((conn) => {
            this.addConnection(
                conn.from_uuid,
                conn.from_port,
                conn.to_uuid,
                conn.to_port,
                conn.data_type,
                conn.is_default || false,
                conn.custom_waypoints || null,
            );
        });

        this.emit("pipeline:loaded", {
            nodeCount: this.state.currentPipeline.nodes.size,
            connectionCount: this.state.currentPipeline.connections.size,
        });
    }

    exportToConfig() {
        const nodes = this.getNodes();
        const config = [];

        nodes.forEach((node) => {
            const nodeConnections = [];

            for (const conn of this.state.currentPipeline.connections.values()) {
                if (conn.fromUuid === node.uuid) {
                    const connData = {
                        from_uuid: conn.fromUuid,
                        from_port: conn.fromPort,
                        to_uuid: conn.toUuid,
                        to_port: conn.toPort,
                        data_type: conn.dataType,
                        is_default: conn.isDefault,
                        custom_waypoints: conn.customWaypoints || null,
                    };

                    nodeConnections.push(connData);
                }
            }

            config.push({
                action_name: node.operationId,
                action_params: node.config,
                position: node.position,
                uuid: node.uuid,
                connections: nodeConnections,
            });
        });

        return config;
    }
}

export const pipelineStore = new PipelineStore();
