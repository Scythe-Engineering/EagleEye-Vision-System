// Central store for pipeline state, nodes, connections, and related UI events.
import { getCachedConfig, prefetchConfigs } from "./operationConfigCache.js";
import { resolveDockingContract } from "./dockingContract.js";
import { uid } from "./utils.js";

/** Return a declared port name from a string or object declaration. */
function declaredPortName(node) {
    return typeof node === "object" && node?.name
        ? String(node.name)
        : String(node);
}

/** Validate a static or indexed dynamic port against cached operation metadata. */
function isDeclaredPort(config, direction, portName) {
    if (!config || typeof config !== "object") return false;
    const nodes = Array.isArray(config[`${direction}_nodes`])
        ? config[`${direction}_nodes`]
        : [];
    const dynamicGroup =
        config.dynamic_group && typeof config.dynamic_group === "object"
            ? config.dynamic_group
            : null;
    let dynamicBase = null;
    let dynamicEnabled = false;
    let dynamicMaximum = null;
    if (dynamicGroup) {
        if (direction === "input") {
            dynamicEnabled = dynamicGroup.input_dynamic_group !== false;
            dynamicBase =
                dynamicGroup.input_base_name ||
                dynamicGroup.input_node ||
                dynamicGroup.input_prefix ||
                null;
            dynamicMaximum = dynamicGroup.max_inputs;
        } else {
            dynamicEnabled = Boolean(
                dynamicGroup.output_dynamic_group ||
                    dynamicGroup.mirrored_output_group,
            );
            dynamicBase =
                dynamicGroup.output_base_name ||
                dynamicGroup.output_node ||
                dynamicGroup.output_prefix ||
                null;
            dynamicMaximum = dynamicGroup.max_outputs;
        }
        if (!dynamicBase && nodes.length) {
            dynamicBase = declaredPortName(nodes.at(-1));
        }
    }

    const staticNames = nodes
        .map(declaredPortName)
        .filter((name) => !dynamicEnabled || name !== dynamicBase);
    if (staticNames.includes(portName)) return true;
    if (!dynamicEnabled || !dynamicBase) return false;
    if (portName === dynamicBase) return true;
    const escapedBase = dynamicBase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const match = portName.match(new RegExp(`^${escapedBase}_(\\d+)$`));
    if (!match || Number(match[1]) < 1) return false;
    const maximum = Number.parseInt(dynamicMaximum, 10);
    return !Number.isFinite(maximum) || Number(match[1]) <= maximum;
}

class PipelineNode {
    /**
     * Create a pipeline node from operation metadata.
     *
     * @param {Object} operationData Operation metadata.
     * @param {string|null} [existingUuid=null] Optional existing node UUID.
     * @param {{x: number, y: number}|null} [existingPosition=null] Optional existing position.
     */
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

    /**
     * Serialize the node for persistence.
     *
     * @returns {Object} Serialized node data.
     */
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
    /**
     * Initialize the pipeline store and its in-memory state.
     */
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
        this.suppressCameraAutoSelect = false;
    }

    /**
     * Subscribe to store events.
     *
     * @param {string} event Event name.
     * @param {Function} callback Listener callback.
     * @returns {Function} Unsubscribe function.
     */
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

    /**
     * Emit a store event to listeners.
     *
     * @param {string} event Event name.
     * @param {*} data Event payload.
     */
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

    /**
     * Replace the available operations list.
     *
     * @param {Array<Object>} operations Operation definitions.
     */
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

    /**
     * Set the available cameras.
     *
     * @param {Array<Object>} cameras Camera definitions.
     */
    setCameras(cameras) {
        this.state.cameras = cameras;
        this.emit("cameras:loaded", cameras);
    }

    /**
     * Set the available pipelines.
     *
     * @param {Array<Object>} pipelines Pipeline definitions.
     */
    setPipelines(pipelines) {
        this.state.pipelines = pipelines;
        this.emit("pipelines:loaded", pipelines);
    }

    /**
     * Select the active camera.
     *
     * @param {string|null} cameraName Camera name.
     */
    setCurrentCamera(cameraName) {
        this.state.currentPipeline.cameraName = cameraName;
        this.emit("camera:selected", cameraName);
    }

    /**
     * Select the active pipeline.
     *
     * @param {string|null} pipelineName Pipeline name.
     */
    setCurrentPipeline(pipelineName) {
        this.state.currentPipeline.pipelineName = pipelineName;
        this.emit("pipeline:selected", pipelineName);
    }

    /**
     * Normalize an operation identifier for lookup.
     *
     * @param {string} id Operation identifier.
     * @returns {string} Normalized identifier.
     */
    normalizeOperationId(id) {
        if (!id) return "";
        return id.replace(/\.py$/, "").toLowerCase().replace(/\s+/g, "_");
    }

    /**
     * Find an operation by its action name.
     *
     * @param {string} actionName Action name.
     * @returns {Object|null} Matching operation.
     */
    findOperation(actionName) {
        const normalized = this.normalizeOperationId(actionName);
        return (
            this.operationLookup.get(actionName) ||
            this.operationLookup.get(normalized) ||
            null
        );
    }

    /**
     * Add a node to the current pipeline.
     *
     * @param {Object} operationData Node operation data.
     * @param {{x: number, y: number}|null} [position=null] Node position.
     * @param {string|null} [existingUuid=null] Optional UUID to reuse.
     * @returns {PipelineNode|null} Created node.
     */
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

    /**
     * Remove a node and its attached connections.
     *
     * @param {string} identifier Node UUID or instance ID.
     * @returns {boolean} True when removed.
     */
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

    /**
     * Update a node's position.
     *
     * @param {string} identifier Node UUID or instance ID.
     * @param {{x: number, y: number}} position New position.
     * @returns {boolean} True when updated.
     */
    updateNodePosition(identifier, position) {
        const uuid = this.resolveToUuid(identifier);
        if (!uuid) return false;

        const node = this.state.currentPipeline.nodes.get(uuid);
        if (!node) return false;

        node.position = position;
        this.emit("node:position:changed", { uuid, position });

        return true;
    }

    /**
     * Update a node's configuration.
     *
     * @param {string} identifier Node UUID or instance ID.
     * @param {Object} config New configuration object.
     * @returns {boolean} True when updated.
     */
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
        if (!this.suppressCameraAutoSelect && this.isDeviceInputNode(node)) {
            this.autoSelectCameraBusIds();
        }

        return true;
    }

    /**
     * Returns whether a node is a device input operation.
     *
     * @param {PipelineNode|object|null} node Pipeline node.
     * @returns {boolean} True for device_input nodes.
     */
    isDeviceInputNode(node) {
        return this.normalizeOperationId(node?.operationId) === "device_input";
    }

    /**
     * Returns a node's declared docking contract, if it has one. The fallback
     * keeps older operation lists usable while the definition cache is loading.
     *
     * @param {PipelineNode|object|null} node Pipeline node.
     * @returns {{source_action:string,source_port:string,target_port:string}|null}
     */
    getDockingMetadata(node) {
        if (!node) return null;
        const config = getCachedConfig(
            node.operationId,
            Boolean(node.isSecondary),
        );
        return resolveDockingContract(
            node.operationId,
            config?.docking,
            this.normalizeOperationId.bind(this),
        );
    }

    /**
     * Checks whether a connection fulfills a target node's docking contract.
     *
     * @param {object} connection Serialized in-memory connection.
     * @returns {boolean} True when the connection docks its target.
     */
    isDockingConnection(connection) {
        const source = this.state.currentPipeline.nodes.get(
            connection?.fromUuid,
        );
        const target = this.state.currentPipeline.nodes.get(connection?.toUuid);
        const docking = this.getDockingMetadata(target);
        return Boolean(
            source &&
                docking &&
                this.normalizeOperationId(source.operationId) ===
                    this.normalizeOperationId(docking.source_action) &&
                connection.fromPort === docking.source_port &&
                connection.toPort === docking.target_port,
        );
    }

    /**
     * Returns docking errors for targets that require a direct dock connection.
     * This public hook is consumed by save/start controls.
     *
     * @returns {Array<{uuid:string,message:string}>} Validation errors.
     */
    getDockingValidationErrors() {
        const errors = [];
        const sourceOwners = new Map();
        for (const node of this.state.currentPipeline.nodes.values()) {
            if (!this.getDockingMetadata(node)) continue;
            const dockingConnection = Array.from(
                this.state.currentPipeline.connections.values(),
            ).find(
                (connection) =>
                    connection.toUuid === node.uuid &&
                    this.isDockingConnection(connection),
            );
            if (!dockingConnection || dockingConnection.isDefault) {
                errors.push({
                    uuid: node.uuid,
                    message: `${node.name || node.operationId} must be connected directly to a Device Input frame.`,
                });
                continue;
            }
            if (sourceOwners.has(dockingConnection.fromUuid)) {
                errors.push({
                    uuid: node.uuid,
                    message: "A Device Input can dock only one MX3 detector.",
                });
            } else {
                sourceOwners.set(dockingConnection.fromUuid, node.uuid);
            }
        }
        return errors;
    }

    /**
     * Validates all required docking relationships.
     *
     * @returns {{valid:boolean,errors:Array<{uuid:string,message:string}>}}
     */
    validateDocking() {
        const errors = this.getDockingValidationErrors();
        return { valid: errors.length === 0, errors };
    }

    /**
     * Infer a camera bus ID for a node from exactly one upstream device_input.
     *
     * @param {string} identifier Node UUID or instance ID.
     * @returns {string|null} Inferred bus ID, or null when ambiguous/unavailable.
     */
    inferCameraBusIdForNode(identifier) {
        const uuid = this.resolveToUuid(identifier);
        if (!uuid) return null;

        const node = this.state.currentPipeline.nodes.get(uuid);
        if (!node || this.isDeviceInputNode(node)) return null;

        const upstreamByTarget = new Map();
        for (const conn of this.state.currentPipeline.connections.values()) {
            if (!upstreamByTarget.has(conn.toUuid)) {
                upstreamByTarget.set(conn.toUuid, []);
            }
            upstreamByTarget.get(conn.toUuid).push(conn.fromUuid);
        }

        const deviceInputBusIds = new Set();
        const visited = new Set();
        const stack = [...(upstreamByTarget.get(uuid) || [])];

        while (stack.length > 0) {
            const currentUuid = stack.pop();
            if (!currentUuid || visited.has(currentUuid)) continue;
            visited.add(currentUuid);

            const upstreamNode =
                this.state.currentPipeline.nodes.get(currentUuid);
            if (!upstreamNode) continue;

            if (this.isDeviceInputNode(upstreamNode)) {
                const busId = upstreamNode.config?.camera_bus_id;
                if (
                    busId !== undefined &&
                    busId !== null &&
                    String(busId) !== ""
                ) {
                    deviceInputBusIds.add(String(busId));
                }
                continue;
            }

            stack.push(...(upstreamByTarget.get(currentUuid) || []));
        }

        return deviceInputBusIds.size === 1
            ? Array.from(deviceInputBusIds)[0]
            : null;
    }

    /**
     * Auto-fill empty camera_bus_id settings from exactly one upstream device_input.
     * Existing values are treated as user-editable and are never overwritten.
     *
     * @returns {number} Number of nodes updated.
     */
    autoSelectCameraBusIds() {
        let updatedCount = 0;

        for (const node of this.state.currentPipeline.nodes.values()) {
            if (this.isDeviceInputNode(node)) continue;

            const config = node.config || {};
            const originalConfig = node.originalConfig || {};
            const exposesCameraBusId =
                Object.prototype.hasOwnProperty.call(config, "camera_bus_id") ||
                Object.prototype.hasOwnProperty.call(
                    originalConfig,
                    "camera_bus_id",
                );
            const currentValue = config.camera_bus_id;

            if (
                !exposesCameraBusId ||
                (currentValue !== undefined &&
                    currentValue !== null &&
                    String(currentValue) !== "")
            ) {
                continue;
            }

            const inferredBusId = this.inferCameraBusIdForNode(node.uuid);
            if (!inferredBusId) continue;

            node.config = { ...config, camera_bus_id: inferredBusId };
            this.emit("node:config:changed", {
                uuid: node.uuid,
                config: node.config,
            });
            this.emit("pipeline:changed", {
                type: "node:config:changed",
                uuid: node.uuid,
            });
            updatedCount += 1;
        }

        return updatedCount;
    }

    /**
     * Add a connection between two nodes.
     *
     * @param {string} fromId Source node UUID or instance ID.
     * @param {string} fromPort Source port.
     * @param {string} toId Target node UUID or instance ID.
     * @param {string} toPort Target port.
     * @param {string|null} [dataType=null] Connection data type.
     * @param {boolean} [isDefault=false] Whether this is the default connection.
     * @param {Array|null} [customWaypoints=null] Optional custom waypoints.
     * @returns {string|null} Connection key.
     */
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

        const sourceNode = this.state.currentPipeline.nodes.get(fromUuid);
        const destinationNode = this.state.currentPipeline.nodes.get(toUuid);
        const sourceConfig = getCachedConfig(
            sourceNode?.operationId,
            Boolean(sourceNode?.isSecondary),
        );
        const destinationConfig = getCachedConfig(
            destinationNode?.operationId,
            Boolean(destinationNode?.isSecondary),
        );
        if (sourceConfig == null) {
            console.warn(
                `Connection output port could not be validated: config unavailable for ${sourceNode?.operationId ?? fromId}`,
            );
        } else if (!isDeclaredPort(sourceConfig, "output", fromPort)) {
            console.error(
                `Connection rejected: ${fromPort} is not a declared output port`,
            );
            return null;
        }
        if (destinationConfig == null) {
            console.warn(
                `Connection input port could not be validated: config unavailable for ${destinationNode?.operationId ?? toId}`,
            );
        } else if (!isDeclaredPort(destinationConfig, "input", toPort)) {
            console.error(
                `Connection rejected: ${toPort} is not a declared input port`,
            );
            return null;
        }

        const connectionKey = `${fromUuid}-${fromPort}-${toUuid}-${toPort}`;

        if (this.state.currentPipeline.connections.has(connectionKey)) {
            console.warn("Connection already exists:", connectionKey);
            return connectionKey;
        }

        const prospectiveConnection = {
            fromUuid,
            fromPort,
            toUuid,
            toPort,
        };
        if (this.isDockingConnection(prospectiveConnection)) {
            const alreadyDocked = Array.from(
                this.state.currentPipeline.connections.values(),
            ).some(
                (connection) =>
                    connection.fromUuid === fromUuid &&
                    this.isDockingConnection(connection),
            );
            if (alreadyDocked) {
                console.error(
                    "Connection rejected: a Device Input can dock only one MX3 detector",
                );
                return null;
            }
        }

        const existingToConnection = Array.from(
            this.state.currentPipeline.connections.values(),
        ).find((conn) => conn.toUuid === toUuid && conn.toPort === toPort);

        if (existingToConnection) {
            const existingKey = `${existingToConnection.fromUuid}-${existingToConnection.fromPort}-${existingToConnection.toUuid}-${existingToConnection.toPort}`;
            this.removeConnection(existingKey, { notify: false });
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
        if (!this.suppressCameraAutoSelect) {
            this.autoSelectCameraBusIds();
        }

        return connectionKey;
    }

    /**
     * Update connection waypoints.
     *
     * @param {string} connectionKey Connection key.
     * @param {Array|null} customWaypoints New waypoint list.
     * @returns {boolean} True when updated.
     */
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

    /**
     * Remove a connection.
     *
     * @param {string} connectionKey Connection key.
     * @param {{notify?: boolean}} [options={}] Event notification options.
     * @returns {boolean} True when removed.
     */
    removeConnection(connectionKey, options = {}) {
        if (!this.state.currentPipeline.connections.has(connectionKey)) {
            return false;
        }

        this.state.currentPipeline.connections.delete(connectionKey);

        if (options.notify !== false) {
            this.emit("connection:removed", { key: connectionKey });
            this.emit("connections:changed", {
                type: "removed",
                key: connectionKey,
            });
            this.emit("pipeline:changed", {
                type: "connection:removed",
                key: connectionKey,
            });
            this.autoSelectCameraBusIds();
        }

        return true;
    }

    /**
     * Toggle a connection's default state.
     *
     * @param {string} connectionKey Connection key.
     * @returns {boolean} True when toggled.
     */
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

    /**
     * Resolve an identifier to a node UUID.
     *
     * @param {string} identifier Node UUID or instance ID.
     * @returns {string|null} UUID.
     */
    resolveToUuid(identifier) {
        if (this.state.currentPipeline.nodes.has(identifier)) {
            return identifier;
        }

        return this.instanceIdToUuid.get(identifier) || null;
    }

    /**
     * Resolve a UUID to an instance ID.
     *
     * @param {string} uuid Node UUID.
     * @returns {string|null} Instance ID.
     */
    resolveToInstanceId(uuid) {
        return this.uuidToInstanceId.get(uuid) || null;
    }

    /**
     * Get a node by UUID or instance ID.
     *
     * @param {string} identifier Node UUID or instance ID.
     * @returns {PipelineNode|null} Node.
     */
    getNode(identifier) {
        const uuid = this.resolveToUuid(identifier);
        return uuid ? this.state.currentPipeline.nodes.get(uuid) : null;
    }

    /**
     * Get all nodes in the current pipeline.
     *
     * @returns {Array<PipelineNode>} Nodes.
     */
    getNodes() {
        return Array.from(this.state.currentPipeline.nodes.values());
    }

    /**
     * Get nodes formatted for the renderer.
     *
     * @returns {Array<Object>} Renderer node data.
     */
    getNodesForRenderer() {
        return this.getNodes().map((node) => ({
            ...node,
            id: node.operationId,
            instanceId: node.instanceId,
        }));
    }

    /**
     * Get all current connections.
     *
     * @returns {Array<Object>} Connections.
     */
    getConnections() {
        return Array.from(this.state.currentPipeline.connections.values());
    }

    /**
     * Get connections formatted for the renderer.
     *
     * @returns {Array<Object>} Renderer connection data.
     */
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

    /**
     * Set whether the pipeline requires a restart.
     *
     * @param {boolean} required Restart flag.
     */
    setRestartRequired(required) {
        this.state.ui.restartRequired = Boolean(required);
        this.emit("restart:changed", {
            restartRequired: this.state.ui.restartRequired,
        });
    }

    /**
     * Check whether the pipeline requires a restart.
     *
     * @returns {boolean} Restart flag.
     */
    isRestartRequired() {
        return this.state.ui.restartRequired;
    }

    /**
     * Clear the restart-required flag.
     */
    clearRestartRequired() {
        this.setRestartRequired(false);
    }

    /**
     * Clear the current pipeline state.
     */
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

    /**
     * Store a profiling snapshot for a pipeline.
     *
     * @param {Object} snapshot Profiling snapshot.
     */
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

    /**
     * Retrieve a profiling snapshot for a pipeline.
     *
     * @param {string} pipelineName Pipeline name.
     * @returns {Object|null} Profiling snapshot.
     */
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

    /**
     * Get the last profiling update timestamp.
     *
     * @param {string} pipelineName Pipeline name.
     * @returns {number} Update time in ms.
     */
    getProfilingLastUpdateMs(pipelineName) {
        return (
            this.state.ui.profilingLastUpdateMsByPipeline.get(pipelineName) || 0
        );
    }

    /**
     * Clear all profiling snapshots.
     */
    clearProfilingSnapshots() {
        this.state.ui.profilingByPipeline.clear();
        this.state.ui.profilingLastUpdateMsByPipeline.clear();
        this.emit("profiling:cleared", {});
    }

    /**
     * Set operation errors and recompute downstream disabled nodes.
     *
     * @param {Array<Object>} errors Error records.
     */
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

    /**
     * Clear all operation errors.
     */
    clearOperationErrors() {
        this.state.ui.operationErrors = new Map();
        this.state.ui.downstreamDisabledNodes.clear();
        this.emit("operation-errors:changed", { errors: [] });
    }

    /**
     * Get the current operation errors.
     *
     * @returns {Array<Object>} Error records.
     */
    getOperationErrors() {
        return Array.from(this.state.ui.operationErrors.values());
    }

    /**
     * Get the set of downstream disabled node UUIDs.
     *
     * @returns {Set<string>} Disabled node UUIDs.
     */
    getDownstreamDisabledNodes() {
        return new Set(this.state.ui.downstreamDisabledNodes);
    }

    /**
     * Recompute nodes disabled by upstream operation errors.
     */
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

    /**
     * Load pipeline data into the store.
     *
     * @param {Array<Object>} configItems Serialized node config items.
     * @param {Array<Object>} [connectionsData=[]] Serialized connection items.
     * @returns {Promise<void>} Resolves once persisted connections are restored.
     */
    async loadPipelineData(configItems, connectionsData = []) {
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

        const operationsToPrefetch = [
            ...new Map(
                Array.from(uuidToNode.values()).map((node) => [
                    `${node.operationId}:${node.isSecondary ? 1 : 0}`,
                    {
                        name: node.operationId,
                        isSecondary: Boolean(node.isSecondary),
                    },
                ]),
            ).values(),
        ];
        await prefetchConfigs(operationsToPrefetch);

        this.suppressCameraAutoSelect = true;
        try {
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
        } finally {
            this.suppressCameraAutoSelect = false;
        }

        this.autoSelectCameraBusIds();

        this.emit("pipeline:loaded", {
            nodeCount: this.state.currentPipeline.nodes.size,
            connectionCount: this.state.currentPipeline.connections.size,
        });
    }

    /**
     * Export the current pipeline to config format.
     *
     * @returns {Array<Object>} Serialized pipeline config.
     */
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
