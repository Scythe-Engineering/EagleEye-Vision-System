// Utilities for analyzing pipeline graphs used by the web UI.
/**
 * Graph utilities for cycle detection in pipeline connections
 */

/**
 * Detects cycles in the pipeline graph using DFS.
 * @param {Map<string, object>} nodes - Map of node instance IDs to node objects.
 * @param {Array<object>} connections - Array of connection objects.
 * @returns {Array<string>} Array of connection IDs that are part of cycles.
 */
export function findCycles(nodes, connections) {
    // Build adjacency list and connection mapping, excluding default connections
    const adj = {};
    const connectionMap = {};

    // Initialize adjacency list for all nodes
    nodes.forEach((node, instanceId) => {
        adj[instanceId] = [];
    });

    // Build graph from connections
    connections.forEach((conn) => {
        // Skip default connections - they break cycles
        if (conn.isDefault) return;

        const fromId = conn.fromNodeId;
        const toId = conn.toNodeId;

        if (!adj[fromId]) adj[fromId] = [];
        adj[fromId].push(toId);

        // Store connection ID for cycle detection
        const key = `${fromId}->${toId}`;
        if (!connectionMap[key]) connectionMap[key] = [];
        connectionMap[key].push(conn.id);
    });

    // Track all connections that are part of any cycle
    const cycleConnectionIds = new Set();

    // DFS for cycle detection - returns array of nodes in current path
    /**
     * Traverses the graph depth-first to identify and mark cycle edges.
     * @param {string} nodeId - Current node instance ID.
     * @param {Set<string>} visited - Nodes already visited in this traversal.
     * @param {Set<string>} recStack - Nodes currently in the recursion stack.
     * @param {Array<string>} path - Ordered path of nodes from the DFS root.
     * @returns {boolean} True when a cycle is found in this branch.
     */
    function dfs(nodeId, visited, recStack, path) {
        visited.add(nodeId);
        recStack.add(nodeId);
        path.push(nodeId);

        const neighbors = adj[nodeId] || [];
        let cycleFound = false;

        for (const neighbor of neighbors) {
            if (!visited.has(neighbor)) {
                // Continue DFS
                const neighborCycleFound = dfs(
                    neighbor,
                    visited,
                    recStack,
                    path,
                );
                if (neighborCycleFound) {
                    cycleFound = true;
                }
            } else if (recStack.has(neighbor)) {
                // Cycle detected - mark ALL connections in the cycle
                const neighborIndex = path.indexOf(neighbor);
                if (neighborIndex === -1) {
                    cycleFound = true;
                    continue;
                }
                const cycleNodes = path.slice(neighborIndex).concat(neighbor);

                // Mark every edge in the cycle
                for (let i = 0; i < cycleNodes.length - 1; i++) {
                    const from = cycleNodes[i];
                    const to = cycleNodes[i + 1];
                    const key = `${from}->${to}`;
                    if (connectionMap[key]) {
                        connectionMap[key].forEach((id) => {
                            cycleConnectionIds.add(id);
                        });
                    }
                }

                // Also mark the edge that closes the cycle (last to first)
                const lastIdx = cycleNodes.length - 1;
                const closeKey = `${cycleNodes[lastIdx]}->${cycleNodes[0]}`;
                if (connectionMap[closeKey]) {
                    connectionMap[closeKey].forEach((id) => {
                        cycleConnectionIds.add(id);
                    });
                }
                cycleFound = true;
            }
        }

        recStack.delete(nodeId);
        path.pop();
        return cycleFound;
    }

    // Run DFS from all nodes to find all cycles
    nodes.forEach((node, instanceId) => {
        const visited = new Set();
        const recStack = new Set();
        const path = [];

        if (!visited.has(instanceId)) {
            dfs(instanceId, visited, recStack, path);
        }
    });

    return Array.from(cycleConnectionIds);
}

/**
 * Finds operation islands that cannot be reached from any data-source node.
 * @param {Map<string, object>} nodes - Map of node instance IDs to node objects.
 * @param {Array<object>} connections - Array of connection objects.
 * @returns {Array<Array<string>>} Groups of unreachable node instance IDs.
 */
export function findUnreachableIslands(nodes, connections) {
    const outgoing = new Map();
    const undirected = new Map();

    nodes.forEach((_node, instanceId) => {
        outgoing.set(instanceId, []);
        undirected.set(instanceId, new Set());
    });

    connections.forEach((conn) => {
        const fromId = conn.fromNodeId;
        const toId = conn.toNodeId;
        if (!nodes.has(fromId) || !nodes.has(toId)) {
            return;
        }

        outgoing.get(fromId).push(toId);
        undirected.get(fromId).add(toId);
        undirected.get(toId).add(fromId);
    });

    const roots = [];
    nodes.forEach((node, instanceId) => {
        if (node?.operationData?.isDataSource) {
            roots.push(instanceId);
        }
    });

    const reachable = new Set();
    const queue = [...roots];
    while (queue.length > 0) {
        const current = queue.shift();
        if (reachable.has(current)) {
            continue;
        }
        reachable.add(current);
        for (const next of outgoing.get(current) || []) {
            if (!reachable.has(next)) {
                queue.push(next);
            }
        }
    }

    const unreachable = new Set();
    nodes.forEach((_node, instanceId) => {
        if (!reachable.has(instanceId)) {
            unreachable.add(instanceId);
        }
    });

    const islands = [];
    const visited = new Set();
    for (const instanceId of unreachable) {
        if (visited.has(instanceId)) {
            continue;
        }

        const island = [];
        const componentQueue = [instanceId];
        visited.add(instanceId);
        while (componentQueue.length > 0) {
            const current = componentQueue.shift();
            island.push(current);

            for (const next of undirected.get(current) || []) {
                if (!unreachable.has(next) || visited.has(next)) {
                    continue;
                }
                visited.add(next);
                componentQueue.push(next);
            }
        }

        islands.push(island);
    }

    return islands;
}
