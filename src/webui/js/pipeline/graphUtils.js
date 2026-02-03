/**
 * Graph utilities for cycle detection in pipeline connections
 */

/**
 * Detects cycles in the pipeline graph using DFS
 * @param {Map} nodes - Map of instanceId to node objects
 * @param {Array} connections - Array of connection objects
 * @returns {Array} - Array of connection IDs that are part of cycles
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
    function dfs(nodeId, visited, recStack, path) {
        visited.add(nodeId);
        recStack.add(nodeId);
        path.push(nodeId);

        const neighbors = adj[nodeId] || [];

        for (const neighbor of neighbors) {
            if (!visited.has(neighbor)) {
                // Continue DFS
                const cycleFound = dfs(neighbor, visited, recStack, path);
                if (cycleFound) {
                    return true;
                }
            } else if (recStack.has(neighbor)) {
                // Cycle detected - mark ALL connections in the cycle
                const neighborIndex = path.indexOf(neighbor);
                if (neighborIndex === -1) {
                    return true;
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

                return true;
            }
        }

        recStack.delete(nodeId);
        path.pop();
        return false;
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
