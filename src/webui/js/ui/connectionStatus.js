// Tracks connection state and notifies subscribers when backend/network status changes.
const listeners = new Set();

let backendConnected = false;
let networkTablesConnected = false;

/**
 * Resolves the combined connection status from backend and NetworkTables state.
 *
 * @returns {"disconnected"|"partial"|"connected"}
 */
function resolveStatus() {
    if (!backendConnected) {
        return "disconnected";
    }

    return networkTablesConnected ? "connected" : "partial";
}

/**
 * Builds a snapshot of the current connection state.
 *
 * @returns {{backendConnected: boolean, networkTablesConnected: boolean, status: string}}
 */
function getSnapshot() {
    return {
        backendConnected,
        networkTablesConnected,
        status: resolveStatus(),
    };
}

/**
 * Notifies all subscribed listeners with the latest connection snapshot.
 */
function notifyListeners() {
    const snapshot = getSnapshot();

    for (const listener of listeners) {
        listener(snapshot);
    }
}

/**
 * Subscribes to connection status updates.
 *
 * @param {Function} listener - Callback invoked immediately and on future changes.
 * @returns {Function} Unsubscribe function.
 */
export function subscribeConnectionStatus(listener) {
    if (typeof listener !== "function") {
        return () => {};
    }

    listeners.add(listener);
    listener(getSnapshot());

    return () => {
        listeners.delete(listener);
    };
}

/**
 * Updates the backend connection flag.
 *
 * @param {*} value - Truthy value indicates connected.
 */
export function setBackendConnected(value) {
    const nextValue = Boolean(value);
    if (backendConnected === nextValue) {
        return;
    }

    backendConnected = nextValue;
    notifyListeners();
}

/**
 * Updates the NetworkTables connection flag.
 *
 * @param {*} value - Truthy value indicates connected.
 */
export function setNetworkTablesConnected(value) {
    const nextValue = Boolean(value);
    if (networkTablesConnected === nextValue) {
        return;
    }

    networkTablesConnected = nextValue;
    notifyListeners();
}
