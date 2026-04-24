const listeners = new Set();

let backendConnected = false;
let networkTablesConnected = false;

function resolveStatus() {
    if (!backendConnected) {
        return "disconnected";
    }

    return networkTablesConnected ? "connected" : "partial";
}

function getSnapshot() {
    return {
        backendConnected,
        networkTablesConnected,
        status: resolveStatus(),
    };
}

function notifyListeners() {
    const snapshot = getSnapshot();

    for (const listener of listeners) {
        listener(snapshot);
    }
}

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

export function setBackendConnected(value) {
    const nextValue = Boolean(value);
    if (backendConnected === nextValue) {
        return;
    }

    backendConnected = nextValue;
    notifyListeners();
}

export function setNetworkTablesConnected(value) {
    const nextValue = Boolean(value);
    if (networkTablesConnected === nextValue) {
        return;
    }

    networkTablesConnected = nextValue;
    notifyListeners();
}
