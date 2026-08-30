// Handles WiFi network discovery, connection, and disconnection in the web UI.
import { BACKEND_BASE_URL } from "../config.js";
import {
    closeOnBackdropClick,
    closeOnEscape,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "../ui/modal.js";
import { confirmDialog } from "../ui/confirmationDialog.js";
import {
    showDanger,
    showSuccess,
    showWarning,
} from "../ui/notificationSystem.js";

const OVERLAY_ID = "networkManagerOverlay";
const MODAL_ID = "networkManagerModal";

let initialized = false;
let networks = [];
let loading = false;
let activeRequestSsid = "";
let networkManagerAvailable = false;

/**
 * Gets or creates the network manager overlay and modal elements.
 *
 * @returns {{overlay: HTMLElement, modal: HTMLElement}} The overlay elements.
 */
function getOverlayElements() {
    return getOrCreateModalElements({
        overlayId: OVERLAY_ID,
        modalId: MODAL_ID,
        modalClassName:
            "bg-[#1a1a1a] rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] flex flex-col border border-[#414141]",
    });
}

/**
 * Fetches JSON from the backend and throws on non-OK responses.
 *
 * @param {string} path The backend path to request.
 * @param {RequestInit} [options={}] Fetch options.
 * @returns {Promise<object>} The parsed JSON payload.
 */
async function fetchJson(path, options = {}) {
    const response = await fetch(`${BACKEND_BASE_URL}${path}`, options);
    let payload = {};
    try {
        payload = await response.json();
    } catch {
        payload = {};
    }
    if (!response.ok) {
        const error = new Error(
            payload.error || `Request failed: ${response.status}`,
        );
        error.status = response.status;
        error.payload = payload;
        throw error;
    }
    return payload;
}

/**
 * Converts a numeric signal strength into a readable label.
 *
 * @param {number} signal The signal strength percentage.
 * @returns {string} A human-readable signal label.
 */
function signalLabel(signal) {
    if (!Number.isFinite(signal)) {
        return "Unknown";
    }
    if (signal >= 75) {
        return "Strong";
    }
    if (signal >= 45) {
        return "Good";
    }
    return "Weak";
}

/**
 * Determines whether a network requires a password.
 *
 * @param {object} network The network record.
 * @returns {boolean} True when a password is required.
 */
function networkNeedsPassword(network) {
    const security = String(network.security || "").toLowerCase();
    return security !== "" && security !== "open" && security !== "--";
}

/**
 * Returns whether NetworkManager reports enterprise authentication support.
 * @param {object} network The target network.
 * @returns {boolean} True for EAP or 802.1X networks.
 */
function networkNeedsUsername(network) {
    const security = String(network.security || "").toLowerCase();
    return security.includes("802.1x") || security.includes("eap");
}

/**
 * Loads the current WiFi network list from the backend.
 *
 * @returns {Promise<void>}
 */
async function loadNetworks() {
    if (!networkManagerAvailable) {
        showWarning("Network Manager requires Linux.");
        return;
    }

    loading = true;
    render();

    try {
        const payload = await fetchJson("/wifi-networks");
        networks = Array.isArray(payload.networks) ? payload.networks : [];
    } catch (error) {
        console.error("Failed to load WiFi networks:", error);
        networks = [];
        showDanger(error.payload?.error || "Failed to load WiFi networks");
    } finally {
        loading = false;
        render();
    }
}

/**
 * Connects to a selected WiFi network.
 *
 * @param {object} network The target network.
 * @param {HTMLInputElement|undefined} usernameInput The username input element.
 * @param {HTMLInputElement|undefined} passwordInput The password input element.
 * @returns {Promise<void>}
 */
async function connectNetwork(network, usernameInput, passwordInput) {
    const username = usernameInput?.value.trim() || "";
    const password = passwordInput?.value || "";
    activeRequestSsid = network.ssid;
    render();

    try {
        await fetchJson("/wifi-networks/connect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                ssid: network.ssid,
                username,
                password,
            }),
        });
        showSuccess(`Connected to ${network.ssid}.`);
        await loadNetworks();
    } catch (error) {
        console.error("Failed to connect WiFi network:", error);
        showDanger(error.payload?.error || "Failed to connect WiFi network");
    } finally {
        activeRequestSsid = "";
        render();
    }
}

/**
 * Disconnects from a selected WiFi network after confirmation.
 *
 * @param {object} network The target network.
 * @returns {Promise<void>}
 */
async function disconnectNetwork(network) {
    const shouldDisconnect = await confirmDialog({
        title: "Disconnect WiFi?",
        message: `Disconnect from "${network.ssid}"?`,
        detail: "Network access to this device may be interrupted.",
        confirmText: "Disconnect",
        variant: "warning",
    });
    if (!shouldDisconnect) {
        return;
    }

    activeRequestSsid = network.ssid;
    render();

    try {
        await fetchJson("/wifi-networks/disconnect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ssid: network.ssid }),
        });
        showWarning(`Disconnected from ${network.ssid}.`);
        await loadNetworks();
    } catch (error) {
        console.error("Failed to disconnect WiFi network:", error);
        showDanger(error.payload?.error || "Failed to disconnect WiFi network");
    } finally {
        activeRequestSsid = "";
        render();
    }
}

/**
 * Renders the network list rows into the provided container.
 *
 * @param {HTMLElement} container The container element for network rows.
 */
function renderNetworkRows(container) {
    container.innerHTML = "";

    if (loading) {
        container.appendChild(
            createElement("div", {
                className: "text-center text-[#ac8a2f] py-8",
                text: "Scanning for WiFi networks...",
            }),
        );
        return;
    }

    if (networks.length === 0) {
        container.appendChild(
            createElement("div", {
                className: "text-center text-[#ac8a2f] py-8",
                text: "No WiFi networks found.",
            }),
        );
        return;
    }

    networks.forEach((network) => {
        const usernameInput = createElement("input", {
            type: "text",
            placeholder: "Username",
            autocomplete: "username",
            className:
                "w-48 bg-[#101010] text-white text-sm px-3 py-2 rounded border border-[#414141] focus:border-[#f9c845] focus:outline-none disabled:opacity-50",
        });
        usernameInput.disabled =
            network.connected || !networkNeedsUsername(network);
        usernameInput.hidden = !networkNeedsUsername(network);

        const passwordInput = createElement("input", {
            type: "password",
            placeholder: "Password",
            autocomplete: "current-password",
            className:
                "w-40 bg-[#101010] text-white text-sm px-3 py-2 rounded border border-[#414141] focus:border-[#f9c845] focus:outline-none disabled:opacity-50",
        });
        passwordInput.disabled =
            network.connected || !networkNeedsPassword(network);

        const isBusy = activeRequestSsid === network.ssid;
        const actionButton = network.connected
            ? createElement("button", {
                  type: "button",
                  className:
                      "w-28 px-3 py-2 bg-red-800 text-white rounded-md hover:bg-red-700 text-sm disabled:opacity-60",
                  text: isBusy ? "Working..." : "Disconnect",
                  disabled: isBusy ? "disabled" : undefined,
                  onclick: () => disconnectNetwork(network),
              })
            : createElement("button", {
                  type: "button",
                  className:
                      "w-28 px-3 py-2 bg-[#2a2a2a] text-[#f9c845] rounded-md border border-[#414141] hover:bg-[#3a3a3a] hover:border-[#f9c845] text-sm disabled:opacity-60",
                  text: isBusy ? "Working..." : "Connect",
                  disabled: isBusy ? "disabled" : undefined,
                  onclick: () =>
                      connectNetwork(network, usernameInput, passwordInput),
              });

        const statusBadge = createElement("span", {
            className: network.connected
                ? "rounded-full border border-emerald-500/40 bg-emerald-900/30 px-2 py-1 text-xs font-semibold text-emerald-200"
                : "rounded-full border border-gray-500/40 bg-gray-800/40 px-2 py-1 text-xs font-semibold text-gray-200",
            text: network.connected ? "Connected" : "Available",
        });

        const info = createElement("div", { className: "flex-1 min-w-0" }, [
            createElement("div", {
                className: "text-white font-medium truncate",
                text: network.ssid,
                title: network.ssid,
            }),
            createElement("div", {
                className: "text-xs text-[#ac8a2f] mt-1",
                text: `${signalLabel(network.signal)} signal (${network.signal ?? 0}%) | ${network.security || "Open"}`,
            }),
        ]);

        const controls = createElement(
            "div",
            {
                className:
                    "flex flex-wrap items-center justify-end gap-2 shrink-0",
            },
            [statusBadge, usernameInput, passwordInput, actionButton],
        );

        container.appendChild(
            createElement(
                "div",
                {
                    className:
                        "flex items-center justify-between gap-3 p-3 border-b border-[#414141] hover:bg-[#232323]",
                },
                [info, controls],
            ),
        );
    });
}

/**
 * Renders the network manager modal content.
 */
function render() {
    const { modal } = getOverlayElements();
    modal.innerHTML = "";

    const closeButton = createElement("button", {
        type: "button",
        className: "absolute top-4 right-4 text-[#ac8a2f] hover:text-white",
        text: "x",
        onclick: close,
        style: "font-size: 1.5rem; line-height: 1;",
    });

    const refreshButton = createElement("button", {
        type: "button",
        className:
            "px-4 py-2 bg-[#2a2a2a] text-[#f9c845] rounded-md border border-[#414141] hover:bg-[#3a3a3a] hover:border-[#f9c845] disabled:opacity-60",
        text: loading ? "Scanning..." : "Refresh",
        disabled: loading ? "disabled" : undefined,
        onclick: loadNetworks,
    });

    const header = createElement(
        "div",
        {
            className: "p-6 border-b border-[#414141] relative",
        },
        [
            createElement("h2", {
                className: "text-xl font-bold text-[#f9c845]",
                text: "Network Manager",
            }),
            createElement("p", {
                className: "text-sm text-gray-300 mt-2",
                text: "Manage WiFi networks visible to the backend host.",
            }),
            closeButton,
        ],
    );

    const listContainer = createElement("div", {
        id: "networkManagerList",
        className:
            "border border-[#414141] rounded-lg bg-[#1f1f1f] max-h-[55vh] overflow-y-auto",
    });

    const body = createElement(
        "div",
        {
            className: "p-6 flex-1 overflow-y-auto",
        },
        [listContainer],
    );

    const footer = createElement(
        "div",
        {
            className: "p-6 border-t border-[#414141] flex justify-end gap-3",
        },
        [
            refreshButton,
            createElement("button", {
                type: "button",
                className:
                    "px-4 py-2 bg-[#414141] text-white rounded-md hover:bg-[#515151]",
                text: "Close",
                onclick: close,
            }),
        ],
    );

    modal.appendChild(header);
    modal.appendChild(body);
    modal.appendChild(footer);
    renderNetworkRows(listContainer);
}

/**
 * Opens the network manager modal and loads networks.
 */
function open() {
    if (!networkManagerAvailable) {
        showWarning("Network Manager requires Linux.");
        return;
    }

    const { overlay } = getOverlayElements();
    render();
    showModal(overlay);
    loadNetworks();
}

/**
 * Closes the network manager modal.
 */
function close() {
    const { overlay } = getOverlayElements();
    hideModal(overlay);
}

/**
 * Updates the manage networks button based on backend availability.
 *
 * @param {{available?: boolean}|undefined} status The backend status payload.
 */
function setManageButtonAvailability(status) {
    const manageButton = document.getElementById("manageNetworksBtn");
    if (!manageButton) {
        return;
    }

    networkManagerAvailable = status?.available === true;
    manageButton.disabled = !networkManagerAvailable;
    manageButton.title = networkManagerAvailable ? "" : "Requires Linux";
}

/**
 * Loads the backend network manager availability status.
 *
 * @returns {Promise<void>}
 */
async function loadNetworkManagerStatus() {
    try {
        const payload = await fetchJson("/wifi-networks/status");
        setManageButtonAvailability(payload);
    } catch (error) {
        console.error("Failed to load Network Manager status:", error);
        setManageButtonAvailability({ available: false });
    }
}

/**
 * Initializes the network manager UI wiring.
 */
export function initializeNetworkManager() {
    if (initialized) {
        return;
    }
    initialized = true;

    const { overlay } = getOverlayElements();
    closeOnBackdropClick(overlay, close);
    closeOnEscape(overlay, close);

    const manageButton = document.getElementById("manageNetworksBtn");
    if (manageButton) {
        setManageButtonAvailability({ available: false });
        manageButton.addEventListener("click", open);
    }
    loadNetworkManagerStatus();

    globalThis.NetworkManager = {
        open,
        close,
        loadNetworks,
        loadNetworkManagerStatus,
    };
}
