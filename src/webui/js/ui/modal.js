// Utilities for creating, showing, hiding, and wiring up backend modal dialogs.
const DEFAULT_OVERLAY_CLASS =
    "fixed inset-0 z-50 hidden flex items-center justify-center";
const DEFAULT_OVERLAY_STYLE =
    "background-color: rgba(0, 0, 0, 0.25); backdrop-filter: blur(6px);";
const DEFAULT_MODAL_CLASS =
    "bg-[#1a1a1a] rounded-lg shadow-xl w-full mx-4 max-h-[90vh] flex flex-col border border-[#414141]";

const managedModalOverlays = new Set();

/**
 * Track an overlay so it can be managed and closed later.
 *
 * @param {HTMLElement|null|undefined} overlay The modal overlay element.
 */
function registerManagedModalOverlay(overlay) {
    if (!overlay) {
        return;
    }
    overlay.dataset.backendModalOverlay = "true";
    managedModalOverlays.add(overlay);
}

/**
 * Hide all overlays that have been registered by this module.
 */
export function closeAllManagedModals() {
    for (const overlay of managedModalOverlays) {
        hideModal(overlay);
    }
}

if (typeof document !== "undefined") {
    document.addEventListener("backend-disconnected", closeAllManagedModals);
}

/**
 * Create a DOM element, assign attributes, and append children.
 *
 * @param {string} tag The element tag name.
 * @param {Object} attrs Attributes and event handlers to apply.
 * @param {Array<Node>} children Child nodes to append.
 * @returns {HTMLElement} The created element.
 */
export function createElement(tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => {
        if (key === "className") {
            el.className = value;
        } else if (key === "text") {
            el.textContent = value;
        } else if (key === "html") {
            el.innerHTML = value;
        } else if (key.startsWith("on") && typeof value === "function") {
            el.addEventListener(key.substring(2).toLowerCase(), value);
        } else if (value !== undefined && value !== null) {
            el.setAttribute(key, String(value));
        }
    });
    (children || []).forEach((child) => el.appendChild(child));
    return el;
}

/**
 * Get existing modal elements or create them if needed.
 *
 * @param {Object} options Configuration for the modal elements.
 * @param {string} options.overlayId Overlay element id.
 * @param {string} options.modalId Modal element id.
 * @param {string} [options.overlayClassName] Overlay CSS classes.
 * @param {string} [options.overlayStyle] Overlay inline style.
 * @param {string} [options.modalClassName] Modal CSS classes.
 * @returns {{overlay: HTMLElement, modal: HTMLElement}} The overlay and modal elements.
 */
export function getOrCreateModalElements({
    overlayId,
    modalId,
    overlayClassName = DEFAULT_OVERLAY_CLASS,
    overlayStyle = DEFAULT_OVERLAY_STYLE,
    modalClassName = DEFAULT_MODAL_CLASS,
}) {
    let overlay = document.getElementById(overlayId);
    let modal = document.getElementById(modalId);

    if (!overlay) {
        overlay = createElement("div", {
            id: overlayId,
            className: overlayClassName,
            style: overlayStyle,
        });
        document.body.appendChild(overlay);
    }
    registerManagedModalOverlay(overlay);

    if (!modal) {
        modal = createElement("div", {
            id: modalId,
            className: modalClassName,
        });
        overlay.appendChild(modal);
    }

    return { overlay, modal };
}

/**
 * Make a modal overlay visible.
 *
 * @param {HTMLElement|null|undefined} overlay The modal overlay element.
 */
export function showModal(overlay) {
    registerManagedModalOverlay(overlay);
    overlay?.classList.remove("hidden");
}

/**
 * Hide a modal overlay.
 *
 * @param {HTMLElement|null|undefined} overlay The modal overlay element.
 */
export function hideModal(overlay) {
    overlay?.classList.add("hidden");
}

/**
 * Close a modal when its backdrop is clicked.
 *
 * @param {HTMLElement|null|undefined} overlay The modal overlay element.
 * @param {Function} close Callback invoked when the backdrop is clicked.
 */
export function closeOnBackdropClick(overlay, close) {
    overlay?.addEventListener("click", (event) => {
        if (event.target === overlay) {
            close();
        }
    });
}

/**
 * Close a modal when Escape is pressed while it is open.
 *
 * @param {HTMLElement|null|undefined} overlay The modal overlay element.
 * @param {Function} close Callback invoked on Escape.
 */
export function closeOnEscape(overlay, close) {
    document.addEventListener(
        "keydown",
        (event) => {
            if (
                event.key === "Escape" &&
                !overlay?.classList.contains("hidden")
            ) {
                close();
            }
        },
        true,
    );
}
