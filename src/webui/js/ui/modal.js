const DEFAULT_OVERLAY_CLASS =
    "fixed inset-0 z-50 hidden flex items-center justify-center";
const DEFAULT_OVERLAY_STYLE =
    "background-color: rgba(0, 0, 0, 0.25); backdrop-filter: blur(6px);";
const DEFAULT_MODAL_CLASS =
    "bg-[#1a1a1a] rounded-lg shadow-xl w-full mx-4 max-h-[90vh] flex flex-col border border-[#414141]";

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

    if (!modal) {
        modal = createElement("div", {
            id: modalId,
            className: modalClassName,
        });
        overlay.appendChild(modal);
    }

    return { overlay, modal };
}

export function showModal(overlay) {
    overlay?.classList.remove("hidden");
}

export function hideModal(overlay) {
    overlay?.classList.add("hidden");
}

export function closeOnBackdropClick(overlay, close) {
    overlay?.addEventListener("click", (event) => {
        if (event.target === overlay) {
            close();
        }
    });
}

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
