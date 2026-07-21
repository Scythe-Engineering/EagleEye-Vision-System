// Manages toast notification creation, stacking, dismissal, and visibility in the web UI.
const activeNotifications = [];
const MAX_VISIBLE_NOTIFICATIONS = 4;
const STACK_OFFSET = 8;
const NOTIFICATION_AUTO_DISMISS_MS = {
    success: 3000,
    warning: 10000,
    danger: 10000,
};

const CHECK_ICON_PATH =
    "M10 .5a9.5 9.5 0 1 0 9.5 9.5A9.51 9.51 0 0 0 10 .5Zm3.707 8.207-4 4a1 1 0 0 1-1.414 0l-2-2a1 1 0 0 1 1.414-1.414L9 10.586l3.293-3.293a1 1 0 0 1 1.414 1.414Z";
const ERROR_ICON_PATH =
    "M10 .5a9.5 9.5 0 1 0 9.5 9.5A9.51 9.51 0 0 0 10 .5Zm3.707 11.793a1 1 0 1 1-1.414 1.414L10 11.414l-2.293 2.293a1 1 0 0 1-1.414-1.414L8.586 10 6.293 7.707a1 1 0 0 1 1.414-1.414L10 8.586l2.293-2.293a1 1 0 0 1 1.414 1.414L11.414 10l2.293 2.293Z";
const UPLOAD_ICON_PATH =
    "M10 1a1 1 0 0 1 1 1v8.586l2.293-2.293a1 1 0 1 1 1.414 1.414l-4 4a1 1 0 0 1-1.414 0l-4-4a1 1 0 0 1 1.414-1.414L9 10.586V2a1 1 0 0 1 1-1Zm-7 13a1 1 0 0 1 1-1h12a1 1 0 1 1 0 2H4a1 1 0 0 1-1-1Z";

/** @type {Map<string, ReturnType<typeof setTimeout>>} */
const notificationDismissTimeouts = new Map();

/**
 * Clears the auto-dismiss timeout for a notification, if one exists.
 *
 * @param {string} notificationId - The notification element ID.
 */
function clearNotificationDismissTimeout(notificationId) {
    const timeoutId = notificationDismissTimeouts.get(notificationId);
    if (timeoutId !== undefined) {
        clearTimeout(timeoutId);
        notificationDismissTimeouts.delete(notificationId);
    }
}

/**
 * Gets the DOM element that contains all notifications.
 *
 * @returns {HTMLElement | null} The notification container element, or null if missing.
 */
function getNotificationContainer() {
    return document.getElementById("notification-container");
}

/**
 * Gets the "Clear All" button element.
 *
 * @returns {HTMLElement | null} The clear-all button element, or null if missing.
 */
function getClearAllButton() {
    return document.getElementById("clearAllNotificationsBtn");
}

/**
 * Updates the visibility of the "Clear All" button based on active notifications.
 */
function updateClearAllButtonVisibility() {
    const clearAllButton = getClearAllButton();
    if (clearAllButton) {
        if (activeNotifications.length > 0) {
            clearAllButton.classList.remove("hidden");
        } else {
            clearAllButton.classList.add("hidden");
        }
    }
}

/**
 * Recomputes notification stack positions, transforms, opacity, and z-index.
 */
function updateNotificationPositions() {
    const notificationContainer = getNotificationContainer();
    if (!notificationContainer) return;

    const baseHeight = 80;
    const totalHeight = activeNotifications.length > 0 
        ? baseHeight + (activeNotifications.length - 1) * STACK_OFFSET 
        : 0;
    notificationContainer.style.height = `${totalHeight}px`;

    // Use reversed copy to process newest to oldest (newest = index 0 visually)
    const reversedNotifications = [...activeNotifications].reverse();

    for (let index = 0; index < reversedNotifications.length; index++) {
        const notificationId = reversedNotifications[index];
        const notificationElement = document.getElementById(notificationId);
        if (!notificationElement) continue;

        const offsetY = index * STACK_OFFSET;
        const offsetX = index * 4; // Slight horizontal offset for depth
        const scale = 1 - index * 0.05;
        
        let opacity = 1;
        if (index > 0) {
            // Fade older notifications
            opacity = Math.max(0.5, 1 - index * 0.15);
        }
        
        // Hide notifications beyond max visible
        if (index >= MAX_VISIBLE_NOTIFICATIONS) {
            opacity = 0;
            notificationElement.style.pointerEvents = "none";
        } else {
            notificationElement.style.pointerEvents = "auto";
        }

        // Reset transform-based animation class conflicts
        notificationElement.classList.remove("notification-enter");

        notificationElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
        notificationElement.style.opacity = opacity.toString();
        // Z-index: Newest (index 0) gets highest
        notificationElement.style.zIndex = (reversedNotifications.length - index + 100).toString();
    }
}

/**
 * Removes a notification from the stack and cleans up its DOM element.
 *
 * @param {string} notificationId - The notification element ID.
 */
function removeNotification(notificationId) {
    clearNotificationDismissTimeout(notificationId);
    const index = activeNotifications.indexOf(notificationId);
    if (index > -1) {
        activeNotifications.splice(index, 1);
    }
    
    const notificationElement = document.getElementById(notificationId);
    if (notificationElement) {
        notificationElement.classList.add("notification-exit");
        
        // Listen for transition end or timeout to remove element
        setTimeout(() => {
            if (notificationElement.parentNode) {
                notificationElement.remove();
            }
            // Force update positions after removal to snap others up
            updateNotificationPositions();
        }, 400); // Match CSS transition duration
    }
    
    updateClearAllButtonVisibility();
}

/**
 * Creates a DOM element for a notification toast.
 *
 * @param {"success" | "warning" | "danger"} type - Notification type.
 * @param {string} message - Notification message content.
 * @returns {HTMLDivElement | null} The created notification element, or null if type is invalid.
 */
function createNotificationElement(type, message) {
    const notificationId = `toast-${type}-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    
    const toastTemplates = {
        success: {
            id: "toast-success",
            // Darker backgrounds for dark mode, distinct text colors
            iconColor: "text-green-400 bg-green-900/30", 
            borderColor: "border-green-500/50",
            iconPath: "M10 .5a9.5 9.5 0 1 0 9.5 9.5A9.51 9.51 0 0 0 10 .5Zm3.707 8.207-4 4a1 1 0 0 1-1.414 0l-2-2a1 1 0 0 1 1.414-1.414L9 10.586l3.293-3.293a1 1 0 0 1 1.414 1.414Z",
            iconLabel: "Check icon"
        },
        warning: {
            id: "toast-warning",
            iconColor: "text-orange-400 bg-orange-900/30",
            borderColor: "border-orange-500/50",
            iconPath: "M10 .5a9.5 9.5 0 1 0 9.5 9.5A9.51 9.51 0 0 0 10 .5ZM10 15a1 1 0 1 1 0-2 1 1 0 0 1 0 2Zm1-4a1 1 0 0 1-2 0V6a1 1 0 0 1 2 0v5Z",
            iconLabel: "Warning icon"
        },
        danger: {
            id: "toast-danger",
            iconColor: "text-red-400 bg-red-900/30",
            borderColor: "border-red-500/50",
            iconPath: "M10 .5a9.5 9.5 0 1 0 9.5 9.5A9.51 9.51 0 0 0 10 .5Zm3.707 11.793a1 1 0 1 1-1.414 1.414L10 11.414l-2.293 2.293a1 1 0 0 1-1.414-1.414L8.586 10 6.293 7.707a1 1 0 0 1 1.414-1.414L10 8.586l2.293-2.293a1 1 0 0 1 1.414 1.414L11.414 10l2.293 2.293Z",
            iconLabel: "Error icon"
        }
    };

    const template = toastTemplates[type];
    if (!template) {
        console.error(`Unknown notification type: ${type}`);
        return null;
    }

    const notificationDiv = document.createElement("div");
    notificationDiv.id = notificationId;
    
    // Enhanced Styling:
    // - bg-[#1f1f1f] to match panel backgrounds
    // - Border with type-specific color for distinction
    // - Stronger shadow for depth
    notificationDiv.className = `notification-item flex items-center w-full max-w-xs p-4 text-gray-200 bg-[#1f1f1f] rounded-lg border-l-4 ${template.borderColor} shadow-[0_10px_15px_-3px_rgba(0,0,0,0.5),0_4px_6px_-2px_rgba(0,0,0,0.3)] border-y border-r border-[#414141]`;
    
    notificationDiv.setAttribute("role", "alert");
    notificationDiv.style.position = "absolute";
    notificationDiv.style.top = "0";
    notificationDiv.style.right = "0";
    notificationDiv.style.width = "100%";
    notificationDiv.style.transformOrigin = "center right"; // Scale from right side to match stack

    notificationDiv.innerHTML = `
        <div class="inline-flex items-center justify-center shrink-0 w-8 h-8 ${template.iconColor} rounded-lg">
            <svg
                class="w-5 h-5"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="currentColor"
                viewBox="0 0 20 20"
            >
                <path d="${template.iconPath}" />
            </svg>
            <span class="sr-only">${template.iconLabel}</span>
        </div>
        <div class="ms-3 text-sm font-normal flex-1 break-words">${message}</div>
        <button
            type="button"
            class="ms-auto -mx-1.5 -my-1.5 bg-[#1f1f1f] text-gray-400 hover:text-gray-200 rounded-lg focus:ring-2 focus:ring-gray-600 p-1.5 hover:bg-[#2a2a2a] inline-flex items-center justify-center h-8 w-8 transition-colors duration-200"
            aria-label="Close"
        >
            <span class="sr-only">Close</span>
            <svg
                class="w-3 h-3"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 14 14"
            >
                <path
                    stroke="currentColor"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="m1 1 6 6m0 0 6 6M7 7l6-6M7 7l-6 6"
                />
            </svg>
        </button>
    `;

    const closeButton = notificationDiv.querySelector("button");
    closeButton.addEventListener("click", () => {
        removeNotification(notificationId);
    });

    return notificationDiv;
}

/**
 * Shows a notification toast of the given type.
 *
 * @param {"success" | "warning" | "danger"} type - Notification type.
 * @param {string} message - Notification message content.
 */
function showNotification(type, message) {
    const notificationContainer = getNotificationContainer();
    if (!notificationContainer) {
        console.error("Notification container not found");
        return;
    }

    const notificationElement = createNotificationElement(type, message);
    if (!notificationElement) {
        return;
    }

    const notificationId = notificationElement.id;
    activeNotifications.push(notificationId);
    notificationContainer.appendChild(notificationElement);
    
    // Trigger enter animation
    notificationElement.classList.add("notification-enter");
    
    // Wait for next frame to allow class to be applied, then remove it for transition
    requestAnimationFrame(() => {
        // Force reflow
        void notificationElement.offsetHeight; 
        notificationElement.classList.remove("notification-enter");
        updateNotificationPositions();
    });

    const dismissMs = NOTIFICATION_AUTO_DISMISS_MS[type] ?? 3000;
    const dismissId = setTimeout(() => {
        removeNotification(notificationId);
    }, dismissMs);
    notificationDismissTimeouts.set(notificationId, dismissId);
    
    updateClearAllButtonVisibility();
}

/**
 * Shows a success notification.
 *
 * @param {string} message - Notification message content.
 */
export function showSuccess(message) {
    showNotification("success", message);
}

/**
 * Shows a warning notification.
 *
 * @param {string} message - Notification message content.
 */
export function showWarning(message) {
    showNotification("warning", message);
}

/**
 * Shows a danger/error notification.
 *
 * @param {string} message - Notification message content.
 */
export function showDanger(message) {
    showNotification("danger", message);
}

/**
 * Shows a persistent upload-progress toast that morphs into success or failure.
 *
 * @param {{label?: string}} [options={}]
 * @returns {{setProgress: (percent: number) => void, complete: (message: string) => void, fail: (message: string) => void, dismiss: () => void}}
 */
export function showUploadToast({ label = "Uploading file..." } = {}) {
    const notificationContainer = getNotificationContainer();
    if (!notificationContainer) {
        console.error("Notification container not found");
        return {
            setProgress() {},
            complete() {},
            fail() {},
            dismiss() {},
        };
    }

    const notificationId = `toast-upload-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    const notificationDiv = document.createElement("div");
    notificationDiv.id = notificationId;
    notificationDiv.className =
        "notification-item flex flex-col w-full max-w-xs p-4 text-gray-200 bg-[#1f1f1f] rounded-lg border-l-4 border-yellow-500/50 shadow-[0_10px_15px_-3px_rgba(0,0,0,0.5),0_4px_6px_-2px_rgba(0,0,0,0.3)] border-y border-r border-[#414141]";
    notificationDiv.setAttribute("role", "alert");
    notificationDiv.style.position = "absolute";
    notificationDiv.style.top = "0";
    notificationDiv.style.right = "0";
    notificationDiv.style.width = "100%";
    notificationDiv.style.transformOrigin = "center right";

    notificationDiv.innerHTML = `
        <div class="flex items-center w-full">
            <div data-upload-icon class="inline-flex items-center justify-center shrink-0 w-8 h-8 text-yellow-400 bg-yellow-900/30 rounded-lg transition-all duration-300">
                <svg class="w-5 h-5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 20 20">
                    <path data-upload-icon-path d="${UPLOAD_ICON_PATH}" />
                </svg>
                <span class="sr-only">Upload icon</span>
            </div>
            <div data-upload-message class="ms-3 text-sm font-normal flex-1 break-words">${label}</div>
            <button
                type="button"
                data-upload-close
                class="ms-auto -mx-1.5 -my-1.5 bg-[#1f1f1f] text-gray-400 hover:text-gray-200 rounded-lg focus:ring-2 focus:ring-gray-600 p-1.5 hover:bg-[#2a2a2a] inline-flex items-center justify-center h-8 w-8 transition-colors duration-200"
                aria-label="Close"
            >
                <span class="sr-only">Close</span>
                <svg class="w-3 h-3" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 14 14">
                    <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m1 1 6 6m0 0 6 6M7 7l6-6M7 7l-6 6" />
                </svg>
            </button>
        </div>
        <div data-upload-progress-wrap class="mt-3 w-full">
            <div class="h-2 w-full overflow-hidden rounded-full bg-[#2a2a2a] border border-[#414141]">
                <div data-upload-progress-bar class="h-full rounded-full bg-yellow-400 transition-[width] duration-150 ease-out" style="width: 0%"></div>
            </div>
            <div data-upload-progress-text class="mt-1 text-xs text-gray-400 text-right">0%</div>
        </div>
    `;

    const closeButton = notificationDiv.querySelector("[data-upload-close]");
    closeButton?.addEventListener("click", () => {
        removeNotification(notificationId);
    });

    activeNotifications.push(notificationId);
    notificationContainer.appendChild(notificationDiv);
    notificationDiv.classList.add("notification-enter");
    requestAnimationFrame(() => {
        void notificationDiv.offsetHeight;
        notificationDiv.classList.remove("notification-enter");
        updateNotificationPositions();
    });
    updateClearAllButtonVisibility();

    let settled = false;

    /**
     * Updates the upload progress bar and percent label.
     * @param {number} percent
     */
    function setProgress(percent) {
        if (settled) {
            return;
        }
        const clampedPercent = Math.max(0, Math.min(100, Math.round(percent)));
        const progressBar = notificationDiv.querySelector("[data-upload-progress-bar]");
        const progressText = notificationDiv.querySelector("[data-upload-progress-text]");
        if (progressBar instanceof HTMLElement) {
            progressBar.style.width = `${clampedPercent}%`;
        }
        if (progressText) {
            progressText.textContent = `${clampedPercent}%`;
        }
    }

    /**
     * Schedules auto-dismiss after a terminal state.
     * @param {"success" | "danger"} type
     */
    function scheduleDismiss(type) {
        const dismissMs = NOTIFICATION_AUTO_DISMISS_MS[type] ?? 3000;
        const dismissId = setTimeout(() => {
            removeNotification(notificationId);
        }, dismissMs);
        notificationDismissTimeouts.set(notificationId, dismissId);
    }

    /**
     * Morphs the toast into a success state and auto-dismisses.
     * @param {string} message
     */
    function complete(message) {
        if (settled) {
            return;
        }
        settled = true;
        setProgress(100);

        const iconWrap = notificationDiv.querySelector("[data-upload-icon]");
        const iconPath = notificationDiv.querySelector("[data-upload-icon-path]");
        const messageEl = notificationDiv.querySelector("[data-upload-message]");
        const progressWrap = notificationDiv.querySelector("[data-upload-progress-wrap]");

        notificationDiv.classList.remove("border-yellow-500/50", "border-red-500/50");
        notificationDiv.classList.add("border-green-500/50");
        if (iconWrap) {
            iconWrap.className =
                "inline-flex items-center justify-center shrink-0 w-8 h-8 text-green-400 bg-green-900/30 rounded-lg transition-all duration-300 upload-toast-complete-icon";
        }
        if (iconPath) {
            iconPath.setAttribute("d", CHECK_ICON_PATH);
        }
        if (messageEl) {
            messageEl.textContent = message;
        }
        if (progressWrap instanceof HTMLElement) {
            progressWrap.classList.add("upload-toast-progress-hide");
            setTimeout(() => {
                progressWrap.classList.add("hidden");
                updateNotificationPositions();
            }, 280);
        }
        scheduleDismiss("success");
    }

    /**
     * Morphs the toast into a failure state and auto-dismisses.
     * @param {string} message
     */
    function fail(message) {
        if (settled) {
            return;
        }
        settled = true;

        const iconWrap = notificationDiv.querySelector("[data-upload-icon]");
        const iconPath = notificationDiv.querySelector("[data-upload-icon-path]");
        const messageEl = notificationDiv.querySelector("[data-upload-message]");
        const progressWrap = notificationDiv.querySelector("[data-upload-progress-wrap]");

        notificationDiv.classList.remove("border-yellow-500/50", "border-green-500/50");
        notificationDiv.classList.add("border-red-500/50");
        if (iconWrap) {
            iconWrap.className =
                "inline-flex items-center justify-center shrink-0 w-8 h-8 text-red-400 bg-red-900/30 rounded-lg transition-all duration-300";
        }
        if (iconPath) {
            iconPath.setAttribute("d", ERROR_ICON_PATH);
        }
        if (messageEl) {
            messageEl.textContent = message;
        }
        if (progressWrap instanceof HTMLElement) {
            progressWrap.classList.add("hidden");
        }
        updateNotificationPositions();
        scheduleDismiss("danger");
    }

    /**
     * Removes the toast without a completion animation.
     */
    function dismiss() {
        if (settled) {
            return;
        }
        settled = true;
        removeNotification(notificationId);
    }

    return { setProgress, complete, fail, dismiss };
}

/**
 * Clears all active notifications with a staggered removal animation.
 */
export function clearAll() {
    const notificationsToRemove = [...activeNotifications];
    // Remove from oldest to newest visually, or newest to oldest
    // Let's remove from top of stack (newest) to bottom
    const reversed = notificationsToRemove.reverse();
    
    for (let index = 0; index < reversed.length; index++) {
        const notificationId = reversed[index];
        setTimeout(() => {
            removeNotification(notificationId);
        }, index * 100);
    }
}
