const activeNotifications = [];
const MAX_VISIBLE_NOTIFICATIONS = 4;
const STACK_OFFSET = 8;

function getNotificationContainer() {
    return document.getElementById("notification-container");
}

function getClearAllButton() {
    return document.getElementById("clearAllNotificationsBtn");
}

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
        notificationElement.style.zIndex = (reversedNotifications.length - index + 10).toString();
    }
}

function removeNotification(notificationId) {
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
    
    updateClearAllButtonVisibility();
}

export function showSuccess(message) {
    showNotification("success", message);
}

export function showWarning(message) {
    showNotification("warning", message);
}

export function showDanger(message) {
    showNotification("danger", message);
}

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
