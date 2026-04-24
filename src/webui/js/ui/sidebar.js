import { init3DView } from "../init3DView.js";
import { loadSettings } from "../settings/settingsHandler.js";
import {
    pauseCameraFeeds,
    resumeCameraFeeds,
    refreshCameraFeeds,
} from "../feeds/cameraFeedHandlers.js";
import { initPipelineCreator } from "../pipeline/pipelineCreator.js";
import { refreshLogMessages } from "../settings/terminalHandler.js";
import {
    initCameraConfigUtils,
    refreshCameraConfigUtils,
} from "../utils/cameraConfigUtils.js";
import { getSelectedFieldModel } from "../dropdown/fieldDropdown.js";

const VIEWS = {
    THREE_D: "view-3d",
    CAMERA: "view-views",
    SETTINGS: "view-settings",
    PIPELINE: "view-pipeline",
    SYSTEM: "view-system",
    UTILS: "view-utils",
};

const FIELD_ASSETS = {
    FIELD_2025_DEFAULT:
        "./assets/fields/2025/field_files/FE-2025-NGP-Simple.glb",
};

class ViewManager {
    constructor() {
        this.sidebarItems = document.querySelectorAll(".sidebar li");
        this.views = document.querySelectorAll("[id^='view-']");
        this.controls = document.querySelectorAll(
            "#fieldDropdown, #robotDropdown, #viewToggles",
        );
    }

    activateView(targetViewId) {
        this.updateActiveSidebarItem(targetViewId);
        this.showTargetView(targetViewId);
        this.toggleControlsVisibility(targetViewId);
        this.handleViewSpecificBehavior(targetViewId);
    }

    updateActiveSidebarItem(targetViewId) {
        for (const sidebarItem of this.sidebarItems) {
            sidebarItem.classList.toggle(
                "active",
                sidebarItem.dataset.view === targetViewId,
            );
        }
    }

    showTargetView(targetViewId) {
        // Hide all views using only Tailwind classes
        for (const view of this.views) {
            view.classList.add("hidden");
        }

        const targetView = document.getElementById(targetViewId);
        if (!targetView) {
            return;
        }

        // Show target view using only Tailwind classes
        targetView.classList.remove("hidden");
    }

    toggleControlsVisibility(targetViewId) {
        for (const element of this.controls) {
            element.classList.toggle("hidden", targetViewId !== VIEWS.THREE_D);
        }
    }

    handleViewSpecificBehavior(viewId) {
        switch (viewId) {
            case VIEWS.THREE_D:
                {
                    const fieldModel = getSelectedFieldModel();
                    init3DView(
                        fieldModel?.url || FIELD_ASSETS.FIELD_2025_DEFAULT,
                        {
                            gamePieceUrls: fieldModel?.gamePieceUrls,
                            aprilTagMapUrl: fieldModel?.aprilTagMapUrl,
                            fieldScale: fieldModel?.fieldScale,
                            fieldYear: fieldModel?.fieldYear,
                            fieldFilename: fieldModel?.fieldFilename,
                        },
                    );
                }
                pauseCameraFeeds();
                break;
            case VIEWS.CAMERA:
                resumeCameraFeeds();
                refreshCameraFeeds();
                break;
            case VIEWS.SETTINGS:
                pauseCameraFeeds();
                loadSettings();
                refreshLogMessages();
                break;
            case VIEWS.PIPELINE:
                pauseCameraFeeds();
                // If pipeline creator is already initialized, refresh it; otherwise initialize it
                if (globalThis.pipelineCreator?.refreshPipelineCreator) {
                    globalThis.pipelineCreator.refreshPipelineCreator();
                } else {
                    initPipelineCreator();
                }
                break;
            case VIEWS.SYSTEM:
                pauseCameraFeeds();
                break;
            case VIEWS.UTILS:
                pauseCameraFeeds();
                initCameraConfigUtils();
                refreshCameraConfigUtils();
                break;
            default:
                pauseCameraFeeds();
        }
    }
}

class URLManager {
    updateTab(viewId) {
        const url = new URL(globalThis.location.href);
        url.searchParams.set("tab", viewId);
        globalThis.history.replaceState({}, "", url.toString());
    }

    getInitialTab() {
        const url = new URL(globalThis.location.href);
        return url.searchParams.get("tab");
    }
}

export function setupSidebar() {
    const viewManager = new ViewManager();
    const urlManager = new URLManager();
    const sidebarItems = document.querySelectorAll(".sidebar li");

    function handleSidebarItemClick(targetViewId) {
        if (!targetViewId) {
            return;
        }
        viewManager.activateView(targetViewId);
        urlManager.updateTab(targetViewId);
    }

    for (const item of sidebarItems) {
        item.addEventListener("click", () => {
            const targetViewId = item.dataset.view;
            handleSidebarItemClick(targetViewId);
        });
    }

    const initialTab = urlManager.getInitialTab();
    if (initialTab && document.getElementById(initialTab)) {
        viewManager.activateView(initialTab);
        urlManager.updateTab(initialTab);
        return;
    }

    const firstItem = sidebarItems[0];
    if (firstItem) {
        const defaultViewId = firstItem.dataset.view;
        if (defaultViewId) {
            viewManager.activateView(defaultViewId);
            urlManager.updateTab(defaultViewId);
        }
    }
}
