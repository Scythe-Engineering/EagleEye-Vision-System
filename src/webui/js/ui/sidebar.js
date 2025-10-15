import { init3DView } from "../init3DView.js";
import { loadSettings } from "../settings/loadSettings.js";
import {
    pauseCameraFeeds,
    resumeCameraFeeds,
} from "../feeds/cameraFeedHandlers.js";
import { initPipelineCreator } from "../pipeline/pipelineCreator.js";

const VIEWS = {
    THREE_D: "view-3d",
    CAMERA: "view-views",
    SETTINGS: "view-settings",
    PIPELINE: "view-pipeline",
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
            "#fieldDropdown, #toggleShadowBtn, #toggleGamePiecesBtn",
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
                init3DView(FIELD_ASSETS.FIELD_2025_DEFAULT);
                pauseCameraFeeds();
                break;
            case VIEWS.CAMERA:
                resumeCameraFeeds();
                break;
            case VIEWS.SETTINGS:
                loadSettings();
                break;
            case VIEWS.PIPELINE:
                // If pipeline creator is already initialized, refresh it; otherwise initialize it
                if (globalThis.pipelineCreator?.refreshPipelineCreator) {
                    globalThis.pipelineCreator.refreshPipelineCreator();
                } else {
                    initPipelineCreator();
                }
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
