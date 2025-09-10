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
        this.sidebarItems.forEach((sidebarItem) => {
            sidebarItem.classList.toggle(
                "active",
                sidebarItem.getAttribute("data-view") === targetViewId,
            );
        });
    }

    showTargetView(targetViewId) {
        // Hide all views using only Tailwind classes
        this.views.forEach((view) => {
            view.classList.add("hidden");
        });

        const targetView = document.getElementById(targetViewId);
        if (!targetView) {
            return;
        }

        // Show target view using only Tailwind classes
        targetView.classList.remove("hidden");
    }

    toggleControlsVisibility(targetViewId) {
        this.controls.forEach((element) => {
            element.classList.toggle("hidden", targetViewId !== VIEWS.THREE_D);
        });
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
                initPipelineCreator();
                break;
            default:
                pauseCameraFeeds();
        }
    }
}

class URLManager {
    updateTab(viewId) {
        const url = new URL(window.location.href);
        url.searchParams.set("tab", viewId);
        window.history.replaceState({}, "", url.toString());
    }

    getInitialTab() {
        const url = new URL(window.location.href);
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

    sidebarItems.forEach((item) => {
        item.addEventListener("click", () => {
            const targetViewId = item.getAttribute("data-view");
            handleSidebarItemClick(targetViewId);
        });
    });

    const initialTab = urlManager.getInitialTab();
    if (initialTab && document.getElementById(initialTab)) {
        viewManager.activateView(initialTab);
        urlManager.updateTab(initialTab);
        return;
    }

    const firstItem = sidebarItems[0];
    if (firstItem) {
        const defaultViewId = firstItem.getAttribute("data-view");
        if (defaultViewId) {
            viewManager.activateView(defaultViewId);
            urlManager.updateTab(defaultViewId);
        }
    }
}
