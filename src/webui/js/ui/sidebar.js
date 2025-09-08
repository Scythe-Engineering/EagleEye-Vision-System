import { init3DView } from "../init3DView.js";
import { loadSettings } from "../settings/loadSettings.js";
import {
    pauseCameraFeeds,
    resumeCameraFeeds,
} from "../feeds/cameraFeedHandlers.js";
import { initPipelineCreator } from "../pipeline/pipelineCreator.js";

export function setupSidebar() {
    const sidebarItems = document.querySelectorAll(".sidebar li");
    const views = document.querySelectorAll("[id^='view-']");

    function activateView(targetViewId) {
        // Update active sidebar item
        sidebarItems.forEach((sidebarItem) => {
            sidebarItem.classList.toggle(
                "active",
                sidebarItem.getAttribute("data-view") === targetViewId,
            );
        });

        // Hide all views using only Tailwind classes
        views.forEach((view) => {
            view.classList.add("hidden");
        });

        const targetView = document.getElementById(targetViewId);
        if (!targetView) {
            return;
        }

        // Show target view using only Tailwind classes
        targetView.classList.remove("hidden");

        // Toggle visibility of controls based on the view
        const controls = document.querySelectorAll(
            "#fieldDropdown, #toggleShadowBtn, #toggleGamePiecesBtn",
        );
        controls.forEach((element) => {
            element.classList.toggle("hidden", targetView.id !== "view-3d");
        });

        if (targetView.id === "view-3d") {
            init3DView(
                "./assets/fields/2025/field_files/FE-2025-NGP-Simple.glb",
            );
            pauseCameraFeeds();
        } else if (targetView.id === "view-views") {
            resumeCameraFeeds();
        } else {
            pauseCameraFeeds();
        }

        if (targetView.id === "view-settings") {
            loadSettings();
        }

        if (targetView.id === "view-pipeline") {
            initPipelineCreator();
        }
    }

    function setTabQueryParam(targetViewId) {
        const url = new URL(window.location.href);
        url.searchParams.set("tab", targetViewId);
        window.history.replaceState({}, "", url.toString());
    }

    sidebarItems.forEach((item) => {
        item.addEventListener("click", () => {
            const targetViewId = item.getAttribute("data-view");
            if (!targetViewId) {
                return;
            }
            activateView(targetViewId);
            setTabQueryParam(targetViewId);
        });
    });

    const initialUrl = new URL(window.location.href);
    const initialTab = initialUrl.searchParams.get("tab");
    if (initialTab && document.getElementById(initialTab)) {
        activateView(initialTab);
        setTabQueryParam(initialTab);
        return;
    }

    const firstItem = sidebarItems[0];
    if (firstItem) {
        const defaultViewId = firstItem.getAttribute("data-view");
        if (defaultViewId) {
            activateView(defaultViewId);
            setTabQueryParam(defaultViewId);
        }
    }
}
