import { init3DView } from "../init3DView.js";
import { loadSettings } from "../settings/loadSettings.js";

export function setupSidebar() {
    const sidebarItems = document.querySelectorAll(".sidebar li");
    const views = document.querySelectorAll("[id^='view-']");

    sidebarItems.forEach((item) => {
        item.addEventListener("click", () => {
            // Update active sidebar item
            sidebarItems.forEach((i) => i.classList.remove("active"));
            item.classList.add("active");
            
            // Hide all views using only Tailwind classes
            views.forEach((view) => {
                view.classList.add("hidden");
            });

            const targetViewId = item.getAttribute("data-view");
            const targetView = document.getElementById(targetViewId);

            if (targetView) {
                // Show target view using only Tailwind classes
                targetView.classList.remove("hidden");
                
                // Toggle visibility of controls based on the view
                const controls = document.querySelectorAll("#fieldDropdown, #toggleShadowBtn, #toggleGamePiecesBtn");
                controls.forEach(el => {
                    el.classList.toggle("hidden", targetView.id !== "view-3d");
                });
                
                if (targetView.id === "view-3d") {
                    init3DView("./assets/fields/2025/field_files/FE-2025-NGP-Simple.glb");
                }
                if (targetView.id === "view-settings") {
                    loadSettings();
                }
            }
        });
    });
}