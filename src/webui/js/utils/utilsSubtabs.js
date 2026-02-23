import {
    initCameraConfigUtils,
    refreshCameraConfigUtils,
} from "./cameraConfigUtils.js";
import { initCustomOpsEditor } from "../custom-ops/index.js";

const SUBTABS = {
    CAMERA_CONFIG: "camera-config",
    CUSTOM_OPS: "custom-ops",
};

const ACTIVE_BUTTON_CLASSES = ["bg-[#2a2a2a]", "text-[#f9c845]", "border-[#f9c845]"];
const INACTIVE_BUTTON_CLASSES = ["bg-[#1f1f1f]", "text-[#bdbdbd]", "border-[#414141]"];

let initialized = false;
let activeSubtab = SUBTABS.CAMERA_CONFIG;

function getInitialSubtabFromUrl() {
    const url = new URL(globalThis.location.href);
    const tab = url.searchParams.get("tab");
    if (tab === "view-custom-ops") {
        return SUBTABS.CUSTOM_OPS;
    }
    return SUBTABS.CAMERA_CONFIG;
}

function getButtons() {
    return document.querySelectorAll("[data-utils-subtab]");
}

function getPanel(subtabId) {
    if (subtabId === SUBTABS.CAMERA_CONFIG) {
        return document.getElementById("utilsSubtabCameraConfig");
    }

    if (subtabId === SUBTABS.CUSTOM_OPS) {
        return document.getElementById("utilsSubtabCustomOps");
    }

    return null;
}

function applyButtonState(button, isActive) {
    button.classList.remove(...ACTIVE_BUTTON_CLASSES, ...INACTIVE_BUTTON_CLASSES);
    button.classList.add(...(isActive ? ACTIVE_BUTTON_CLASSES : INACTIVE_BUTTON_CLASSES));
}

function refreshActiveSubtabContent() {
    if (activeSubtab === SUBTABS.CAMERA_CONFIG) {
        initCameraConfigUtils();
        refreshCameraConfigUtils();
        return;
    }

    if (activeSubtab === SUBTABS.CUSTOM_OPS) {
        initCustomOpsEditor();
    }
}

function setActiveSubtab(subtabId) {
    if (subtabId !== SUBTABS.CAMERA_CONFIG && subtabId !== SUBTABS.CUSTOM_OPS) {
        return;
    }

    activeSubtab = subtabId;

    const buttons = getButtons();
    buttons.forEach((button) => {
        const isActive = button.dataset.utilsSubtab === activeSubtab;
        applyButtonState(button, isActive);
    });

    const cameraConfigPanel = getPanel(SUBTABS.CAMERA_CONFIG);
    const customOpsPanel = getPanel(SUBTABS.CUSTOM_OPS);

    cameraConfigPanel?.classList.toggle("hidden", activeSubtab !== SUBTABS.CAMERA_CONFIG);
    customOpsPanel?.classList.toggle("hidden", activeSubtab !== SUBTABS.CUSTOM_OPS);

    refreshActiveSubtabContent();
}

export function initUtilsSubtabs() {
    if (!initialized) {
        activeSubtab = getInitialSubtabFromUrl();
        const buttons = getButtons();
        buttons.forEach((button) => {
            button.addEventListener("click", () => {
                setActiveSubtab(button.dataset.utilsSubtab || SUBTABS.CAMERA_CONFIG);
            });
        });
        initialized = true;
    }

    setActiveSubtab(activeSubtab);
}
