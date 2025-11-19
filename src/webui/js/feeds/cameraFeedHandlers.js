import { BACKEND_BASE_URL } from "../config.js";

let cameraFeedsPaused = false;
let cameraListPollIntervalId = null;
let cameraFetchFn = null;

export function setupCameraFeedHandlers() {
    const cameraList = document.getElementById("cameraList");
    const noCamerasMessage = document.getElementById("noCamerasMessage");

    // Hide manual feed control elements if they exist
    const feedControls = document.querySelector(".feed-controls");
    if (feedControls) {
        feedControls.style.display = "none";
    }
    const addFeedBackgroundDiv = document.getElementById(
        "addFeedBackgroundDiv",
    );
    if (addFeedBackgroundDiv) {
        addFeedBackgroundDiv.remove();
    }

    function updateGridLayout() {
        const cameraCount = cameraList.children.length;
        let columns;
        if (cameraCount === 2) {
            columns = 2;
        } else if (cameraCount <= 2) {
            columns = 1;
        } else if (cameraCount <= 4) {
            columns = 2;
        } else if (cameraCount <= 9) {
            columns = 3;
        } else {
            columns = 4;
        }
        cameraList.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
    }

    function renderCameras(cameraNames) {
        cameraList.innerHTML = "";

        if (cameraNames.length === 0) {
            if (noCamerasMessage) noCamerasMessage.style.display = "block";
            updateGridLayout();
            return;
        }

        if (noCamerasMessage) noCamerasMessage.style.display = "none";

        for (const name of cameraNames) {
            const cameraBox = document.createElement("div");
            cameraBox.className =
                "relative flex items-center justify-center min-h-[100px] bg-[#1E1E1E] text-[#f9c84a] border-2 border-[#444] rounded-xl py-[30px] px-[15px] text-lg text-center";
            cameraBox.style.boxShadow = "4px 4px 8px rgba(0, 0, 0, 0.4)";
            cameraBox.dataset.cameraName = name;

            const cameraNameLabel = document.createElement("div");
            cameraNameLabel.className =
                "absolute top-2 left-3 bg-[#1E1E1E] text-[#f9c84a] px-2 py-1 rounded-md text-sm font-semibold border border-[#333] z-10 pointer-events-none";
            cameraNameLabel.textContent = name;
            cameraNameLabel.style.boxShadow = "2px 2px 4px rgba(0, 0, 0, 0.4)";
            cameraBox.appendChild(cameraNameLabel);

            const cameraView = document.createElement("img");
            cameraView.className = "camera-view rounded-lg";
            const feedSrc = `${BACKEND_BASE_URL}/feed/${name.replaceAll(' ', "_")}`;
            if (cameraFeedsPaused) {
                cameraView.dataset.pausedSrc = feedSrc;
                cameraView.src = "";
            } else {
                cameraView.src = feedSrc;
            }
            cameraBox.appendChild(cameraView);

            cameraList.appendChild(cameraBox);
        }

        updateGridLayout();
    }

    function fetchAndUpdateCameras() {
        fetch(`${BACKEND_BASE_URL}/get-available-cameras`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => {
                const cameraNames = Object.keys(data || {});
                renderCameras(cameraNames);
            })
            .catch((error) => {
                console.error("Error fetching cameras:", error);
            });
    }

    cameraFetchFn = fetchAndUpdateCameras;

    fetchAndUpdateCameras();
}

export function pauseCameraFeeds() {
    cameraFeedsPaused = true;
    if (cameraListPollIntervalId !== null) {
        clearInterval(cameraListPollIntervalId);
        cameraListPollIntervalId = null;
    }

    const imageElements = document.querySelectorAll("img.camera-view");
    for (const img of imageElements) {
        if (img?.src && img.src !== "") {
            img.dataset.pausedSrc = img.src;
            img.src = "";
        }
    }
}

export function resumeCameraFeeds() {
    cameraFeedsPaused = false;

    const imageElements = document.querySelectorAll("img.camera-view");
    for (const img of imageElements) {
        if (img.dataset?.pausedSrc) {
            img.src = img.dataset.pausedSrc;
            delete img.dataset.pausedSrc;
        } else if (img && (!img.src || img.src.trim() === "")) {
            const container = img.closest("[data-camera-name]");
            if (container?.dataset?.cameraName) {
                const name = container.dataset.cameraName;
                img.src = `${BACKEND_BASE_URL}/feed/${name.replaceAll(' ', "_")}`;
            }
        }
    }
}

export function refreshCameraFeeds() {
    if (typeof cameraFetchFn === "function") {
        cameraFetchFn();
    }
}
