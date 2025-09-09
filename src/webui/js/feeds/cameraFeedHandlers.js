let cameraFeedsPaused = false;
let cameraListPollIntervalId = null;
let cameraFetchFn = null;
const BACKEND_BASE_URL = "http://localhost:5001";

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

        cameraNames.forEach((name) => {
            const cameraBox = document.createElement("div");
            cameraBox.className =
                "relative flex items-center justify-center min-h-[100px] bg-[#222] text-[#f9c84a] border-2 border-[#444] rounded-xl py-[30px] px-[15px] text-lg text-center";
            cameraBox.dataset.cameraName = name;

            const cameraNameLabel = document.createElement("div");
            cameraNameLabel.className =
                "absolute top-2 left-3 bg-[#111]/90 text-[#f9c84a] px-2 py-1 rounded-md text-sm font-semibold border border-[#333] z-10 pointer-events-none";
            cameraNameLabel.textContent = name;
            cameraBox.appendChild(cameraNameLabel);

            const cameraView = document.createElement("img");
            cameraView.className = "camera-view";
            const feedSrc = `${BACKEND_BASE_URL}/feed/${name.replace(/ /g, "_")}`;
            if (cameraFeedsPaused) {
                cameraView.dataset.pausedSrc = feedSrc;
                cameraView.src = "";
            } else {
                cameraView.src = feedSrc;
            }
            cameraBox.appendChild(cameraView);

            cameraList.appendChild(cameraBox);
        });

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

    // Initial fetch
    fetchAndUpdateCameras();

    // Poll every 5 seconds and keep interval id so we can stop/start
    if (cameraListPollIntervalId === null) {
        cameraListPollIntervalId = setInterval(fetchAndUpdateCameras, 5000);
    }
}

export function pauseCameraFeeds() {
    cameraFeedsPaused = true;
    if (cameraListPollIntervalId !== null) {
        clearInterval(cameraListPollIntervalId);
        cameraListPollIntervalId = null;
    }

    const imageElements = document.querySelectorAll("img.camera-view");
    imageElements.forEach((img) => {
        if (img?.src && img.src !== "") {
            img.dataset.pausedSrc = img.src;
            img.src = "";
        }
    });
}

export function resumeCameraFeeds() {
    cameraFeedsPaused = false;

    const imageElements = document.querySelectorAll("img.camera-view");
    imageElements.forEach((img) => {
        if (img.dataset?.pausedSrc) {
            img.src = img.dataset.pausedSrc;
            delete img.dataset.pausedSrc;
        } else if (img && (!img.src || img.src.trim() === "")) {
            const container = img.closest("[data-camera-name]");
            if (container?.dataset?.cameraName) {
                const name = container.dataset.cameraName;
                img.src = `${BACKEND_BASE_URL}/feed/${name.replace(/ /g, "_")}`;
            }
        }
    });

    if (
        cameraListPollIntervalId === null &&
        typeof cameraFetchFn === "function"
    ) {
        cameraFetchFn();
        cameraListPollIntervalId = setInterval(cameraFetchFn, 5000);
    }
}
