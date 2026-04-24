import { BACKEND_BASE_URL } from "../config.js";

const EMPTY_IMAGE_SRC =
    "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=";

let cameraFeedsPaused = false;
let cameraListPollIntervalId = null;
let cameraFetchFn = null;
let cameraListFetchController = null;

function isCameraViewActive() {
    const cameraView = document.getElementById("view-views");
    return Boolean(cameraView && !cameraView.classList.contains("hidden"));
}

function buildCameraFeedSrc(cameraName) {
    return `${BACKEND_BASE_URL}/feed/${cameraName.replaceAll(" ", "_")}`;
}

function stopCameraImage(img) {
    if (!img) {
        return;
    }

    if (img.src && img.src !== EMPTY_IMAGE_SRC) {
        img.dataset.pausedSrc = img.src;
    }

    img.src = EMPTY_IMAGE_SRC;
    if (typeof img.load === "function") {
        img.load();
    }
}

export function setupCameraFeedHandlers() {
    const cameraList = document.getElementById("cameraList");
    const noCamerasMessage = document.getElementById("noCamerasMessage");
    const bottomBlur = document.getElementById("cameraListBottomBlur");

    cameraFeedsPaused = !isCameraViewActive();

    // Handle bottom blur visibility on scroll
    if (cameraList && bottomBlur) {
        const updateBlurVisibility = () => {
            const isScrollable =
                cameraList.scrollHeight > cameraList.clientHeight;
            const isAtBottom =
                cameraList.scrollHeight -
                    cameraList.scrollTop -
                    cameraList.clientHeight <
                10;

            if (isScrollable && !isAtBottom) {
                bottomBlur.classList.remove("opacity-0");
                bottomBlur.classList.add("opacity-100");
            } else {
                bottomBlur.classList.remove("opacity-100");
                bottomBlur.classList.add("opacity-0");
            }
        };

        cameraList.addEventListener("scroll", updateBlurVisibility);
        // Also check on resize and when content changes
        const resizeObserver = new ResizeObserver(updateBlurVisibility);
        resizeObserver.observe(cameraList);

        // Export it so it can be called after rendering
        cameraList.updateBlurVisibility = updateBlurVisibility;
    }

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
                "relative flex items-center justify-center min-h-[100px] bg-[#1f1f1f] text-[#f9c84a] border-2 border-[#414141] rounded-[15px] p-[15px] text-lg text-center";
            cameraBox.style.boxShadow =
                "8px 8px 16px rgba(0, 0, 0, 0.4)";
            cameraBox.dataset.cameraName = name;

            const cameraNameLabel = document.createElement("div");
            cameraNameLabel.className =
                "absolute top-0 left-0 bg-[#1E1E1E] text-[#f9c84a] px-[15px] py-2 rounded-tl-[13px] rounded-br-xl text-sm font-semibold border-r-2 border-b-2 border-[#414141] z-10 pointer-events-none";
            cameraNameLabel.textContent = name;
            cameraNameLabel.style.boxShadow = "2px 2px 4px rgba(0, 0, 0, 0.4)";
            cameraBox.appendChild(cameraNameLabel);

            const cameraView = document.createElement("img");
            cameraView.className = "camera-view rounded-lg";
            const feedSrc = buildCameraFeedSrc(name);
            if (cameraFeedsPaused) {
                cameraView.dataset.pausedSrc = feedSrc;
                cameraView.src = EMPTY_IMAGE_SRC;
            } else {
                cameraView.src = feedSrc;
            }
            cameraBox.appendChild(cameraView);

            cameraList.appendChild(cameraBox);
        }

        updateGridLayout();
        if (cameraList.updateBlurVisibility) {
            setTimeout(cameraList.updateBlurVisibility, 100);
        }
    }

    function fetchAndUpdateCameras() {
        if (cameraListFetchController) {
            cameraListFetchController.abort();
        }
        const fetchController = new AbortController();
        cameraListFetchController = fetchController;

        fetch(`${BACKEND_BASE_URL}/get-available-cameras`, {
            method: "GET",
            signal: fetchController.signal,
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
                if (error?.name === "AbortError") {
                    return;
                }
                console.error("Error fetching cameras:", error);
            })
            .finally(() => {
                if (cameraListFetchController === fetchController) {
                    cameraListFetchController = null;
                }
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
    if (cameraListFetchController) {
        cameraListFetchController.abort();
        cameraListFetchController = null;
    }

    const imageElements = document.querySelectorAll("img.camera-view");
    for (const img of imageElements) {
        stopCameraImage(img);
    }
}

export function resumeCameraFeeds() {
    cameraFeedsPaused = false;

    const imageElements = document.querySelectorAll("img.camera-view");
    for (const img of imageElements) {
        if (img.dataset?.pausedSrc) {
            img.src = img.dataset.pausedSrc;
            delete img.dataset.pausedSrc;
        } else if (img && (!img.src || img.src === EMPTY_IMAGE_SRC)) {
            const container = img.closest("[data-camera-name]");
            if (container?.dataset?.cameraName) {
                const name = container.dataset.cameraName;
                img.src = buildCameraFeedSrc(name);
            }
        }
    }
}

export function refreshCameraFeeds() {
    if (typeof cameraFetchFn === "function") {
        cameraFetchFn();
    }
}
