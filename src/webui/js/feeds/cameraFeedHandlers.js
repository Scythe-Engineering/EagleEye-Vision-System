export function setupCameraFeedHandlers() {
    const cameraList = document.getElementById("cameraList");
    const noCamerasMessage = document.getElementById("noCamerasMessage");

    // Hide manual feed control elements if they exist
    const feedControls = document.querySelector(".feed-controls");
    if (feedControls) {
        feedControls.style.display = "none";
    }
    const addFeedBackgroundDiv = document.getElementById("addFeedBackgroundDiv");
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
        // Clear current list then rebuild based on latest cameraNames
        cameraList.innerHTML = "";

        if (cameraNames.length === 0) {
            noCamerasMessage.style.display = "block";
            updateGridLayout();
            return;
        }

        noCamerasMessage.style.display = "none";

        cameraNames.forEach((name) => {
            const cameraBox = document.createElement("div");
            cameraBox.className = "camera-box";
            cameraBox.dataset.cameraName = name;
            cameraBox.textContent = name;

            const cameraView = document.createElement("img");
            cameraView.className = "camera-view";
            cameraView.src = `/feed/${name.replace(/ /g, "_")}`;
            cameraBox.appendChild(cameraView);

            cameraList.appendChild(cameraBox);
        });

        updateGridLayout();
    }

    function fetchAndUpdateCameras() {
        fetch("/get-available-cameras", {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => {
                const cameraNames = Object.keys(data);
                renderCameras(cameraNames);
            })
            .catch((error) => {
                console.error("Error fetching cameras:", error);
            });
    }

    // Initial fetch
    fetchAndUpdateCameras();
    // Poll every 5 seconds
    setInterval(fetchAndUpdateCameras, 5000);
}
