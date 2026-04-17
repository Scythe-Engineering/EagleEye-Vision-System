import { BACKEND_BASE_URL } from "../config.js";

let latestRobotRecords = [];

function normalizeScale(scale) {
    const numericScale = Number.parseFloat(scale);
    return Number.isFinite(numericScale) && numericScale > 0 ? numericScale : 1;
}

function normalizeRobotRecord(record) {
    if (typeof record === "string") {
        return {
            filename: record,
            scale: 1,
        };
    }

    if (!record?.filename) {
        return null;
    }

    return {
        ...record,
        scale: normalizeScale(record.scale),
    };
}

function recordsFromPayload(data) {
    const detailRecords = Array.isArray(data?.file_details)
        ? data.file_details.map(normalizeRobotRecord).filter(Boolean)
        : [];
    if (detailRecords.length > 0) {
        return detailRecords;
    }

    return Array.isArray(data?.robots)
        ? data.robots.map(normalizeRobotRecord).filter(Boolean)
        : [];
}

export function getRobotFileScale(robotFile) {
    const record = latestRobotRecords.find(
        (robotRecord) => robotRecord.filename === robotFile,
    );
    return normalizeScale(record?.scale);
}

export async function populateRobotDropdown(selectedRobotFile = null) {
    const robotFileSelect = document.getElementById("robotFileSelect");
    if (!robotFileSelect) {
        return;
    }

    async function fetchAvailableRobots() {
        try {
            const response = await fetch(
                `${BACKEND_BASE_URL}/get-available-robots`,
            );
            const data = await response.json();
            return recordsFromPayload(data);
        } catch (error) {
            console.error("Error fetching available robots:", error);
            return [];
        }
    }

    async function loadRobots() {
        latestRobotRecords = await fetchAvailableRobots();
        const robotNames = latestRobotRecords.map((robot) => robot.filename);

        // Save the currently selected value before clearing
        const previouslySelectedValue =
            selectedRobotFile ||
            (robotFileSelect.selectedIndex > 0 ? robotFileSelect.value : null);

        robotFileSelect.innerHTML =
            "<option disabled selected>Select Robot File</option>";

        latestRobotRecords.forEach((robot) => {
            const option = document.createElement("option");
            option.value = robot.filename;
            option.textContent = robot.filename;
            option.dataset.scale = String(robot.scale);
            robotFileSelect.appendChild(option);
        });

        // Restore previous selection if it still exists in the new list
        if (
            previouslySelectedValue &&
            robotNames.includes(previouslySelectedValue)
        ) {
            robotFileSelect.value = previouslySelectedValue;
        }
        // Otherwise, select first robot if robots are available and no previous selection
        else if (latestRobotRecords.length > 0 && !previouslySelectedValue) {
            robotFileSelect.selectedIndex = 1; // Index 1 because index 0 is the disabled placeholder
        }
    }

    await loadRobots();
}
