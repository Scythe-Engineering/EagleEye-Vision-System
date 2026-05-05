import { BACKEND_BASE_URL } from "../config.js";

// Manages loading robot file options and scale metadata for the robot dropdown UI.
let latestRobotRecords = [];

/**
 * Normalize a robot scale value to a positive finite number, defaulting to 1.
 *
 * @param {unknown} scale - The scale value to normalize.
 * @returns {number}
 */
function normalizeScale(scale) {
    const numericScale = Number.parseFloat(scale);
    return Number.isFinite(numericScale) && numericScale > 0 ? numericScale : 1;
}

/**
 * Normalize a robot record into a consistent shape used by the dropdown.
 *
 * @param {string|object|null|undefined} record - The raw robot record.
 * @returns {{filename: string, scale: number}|null}
 */
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

/**
 * Extract normalized robot records from an API payload.
 *
 * @param {object|null|undefined} data - The backend payload.
 * @returns {Array<{filename: string, scale: number}>}
 */
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

/**
 * Get the normalized scale for a robot file currently loaded in the dropdown.
 *
 * @param {string} robotFile - The robot filename.
 * @returns {number}
 */
export function getRobotFileScale(robotFile) {
    const record = latestRobotRecords.find(
        (robotRecord) => robotRecord.filename === robotFile,
    );
    return normalizeScale(record?.scale);
}

/**
 * Fetch available robots and populate the robot file dropdown.
 *
 * @param {string|null} selectedRobotFile - Optional filename to preselect.
 * @returns {Promise<void>}
 */
export async function populateRobotDropdown(selectedRobotFile = null) {
    const robotFileSelect = document.getElementById("robotFileSelect");
    if (!robotFileSelect) {
        return;
    }

    /**
     * Fetch available robot records from the backend.
     *
     * @returns {Promise<Array<{filename: string, scale: number}>>}
     */
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

    /**
     * Load robot records into the dropdown and preserve selection when possible.
     *
     * @returns {Promise<void>}
     */
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
