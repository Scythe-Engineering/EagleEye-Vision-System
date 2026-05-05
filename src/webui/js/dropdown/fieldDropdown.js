import { BACKEND_BASE_URL } from "../config.js";
import { init3DView } from "../init3DView.js";

/**
 * Builds and manages the field/year dropdown state and selection handling for the web UI.
 */

/**
 * Creates a normalized field record for a given year and field file.
 *
 * @param {string} year - Field year.
 * @param {string} filename - Field file name.
 * @returns {object} Field record object.
 */
function createFieldRecord(year, filename) {
    return {
        year,
        filename,
        scale: 1,
        url: `/assets/fields/${year}/field_files/${filename}`,
        rotation_offset: { x: 0, y: 0, z: 0 },
        game_piece_urls: [
            `/assets/fields/${year}/game_pieces/FE-${year}-GP.glb`,
        ],
    };
}

const FALLBACK_FIELD_RECORDS = {
    2025: [
        createFieldRecord("2025", "FE-2025-NGP-Simple.glb"),
        createFieldRecord("2025", "FE-2025-NGP.glb"),
    ],
};

let latestFieldRecords = FALLBACK_FIELD_RECORDS;
let listenersAttached = false;

/**
 * Converts a scale value to a positive finite number, defaulting to 1.
 *
 * @param {unknown} scale - Scale value to normalize.
 * @returns {number} Normalized scale.
 */
function normalizeScale(scale) {
    const numericScale = Number.parseFloat(scale);
    return Number.isFinite(numericScale) && numericScale > 0 ? numericScale : 1;
}

/**
 * Converts records grouped by year into a year-to-filename array map.
 *
 * @param {Object<string, Array<object>>} recordsByYear - Records grouped by year.
 * @returns {Object<string, string[]>} Map of years to filenames.
 */
function recordsToFilenameMap(recordsByYear) {
    return Object.fromEntries(
        Object.entries(recordsByYear).map(([year, records]) => [
            year,
            records.map((record) => record.filename),
        ]),
    );
}

/**
 * Normalizes a raw field record into the shape used by the dropdown logic.
 *
 * @param {object} record - Raw field record.
 * @returns {object|null} Normalized field record, or null when invalid.
 */
function normalizeFieldRecord(record) {
    if (!record?.year || !record?.filename) {
        return null;
    }

    return {
        ...record,
        scale: normalizeScale(record.scale),
        rotation_offset: record.rotation_offset || { x: 0, y: 0, z: 0 },
        url:
            record.url ||
            `/assets/fields/${record.year}/field_files/${record.filename}`,
        game_piece_urls: Array.isArray(record.game_piece_urls)
            ? record.game_piece_urls
            : [],
    };
}

/**
 * Groups valid field records by year from backend file details.
 *
 * @param {Array<object>} fileDetails - File detail records from the backend.
 * @returns {Object<string, Array<object>>} Records grouped by year.
 */
function recordsByYearFromDetails(fileDetails) {
    const recordsByYear = {};
    fileDetails.forEach((fileDetail) => {
        const record = normalizeFieldRecord(fileDetail);
        if (!record) {
            return;
        }

        if (!recordsByYear[record.year]) {
            recordsByYear[record.year] = [];
        }
        recordsByYear[record.year].push(record);
    });

    Object.values(recordsByYear).forEach((records) => {
        records.sort((a, b) => a.filename.localeCompare(b.filename));
    });

    return recordsByYear;
}

/**
 * Fetches available field records from the backend, falling back to built-in defaults.
 *
 * @returns {Promise<Object<string, Array<object>>>} Records grouped by year.
 */
async function fetchAvailableFields() {
    try {
        const response = await fetch(`${BACKEND_BASE_URL}/field-files`);
        const data = await response.json();
        const recordsByYear = recordsByYearFromDetails(
            data?.file_details || [],
        );
        if (Object.keys(recordsByYear).length > 0) {
            return recordsByYear;
        }
    } catch (error) {
        console.error("Error fetching available fields:", error);
    }
    return FALLBACK_FIELD_RECORDS;
}

/**
 * Builds a relative field model URL for the selected year and file.
 *
 * @param {string} year - Field year.
 * @param {string} file - Field file name.
 * @returns {string} Relative field model URL.
 */
export function fieldModelUrl(year, file) {
    return `./assets/fields/${year}/field_files/${file}`;
}

/**
 * Returns the currently selected field model information from the dropdowns.
 *
 * @returns {object|null} Selected field model data, or null if nothing is selected.
 */
export function getSelectedFieldModel() {
    const yearSelect = document.getElementById("yearSelect");
    const fileSelect = document.getElementById("fieldFileSelect");

    if (
        !yearSelect ||
        !fileSelect ||
        yearSelect.selectedIndex <= 0 ||
        fileSelect.selectedIndex <= 0
    ) {
        return null;
    }

    const fieldRecord = getFieldRecord(yearSelect.value, fileSelect.value);
    if (fieldRecord) {
        return {
            url: fieldRecord.url,
            gamePieceUrls: fieldRecord.game_piece_urls,
            aprilTagMapUrl: fieldRecord.apriltag_map_url,
            fieldScale: fieldRecord.scale,
            fieldRotationOffset: fieldRecord.rotation_offset,
            fieldYear: fieldRecord.year,
            fieldFilename: fieldRecord.filename,
        };
    }

    return {
        url: fieldModelUrl(yearSelect.value, fileSelect.value),
        gamePieceUrls: undefined,
        aprilTagMapUrl: undefined,
        fieldScale: 1,
        fieldRotationOffset: { x: 0, y: 0, z: 0 },
        fieldYear: yearSelect.value,
        fieldFilename: fileSelect.value,
    };
}

/**
 * Returns the URL for the currently selected field model.
 *
 * @returns {string|null} Selected field model URL, or null if unavailable.
 */
export function getSelectedFieldModelUrl() {
    return getSelectedFieldModel()?.url || null;
}

/**
 * Finds a field record by year and filename in the cached records.
 *
 * @param {string} year - Field year.
 * @param {string} filename - Field filename.
 * @returns {object|undefined} Matching field record, if present.
 */
function getFieldRecord(year, filename) {
    return latestFieldRecords[year]?.find(
        (record) => record.filename === filename,
    );
}

/**
 * Loads the selected field into the 3D view when both dropdowns have a valid selection.
 *
 * @param {HTMLSelectElement} yearSelect - Year dropdown element.
 * @param {HTMLSelectElement} fileSelect - Field file dropdown element.
 */
function loadSelectedField(yearSelect, fileSelect) {
    if (yearSelect.selectedIndex <= 0 || fileSelect.selectedIndex <= 0) {
        return;
    }

    const fieldModel = getSelectedFieldModel();
    if (!fieldModel) {
        return;
    }

    init3DView(fieldModel.url, {
        gamePieceUrls: fieldModel.gamePieceUrls,
        aprilTagMapUrl: fieldModel.aprilTagMapUrl,
        fieldScale: fieldModel.fieldScale,
        fieldRotationOffset: fieldModel.fieldRotationOffset,
        fieldYear: fieldModel.fieldYear,
        fieldFilename: fieldModel.fieldFilename,
    });
}

/**
 * Renders the year and field dropdown options based on the latest field records.
 *
 * @param {HTMLSelectElement} yearSelect - Year dropdown element.
 * @param {HTMLSelectElement} fileSelect - Field file dropdown element.
 * @param {string|null} selectedYear - Previously selected year, if any.
 * @param {string|null} selectedFile - Previously selected field file, if any.
 */
function renderFieldDropdowns(
    yearSelect,
    fileSelect,
    selectedYear,
    selectedFile,
) {
    const latestFields = recordsToFilenameMap(latestFieldRecords);
    const years = Object.keys(latestFields).sort(
        (a, b) => Number.parseInt(b, 10) - Number.parseInt(a, 10),
    );
    const nextYear =
        selectedYear && latestFields[selectedYear]
            ? selectedYear
            : years[0] || "";

    yearSelect.innerHTML = "<option disabled selected>Select Year</option>";
    years.forEach((year) => {
        const option = document.createElement("option");
        option.value = year;
        option.textContent = year;
        yearSelect.appendChild(option);
    });

    if (nextYear) {
        yearSelect.value = nextYear;
    }

    const files = latestFields[nextYear] || [];
    const nextFile =
        selectedFile && files.includes(selectedFile)
            ? selectedFile
            : files[0] || "";

    fileSelect.innerHTML =
        "<option disabled selected>Select Field File</option>";
    files.forEach((file) => {
        const option = document.createElement("option");
        option.value = file;
        option.textContent = file;
        fileSelect.appendChild(option);
    });

    if (nextFile) {
        fileSelect.value = nextFile;
    }
}

/**
 * Populates the field dropdowns, wires change listeners, and optionally loads the selection.
 *
 * @param {object} [options={}] - Population options.
 * @param {string|null} [options.selectedYear] - Year to preserve when re-rendering.
 * @param {string|null} [options.selectedFile] - File to preserve when re-rendering.
 * @param {boolean} [options.loadSelected=false] - Whether to load the selected field immediately.
 * @returns {Promise<void>}
 */
export async function populateFieldDropdown(options = {}) {
    const yearSelect = document.getElementById("yearSelect");
    const fileSelect = document.getElementById("fieldFileSelect");

    if (!yearSelect || !fileSelect) {
        return;
    }

    const selectedYear =
        options.selectedYear ||
        (yearSelect.selectedIndex > 0 ? yearSelect.value : null);
    const selectedFile =
        options.selectedFile ||
        (fileSelect.selectedIndex > 0 ? fileSelect.value : null);

    latestFieldRecords = await fetchAvailableFields();
    renderFieldDropdowns(yearSelect, fileSelect, selectedYear, selectedFile);

    if (!listenersAttached) {
        yearSelect.addEventListener("change", () => {
            renderFieldDropdowns(
                yearSelect,
                fileSelect,
                yearSelect.value,
                null,
            );
            loadSelectedField(yearSelect, fileSelect);
        });

        fileSelect.addEventListener("change", () => {
            loadSelectedField(yearSelect, fileSelect);
        });

        listenersAttached = true;
    }

    if (options.loadSelected) {
        loadSelectedField(yearSelect, fileSelect);
    }
}
