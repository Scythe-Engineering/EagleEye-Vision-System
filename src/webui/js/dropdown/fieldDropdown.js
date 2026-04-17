import { BACKEND_BASE_URL } from "../config.js";
import { init3DView } from "../init3DView.js";

function createFieldRecord(year, filename) {
    return {
        year,
        filename,
        url: `/assets/fields/${year}/field_files/${filename}`,
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

function recordsToFilenameMap(recordsByYear) {
    return Object.fromEntries(
        Object.entries(recordsByYear).map(([year, records]) => [
            year,
            records.map((record) => record.filename),
        ]),
    );
}

function normalizeFieldRecord(record) {
    if (!record?.year || !record?.filename) {
        return null;
    }

    return {
        ...record,
        url:
            record.url ||
            `/assets/fields/${record.year}/field_files/${record.filename}`,
        game_piece_urls: Array.isArray(record.game_piece_urls)
            ? record.game_piece_urls
            : [],
    };
}

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

export function fieldModelUrl(year, file) {
    return `./assets/fields/${year}/field_files/${file}`;
}

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
        };
    }

    return {
        url: fieldModelUrl(yearSelect.value, fileSelect.value),
        gamePieceUrls: undefined,
        aprilTagMapUrl: undefined,
    };
}

export function getSelectedFieldModelUrl() {
    return getSelectedFieldModel()?.url || null;
}

function getFieldRecord(year, filename) {
    return latestFieldRecords[year]?.find(
        (record) => record.filename === filename,
    );
}

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
    });
}

function renderFieldDropdowns(
    yearSelect,
    fileSelect,
    selectedYear,
    selectedFile,
) {
    const latestFields = recordsToFilenameMap(latestFieldRecords);
    const years = Object.keys(latestFields).sort();
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
