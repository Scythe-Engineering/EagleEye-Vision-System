import { BACKEND_BASE_URL } from "../../config.js";
import { showDanger } from "../../ui/notificationSystem.js";

/**
 * Data access helpers for the pipeline creator UI.
 */

/**
 * Convert a backend operation name into a title-cased display name.
 *
 * @param {string} name - Raw operation name.
 * @returns {string} Title-cased display name.
 */
function toTitleCaseName(name) {
    return String(name)
        .replaceAll(".py", "")
        .replaceAll("_", " ")
        .replaceAll(/\b\w/g, (l) => l.toUpperCase());
}

/**
 * Fetch available operations and store them in the pipeline store.
 *
 * @param {object} pipelineStore - Store used to persist operations.
 * @returns {Promise<Array<object>>} Resolved operations list.
 */
export async function fetchAvailableOperations(pipelineStore) {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-available-operations`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        const operations = (data.operations || []).map((op) => ({
            id: op.name,
            name: toTitleCaseName(op.name),
            type: op.category.toUpperCase(),
            folder: op.folder || "Uncategorized",
            description: op.description,
            path: op.path,
            configDataPath: op.config_data_path,
            isSecondary: op.is_secondary,
            isDataSource: Boolean(op.is_data_source),
            hasVisualization: Boolean(op.has_visualization),
        }));

        pipelineStore.setOperations(operations);
        console.log("Loaded operations from server:", operations);
        return operations;
    } catch (error) {
        showDanger("Failed to fetch operations");
        console.error("Failed to fetch operations:", error);
        pipelineStore.setOperations([]);
        return [];
    }
}

/**
 * Fetch available cameras and store them in the pipeline store.
 *
 * @param {object} pipelineStore - Store used to persist cameras.
 * @returns {Promise<Array<object>>} Resolved cameras list.
 */
export async function fetchAvailableCameras(pipelineStore) {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-available-cameras`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        const cameras = Object.entries(data || {}).map(([name, cameraInfo]) => {
            let resolvedCameraId = name;
            if (cameraInfo?.bus_id != null) {
                resolvedCameraId = String(cameraInfo.bus_id);
            } else if (cameraInfo?.id != null) {
                resolvedCameraId = String(cameraInfo.id);
            }

            return {
                name,
                urlSafeName: cameraInfo?.name ?? name.replaceAll(" ", "_"),
                id: resolvedCameraId,
            };
        });

        pipelineStore.setCameras(cameras);
        console.log("Loaded cameras from server:", cameras);
        return cameras;
    } catch (error) {
        showDanger("Failed to fetch cameras");
        console.error("Failed to fetch cameras:", error);
        pipelineStore.setCameras([]);
        return [];
    }
}

/**
 * Fetch pipeline names and store them in the pipeline store.
 *
 * @param {object} pipelineStore - Store used to persist pipelines.
 * @returns {Promise<Array<object>>} Resolved pipelines list.
 */
export async function fetchPipelines(pipelineStore) {
    try {
        const response = await fetch(`${BACKEND_BASE_URL}/get-pipeline-names`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const pipelineNames = await response.json();

        const pipelines = pipelineNames.map((name) => ({
            name,
            displayName: name
                .replaceAll("_", " ")
                .replaceAll(/\b\w/g, (l) => l.toUpperCase()),
        }));

        pipelineStore.setPipelines(pipelines);
        console.log("Loaded pipelines from server:", pipelines);
        return pipelines;
    } catch (error) {
        showDanger("Failed to fetch pipelines");
        console.error("Failed to fetch pipelines:", error);
        pipelineStore.setPipelines([]);
        return [];
    }
}

/**
 * Fetch the configuration for a single pipeline.
 *
 * @param {string} pipelineName - Pipeline name to load.
 * @returns {Promise<Array|object>} Pipeline configuration payload.
 */
export async function fetchPipelineConfig(pipelineName) {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-pipeline-config/${encodeURIComponent(pipelineName)}`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const pipelineConfig = await response.json();
        console.log("Loaded pipeline config from server:", pipelineConfig);
        return pipelineConfig;
    } catch (error) {
        showDanger("Failed to fetch pipeline config");
        console.error("Failed to fetch pipeline config:", error);
        return [];
    }
}

/**
 * Fetch the complete pipeline configuration as raw text.
 *
 * @returns {Promise<{content: string, revision: string}>} Raw content and revision.
 */
export async function fetchPipelineConfigJson() {
    const response = await fetch(`${BACKEND_BASE_URL}/pipeline-config/json`);
    let payload = {};
    try {
        payload = await response.json();
    } catch {
        payload = {};
    }
    if (
        !response.ok ||
        typeof payload.content !== "string" ||
        typeof payload.revision !== "string"
    ) {
        throw new Error(
            payload.error || `HTTP error! status: ${response.status}`,
        );
    }
    return payload;
}

/**
 * Validate and save the complete raw pipeline configuration.
 *
 * @param {string} content - Complete pipeline_config.json text.
 * @param {string} revision - Revision returned when the editor loaded the file.
 * @returns {Promise<object>} Backend response payload.
 */
export async function savePipelineConfigJson(content, revision) {
    const response = await fetch(`${BACKEND_BASE_URL}/pipeline-config/json`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content, revision }),
    });
    let payload = {};
    try {
        payload = await response.json();
    } catch {
        payload = {};
    }
    if (!response.ok) {
        const location =
            payload.line && payload.column
                ? ` at line ${payload.line}, column ${payload.column}`
                : "";
        throw new Error(
            `${payload.error || `HTTP error! status: ${response.status}`}${location}${payload.detail ? `: ${payload.detail}` : ""}`,
        );
    }
    return payload;
}

/**
 * Save a pipeline configuration to the backend.
 *
 * @param {string} pipelineName - Pipeline name to save.
 * @param {object|Array} pipelineConfig - Configuration payload.
 * @returns {Promise<object>} Backend response payload.
 */
export async function savePipelineConfig(pipelineName, pipelineConfig) {
    const response = await fetch(
        `${BACKEND_BASE_URL}/save-pipeline-config/${encodeURIComponent(pipelineName)}`,
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(pipelineConfig),
        },
    );
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

/**
 * Delete a pipeline configuration from the backend.
 *
 * @param {string} pipelineName - Pipeline name to delete.
 * @returns {Promise<object>} Backend response payload.
 */
export async function deletePipelineConfig(pipelineName) {
    const response = await fetch(
        `${BACKEND_BASE_URL}/delete-pipeline/${encodeURIComponent(pipelineName)}`,
        {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
        },
    );
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

/**
 * Fetch settings for a single pipeline.
 *
 * @param {string} pipelineName - Pipeline name whose settings to load.
 * @returns {Promise<object>} Pipeline settings payload.
 */
export async function fetchPipelineSettings(pipelineName) {
    const response = await fetch(
        `${BACKEND_BASE_URL}/pipeline-settings/${encodeURIComponent(pipelineName)}`,
    );
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

/**
 * Save settings for a single pipeline.
 *
 * @param {string} pipelineName - Pipeline name whose settings to save.
 * @param {{limit_frames_to_camera_capture_speed: boolean}} settings - Settings payload.
 * @returns {Promise<object>} Backend response payload.
 */
export async function savePipelineSettings(pipelineName, settings) {
    const response = await fetch(
        `${BACKEND_BASE_URL}/pipeline-settings/${encodeURIComponent(pipelineName)}`,
        {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings),
        },
    );
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

/**
 * Fetch thread information for a pipeline.
 *
 * @param {string} pipelineName - Pipeline name to inspect.
 * @returns {Promise<object>} Thread info payload.
 */
export async function fetchPipelineThreadInfo(pipelineName) {
    const response = await fetch(
        `${BACKEND_BASE_URL}/get-pipeline-thread-info/${encodeURIComponent(pipelineName)}`,
    );
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

/**
 * Fetch the backend restart-required flag.
 *
 * @returns {Promise<object>} Restart-required payload.
 */
export async function getRestartRequired() {
    const response = await fetch(`${BACKEND_BASE_URL}/get_restart_required`);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

/**
 * Update the backend restart-required flag.
 *
 * @param {boolean} required - Whether a restart is required.
 * @returns {Promise<object>} Backend response payload.
 */
export async function setRestartRequired(required) {
    const response = await fetch(`${BACKEND_BASE_URL}/set_restart_required`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ required }),
    });
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

/**
 * Request a backend restart.
 *
 * @returns {Promise<object>} Backend response payload.
 */
export async function restartBackend() {
    const response = await fetch(`${BACKEND_BASE_URL}/restart-backend`, {
        method: "POST",
    });
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}
