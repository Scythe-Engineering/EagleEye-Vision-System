import { BACKEND_BASE_URL } from "../../config.js";
import { showDanger } from "../../ui/notificationSystem.js";

function toTitleCaseName(name) {
    return String(name)
        .replaceAll(".py", "")
        .replaceAll("_", " ")
        .replaceAll(/\b\w/g, (l) => l.toUpperCase());
}

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

export async function fetchAvailableCameras(pipelineStore) {
    try {
        const response = await fetch(`${BACKEND_BASE_URL}/get-available-cameras`);
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

export async function fetchPipelines(pipelineStore) {
    try {
        const response = await fetch(`${BACKEND_BASE_URL}/get-pipeline-names`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const pipelineNames = await response.json();

        const pipelines = pipelineNames.map((name) => ({
            name,
            displayName: name.replaceAll("_", " ").replaceAll(/\b\w/g, (l) => l.toUpperCase()),
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

export async function fetchPipelineThreadInfo(pipelineName) {
    const response = await fetch(
        `${BACKEND_BASE_URL}/get-pipeline-thread-info/${encodeURIComponent(pipelineName)}`,
    );
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

export async function getRestartRequired() {
    const response = await fetch(`${BACKEND_BASE_URL}/get_restart_required`);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}

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

export async function restartBackend() {
    const response = await fetch(`${BACKEND_BASE_URL}/restart-backend`, {
        method: "POST",
    });
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
}
