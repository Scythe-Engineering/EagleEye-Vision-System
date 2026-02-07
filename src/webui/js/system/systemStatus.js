const DEFAULT_TEXT = "N/A";

function formatPercent(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return DEFAULT_TEXT;
    }
    return `${value.toFixed(1)}%`;
}

function formatNumber(value, unit) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return DEFAULT_TEXT;
    }
    return `${value.toFixed(1)} ${unit}`;
}

function formatInteger(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return DEFAULT_TEXT;
    }
    return `${Math.round(value)}`;
}

function resolveStatusText(status) {
    if (typeof status === "string" && status.trim().length > 0) {
        return status;
    }
    return DEFAULT_TEXT;
}

function toDialPercent(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return null;
    }
    return Math.min(100, Math.max(0, value));
}

function setDial(dial, label, percent) {
    if (!dial || !label) {
        return;
    }
    if (typeof percent !== "number" || !Number.isFinite(percent)) {
        dial.setAttribute("stroke-dasharray", "0, 100");
        label.textContent = DEFAULT_TEXT;
        return;
    }
    dial.setAttribute("stroke-dasharray", `${percent.toFixed(1)}, 100`);
    label.textContent = `${percent.toFixed(0)}%`;
}

function normalizePipelineList(pipelines) {
    if (!Array.isArray(pipelines)) {
        return [];
    }
    return pipelines
        .filter((pipeline) => pipeline && typeof pipeline.name === "string")
        .map((pipeline) => ({
            name: pipeline.name,
            active: Boolean(pipeline.active),
        }));
}

function renderPipelines(container, pipelines) {
    if (!container) {
        return;
    }

    container.innerHTML = "";
    if (pipelines.length === 0) {
        const emptyState = document.createElement("div");
        emptyState.className = "text-sm text-[#ac8a2f]";
        emptyState.textContent = "No pipelines available";
        container.appendChild(emptyState);
        return;
    }

    const fragment = document.createDocumentFragment();
    for (const pipeline of pipelines) {
        const row = document.createElement("div");
        row.className =
            "flex items-center justify-between bg-[#1f1f1f] border border-[#414141] rounded-lg px-3 py-2";
        row.style.boxShadow = "4px 4px 8px rgba(0, 0, 0, 0.4)";

        const name = document.createElement("span");
        name.className = "text-sm text-white font-medium";
        name.textContent = pipeline.name;

        const badge = document.createElement("span");
        badge.className =
            "text-xs font-semibold px-2 py-1 rounded-full border";
        if (pipeline.active) {
            badge.classList.add(
                "text-emerald-200",
                "border-emerald-500/40",
                "bg-emerald-900/30",
            );
            badge.textContent = "Active";
        } else {
            badge.classList.add(
                "text-gray-200",
                "border-gray-500/40",
                "bg-gray-800/40",
            );
            badge.textContent = "Inactive";
        }

        row.appendChild(name);
        row.appendChild(badge);
        fragment.appendChild(row);
    }

    container.appendChild(fragment);
}

export function createSystemStatusModule() {
    const cpuPercent = document.getElementById("systemCpuPercent");
    const cpuDetail = document.getElementById("systemCpuDetail");
    const memoryPercent = document.getElementById("systemMemoryPercent");
    const memoryDetail = document.getElementById("systemMemoryDetail");
    const storagePercent = document.getElementById("systemStoragePercent");
    const storageDetail = document.getElementById("systemStorageDetail");
    const cpuDial = document.getElementById("systemCpuDial");
    const cpuDialValue = document.getElementById("systemCpuDialValue");
    const memoryDial = document.getElementById("systemMemoryDial");
    const memoryDialValue = document.getElementById("systemMemoryDialValue");
    const storageDial = document.getElementById("systemStorageDial");
    const storageDialValue = document.getElementById("systemStorageDialValue");
    const pipelineList = document.getElementById("systemPipelineList");
    const pipelineCount = document.getElementById("systemPipelineCount");

    return {
        render(data) {
            if (!data || typeof data !== "object") {
                return;
            }

            if (cpuPercent) {
                cpuPercent.textContent = resolveStatusText(
                    formatPercent(data.cpu?.percent),
                );
            }
            setDial(cpuDial, cpuDialValue, toDialPercent(data.cpu?.percent));
            if (cpuDetail) {
                cpuDetail.textContent = `Cores: ${resolveStatusText(
                    formatInteger(data.cpu?.cores),
                )}`;
            }

            if (memoryPercent) {
                memoryPercent.textContent = resolveStatusText(
                    formatPercent(data.memory?.percent),
                );
            }
            setDial(
                memoryDial,
                memoryDialValue,
                toDialPercent(data.memory?.percent),
            );
            if (memoryDetail) {
                const used = resolveStatusText(
                    formatNumber(data.memory?.used_mb, "MB"),
                );
                const total = resolveStatusText(
                    formatNumber(data.memory?.total_mb, "MB"),
                );
                memoryDetail.textContent = `${used} / ${total}`;
            }

            if (storagePercent) {
                storagePercent.textContent = resolveStatusText(
                    formatPercent(data.storage?.percent),
                );
            }
            setDial(
                storageDial,
                storageDialValue,
                toDialPercent(data.storage?.percent),
            );
            if (storageDetail) {
                const used = resolveStatusText(
                    formatNumber(data.storage?.used_gb, "GB"),
                );
                const total = resolveStatusText(
                    formatNumber(data.storage?.total_gb, "GB"),
                );
                storageDetail.textContent = `${used} / ${total}`;
            }

            const pipelines = normalizePipelineList(data.pipelines);
            renderPipelines(pipelineList, pipelines);
            if (pipelineCount) {
                const activeCount = pipelines.filter(
                    (pipeline) => pipeline.active,
                ).length;
                pipelineCount.textContent = `${activeCount} active`;
            }
        },
    };
}
