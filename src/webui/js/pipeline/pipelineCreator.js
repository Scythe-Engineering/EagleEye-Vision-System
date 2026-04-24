import {
    createDescriptionPopup,
    renderOperations,
    FlowchartRenderer,
} from "./rendering.js";
import { handleDragStart } from "./dragDrop.js";
import { debounce, escapeHtml } from "./utils.js";
import { BACKEND_BASE_URL } from "../config.js";
import { pipelineStore } from "./PipelineStore.js";
import { showDanger, showWarning } from "../ui/notificationSystem.js";
import { registerSettingsPopup } from "./settingsPopup.js";

registerSettingsPopup();

function handleDragStartWithLogging(
    event,
    item,
    fromIndex = null,
    collection = null,
) {
    console.log("[PIPELINE] Drag start initiated", {
        draggedElement: event.target,
        itemInstanceId: item?.instanceId || null,
        fromIndex: fromIndex,
        timestamp: new Date().toISOString(),
    });
    return handleDragStart(event, item, collection, fromIndex);
}

let isInitialized = false;

let flowchartRenderer = null;

const restartRequiredOperations = new Map();
const PROFILING_STALE_TIMEOUT_MS = 2000;

let pipelineArea;
let pipelinePlaceholder;
let operationsList;
let runButton;
let pipelineSelect;
let pipelineCameraNote;
let newPipelineButton;
let deletePipelineButton;
let restartIndicator;
let flowchartCanvas;
let executionTimestepsList;
let executionSummaryContent;
let profilingDetailsOverlay;
let profilingDetailsBackdrop;
let profilingDetailsBody;
let profilingDetailsTitle;
let profilingDetailsCloseButton;
let profilingDetailsInfoButton;
let profilingDetailsAverageCheckbox;
/**
 * Running sums for cumulative arithmetic mean of profiling samples (since mode enabled).
 *
 * @type {{
 *   pipelineName: string,
 *   snapshotCount: number,
 *   frameWall: { sum: number, n: number },
 *   operations: Map<
 *     string,
 *     { sumMs: number, n: number, name: string, timestep: number, thread: number }
 *   >,
 *   timesteps: Map<
 *     number,
 *     {
 *       sumWall: number,
 *       sumMaxOpTime: number,
 *       sumOpCount: number,
 *       n: number,
 *       lastMaxName: string | null,
 *     }
 *   >,
 * } | null}
 */
let profilingAverageAccumulator = null;
let profilingStaleIntervalId = null;

let pipelineErrorPopup;
const operationErrorsByUuid = new Map();
const downstreamErrorUuids = new Set();

const autoSavePipeline = debounce(autoSavePipelineImpl, 500);

function getOperations() {
    return pipelineStore.state.operations;
}

function getPipeline() {
    return pipelineStore.getNodesForRenderer();
}

function getPipelines() {
    return pipelineStore.state.pipelines;
}

function getSelectedPipeline() {
    const pipelineName = pipelineStore.state.currentPipeline?.pipelineName;
    if (!pipelineName) {
        return null;
    }
    const selectedPipeline = pipelineStore.state.pipelines.find(
        (p) => p.name === pipelineName,
    );
    return selectedPipeline ?? null;
}

function hideAllProfilingBadges() {
    if (!flowchartRenderer) {
        return;
    }
    for (const node of flowchartRenderer.nodes.values()) {
        node.hideProfilingBadge?.();
    }
}

function getProfilingThreadColor(threadNumber) {
    const threadColors = [
        "#ff6b6b",
        "#4ecdc4",
        "#45b7d1",
        "#96ceb4",
        "#ffeaa7",
        "#dfe6e9",
        "#fd79a8",
        "#a29bfe",
        "#00b894",
        "#e17055",
    ];
    const normalizedThread = Number(threadNumber);
    if (!Number.isFinite(normalizedThread) || normalizedThread <= 0) {
        return "#5a5a5a";
    }
    return threadColors[(normalizedThread - 1) % threadColors.length];
}

/**
 * Distinct threads scheduled at the timestep and sum of execution_time_ms.
 *
 * @param {Record<string, any>} operationsByUuid
 * @param {number} timestep
 * @returns {{ threadsSorted: number[], sumMs: number }}
 */
function analyzeTimestepOperations(operationsByUuid, timestep) {
    const presentThreads = new Set();
    let sumMs = 0;
    for (const row of Object.values(operationsByUuid)) {
        if (Number(row?.timestep) !== timestep) {
            continue;
        }
        let thread = Number(row?.thread);
        if (!Number.isFinite(thread)) {
            thread = 0;
        }
        presentThreads.add(thread);
        const ms = Number(row?.execution_time_ms);
        if (Number.isFinite(ms) && ms > 0) {
            sumMs += ms;
        }
    }
    const threadsSorted = [...presentThreads].sort((a, b) => a - b);
    return { threadsSorted, sumMs };
}

/**
 * One circular badge: timestep index in the center; ring split into equal
 * slices (one per thread at this timestep), colored by thread id.
 *
 * @param {number[]} threadsSorted
 * @param {number} timestep
 * @param {number} size
 * @returns {string}
 */
function buildTimestepThreadBadgeSvg(threadsSorted, timestep, size) {
    const dim = Number.isFinite(size) && size > 8 ? size : 22;
    const cx = dim / 2;
    const cy = dim / 2;
    const R = dim / 2 - 0.5;
    const n = threadsSorted.length;
    const ts = String(timestep);
    const innerR = R * 0.58;
    const singleThread = n === 1 ? threadsSorted[0] : null;
    let textFill = "#f5f5f5";
    if (n === 1 && singleThread > 0) {
        textFill = "#0a0a0a";
    }
    const textEl = `<text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="central" fill="${textFill}" font-size="10" font-weight="600" font-family="ui-sans-serif,system-ui,sans-serif">${escapeHtml(ts)}</text>`;
    const svgOpen = `<svg viewBox="0 0 ${dim} ${dim}" width="${dim}" height="${dim}" class="shrink-0" aria-hidden="true">`;

    if (n <= 0) {
        const tip = escapeHtml(`Timestep ${timestep} (no thread data)`);
        return `${svgOpen}<circle cx="${cx}" cy="${cy}" r="${R}" fill="#3a3a3a"><title>${tip}</title></circle>${textEl}</svg>`;
    }
    if (n === 1) {
        const fill = getProfilingThreadColor(threadsSorted[0]);
        const tip = escapeHtml(
            `Timestep ${timestep} · thread ${threadsSorted[0]}`,
        );
        return `${svgOpen}<circle cx="${cx}" cy="${cy}" r="${R}" fill="${fill}"><title>${tip}</title></circle>${textEl}</svg>`;
    }

    const sliceAngle = (2 * Math.PI) / n;
    let angle = -Math.PI / 2;
    const paths = [];
    for (let i = 0; i < n; i++) {
        const endAngle = angle + sliceAngle;
        const x1 = cx + R * Math.cos(angle);
        const y1 = cy + R * Math.sin(angle);
        const x2 = cx + R * Math.cos(endAngle);
        const y2 = cy + R * Math.sin(endAngle);
        const largeArc = sliceAngle > Math.PI ? 1 : 0;
        const d = `M ${cx} ${cy} L ${x1} ${y1} A ${R} ${R} 0 ${largeArc} 1 ${x2} ${y2} Z`;
        const fill = getProfilingThreadColor(threadsSorted[i]);
        const tip = escapeHtml(
            `Timestep ${timestep} · thread ${threadsSorted[i]} (${n} threads, equal split)`,
        );
        paths.push(
            `<path d="${d}" fill="${fill}"><title>${tip}</title></path>`,
        );
        angle = endAngle;
    }
    const mask = `<circle cx="${cx}" cy="${cy}" r="${innerR}" fill="#1f1f1f"/>`;
    const tipAll = escapeHtml(
        `Timestep ${timestep} · ${n} threads (equal segments)`,
    );
    return `${svgOpen}<g><title>${tipAll}</title>${paths.join("")}</g>${mask}${textEl}</svg>`;
}

function renderExecutionTimestepsPanel(snapshot) {
    if (!executionTimestepsList) {
        return;
    }

    const timesteps = Array.isArray(snapshot?.timesteps)
        ? [...snapshot.timesteps]
        : [];
    if (timesteps.length === 0) {
        executionTimestepsList.innerHTML =
            '<div class="text-[#888]">No profiling data</div>';
        return;
    }

    const operationsByUuid = snapshot?.operations || {};
    timesteps.sort((a, b) => (a?.timestep ?? 0) - (b?.timestep ?? 0));
    const badgeSize = 22;
    executionTimestepsList.innerHTML = timesteps
        .map((row) => {
            const timestep = Number(row?.timestep) || 0;
            const { threadsSorted, sumMs } = analyzeTimestepOperations(
                operationsByUuid,
                timestep,
            );
            const wallMs = Number(row?.total_time_ms) || 0;
            const displayMs = sumMs > 0 ? sumMs : wallMs;
            const badgeSvg = buildTimestepThreadBadgeSvg(
                threadsSorted,
                timestep,
                badgeSize,
            );
            const titleParts = [
                `Timestep ${timestep}: Σ ops ${displayMs.toFixed(2)}ms`,
            ];
            if (sumMs > 0 && wallMs > 0 && Math.abs(sumMs - wallMs) > 0.05) {
                titleParts.push(`wall ${wallMs.toFixed(2)}ms`);
            }
            const rowTitle = escapeHtml(titleParts.join(" · "));
            return `<div class="flex items-center gap-2 py-1" title="${rowTitle}">
                ${badgeSvg}
                <div class="text-[#f1f1f1]">${displayMs.toFixed(2)}ms</div>
            </div>`;
        })
        .join("");
}

function renderExecutionSummary(snapshot) {
    if (!executionSummaryContent) {
        return;
    }

    const frameTimeMs = Number(snapshot?.frame_time_ms);
    if (!Number.isFinite(frameTimeMs) || frameTimeMs <= 0) {
        executionSummaryContent.innerHTML =
            '<div class="text-[#888]">No profiling data</div>';
        return;
    }

    const estimatedFps = 1000 / frameTimeMs;
    executionSummaryContent.innerHTML = `<div class="text-[#f1f1f1]">Flow: ${frameTimeMs.toFixed(2)}ms</div>
        <div class="text-[#9ad1a8]">FPS: ${estimatedFps.toFixed(1)}</div>`;
}

/**
 * @param {number} thread
 * @returns {string}
 */
function profilingThreadSwatchHtml(thread) {
    const t = Number(thread);
    const color = getProfilingThreadColor(t);
    const label = !Number.isFinite(t) || t <= 0 ? "—" : `Thread ${t}`;
    return `<span class="inline-flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-full shrink-0 ring-1 ring-white/10" style="background-color:${color}"></span><span class="text-[#ddd]">${label}</span></span>`;
}

function clearProfilingAverageState() {
    profilingAverageAccumulator = null;
}

/**
 * @param {string} pipelineName
 */
function ensureProfilingAverageAccumulator(pipelineName) {
    if (
        !profilingAverageAccumulator ||
        profilingAverageAccumulator.pipelineName !== pipelineName
    ) {
        profilingAverageAccumulator = {
            pipelineName,
            snapshotCount: 0,
            frameWall: { sum: 0, n: 0 },
            operations: new Map(),
            timesteps: new Map(),
        };
    }
}

/**
 * Incorporates one profiling snapshot into cumulative means.
 *
 * @param {Record<string, any>} snapshot
 * @param {string} pipelineName
 */
function mergeProfilingSnapshotIntoAverage(snapshot, pipelineName) {
    ensureProfilingAverageAccumulator(pipelineName);
    const a = profilingAverageAccumulator;
    if (!a) {
        return;
    }
    const ft = Number(snapshot.frame_time_ms);
    if (Number.isFinite(ft) && ft > 0) {
        a.frameWall.sum += ft;
        a.frameWall.n += 1;
    }
    for (const [uuid, row] of Object.entries(snapshot.operations || {})) {
        const ms = Number(row.execution_time_ms);
        if (!Number.isFinite(ms) || ms < 0) {
            continue;
        }
        let st = a.operations.get(uuid);
        if (!st) {
            st = {
                sumMs: 0,
                n: 0,
                name: "",
                timestep: -1,
                thread: 0,
            };
        }
        st.sumMs += ms;
        st.n += 1;
        st.name = String(row.name ?? "");
        const ts = Number(row.timestep);
        st.timestep = Number.isFinite(ts) ? ts : st.timestep;
        const th = Number(row.thread);
        st.thread = Number.isFinite(th) ? th : st.thread;
        a.operations.set(uuid, st);
    }
    for (const row of snapshot.timesteps || []) {
        const t = Number(row.timestep);
        if (!Number.isFinite(t)) {
            continue;
        }
        let ts = a.timesteps.get(t);
        if (!ts) {
            ts = {
                sumWall: 0,
                sumMaxOpTime: 0,
                sumOpCount: 0,
                n: 0,
                lastMaxName: null,
            };
        }
        const wall = Number(row.total_time_ms);
        const mx = Number(row.max_operation_time_ms);
        const cnt = Number(row.operation_count);
        if (Number.isFinite(wall) && wall >= 0) {
            ts.sumWall += wall;
        }
        if (Number.isFinite(mx) && mx >= 0) {
            ts.sumMaxOpTime += mx;
        }
        if (Number.isFinite(cnt) && cnt >= 0) {
            ts.sumOpCount += cnt;
        }
        ts.n += 1;
        if (row.max_operation_name) {
            ts.lastMaxName = String(row.max_operation_name);
        }
        a.timesteps.set(t, ts);
    }
    a.snapshotCount += 1;
}

/**
 * @param {Record<string, any>} latest
 * @returns {Record<string, any>}
 */
function buildProfilingAverageDisplaySnapshot(latest) {
    const a = profilingAverageAccumulator;
    if (!a || a.snapshotCount <= 0) {
        return latest;
    }
    const operations = {};
    for (const [uuid, st] of a.operations) {
        const avgMs = st.n > 0 ? st.sumMs / st.n : 0;
        operations[uuid] = {
            name: st.name,
            timestep: st.timestep,
            thread: st.thread,
            execution_time_ms: avgMs,
        };
    }
    const latestTimesteps = latest.timesteps || [];
    const timesteps = [...a.timesteps.keys()]
        .sort((x, y) => x - y)
        .map((t) => {
            const ts = a.timesteps.get(t);
            const fromLatest = latestTimesteps.find(
                (r) => Number(r?.timestep) === t,
            );
            return {
                timestep: t,
                total_time_ms: ts && ts.n > 0 ? ts.sumWall / ts.n : 0,
                max_operation_time_ms:
                    ts && ts.n > 0 ? ts.sumMaxOpTime / ts.n : 0,
                operation_count: ts && ts.n > 0 ? ts.sumOpCount / ts.n : 0,
                max_operation_name:
                    (ts && ts.lastMaxName) || fromLatest?.max_operation_name,
                max_operation_uuid: fromLatest?.max_operation_uuid ?? null,
            };
        });
    const frameMs =
        a.frameWall.n > 0
            ? a.frameWall.sum / a.frameWall.n
            : Number(latest.frame_time_ms);
    return {
        ...latest,
        frame_time_ms: Number.isFinite(frameMs)
            ? frameMs
            : latest.frame_time_ms,
        operations,
        timesteps,
    };
}

/**
 * @param {Record<string, any>|null|undefined} snapshot
 * @param {string} pipelineName
 * @param {{ cumulativeAverage?: boolean, mergeCount?: number }} [hint]
 * @returns {string}
 */
function buildProfilingDetailsHtml(snapshot, pipelineName, hint = {}) {
    if (!snapshot) {
        return `<p class="text-[#888] text-sm leading-relaxed">No profiling snapshot is stored yet for <span class="text-[#f9c845]">${escapeHtml(pipelineName)}</span>. Run the pipeline and wait for updates over SSE.</p>`;
    }

    const ops = snapshot.operations || {};
    const timestepRows = [...(snapshot.timesteps || [])].sort(
        (a, b) => (a?.timestep ?? 0) - (b?.timestep ?? 0),
    );
    const frameMs = Number(snapshot.frame_time_ms);
    const fps =
        Number.isFinite(frameMs) && frameMs > 0
            ? (1000 / frameMs).toFixed(1)
            : "—";
    const seq = snapshot.frame_seq != null ? String(snapshot.frame_seq) : "—";
    const tsMs = Number(snapshot.timestamp_ms);
    const tsLabel = Number.isFinite(tsMs)
        ? new Date(tsMs).toLocaleString()
        : "—";

    const parts = [];
    parts.push(
        `<div class="rounded-lg border border-[#3a3a3a] bg-[#181818] p-4 mb-5">`,
    );
    parts.push(
        `<div class="text-xs uppercase tracking-wide text-[#888] mb-2">Current snapshot</div>`,
    );
    if (hint.cumulativeAverage && (hint.mergeCount ?? 0) > 0) {
        parts.push(
            `<p class="text-[#9ad1a8] text-xs mb-2 leading-relaxed">Cumulative arithmetic mean of each numeric field over <span class="font-semibold">${hint.mergeCount}</span> profiling update(s) since this mode was enabled (converges toward stable values as samples grow; not a rolling window).</p>`,
        );
    }
    parts.push(`<div class="grid gap-2 text-sm">`);
    parts.push(
        `<div><span class="text-[#888]">Pipeline</span> · <span class="text-[#f1f1f1] font-medium">${escapeHtml(pipelineName)}</span></div>`,
    );
    parts.push(
        `<div><span class="text-[#888]">Frame wall time</span> · <span class="text-[#f1f1f1]">${Number.isFinite(frameMs) && frameMs > 0 ? `${frameMs.toFixed(2)} ms` : "—"}</span>` +
            (fps !== "—"
                ? ` <span class="text-[#9ad1a8]">(~${fps} FPS)</span>`
                : "") +
            `</div>`,
    );
    parts.push(
        `<div><span class="text-[#888]">Frame sequence</span> · <span class="text-[#f1f1f1]">${escapeHtml(seq)}</span></div>`,
    );
    parts.push(
        `<div><span class="text-[#888]">Recorded at</span> · <span class="text-[#f1f1f1]">${escapeHtml(tsLabel)}</span></div>`,
    );
    parts.push(`</div></div>`);

    parts.push(
        `<h4 class="text-[#f9c845] text-sm font-semibold mb-2 border-b border-[#414141] pb-1">By timestep</h4>`,
    );
    parts.push(
        `<p class="text-[#888] text-xs mb-3 leading-relaxed">Each timestep runs a group of operations (they may overlap on different threads). <span class="text-[#c9c9c9]">Wall</span> is measured around the whole group; <span class="text-[#c9c9c9]">Σ ops</span> is the sum of per-operation execution times for that timestep.</p>`,
    );

    if (timestepRows.length === 0) {
        parts.push(
            `<p class="text-[#666] text-xs mb-4">No timestep rows in this snapshot.</p>`,
        );
    } else {
        for (const row of timestepRows) {
            const tsN = Number(row?.timestep) || 0;
            const { sumMs } = analyzeTimestepOperations(ops, tsN);
            const wallMs = Number(row?.total_time_ms) || 0;
            const opEntries = Object.entries(ops).filter(
                ([, op]) => Number(op?.timestep) === tsN,
            );
            opEntries.sort((a, b) => {
                const ta = Number(a[1]?.thread) || 0;
                const tb = Number(b[1]?.thread) || 0;
                if (ta !== tb) {
                    return ta - tb;
                }
                return String(a[1]?.name || "").localeCompare(
                    String(b[1]?.name || ""),
                );
            });
            const dispSum = sumMs > 0 ? sumMs : wallMs;
            const countFromPayload = Number(row?.operation_count);
            parts.push(
                `<div class="mb-4 rounded-md border border-[#3a3a3a] overflow-hidden bg-[#1a1a1a]">`,
            );
            parts.push(
                `<div class="px-3 py-2 bg-[#252525] flex flex-wrap gap-x-3 gap-y-1 text-xs">`,
            );
            parts.push(
                `<span class="font-semibold text-[#f9c845]">Timestep ${tsN}</span>`,
            );
            parts.push(
                `<span class="text-[#aaa]">wall <span class="text-[#f1f1f1]">${wallMs.toFixed(2)} ms</span></span>`,
            );
            parts.push(
                `<span class="text-[#aaa]">Σ ops <span class="text-[#f1f1f1]">${dispSum.toFixed(2)} ms</span></span>`,
            );
            if (Number.isFinite(countFromPayload)) {
                parts.push(
                    `<span class="text-[#aaa]">count <span class="text-[#f1f1f1]">${countFromPayload}</span></span>`,
                );
            }
            parts.push(`</div>`);
            if (row?.max_operation_name) {
                parts.push(
                    `<div class="px-3 py-1.5 text-xs text-[#ac8a2f] border-b border-[#333]">Heaviest op this frame: <span class="text-[#e8e8e8]">${escapeHtml(String(row.max_operation_name))}</span> (${Number(row?.max_operation_time_ms || 0).toFixed(2)} ms)</div>`,
                );
            }
            parts.push(
                `<table class="w-full text-xs border-collapse"><thead><tr class="bg-[#222] text-left text-[#c9c9c9]"><th class="px-3 py-1.5 font-medium">Operation</th><th class="px-3 py-1.5 font-medium w-32">Thread</th><th class="px-3 py-1.5 font-medium w-24 text-right">Time (ms)</th></tr></thead><tbody>`,
            );
            if (!opEntries.length) {
                parts.push(
                    `<tr><td colspan="3" class="px-3 py-2 text-[#666]">No operation rows for this timestep.</td></tr>`,
                );
            } else {
                for (const [uuid, op] of opEntries) {
                    const name = escapeHtml(
                        String(op?.name ?? uuid.slice(0, 8)),
                    );
                    const th = Number(op?.thread);
                    const ms = Number(op?.execution_time_ms);
                    const msStr = Number.isFinite(ms) ? ms.toFixed(2) : "—";
                    parts.push(
                        `<tr class="border-t border-[#2f2f2f]"><td class="px-3 py-1.5 text-[#e4e4e4]">${name}</td><td class="px-3 py-1.5">${profilingThreadSwatchHtml(th)}</td><td class="px-3 py-1.5 text-right text-[#e4e4e4] font-mono">${msStr}</td></tr>`,
                    );
                }
            }
            parts.push(`</tbody></table></div>`);
        }
    }

    parts.push(
        `<h4 class="text-[#f9c845] text-sm font-semibold mt-6 mb-2 border-b border-[#414141] pb-1">By thread</h4>`,
    );
    parts.push(
        `<p class="text-[#888] text-xs mb-3 leading-relaxed">All operations in this snapshot, grouped by worker thread. Each row is one operation at a timestep with its measured execution time.</p>`,
    );

    const byThread = new Map();
    for (const [uuid, op] of Object.entries(ops)) {
        let th = Number(op?.thread);
        if (!Number.isFinite(th)) {
            th = 0;
        }
        const step = Number(op?.timestep);
        const stepLabel =
            Number.isFinite(step) && step >= 0 ? String(step) : "—";
        const arr = byThread.get(th) || [];
        arr.push({
            timestep: stepLabel,
            name: String(op?.name ?? "—"),
            ms: Number(op?.execution_time_ms),
            uuid,
        });
        byThread.set(th, arr);
    }
    const threads = [...byThread.keys()].sort((a, b) => a - b);
    if (!threads.length) {
        parts.push(
            `<p class="text-[#666] text-xs">No operations in this snapshot.</p>`,
        );
    } else {
        for (const th of threads) {
            const rows = byThread.get(th) || [];
            rows.sort((a, b) => {
                const cmp = String(a.timestep).localeCompare(
                    String(b.timestep),
                    undefined,
                    { numeric: true },
                );
                if (cmp !== 0) {
                    return cmp;
                }
                return a.name.localeCompare(b.name);
            });
            const subtotal = rows.reduce((s, r) => {
                const m = Number(r.ms);
                return s + (Number.isFinite(m) && m > 0 ? m : 0);
            }, 0);
            parts.push(
                `<div class="mb-4 rounded-md border border-[#3a3a3a] overflow-hidden bg-[#1a1a1a]">`,
            );
            parts.push(
                `<div class="px-3 py-2 bg-[#252525] text-xs font-semibold text-[#e8e8e8] flex items-center gap-2">${profilingThreadSwatchHtml(th)}<span class="text-[#888] font-normal">Σ ${subtotal.toFixed(2)} ms</span></div>`,
            );
            parts.push(
                `<table class="w-full text-xs border-collapse"><thead><tr class="bg-[#222] text-left text-[#c9c9c9]"><th class="px-3 py-1.5 font-medium w-20">Timestep</th><th class="px-3 py-1.5 font-medium">Operation</th><th class="px-3 py-1.5 font-medium w-24 text-right">Time (ms)</th></tr></thead><tbody>`,
            );
            for (const r of rows) {
                const ms = Number(r.ms);
                const msStr = Number.isFinite(ms) ? ms.toFixed(2) : "—";
                parts.push(
                    `<tr class="border-t border-[#2f2f2f]"><td class="px-3 py-1.5 text-[#c9c9c9] font-mono">${escapeHtml(String(r.timestep))}</td><td class="px-3 py-1.5 text-[#e4e4e4]">${escapeHtml(r.name)}</td><td class="px-3 py-1.5 text-right font-mono text-[#e4e4e4]">${msStr}</td></tr>`,
                );
            }
            parts.push(`</tbody></table></div>`);
        }
    }

    return parts.join("");
}

/**
 * Refreshes the profiling details modal when it is open.
 *
 * @param {Record<string, any>|null|undefined} snapshot - Snapshot from
 *   `applyProfilingSnapshot`, `null` to show the empty state, or `undefined`
 *   to load the latest snapshot for the currently selected pipeline.
 */
function refreshProfilingDetailsPopupIfVisible(snapshot) {
    if (
        !profilingDetailsOverlay ||
        profilingDetailsOverlay.classList.contains("hidden")
    ) {
        return;
    }
    const selected = getSelectedPipeline();
    const pipelineName = selected?.name || "—";
    let snapshotForView = snapshot;
    if (snapshot === undefined) {
        snapshotForView = selected?.name
            ? pipelineStore.getProfilingSnapshot(selected.name)
            : null;
    }
    if (!profilingDetailsBody || !profilingDetailsTitle) {
        return;
    }

    const useAvg = Boolean(profilingDetailsAverageCheckbox?.checked);
    let displaySnapshot = snapshotForView;
    const hint = {
        cumulativeAverage: false,
        mergeCount: 0,
    };

    if (useAvg && snapshotForView && selected?.name) {
        mergeProfilingSnapshotIntoAverage(snapshotForView, selected.name);
        displaySnapshot = buildProfilingAverageDisplaySnapshot(snapshotForView);
        hint.cumulativeAverage = true;
        hint.mergeCount = profilingAverageAccumulator?.snapshotCount ?? 0;
    }

    profilingDetailsBody.innerHTML = buildProfilingDetailsHtml(
        displaySnapshot,
        pipelineName,
        hint,
    );
    profilingDetailsTitle.textContent = selected?.name
        ? `Profiling — ${selected.name}`
        : "Profiling details";
}

function profilingDetailsOnKeydown(event) {
    if (event.key === "Escape") {
        event.preventDefault();
        closeProfilingDetailsPopup();
    }
}

function closeProfilingDetailsPopup() {
    if (
        !profilingDetailsOverlay ||
        profilingDetailsOverlay.classList.contains("hidden")
    ) {
        return;
    }
    profilingDetailsOverlay.classList.add("hidden");
    profilingDetailsOverlay.setAttribute("aria-hidden", "true");
    document.removeEventListener("keydown", profilingDetailsOnKeydown, true);
}

function openProfilingDetailsPopup() {
    const selected = getSelectedPipeline();
    if (!selected?.name) {
        showWarning("Select a pipeline first.");
        return;
    }
    if (!profilingDetailsOverlay) {
        return;
    }
    document.removeEventListener("keydown", profilingDetailsOnKeydown, true);
    profilingDetailsOverlay.classList.remove("hidden");
    profilingDetailsOverlay.setAttribute("aria-hidden", "false");
    document.addEventListener("keydown", profilingDetailsOnKeydown, true);
    refreshProfilingDetailsPopupIfVisible(undefined);
}

function clearProfilingUI() {
    hideAllProfilingBadges();
    renderExecutionTimestepsPanel(null);
    renderExecutionSummary(null);
    refreshProfilingDetailsPopupIfVisible(null);
}

function applyProfilingSnapshot(snapshot) {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline || !snapshot) {
        clearProfilingUI();
        return;
    }

    if (snapshot.pipeline_name !== selectedPipeline.name) {
        return;
    }

    if (flowchartRenderer) {
        const operations = snapshot.operations || {};
        for (const [instanceId, node] of flowchartRenderer.nodes) {
            const uuid = pipelineStore.instanceIdToUuid.get(instanceId);
            const profilingInfo = uuid ? operations[uuid] : null;
            if (profilingInfo) {
                node.updateProfilingInfo?.(profilingInfo);
            } else {
                node.hideProfilingBadge?.();
            }
        }
    }

    renderExecutionTimestepsPanel(snapshot);
    renderExecutionSummary(snapshot);
    refreshProfilingDetailsPopupIfVisible(snapshot);
}

function applySelectedPipelineProfiling() {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline) {
        clearProfilingUI();
        return;
    }

    const snapshot = pipelineStore.getProfilingSnapshot(selectedPipeline.name);
    if (!snapshot) {
        clearProfilingUI();
        return;
    }
    applyProfilingSnapshot(snapshot);
}

function handleProfilingUpdate(payload) {
    if (!payload || typeof payload.pipeline_name !== "string") {
        return;
    }

    pipelineStore.setProfilingSnapshot(payload);
}

function checkAndClearStaleProfiling() {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline) {
        clearProfilingUI();
        return;
    }

    const lastUpdateMs = pipelineStore.getProfilingLastUpdateMs(
        selectedPipeline.name,
    );
    if (lastUpdateMs <= 0) {
        clearProfilingUI();
        return;
    }

    if (Date.now() - lastUpdateMs > PROFILING_STALE_TIMEOUT_MS) {
        clearProfilingUI();
    }
}

function getDeviceInputNodes() {
    return pipelineStore.getNodes().filter((node) => {
        return (
            pipelineStore.normalizeOperationId(node.operationId) ===
            "device_input"
        );
    });
}

function getDeviceInputBusIds() {
    const busIds = new Set();
    pipelineStore.getNodes().forEach((node) => {
        const operationId = pipelineStore.normalizeOperationId(
            node.operationId,
        );
        if (operationId === "device_input") {
            const busId = node.config?.bus_id;
            if (busId !== undefined && busId !== null) {
                busIds.add(String(busId));
            }
        }
    });
    return Array.from(busIds);
}

function formatPipelineCameraNote(busIds) {
    if (busIds.length === 0) {
        return { text: "No camera bus IDs configured", title: "" };
    }
    const sortedBusIds = [...busIds].sort((first, second) =>
        first.localeCompare(second, undefined, { sensitivity: "accent" }),
    );
    if (sortedBusIds.length <= 2) {
        return {
            text: `Bus IDs: ${sortedBusIds.join(", ")}`,
            title: sortedBusIds.join(", "),
        };
    }
    const visibleBusIds = sortedBusIds.slice(0, 2).join(", ");
    return {
        text: `Bus IDs: ${visibleBusIds} (+${sortedBusIds.length - 2} more)`,
        title: sortedBusIds.join(", "),
    };
}

function updatePipelineCameraNote() {
    if (!pipelineCameraNote) {
        return;
    }
    const busIds = getDeviceInputBusIds();
    const note = formatPipelineCameraNote(busIds);
    pipelineCameraNote.textContent = note.text;
    pipelineCameraNote.title = note.title;
}

async function fetchAvailableOperations() {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-available-operations`,
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        const operations = data.operations.map((op) => ({
            id: op.name,
            name: op.name
                .replaceAll(".py", "")
                .replaceAll("_", " ")
                .replaceAll(/\b\w/g, (l) => l.toUpperCase()),
            type: op.category.toUpperCase(),
            folder: op.folder || "Uncategorized",
            description: op.description,
            path: op.path,
            configDataPath: op.config_data_path,
            isSecondary: op.is_secondary,
            hasVisualization: Boolean(op.has_visualization),
        }));

        pipelineStore.setOperations(operations);
        console.log("Loaded operations from server:", operations);
    } catch (error) {
        showDanger("Failed to fetch operations");
        console.error("Failed to fetch operations:", error);
        pipelineStore.setOperations([]);
    }
}

async function fetchAvailableCameras() {
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
                name: name,
                urlSafeName: cameraInfo?.name ?? name.replaceAll(" ", "_"),
                id: resolvedCameraId,
            };
        });

        pipelineStore.setCameras(cameras);
        console.log("Loaded cameras from server:", cameras);
    } catch (error) {
        showDanger("Failed to fetch cameras");
        console.error("Failed to fetch cameras:", error);
        pipelineStore.setCameras([]);
    }
}

function populateCameraDropdown() {
    cameraSelect.innerHTML = "";

    const cameras = pipelineStore.state.cameras;
    if (!Array.isArray(cameras) || cameras.length === 0) {
        const option = document.createElement("option");
        option.disabled = true;
        option.selected = true;
        option.textContent = "No cameras available";
        cameraSelect.appendChild(option);
        pipelineStore.setCurrentCamera(null);
        return;
    }

    for (let index = 0; index < cameras.length; index++) {
        const camera = cameras[index];
        const option = document.createElement("option");
        option.value = camera.urlSafeName;
        option.textContent = camera.name;
        if (index === 0) {
            option.selected = true;
            pipelineStore.setCurrentCamera(camera.name);
        }
        cameraSelect.appendChild(option);
    }
}

async function fetchPipelines() {
    try {
        const response = await fetch(`${BACKEND_BASE_URL}/get-pipeline-names`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const pipelineNames = await response.json();

        const pipelines = pipelineNames.map((name) => ({
            name: name,
            displayName: name
                .replaceAll("_", " ")
                .replaceAll(/\b\w/g, (l) => l.toUpperCase()),
        }));

        pipelineStore.setPipelines(pipelines);
        console.log("Loaded pipelines from server:", pipelines);
    } catch (error) {
        showDanger("Failed to fetch pipelines");
        console.error("Failed to fetch pipelines:", error);
        pipelineStore.setPipelines([]);
    }
}

async function fetchPipelineConfig(pipelineName) {
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

function populatePipelineDropdown(selectedPipelineName = null) {
    pipelineSelect.innerHTML = "";

    const defaultOption = document.createElement("option");
    defaultOption.disabled = true;
    defaultOption.textContent = "Select Pipeline";
    pipelineSelect.appendChild(defaultOption);

    let foundSelectedPipeline = false;
    const pipelines = getPipelines();

    for (let index = 0; index < pipelines.length; index++) {
        const pipelineItem = pipelines[index];
        const option = document.createElement("option");
        option.value = pipelineItem.name;
        option.textContent = pipelineItem.displayName;

        if (
            selectedPipelineName === pipelineItem.name ||
            (selectedPipelineName === null && index === 0)
        ) {
            option.selected = true;
            pipelineStore.setCurrentPipeline(pipelineItem.name);
            foundSelectedPipeline = true;
        }

        pipelineSelect.appendChild(option);
    }

    if (
        selectedPipelineName &&
        !foundSelectedPipeline &&
        pipelines.length > 0
    ) {
        console.warn(
            `Pipeline "${selectedPipelineName}" not found in pipelines list, selecting first pipeline`,
        );
        const firstOption = pipelineSelect.querySelector(
            "option:not([disabled])",
        );
        if (firstOption) {
            firstOption.selected = true;
            pipelineStore.setCurrentPipeline(pipelines[0].name);
        }
    }

    if (pipelines.length === 0) {
        pipelineStore.setCurrentPipeline(null);
    }
}

async function handlePipelineSelection() {
    const selectedValue = pipelineSelect.value;
    const pipelines = getPipelines();
    const selectedPipeline = pipelines.find(
        (pipelineItem) => pipelineItem.name === selectedValue,
    );
    console.log("Selected pipeline:", selectedPipeline);

    if (selectedPipeline) {
        pipelineStore.setCurrentPipeline(selectedPipeline.name);
        await loadPipelineIntoBuilder(selectedPipeline.name);
        applySelectedPipelineProfiling();
    }

    updateDeleteButtonVisibility();
}

async function loadPipelineIntoBuilder(pipelineName) {
    try {
        const operations = getOperations();
        if (operations.length === 0) {
            console.warn("Operations not loaded yet, cannot load pipeline");
            return;
        }

        const pipelineConfig = await fetchPipelineConfig(pipelineName);

        const allConnections = [];
        pipelineConfig.forEach((configItem) => {
            if (
                configItem.connections &&
                Array.isArray(configItem.connections)
            ) {
                allConnections.push(...configItem.connections);
            }
        });

        pipelineStore.loadPipelineData(pipelineConfig, allConnections);

        await renderCurrentPipeline();

        updateRunButton();
        updatePipelineCameraNote();
    } catch (error) {
        showDanger("Failed to load pipeline");
        console.error("Failed to load pipeline:", error);
    }
}

/**
 * Re-applies error highlights, thread badges, profiling UI, and camera copy after
 * a local graph add/remove (no full `renderPipeline`).
 */
async function postFlowchartStructureRefresh() {
    applyPipelineErrorHighlights();
    await fetchAndUpdateThreadInfo();
    applySelectedPipelineProfiling();
    updatePipelineCameraNote();
}

async function renderCurrentPipeline() {
    if (!flowchartRenderer) {
        return;
    }

    const pipeline = getPipeline();
    const connections = pipelineStore.getConnectionsForRenderer();
    const options = { connections, centerView: true };
    await flowchartRenderer.renderPipeline(pipeline, options);
    await postFlowchartStructureRefresh();
}

function extractMissingArgumentNames(message) {
    if (!message) {
        return null;
    }

    const match = message.match(
        /missing\s+\d+\s+required positional arguments?:\s*(.+)$/i,
    );

    if (!match) {
        return null;
    }

    const rawList = match[1].trim();
    const quotedMatches = Array.from(rawList.matchAll(/'([^']+)'/g)).map(
        (item) => item[1],
    );

    if (quotedMatches.length > 0) {
        return quotedMatches;
    }

    const normalized = rawList
        .replaceAll(/\band\b/gi, ",")
        .replaceAll(/\s+/g, " ")
        .trim();
    const parts = normalized
        .split(",")
        .map((item) => item.trim())
        .filter((item) => item.length > 0);

    return parts.length > 0 ? parts : null;
}

function buildPipelineErrorPopupContent(errorRecord) {
    const message = errorRecord?.message || "Unknown error";
    const missingArgs = extractMissingArgumentNames(message);
    const displayMessage = missingArgs
        ? `Please fill out the following settings fields in this operation: ${missingArgs.join(", ")}`
        : message;
    const count = errorRecord?.count || 1;
    const name = errorRecord?.name || "Operation Error";

    return `
        <div class="text-red-200 font-semibold text-sm mb-2 border-b border-[#3a1d1d] pb-2">${escapeHtml(
            name,
        )}</div>
        <div class="text-red-100 text-xs whitespace-pre-wrap" style="word-break: break-word; overflow-wrap: anywhere;">${escapeHtml(
            displayMessage,
        )}</div>
        <div class="text-red-300 text-xs mt-2">Seen ${count} time${
            count === 1 ? "" : "s"
        }</div>
    `;
}

function ensurePipelineErrorPopup() {
    if (pipelineErrorPopup) {
        return pipelineErrorPopup;
    }

    pipelineErrorPopup = document.createElement("div");
    pipelineErrorPopup.id = "pipeline-error-popup";
    pipelineErrorPopup.className =
        "fixed z-50 bg-[#2b1f1f] border-2 border-[#ff5c5c] rounded-lg p-3 shadow-lg max-w-sm pointer-events-none opacity-0 transition-opacity duration-200";
    pipelineErrorPopup.style.fontSize = "0.875rem";
    pipelineErrorPopup.style.lineHeight = "1.25rem";
    pipelineErrorPopup.style.width = "max-content";
    pipelineErrorPopup.style.maxWidth = "320px";
    pipelineErrorPopup.style.height = "auto";
    pipelineErrorPopup.style.boxShadow =
        "4px 4px 12px rgba(0,0,0,0.45), 8px 8px 20px rgba(0,0,0,0.25), 2px 2px 6px rgba(255,92,92,0.15)";
    document.body.appendChild(pipelineErrorPopup);

    return pipelineErrorPopup;
}

function positionPipelineErrorPopup(popup, anchorX, anchorY) {
    const margin = 12;
    const offset = 12;

    popup.style.left = `${anchorX + offset}px`;
    popup.style.top = `${anchorY + offset}px`;

    const rect = popup.getBoundingClientRect();
    const maxLeft = window.innerWidth - rect.width - margin;
    const maxTop = window.innerHeight - rect.height - margin;

    const clampedLeft = Math.min(
        Math.max(anchorX + offset, margin),
        Math.max(maxLeft, margin),
    );
    const clampedTop = Math.min(
        Math.max(anchorY + offset, margin),
        Math.max(maxTop, margin),
    );

    popup.style.left = `${clampedLeft}px`;
    popup.style.top = `${clampedTop}px`;
}

function showPipelineErrorPopup(errorRecord, event) {
    const popup = ensurePipelineErrorPopup();
    popup.innerHTML = buildPipelineErrorPopupContent(errorRecord);

    positionPipelineErrorPopup(popup, event.clientX, event.clientY);
    popup.classList.remove("opacity-0");
    popup.classList.add("opacity-100");
}

function hidePipelineErrorPopup() {
    if (!pipelineErrorPopup) {
        return;
    }
    pipelineErrorPopup.classList.remove("opacity-100");
    pipelineErrorPopup.classList.add("opacity-0");
}

function computeDownstreamErrorUuids() {
    downstreamErrorUuids.clear();

    const errorUuids = new Set(operationErrorsByUuid.keys());
    if (errorUuids.size === 0) {
        return;
    }

    const connections = pipelineStore.getConnections();
    const outgoing = new Map();
    for (const connection of connections) {
        if (!outgoing.has(connection.fromUuid)) {
            outgoing.set(connection.fromUuid, []);
        }
        outgoing.get(connection.fromUuid).push(connection.toUuid);
    }

    const queue = Array.from(errorUuids);
    const visited = new Set(errorUuids);
    while (queue.length > 0) {
        const current = queue.shift();
        const nextNodes = outgoing.get(current) || [];
        for (const next of nextNodes) {
            if (visited.has(next)) {
                continue;
            }
            visited.add(next);
            downstreamErrorUuids.add(next);
            queue.push(next);
        }
    }
}

function applyPipelineErrorHighlights() {
    computeDownstreamErrorUuids();
    if (!flowchartRenderer) {
        return;
    }
    for (const node of flowchartRenderer.nodes.values()) {
        const uuid = pipelineStore.instanceIdToUuid.get(node.instanceId);
        const errorRecord = uuid ? operationErrorsByUuid.get(uuid) : null;
        const isDownstream = uuid ? downstreamErrorUuids.has(uuid) : false;
        if (node.setErrorState) {
            node.setErrorState(errorRecord, isDownstream);
            applyFlowchartNodeErrorIcon(node, errorRecord);
        } else {
            applyFlowchartNodeErrorFallback(node, errorRecord, isDownstream);
        }
    }
}

function applyFlowchartNodeErrorIcon(node, errorRecord) {
    const element = node.element;
    if (!element) {
        return;
    }

    const icon = element.querySelector(".node-error-icon");
    if (!icon) {
        return;
    }

    if (errorRecord) {
        icon.style.display = "inline-flex";
        if (!icon.dataset.pipelineErrorBound) {
            icon.dataset.pipelineErrorBound = "true";
            icon.addEventListener("mouseenter", (event) => {
                const uuid = pipelineStore.instanceIdToUuid.get(
                    node.instanceId,
                );
                const currentError = uuid
                    ? operationErrorsByUuid.get(uuid)
                    : null;
                if (currentError) {
                    showPipelineErrorPopup(currentError, event);
                }
            });
            icon.addEventListener("mousemove", (event) => {
                if (pipelineErrorPopup?.classList.contains("opacity-100")) {
                    positionPipelineErrorPopup(
                        pipelineErrorPopup,
                        event.clientX,
                        event.clientY,
                    );
                }
            });
            icon.addEventListener("mouseleave", () => {
                hidePipelineErrorPopup();
            });
        }
    } else {
        icon.style.display = "none";
    }
}

function applyFlowchartNodeErrorFallback(node, errorRecord, isDownstream) {
    const element = node.element;
    if (!element) {
        return;
    }

    if (errorRecord) {
        element.style.borderColor = "#ff5c5c";
        element.style.boxShadow =
            "0 0 0 2px rgba(255,92,92,0.35), 4px 4px 12px rgba(0, 0, 0, 0.5)";
    } else if (!node.isDragging) {
        element.style.borderColor = "#404040";
        element.style.boxShadow = "4px 4px 12px rgba(0, 0, 0, 0.5)";
    }

    element.classList.toggle("pipeline-error-node", Boolean(errorRecord));
    element.classList.toggle(
        "pipeline-downstream-disabled",
        Boolean(isDownstream),
    );

    let infoIcon = element.querySelector(".error-info-icon");
    if (errorRecord && !infoIcon) {
        const header = element.querySelector(".node-header");
        if (!header) {
            return;
        }
        infoIcon = document.createElement("div");
        infoIcon.className = "error-info-icon";
        infoIcon.textContent = "i";
        infoIcon.style.width = "18px";
        infoIcon.style.height = "18px";
        infoIcon.style.borderRadius = "50%";
        infoIcon.style.backgroundColor = "#ff5c5c";
        infoIcon.style.color = "#1a1a1a";
        infoIcon.style.fontSize = "12px";
        infoIcon.style.fontWeight = "700";
        infoIcon.style.display = "inline-flex";
        infoIcon.style.alignItems = "center";
        infoIcon.style.justifyContent = "center";
        infoIcon.style.marginLeft = "8px";
        infoIcon.style.cursor = "default";

        infoIcon.addEventListener("mouseenter", (event) => {
            showPipelineErrorPopup(errorRecord, event);
        });
        infoIcon.addEventListener("mousemove", (event) => {
            if (pipelineErrorPopup?.classList.contains("opacity-100")) {
                positionPipelineErrorPopup(
                    pipelineErrorPopup,
                    event.clientX,
                    event.clientY,
                );
            }
        });
        infoIcon.addEventListener("mouseleave", () => {
            hidePipelineErrorPopup();
        });

        header.appendChild(infoIcon);
    } else if (!errorRecord && infoIcon) {
        infoIcon.remove();
    }
}

function handleOperationErrorUpdate(payload) {
    if (!payload) {
        return;
    }

    const selectedPipeline = getSelectedPipeline();
    if (payload.pipeline_name && selectedPipeline) {
        if (payload.pipeline_name !== selectedPipeline.name) {
            return;
        }
    }

    operationErrorsByUuid.clear();
    const errors = Array.isArray(payload.errors) ? payload.errors : [];
    errors.forEach((errorRecord) => {
        if (errorRecord?.uuid) {
            operationErrorsByUuid.set(errorRecord.uuid, errorRecord);
        }
    });

    pipelineStore.setOperationErrors(errors);
    applyPipelineErrorHighlights();
}

async function fetchAndUpdateThreadInfo() {
    const selectedPipeline = getSelectedPipeline();

    if (!selectedPipeline || pipelineStore.isRestartRequired()) {
        hideAllThreadBadges();
        return;
    }

    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-pipeline-thread-info/${encodeURIComponent(selectedPipeline.name)}`,
        );

        if (!response.ok) {
            console.warn("Failed to fetch thread info:", response.status);
            hideAllThreadBadges();
            return;
        }

        const data = await response.json();

        if (flowchartRenderer) {
            const nodes = flowchartRenderer.nodes;

            for (const [instanceId, node] of nodes) {
                const uuid = pipelineStore.instanceIdToUuid.get(instanceId);
                if (uuid && data.operations) {
                    const threadInfo = data.operations[uuid];
                    node.updateThreadInfo(threadInfo);
                } else {
                    node.hideThreadBadge();
                }
            }
        }
    } catch (error) {
        console.error("Error fetching thread info:", error);
        hideAllThreadBadges();
    }
}

function hideAllThreadBadges() {
    if (flowchartRenderer) {
        for (const node of flowchartRenderer.nodes.values()) {
            node.hideThreadBadge();
        }
    }
}

function hideAllThreadAndProfilingBadges() {
    hideAllThreadBadges();
    clearProfilingUI();
}

async function checkAndTriggerAutoFill() {
    try {
        const pipelines = getPipelines();

        if (!pipelineSelect?.value) {
            console.log("No pipeline selected, skipping auto-fill");
            return;
        }

        const selectedPipelineName = pipelineSelect.value;

        const pipelineObj = pipelines.find(
            (p) => p.name === selectedPipelineName,
        );

        if (!pipelineObj) {
            console.log("Selected pipeline not found in pipelines list");
            return;
        }

        pipelineStore.setCurrentPipeline(pipelineObj.name);

        console.log("Pipeline pre-selected, triggering auto-fill");
        await loadPipelineIntoBuilder(pipelineObj.name);
    } catch (error) {
        console.error("Error during auto-fill check:", error);
    }
}

async function removeFromPipeline(instanceId) {
    const removedNode = pipelineStore.getNode(instanceId);
    const deviceInputCountBefore = getDeviceInputNodes().length;

    console.log("[PIPELINE] Removing operation from pipeline", {
        removedOperation: removedNode
            ? {
                  id: removedNode.operationId,
                  name: removedNode.name,
                  instanceId: removedNode.instanceId,
              }
            : null,
        pipelineLengthBefore: pipelineStore.getNodes().length,
        timestamp: new Date().toISOString(),
    });

    pipelineStore.removeNode(instanceId);

    if (flowchartRenderer) {
        flowchartRenderer.removeNode(instanceId);
    }

    console.log("[PIPELINE] Pipeline after removal", {
        pipelineLengthAfter: pipelineStore.getNodes().length,
        remainingOperations: pipelineStore.getNodes().map((node) => ({
            id: node.operationId,
            name: node.name,
            instanceId: node.instanceId,
        })),
        timestamp: new Date().toISOString(),
    });

    await postFlowchartStructureRefresh();
    autoSavePipeline();

    const deviceInputCountAfter = getDeviceInputNodes().length;
    if (deviceInputCountBefore > 0 && deviceInputCountAfter === 0) {
        showWarning(
            "No device_input nodes configured; bus_id required for camera input.",
        );
    }

    console.log("Operation removed from pipeline - requiring backend restart");
    await updateRestartIndicator(true);
    pipelineStore.clearRestartRequired();
}

function runPipeline() {
    console.log("Running pipeline:", getPipeline());
    alert("Pipeline run! Check console for details.");
}

function openOperationSettings(opOrItem) {
    const title = `${opOrItem.name || opOrItem.id || "Operation"} Settings`;
    const operationName = opOrItem.name || opOrItem.id;
    const operationId = opOrItem.operationId || opOrItem.id || opOrItem.name;
    const operationUuid = opOrItem.uuid || opOrItem.instanceId;
    const isSecondary = opOrItem.isSecondary || false;
    const initialValues = opOrItem.config || {};

    if (!opOrItem.originalConfig) {
        opOrItem.originalConfig = { ...initialValues };
    }

    const onSave = (values) => {
        console.log("Saved settings for", opOrItem, values);
        const isAutoSaveFlag = values._isAutoSave;
        const requiresRestart = values._requiresRestart;
        console.log("isAutoSave flag:", isAutoSaveFlag);
        console.log("requiresRestart flag:", requiresRestart);

        delete values._isAutoSave;
        delete values._requiresRestart;

        const previousConfig = { ...opOrItem.config };

        // Update the actual node in PipelineStore, not just the copy
        const node = pipelineStore.getNode(opOrItem.instanceId);
        if (node) {
            pipelineStore.updateNodeConfig(opOrItem.instanceId, values);
            node.requiresRestart = requiresRestart || false;
            console.log("Updated node.config:", node.config);
            console.log("Updated node.requiresRestart:", node.requiresRestart);
            updatePipelineCameraNote();
        } else {
            // Fallback to updating the copy if node not found (shouldn't happen)
            opOrItem.config = values;
            opOrItem.requiresRestart = requiresRestart || false;
            console.log("Updated opOrItem.config:", opOrItem.config);
            console.log(
                "Updated opOrItem.requiresRestart:",
                opOrItem.requiresRestart,
            );
            updatePipelineCameraNote();
        }

        console.log("Calling autoSavePipeline...");
        autoSavePipeline();

        const changedParams = [];
        for (const [key, value] of Object.entries(values)) {
            if (JSON.stringify(previousConfig[key]) !== JSON.stringify(value)) {
                changedParams.push({ paramName: key, value: value });
            }
        }

        if (changedParams.length > 0) {
            for (const { paramName, value } of changedParams) {
                checkPipelineRestartRequirements(opOrItem, paramName, value);
            }
        } else {
            checkPipelineRestartRequirements();
        }
    };

    const doOpen = () => {
        try {
            globalThis.SettingsPopup.open({
                title,
                operationName,
                operationId,
                operationUuid,
                isSecondary,
                initialValues,
                onSave,
            });
        } catch (err) {
            console.error("Failed to open SettingsPopup:", err);
        }
    };

    if (!globalThis.FileManagerPopup) {
        console.error("FileManagerPopup not available");
        return;
    }

    if (globalThis.SettingsPopup) {
        doOpen();
        return;
    }

    console.error("SettingsPopup not available");
}

function updateRunButton() {
    if (runButton) {
        runButton.disabled = pipelineStore.getNodes().length === 0;
    }
}

async function autoSavePipelineImpl() {
    const selectedPipeline = getSelectedPipeline();

    if (!selectedPipeline) {
        console.log("No pipeline selected, skipping auto-save");
        return;
    }

    try {
        const pipelineConfig = pipelineStore.exportToConfig();

        const response = await fetch(
            `${BACKEND_BASE_URL}/save-pipeline-config/${encodeURIComponent(selectedPipeline.name)}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(pipelineConfig),
            },
        );
        if (!response.ok)
            throw new Error(`HTTP error! status: ${response.status}`);
        await response.json();
        console.log("Pipeline auto-saved successfully");
    } catch (error) {
        console.error("Failed to auto-save pipeline:", error);
    }
}

async function createNewPipeline() {
    const newPipelineName = prompt("Enter a name for the new pipeline:");
    if (!newPipelineName || newPipelineName.trim() === "") {
        return;
    }

    const pipelineFileName = newPipelineName.trim().replaceAll(/\s+/g, "_");

    const pipelines = getPipelines();
    const existingPipeline = pipelines.find((p) => p.name === pipelineFileName);
    if (existingPipeline) {
        if (
            !confirm(
                `Pipeline "${newPipelineName}" already exists. Do you want to overwrite it?`,
            )
        ) {
            return;
        }
    }

    try {
        pipelineStore.clearPipeline();

        const newPipelineObj = {
            name: pipelineFileName,
            displayName: newPipelineName.trim(),
        };

        const currentPipelines = pipelineStore.state.pipelines;
        const existingIndex = currentPipelines.findIndex(
            (p) => p.name === pipelineFileName,
        );
        if (existingIndex >= 0) {
            currentPipelines[existingIndex] = newPipelineObj;
        } else {
            currentPipelines.push(newPipelineObj);
        }

        pipelineStore.setCurrentPipeline(pipelineFileName);
        populatePipelineDropdown(pipelineFileName);

        setTimeout(() => {
            if (pipelineSelect) {
                pipelineSelect.value = pipelineFileName;
                console.log("Dropdown value set to:", pipelineFileName);
            }
        }, 10);

        // Automatically add device_input operation
        const operations = getOperations();
        const deviceInputOp = operations.find(
            (op) => op.id === "device_input.py",
        );
        if (deviceInputOp) {
            pipelineStore.addNode(
                { id: deviceInputOp.id, config: {} },
                { x: 100, y: 100 },
            );
        }

        console.log(
            "[PIPELINE] Re-rendering pipeline with device_input for new pipeline creation",
            {
                pipelineName: newPipelineName,
                timestamp: new Date().toISOString(),
            },
        );

        await renderCurrentPipeline();
        updateRunButton();
        updateDeleteButtonVisibility();

        // Save the empty pipeline to backend so it persists
        await autoSavePipelineImpl();

        pipelineStore.clearRestartRequired();
        await updateRestartIndicator(false);

        console.log("New pipeline created:", newPipelineName);
        console.log("Pipeline state:", pipelineStore.getNodes());
        console.log("Selected pipeline:", getSelectedPipeline());
        console.log("Pipelines list:", getPipelines());
    } catch (error) {
        console.error("Failed to create new pipeline:", error);
        alert(
            "Failed to create new pipeline. Please check the console for details.",
        );
    }
}

async function deleteCurrentPipeline() {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline) {
        alert("No pipeline selected to delete.");
        return;
    }

    const pipelineToDelete = selectedPipeline;

    const confirmed = confirm(
        `Are you sure you want to delete the pipeline "${pipelineToDelete.displayName}"?\n\nThis action cannot be undone.`,
    );

    if (!confirmed) {
        return;
    }

    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/delete-pipeline/${encodeURIComponent(pipelineToDelete.name)}`,
            {
                method: "DELETE",
                headers: {
                    "Content-Type": "application/json",
                },
            },
        );

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Pipeline deleted from backend:", result);

        const currentPipelines = pipelineStore.state.pipelines;
        const pipelineIndex = currentPipelines.findIndex(
            (p) => p.name === pipelineToDelete.name,
        );

        if (pipelineIndex === -1) {
            console.error("Pipeline not found in pipelines array");
            alert("Failed to delete pipeline. Pipeline not found.");
            return;
        }

        currentPipelines.splice(pipelineIndex, 1);

        console.log("Deleted pipeline:", pipelineToDelete.name);
        console.log("Remaining pipelines:", currentPipelines);

        pipelineStore.clearPipeline();
        pipelineStore.setCurrentPipeline(null);

        populatePipelineDropdown();

        await renderCurrentPipeline();
        updateRunButton();
        updateDeleteButtonVisibility();

        restartRequiredOperations.clear();
        await updateRestartIndicator(false);
    } catch (error) {
        console.error("Failed to delete pipeline:", error);
        alert(
            "Failed to delete pipeline. Please check the console for details.",
        );
    }
}

function updateDeleteButtonVisibility() {
    if (deletePipelineButton) {
        const selectedPipeline = getSelectedPipeline();
        if (selectedPipeline) {
            deletePipelineButton.classList.remove("hidden");
        } else {
            deletePipelineButton.classList.add("hidden");
        }
    }
}

async function updateRestartIndicator(show = false) {
    try {
        await fetch(`${BACKEND_BASE_URL}/set_restart_required`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ required: show }),
        });
        console.log(
            `Backend notified: restart ${show ? "required" : "not required"}`,
        );
    } catch (error) {
        showDanger("Failed to notify backend about restart requirement");
        console.error(
            "Failed to notify backend about restart requirement:",
            error,
        );
    }

    if (!restartIndicator) return;

    const restartMessage =
        restartIndicator.querySelector(".text-red-100") ||
        restartIndicator.querySelector("span");

    if (show) {
        restartIndicator.classList.remove("hidden");
        if (restartMessage)
            restartMessage.textContent = "Backend restart required";
        restartIndicator.classList.add("backend-state-warning");
    } else {
        restartIndicator.classList.add("hidden");
        restartIndicator.classList.remove("backend-state-warning");
    }

    if (show) {
        hideAllThreadAndProfilingBadges();
    }
}

async function handleRestartBackend() {
    try {
        const restartButton = restartIndicator?.querySelector(
            "#restartBackendButton",
        );

        if (restartButton) {
            restartButton.disabled = true;
            restartButton.textContent = "Restarting...";
        }

        try {
            await fetch(`${BACKEND_BASE_URL}/restart-backend`, {
                method: "POST",
            });
        } catch (error) {
            console.warn("Failed to send restart request:", error);
        }

        console.log("Backend restarted successfully");

        restartRequiredOperations.clear();

        globalThis.location.reload();
    } catch (error) {
        console.error("Failed to restart backend:", error);
    }
}

async function checkBackendRestartStatus() {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get_restart_required`,
        );

        if (response.ok) {
            const data = await response.json();
            const restartRequired = data.restart_required || false;

            if (restartRequired) {
                console.log(
                    "Backend indicates restart is required - showing indicator",
                );
                await updateRestartIndicator(true);
            } else {
                console.log("Backend indicates no restart required");
            }
        } else {
            console.warn(
                "Failed to get restart status from backend:",
                response.status,
            );
        }
    } catch (error) {
        console.error("Error checking backend restart status:", error);
    }
}

async function checkPipelineRestartRequirements(
    operationItem = null,
    changedParamName = null,
    changedValue = null,
) {
    const restartIndicatorEl = document.getElementById("restartIndicator");
    if (
        restartIndicatorEl &&
        !restartIndicatorEl.classList.contains("hidden") &&
        restartIndicatorEl.classList.contains("backend-state-warning")
    ) {
        return;
    }

    if (operationItem && changedParamName !== null && changedValue !== null) {
        await checkSpecificParameterRestart(
            operationItem,
            changedParamName,
            changedValue,
        );
    } else if (operationItem) {
        await checkOperationRestartRequirements(operationItem);
    }

    const hasRestartRequirements = restartRequiredOperations.size > 0;
    await updateRestartIndicator(hasRestartRequirements);
}

async function checkSpecificParameterRestart(
    operationItem,
    paramName,
    currentValue,
) {
    try {
        const isSecondary = operationItem.isSecondary || false;
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-operation-config-data/${encodeURIComponent(operationItem.id)}/${isSecondary ? 1 : 0}`,
        );

        if (response.ok) {
            const configData = await response.json();
            const params = configData.parameters || {};
            const paramDef = params[paramName];

            if (paramDef?.restart_for_change) {
                const originalValue = operationItem.originalConfig[paramName];

                const requiresRestart =
                    currentValue !== undefined &&
                    currentValue !== null &&
                    originalValue !== undefined &&
                    originalValue !== null &&
                    JSON.stringify(currentValue) !==
                        JSON.stringify(originalValue);

                const instanceId = operationItem.instanceId;
                if (!restartRequiredOperations.has(instanceId)) {
                    restartRequiredOperations.set(instanceId, new Set());
                }

                const paramSet = restartRequiredOperations.get(instanceId);
                if (requiresRestart) {
                    paramSet.add(paramName);
                    console.log(
                        `Operation ${operationItem.name} parameter ${paramName} requires restart (current: ${JSON.stringify(currentValue)}, original: ${JSON.stringify(originalValue)})`,
                    );
                } else {
                    paramSet.delete(paramName);
                    if (paramSet.size === 0) {
                        restartRequiredOperations.delete(instanceId);
                    }
                }

                operationItem.requiresRestart = paramSet.size > 0;
            }
        }
    } catch (error) {
        console.warn(
            `Failed to check restart requirements for ${operationItem.name} parameter ${paramName}:`,
            error,
        );
    }
}

async function checkOperationRestartRequirements(operationItem) {
    for (const [paramName, value] of Object.entries(
        operationItem.config || {},
    )) {
        await checkSpecificParameterRestart(operationItem, paramName, value);
    }
}

async function refreshPipelineCreator() {
    try {
        console.log(
            "[PIPELINE] Refreshing pipeline creator after reconnection",
        );

        await fetchAvailableOperations();

        const operations = getOperations();
        if (operationsList && operations.length > 0) {
            renderOperations(
                operations,
                operationsList,
                openOperationSettings,
                handleDragStartWithLogging,
            );
        }

        await fetchAvailableCameras();

        await fetchPipelines();
        populatePipelineDropdown();

        const selectedPipeline = getSelectedPipeline();
        if (selectedPipeline) {
            await loadPipelineIntoBuilder(selectedPipeline.name);
        }

        updateDeleteButtonVisibility();

        await checkBackendRestartStatus();
    } catch (error) {
        console.error("[PIPELINE] Error refreshing pipeline creator:", error);
    }
}

async function handleFlowchartPipelineChange(changeEvent) {
    const selectedPipeline = getSelectedPipeline();

    if (!selectedPipeline) {
        const shouldCreate = confirm(
            "You need to create a pipeline before adding operations. Would you like to create a new pipeline now?",
        );
        if (!shouldCreate) {
            return;
        }

        await createNewPipeline();

        if (!getSelectedPipeline()) {
            return;
        }
    }

    if (changeEvent.type === "add") {
        const node = pipelineStore.addNode(
            { id: changeEvent.operationId },
            changeEvent.position,
        );
        if (!node) {
            console.warn(`Operation ${changeEvent.operationId} not found`);
            return;
        }

        if (flowchartRenderer) {
            await flowchartRenderer.addNodeFromStore(node.instanceId);
        }
        autoSavePipeline();
        await updateRestartIndicator(true);
        pipelineStore.clearRestartRequired();
        hideAllThreadBadges();
        await postFlowchartStructureRefresh();
    }
}

function initFlowchartRenderer() {
    flowchartCanvas = document.getElementById("flowchartCanvas");

    if (!flowchartCanvas) {
        console.error("Flowchart canvas not found (#flowchartCanvas)");
        showDanger(
            "Pipeline builder could not start: the flowchart canvas is missing from the page.",
        );
        return;
    }

    flowchartRenderer = new FlowchartRenderer(flowchartCanvas, {
        gridSpacing: 20,
        nodeSpacingX: 300,
        nodeSpacingY: 150,
        openOperationSettings,
        updateRunButton,
        removeFromPipeline,
        onPipelineChange: handleFlowchartPipelineChange,
        autoSavePipeline,
    });

    globalThis.flowchartRenderer = flowchartRenderer;
}
export async function initPipelineCreator() {
    if (isInitialized) return;

    pipelineArea = document.getElementById("pipelineArea");
    pipelinePlaceholder = document.getElementById("pipelinePlaceholder");
    operationsList = document.getElementById("operationsList");
    runButton = document.getElementById("runButton");
    pipelineSelect = document.getElementById("pipelineSelect");
    pipelineCameraNote = document.getElementById("pipelineCameraNote");
    newPipelineButton = document.getElementById("newPipelineButton");
    deletePipelineButton = document.getElementById("deletePipelineButton");
    restartIndicator = document.getElementById("restartIndicator");
    executionTimestepsList = document.getElementById("executionTimestepsList");
    executionSummaryContent = document.getElementById(
        "executionSummaryContent",
    );
    profilingDetailsOverlay = document.getElementById(
        "profilingDetailsOverlay",
    );
    profilingDetailsBackdrop = document.getElementById(
        "profilingDetailsBackdrop",
    );
    profilingDetailsBody = document.getElementById("profilingDetailsBody");
    profilingDetailsTitle = document.getElementById("profilingDetailsTitle");
    profilingDetailsCloseButton = document.getElementById(
        "profilingDetailsCloseButton",
    );
    profilingDetailsInfoButton = document.getElementById(
        "profilingDetailsInfoButton",
    );
    profilingDetailsAverageCheckbox = document.getElementById(
        "profilingDetailsAverageCheckbox",
    );

    if (profilingDetailsBackdrop) {
        profilingDetailsBackdrop.addEventListener("click", () => {
            closeProfilingDetailsPopup();
        });
    }
    if (profilingDetailsCloseButton) {
        profilingDetailsCloseButton.addEventListener("click", () => {
            closeProfilingDetailsPopup();
        });
    }
    if (profilingDetailsInfoButton) {
        profilingDetailsInfoButton.addEventListener("click", () => {
            openProfilingDetailsPopup();
        });
    }
    if (profilingDetailsAverageCheckbox) {
        profilingDetailsAverageCheckbox.addEventListener("change", () => {
            clearProfilingAverageState();
            refreshProfilingDetailsPopupIfVisible(undefined);
        });
    }

    createDescriptionPopup();

    const styleElementId = "pipeline-creator-styles";
    if (!document.getElementById(styleElementId)) {
        const styleEl = document.createElement("style");
        styleEl.id = styleElementId;
        styleEl.textContent = `
.op-settings-btn, .remove-btn { display: none !important; }
#pipelineArea .op-settings-btn, #pipelineArea .remove-btn { display: inline-flex !important; }
.icon-grayscale { filter: grayscale(100%); transition: filter .15s ease-in-out; }
#pipelineArea .group:hover .icon-grayscale, #pipelineArea .group:focus-within .icon-grayscale { filter: none; }
#flowchartCanvas { background-color: #1a1a1a; }
.flowchart-node .node-settings-btn:hover img,
.flowchart-node .node-remove-btn:hover img { filter: none !important; }
.pipeline-error-node { border-color: #ff5c5c !important; }
.pipeline-downstream-disabled { filter: grayscale(100%); opacity: 0.55; }
.pipeline-downstream-disabled .icon-grayscale { filter: grayscale(100%) !important; }
.pipeline-error-icon, .error-info-icon { pointer-events: auto; }
`;
        document.head.appendChild(styleEl);
    }

    initFlowchartRenderer();

    pipelineStore.subscribe(
        "profiling:updated",
        ({ snapshot, pipelineName }) => {
            const selectedPipeline = getSelectedPipeline();
            if (!selectedPipeline || selectedPipeline.name !== pipelineName) {
                return;
            }
            applyProfilingSnapshot(snapshot);
        },
    );

    await fetchAvailableOperations();

    await fetchAvailableCameras();

    if (pipelineSelect) {
        pipelineSelect.addEventListener("change", handlePipelineSelection);
    }

    await fetchPipelines();
    populatePipelineDropdown();

    await checkAndTriggerAutoFill();

    updateDeleteButtonVisibility();

    if (runButton) {
        runButton.addEventListener("click", runPipeline);
    }

    if (newPipelineButton) {
        newPipelineButton.addEventListener("click", createNewPipeline);
    }

    if (deletePipelineButton) {
        deletePipelineButton.addEventListener("click", deleteCurrentPipeline);
    }

    if (restartIndicator) {
        const restartButton = restartIndicator.querySelector(
            "#restartBackendButton",
        );
        if (restartButton) {
            restartButton.addEventListener("click", handleRestartBackend);
        }
    }

    renderOperations(
        getOperations(),
        operationsList,
        openOperationSettings,
        handleDragStartWithLogging,
    );

    // Initialize globalThis.pipelineCreator BEFORE rendering so it's available during placeholder checks
    globalThis.pipelineCreator = {
        autoSavePipeline: autoSavePipeline,
        updateRestartIndicator: updateRestartIndicator,
        checkPipelineRestartRequirements: checkPipelineRestartRequirements,
        checkBackendRestartStatus: checkBackendRestartStatus,
        restartIndicator: restartIndicator,
        refreshPipelineCreator: refreshPipelineCreator,
        flowchartRenderer: flowchartRenderer,
        selectedPipeline: null,
        getAvailableCameras: () => pipelineStore.state.cameras,
        refreshAvailableCameras: () => fetchAvailableCameras(),
        handleOperationErrorUpdate: handleOperationErrorUpdate,
        handleProfilingUpdate: handleProfilingUpdate,
        getOperations: () => pipelineStore.state.operations,
    };

    Object.defineProperty(globalThis.pipelineCreator, "selectedPipeline", {
        get: () => getSelectedPipeline(),
        enumerable: true,
    });

    await renderCurrentPipeline();

    if (!profilingStaleIntervalId) {
        profilingStaleIntervalId = setInterval(
            checkAndClearStaleProfiling,
            500,
        );
    }

    await checkBackendRestartStatus();

    isInitialized = true;

    if (globalThis.showBackendRestartIndicator) {
        globalThis.showBackendRestartIndicator();
    }
}
