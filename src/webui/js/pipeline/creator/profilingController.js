// Profiling controller: manages profiling snapshot rendering, overlays, and UI state.
import { escapeHtml } from "../utils.js";
import { pipelineStore } from "../PipelineStore.js";
import { creatorContext } from "./context.js";
import { getSelectedPipeline } from "./stateHelpers.js";

const PROFILING_STALE_TIMEOUT_MS = 2000;
const PROFILING_DETAILS_REFRESH_MS = 250;

let profilingAverageAccumulator = null;
let profilingStaleIntervalId = null;
let profilingUiFrameId = null;
let pendingProfilingSnapshot = null;
let profilingDetailsRefreshTimerId = null;
let pendingProfilingDetailsSnapshot = undefined;

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

/**
 * Hides all profiling badges currently shown on flowchart nodes.
 */
function hideAllProfilingBadges() {
    const flowchartRenderer = creatorContext.flowchartRenderer;
    if (!flowchartRenderer) {
        return;
    }
    for (const node of flowchartRenderer.nodes.values()) {
        node.hideProfilingBadge?.();
    }
}

/**
 * Returns the color assigned to a profiling thread number.
 *
 * @param {number|string} threadNumber
 * @returns {string}
 */
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
 * Collects all operations for a timestep and summarizes thread participation and total time.
 *
 * @param {Object<string, Object>} operationsByUuid
 * @param {number} timestep
 * @returns {{threadsSorted: number[], sumMs: number}}
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
 * Builds the SVG badge used to visualize timestep thread participation.
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
        const tip = escapeHtml(`Timestep ${timestep} · thread ${threadsSorted[0]}`);
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
        const tip = escapeHtml(`Timestep ${timestep} · thread ${threadsSorted[i]} (${n} threads, equal split)`);
        paths.push(`<path d="${d}" fill="${fill}"><title>${tip}</title></path>`);
        angle = endAngle;
    }
    const mask = `<circle cx="${cx}" cy="${cy}" r="${innerR}" fill="#1f1f1f"/>`;
    const tipAll = escapeHtml(`Timestep ${timestep} · ${n} threads (equal segments)`);
    return `${svgOpen}<g><title>${tipAll}</title>${paths.join("")}</g>${mask}${textEl}</svg>`;
}

/**
 * Renders the timestep breakdown panel in the profiling sidebar.
 *
 * @param {Object|null} snapshot
 */
function renderExecutionTimestepsPanel(snapshot) {
    if (!creatorContext.elements.executionTimestepsList) return;
    const executionTimestepsList = creatorContext.elements.executionTimestepsList;
    const timesteps = Array.isArray(snapshot?.timesteps) ? [...snapshot.timesteps] : [];
    if (timesteps.length === 0) {
        executionTimestepsList.innerHTML = '<div class="text-[#888]">No profiling data</div>';
        return;
    }
    const operationsByUuid = snapshot?.operations || {};
    timesteps.sort((a, b) => (a?.timestep ?? 0) - (b?.timestep ?? 0));
    const badgeSize = 22;
    executionTimestepsList.innerHTML = timesteps
        .map((row) => {
            const timestep = Number(row?.timestep) || 0;
            const { threadsSorted, sumMs } = analyzeTimestepOperations(operationsByUuid, timestep);
            const wallMs = Number(row?.total_time_ms) || 0;
            const displayMs = sumMs > 0 ? sumMs : wallMs;
            const badgeSvg = buildTimestepThreadBadgeSvg(threadsSorted, timestep, badgeSize);
            const titleParts = [`Timestep ${timestep}: Σ ops ${displayMs.toFixed(2)}ms`];
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

/**
 * Renders the compact execution summary block.
 *
 * @param {Object|null} snapshot
 */
function renderExecutionSummary(snapshot) {
    if (!creatorContext.elements.executionSummaryContent) return;
    const executionSummaryContent = creatorContext.elements.executionSummaryContent;
    const frameTimeMs = Number(snapshot?.frame_time_ms);
    if (!Number.isFinite(frameTimeMs) || frameTimeMs <= 0) {
        executionSummaryContent.innerHTML = '<div class="text-[#888]">No profiling data</div>';
        return;
    }
    const cycleTimeMs = Number(snapshot?.cycle_time_ms);
    const fpsIntervalMs = Number.isFinite(cycleTimeMs) && cycleTimeMs > 0
        ? cycleTimeMs
        : frameTimeMs;
    const estimatedFps = 1000 / fpsIntervalMs;
    executionSummaryContent.innerHTML = `<div class="text-[#f1f1f1]">Flow: ${frameTimeMs.toFixed(2)}ms</div>
        <div class="text-[#9ad1a8]" title="Includes time waiting for fresh device input">FPS: ${estimatedFps.toFixed(1)}</div>`;
}

/**
 * Returns a colored swatch HTML snippet for a thread label.
 *
 * @param {number|string} thread
 * @returns {string}
 */
function profilingThreadSwatchHtml(thread) {
    const t = Number(thread);
    const color = getProfilingThreadColor(t);
    const label = !Number.isFinite(t) || t <= 0 ? "—" : `Thread ${t}`;
    return `<span class="inline-flex items-center gap-1.5"><span class="inline-block w-2.5 h-2.5 rounded-full shrink-0 ring-1 ring-white/10" style="background-color:${color}"></span><span class="text-[#ddd]">${label}</span></span>`;
}

/**
 * Clears the cumulative average accumulator.
 */
function clearProfilingAverageState() {
    profilingAverageAccumulator = null;
}

/**
 * Ensures the cumulative average accumulator matches the current pipeline.
 *
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
            cycleWall: { sum: 0, n: 0 },
            operations: new Map(),
            timesteps: new Map(),
        };
    }
}

/**
 * Merges a profiling snapshot into the cumulative average accumulator.
 *
 * @param {Object} snapshot
 * @param {string} pipelineName
 */
function mergeProfilingSnapshotIntoAverage(snapshot, pipelineName) {
    ensureProfilingAverageAccumulator(pipelineName);
    const a = profilingAverageAccumulator;
    if (!a) return;
    const ft = Number(snapshot.frame_time_ms);
    if (Number.isFinite(ft) && ft > 0) {
        a.frameWall.sum += ft;
        a.frameWall.n += 1;
    }
    const ct = Number(snapshot.cycle_time_ms);
    if (Number.isFinite(ct) && ct > 0) {
        a.cycleWall.sum += ct;
        a.cycleWall.n += 1;
    }
    for (const [uuid, row] of Object.entries(snapshot.operations || {})) {
        const ms = Number(row.execution_time_ms);
        if (!Number.isFinite(ms) || ms < 0) continue;
        let st = a.operations.get(uuid);
        if (!st) {
            st = { sumMs: 0, n: 0, name: "", timestep: -1, thread: 0 };
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
        if (!Number.isFinite(t)) continue;
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
        if (Number.isFinite(wall) && wall >= 0) ts.sumWall += wall;
        if (Number.isFinite(mx) && mx >= 0) ts.sumMaxOpTime += mx;
        if (Number.isFinite(cnt) && cnt >= 0) ts.sumOpCount += cnt;
        ts.n += 1;
        if (row.max_operation_name) ts.lastMaxName = String(row.max_operation_name);
        a.timesteps.set(t, ts);
    }
    a.snapshotCount += 1;
}

/**
 * Builds a snapshot that represents the current cumulative average view.
 *
 * @param {Object} latest
 * @returns {Object}
 */
function buildProfilingAverageDisplaySnapshot(latest) {
    const a = profilingAverageAccumulator;
    if (!a || a.snapshotCount <= 0) return latest;
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
            const fromLatest = latestTimesteps.find((r) => Number(r?.timestep) === t);
            return {
                timestep: t,
                total_time_ms: ts && ts.n > 0 ? ts.sumWall / ts.n : 0,
                max_operation_time_ms: ts && ts.n > 0 ? ts.sumMaxOpTime / ts.n : 0,
                operation_count: ts && ts.n > 0 ? ts.sumOpCount / ts.n : 0,
                max_operation_name: (ts && ts.lastMaxName) || fromLatest?.max_operation_name,
                max_operation_uuid: fromLatest?.max_operation_uuid ?? null,
            };
        });
    const frameMs = a.frameWall.n > 0 ? a.frameWall.sum / a.frameWall.n : Number(latest.frame_time_ms);
    const cycleMs = a.cycleWall.n > 0 ? a.cycleWall.sum / a.cycleWall.n : Number(latest.cycle_time_ms);
    return {
        ...latest,
        frame_time_ms: Number.isFinite(frameMs) ? frameMs : latest.frame_time_ms,
        cycle_time_ms: Number.isFinite(cycleMs) ? cycleMs : latest.cycle_time_ms,
        operations,
        timesteps,
    };
}

/**
 * Builds the HTML content for the profiling details popup.
 *
 * @param {Object|null} snapshot
 * @param {string} pipelineName
 * @param {{cumulativeAverage?: boolean, mergeCount?: number}} [hint={}]
 * @returns {string}
 */
function buildProfilingDetailsHtml(snapshot, pipelineName, hint = {}) {
    if (!snapshot) {
        return `<p class="text-[#888] text-sm leading-relaxed">No profiling snapshot is stored yet for <span class="text-[#f9c845]">${escapeHtml(pipelineName)}</span>. Run the pipeline and wait for updates over SSE.</p>`;
    }
    const ops = snapshot.operations || {};
    const timestepRows = [...(snapshot.timesteps || [])].sort((a, b) => (a?.timestep ?? 0) - (b?.timestep ?? 0));
    const frameMs = Number(snapshot.frame_time_ms);
    const cycleMs = Number(snapshot.cycle_time_ms);
    const fpsIntervalMs = Number.isFinite(cycleMs) && cycleMs > 0 ? cycleMs : frameMs;
    const fps = Number.isFinite(fpsIntervalMs) && fpsIntervalMs > 0 ? (1000 / fpsIntervalMs).toFixed(1) : "—";
    const seq = snapshot.frame_seq != null ? String(snapshot.frame_seq) : "—";
    const tsMs = Number(snapshot.timestamp_ms);
    const tsLabel = Number.isFinite(tsMs) ? new Date(tsMs).toLocaleString() : "—";

    const parts = [
        `<div class="rounded-lg border border-[#3a3a3a] bg-[#181818] p-4 mb-5">`,
        `<div class="text-xs uppercase tracking-wide text-[#888] mb-2">Current snapshot</div>`,
    ];
    if (hint.cumulativeAverage && (hint.mergeCount ?? 0) > 0) {
        parts.push(`<p class="text-[#9ad1a8] text-xs mb-2 leading-relaxed">Cumulative arithmetic mean of each numeric field over <span class="font-semibold">${hint.mergeCount}</span> profiling update(s) since this mode was enabled (converges toward stable values as samples grow; not a rolling window).</p>`);
    }
    parts.push(
        `<div class="grid gap-2 text-sm">`,
        `<div><span class="text-[#888]">Pipeline</span> · <span class="text-[#f1f1f1] font-medium">${escapeHtml(pipelineName)}</span></div>`,
        `<div><span class="text-[#888]">Frame wall time</span> · <span class="text-[#f1f1f1]">${Number.isFinite(frameMs) && frameMs > 0 ? `${frameMs.toFixed(2)} ms` : "—"}</span></div>`,
        `<div><span class="text-[#888]">Pipeline cycle</span> · <span class="text-[#f1f1f1]">${Number.isFinite(cycleMs) && cycleMs > 0 ? `${cycleMs.toFixed(2)} ms` : "—"}</span>` + (fps !== "—" ? ` <span class="text-[#9ad1a8]">(~${fps} FPS)</span>` : "") + ` <span class="text-[#777] text-xs">includes fresh-input wait</span></div>`,
        `<div><span class="text-[#888]">Frame sequence</span> · <span class="text-[#f1f1f1]">${escapeHtml(seq)}</span></div>`,
        `<div><span class="text-[#888]">Recorded at</span> · <span class="text-[#f1f1f1]">${escapeHtml(tsLabel)}</span></div>`,
        `</div></div>`,
        `<h4 class="text-[#f9c845] text-sm font-semibold mb-2 border-b border-[#414141] pb-1">By timestep</h4>`,
        `<p class="text-[#888] text-xs mb-3 leading-relaxed">Each timestep runs a group of operations (they may overlap on different threads). <span class="text-[#c9c9c9]">Wall</span> is measured around the whole group; <span class="text-[#c9c9c9]">Σ ops</span> is the sum of per-operation execution times for that timestep.</p>`,
    );

    if (timestepRows.length === 0) {
        parts.push(`<p class="text-[#666] text-xs mb-4">No timestep rows in this snapshot.</p>`);
    } else {
        for (const row of timestepRows) {
            const tsN = Number(row?.timestep) || 0;
            const { sumMs } = analyzeTimestepOperations(ops, tsN);
            const wallMs = Number(row?.total_time_ms) || 0;
            const opEntries = Object.entries(ops).filter(([, op]) => Number(op?.timestep) === tsN);
            opEntries.sort((a, b) => {
                const ta = Number(a[1]?.thread) || 0;
                const tb = Number(b[1]?.thread) || 0;
                if (ta !== tb) return ta - tb;
                return String(a[1]?.name || "").localeCompare(String(b[1]?.name || ""));
            });
            const dispSum = sumMs > 0 ? sumMs : wallMs;
            const countFromPayload = Number(row?.operation_count);
            parts.push(`<div class="mb-4 rounded-md border border-[#3a3a3a] overflow-hidden bg-[#1a1a1a]">`, `<div class="px-3 py-2 bg-[#252525] flex flex-wrap gap-x-3 gap-y-1 text-xs">`, `<span class="font-semibold text-[#f9c845]">Timestep ${tsN}</span>`, `<span class="text-[#aaa]">wall <span class="text-[#f1f1f1]">${wallMs.toFixed(2)} ms</span></span>`, `<span class="text-[#aaa]">Σ ops <span class="text-[#f1f1f1]">${dispSum.toFixed(2)} ms</span></span>`);
            if (Number.isFinite(countFromPayload)) {
                parts.push(`<span class="text-[#aaa]">count <span class="text-[#f1f1f1]">${countFromPayload}</span></span>`);
            }
            parts.push(`</div>`);
            if (row?.max_operation_name) {
                parts.push(`<div class="px-3 py-1.5 text-xs text-[#ac8a2f] border-b border-[#333]">Heaviest op this frame: <span class="text-[#e8e8e8]">${escapeHtml(String(row.max_operation_name))}</span> (${Number(row?.max_operation_time_ms || 0).toFixed(2)} ms)</div>`);
            }
            parts.push(`<table class="w-full text-xs border-collapse"><thead><tr class="bg-[#222] text-left text-[#c9c9c9]"><th class="px-3 py-1.5 font-medium">Operation</th><th class="px-3 py-1.5 font-medium w-32">Thread</th><th class="px-3 py-1.5 font-medium w-24 text-right">Time (ms)</th></tr></thead><tbody>`);
            if (!opEntries.length) {
                parts.push(`<tr><td colspan="3" class="px-3 py-2 text-[#666]">No operation rows for this timestep.</td></tr>`);
            } else {
                for (const [uuid, op] of opEntries) {
                    const name = escapeHtml(String(op?.name ?? uuid.slice(0, 8)));
                    const th = Number(op?.thread);
                    const ms = Number(op?.execution_time_ms);
                    const msStr = Number.isFinite(ms) ? ms.toFixed(2) : "—";
                    parts.push(`<tr class="border-t border-[#2f2f2f]"><td class="px-3 py-1.5 text-[#e4e4e4]">${name}</td><td class="px-3 py-1.5">${profilingThreadSwatchHtml(th)}</td><td class="px-3 py-1.5 text-right text-[#e4e4e4] font-mono">${msStr}</td></tr>`);
                }
            }
            parts.push(`</tbody></table></div>`);
        }
    }

    parts.push(`<h4 class="text-[#f9c845] text-sm font-semibold mt-6 mb-2 border-b border-[#414141] pb-1">By thread</h4>`, `<p class="text-[#888] text-xs mb-3 leading-relaxed">All operations in this snapshot, grouped by worker thread. Each row is one operation at a timestep with its measured execution time.</p>`);

    const byThread = new Map();
    for (const [uuid, op] of Object.entries(ops)) {
        let th = Number(op?.thread);
        if (!Number.isFinite(th)) th = 0;
        const step = Number(op?.timestep);
        const stepLabel = Number.isFinite(step) && step >= 0 ? String(step) : "—";
        const arr = byThread.get(th) || [];
        arr.push({ timestep: stepLabel, name: String(op?.name ?? "—"), ms: Number(op?.execution_time_ms), uuid });
        byThread.set(th, arr);
    }
    const threads = [...byThread.keys()].sort((a, b) => a - b);
    if (!threads.length) {
        parts.push(`<p class="text-[#666] text-xs">No operations in this snapshot.</p>`);
    } else {
        for (const th of threads) {
            const rows = byThread.get(th) || [];
            rows.sort((a, b) => {
                const cmp = String(a.timestep).localeCompare(String(b.timestep), undefined, { numeric: true });
                if (cmp !== 0) return cmp;
                return a.name.localeCompare(b.name);
            });
            const subtotal = rows.reduce((s, r) => {
                const m = Number(r.ms);
                return s + (Number.isFinite(m) && m > 0 ? m : 0);
            }, 0);
            parts.push(`<div class="mb-4 rounded-md border border-[#3a3a3a] overflow-hidden bg-[#1a1a1a]">`, `<div class="px-3 py-2 bg-[#252525] text-xs font-semibold text-[#e8e8e8] flex items-center gap-2">${profilingThreadSwatchHtml(th)}<span class="text-[#888] font-normal">Σ ${subtotal.toFixed(2)} ms</span></div>`, `<table class="w-full text-xs border-collapse"><thead><tr class="bg-[#222] text-left text-[#c9c9c9]"><th class="px-3 py-1.5 font-medium w-20">Timestep</th><th class="px-3 py-1.5 font-medium">Operation</th><th class="px-3 py-1.5 font-medium w-24 text-right">Time (ms)</th></tr></thead><tbody>`);
            for (const r of rows) {
                const ms = Number(r.ms);
                const msStr = Number.isFinite(ms) ? ms.toFixed(2) : "—";
                parts.push(`<tr class="border-t border-[#2f2f2f]"><td class="px-3 py-1.5 text-[#c9c9c9] font-mono">${escapeHtml(String(r.timestep))}</td><td class="px-3 py-1.5 text-[#e4e4e4]">${escapeHtml(r.name)}</td><td class="px-3 py-1.5 text-right font-mono text-[#e4e4e4]">${msStr}</td></tr>`);
            }
            parts.push(`</tbody></table></div>`);
        }

    }

    return parts.join("");
}

/**
 * Refreshes the profiling details popup if it is currently visible.
 *
 * @param {Object|null|undefined} snapshot
 */
function refreshProfilingDetailsPopupIfVisible(snapshot) {
    if (!creatorContext.elements.profilingDetailsOverlay || creatorContext.elements.profilingDetailsOverlay.classList.contains("hidden")) {
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
    if (!creatorContext.elements.profilingDetailsBody || !creatorContext.elements.profilingDetailsTitle) {
        return;
    }
    const useAvg = Boolean(creatorContext.elements.profilingDetailsAverageCheckbox?.checked);
    let displaySnapshot = snapshotForView;
    const hint = { cumulativeAverage: false, mergeCount: 0 };
    if (useAvg && snapshotForView && selected?.name) {
        mergeProfilingSnapshotIntoAverage(snapshotForView, selected.name);
        displaySnapshot = buildProfilingAverageDisplaySnapshot(snapshotForView);
        hint.cumulativeAverage = true;
        hint.mergeCount = profilingAverageAccumulator?.snapshotCount ?? 0;
    }
    creatorContext.elements.profilingDetailsBody.innerHTML = buildProfilingDetailsHtml(displaySnapshot, pipelineName, hint);
    creatorContext.elements.profilingDetailsTitle.textContent = selected?.name ? `Profiling — ${selected.name}` : "Profiling details";
}

/**
 * Schedules a throttled refresh of the profiling details popup.
 *
 * @param {Object|null} snapshot
 */
function scheduleProfilingDetailsRefresh(snapshot) {
    pendingProfilingDetailsSnapshot = snapshot;
    if (!creatorContext.elements.profilingDetailsOverlay || creatorContext.elements.profilingDetailsOverlay.classList.contains("hidden")) {
        return;
    }
    if (profilingDetailsRefreshTimerId) return;
    profilingDetailsRefreshTimerId = setTimeout(() => {
        profilingDetailsRefreshTimerId = null;
        const latestSnapshot = pendingProfilingDetailsSnapshot;
        pendingProfilingDetailsSnapshot = undefined;
        refreshProfilingDetailsPopupIfVisible(latestSnapshot);
    }, PROFILING_DETAILS_REFRESH_MS);
}

/**
 * Handles keydown events for the profiling details popup.
 *
 * @param {KeyboardEvent} event
 */
function profilingDetailsOnKeydown(event) {
    if (event.key === "Escape") {
        event.preventDefault();
        closeProfilingDetailsPopup();
    }
}

/**
 * Closes the profiling details popup.
 */
function closeProfilingDetailsPopup() {
    if (!creatorContext.elements.profilingDetailsOverlay || creatorContext.elements.profilingDetailsOverlay.classList.contains("hidden")) {
        return;
    }
    creatorContext.elements.profilingDetailsOverlay.classList.add("hidden");
    creatorContext.elements.profilingDetailsOverlay.setAttribute("aria-hidden", "true");
    creatorContext.elements.profilingDetailsModal?.removeAttribute("open");
    document.removeEventListener("keydown", profilingDetailsOnKeydown, true);
}

/**
 * Opens the profiling details popup for the selected pipeline.
 */
function openProfilingDetailsPopup() {
    const selected = getSelectedPipeline();
    if (!selected?.name) {
        showWarning("Select a pipeline first.");
        return;
    }
    if (!creatorContext.elements.profilingDetailsOverlay) return;
    document.removeEventListener("keydown", profilingDetailsOnKeydown, true);
    creatorContext.elements.profilingDetailsOverlay.classList.remove("hidden");
    creatorContext.elements.profilingDetailsOverlay.setAttribute("aria-hidden", "false");
    creatorContext.elements.profilingDetailsModal?.setAttribute("open", "");
    document.addEventListener("keydown", profilingDetailsOnKeydown, true);
    refreshProfilingDetailsPopupIfVisible(undefined);
}

/**
 * Clears all profiling UI regions.
 */
function clearProfilingUI() {
    hideAllProfilingBadges();
    renderExecutionTimestepsPanel(null);
    renderExecutionSummary(null);
    refreshProfilingDetailsPopupIfVisible(null);
}

/**
 * Applies a profiling snapshot to the current UI.
 *
 * @param {Object|null} snapshot
 */
function applyProfilingSnapshot(snapshot) {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline || !snapshot) {
        clearProfilingUI();
        return;
    }
    if (snapshot.pipeline_name !== selectedPipeline.name) {
        return;
    }
    const flowchartRenderer = creatorContext.flowchartRenderer;
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
    scheduleProfilingDetailsRefresh(snapshot);
}

/**
 * Queues an animation-frame update for applying a profiling snapshot.
 *
 * @param {Object|null} snapshot
 */
function scheduleProfilingUiApply(snapshot) {
    pendingProfilingSnapshot = snapshot;
    if (profilingUiFrameId) return;
    profilingUiFrameId = requestAnimationFrame(() => {
        profilingUiFrameId = null;
        const latestSnapshot = pendingProfilingSnapshot;
        pendingProfilingSnapshot = null;
        applyProfilingSnapshot(latestSnapshot);
    });
}

/**
 * Applies the current selected pipeline's stored profiling snapshot, if any.
 */
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

/**
 * Stores an incoming profiling update payload in the pipeline store.
 *
 * @param {Object} payload
 */
function handleProfilingUpdate(payload) {
    if (!payload || typeof payload.pipeline_name !== "string") {
        return;
    }
    pipelineStore.setProfilingSnapshot(payload);
}

/**
 * Clears stale profiling UI for the selected pipeline if updates have timed out.
 */
function checkAndClearStaleProfiling() {
    const selectedPipeline = getSelectedPipeline();
    if (!selectedPipeline) {
        clearProfilingUI();
        return;
    }
    const lastUpdateMs = pipelineStore.getProfilingLastUpdateMs(selectedPipeline.name);
    if (lastUpdateMs <= 0) {
        clearProfilingUI();
        return;
    }
    if (Date.now() - lastUpdateMs > PROFILING_STALE_TIMEOUT_MS) {
        clearProfilingUI();
    }
}

export {
    PROFILING_STALE_TIMEOUT_MS,
    PROFILING_DETAILS_REFRESH_MS,
    clearProfilingAverageState,
    hideAllProfilingBadges,
    applyProfilingSnapshot,
    applySelectedPipelineProfiling,
    buildProfilingDetailsHtml,
    buildProfilingAverageDisplaySnapshot,
    buildTimestepThreadBadgeSvg,
    checkAndClearStaleProfiling,
    clearProfilingUI,
    closeProfilingDetailsPopup,
    handleProfilingUpdate,
    getProfilingThreadColor,
    mergeProfilingSnapshotIntoAverage,
    openProfilingDetailsPopup,
    refreshProfilingDetailsPopupIfVisible,
    renderExecutionSummary,
    renderExecutionTimestepsPanel,
    scheduleProfilingDetailsRefresh,
    scheduleProfilingUiApply,
    profilingThreadSwatchHtml,
};
