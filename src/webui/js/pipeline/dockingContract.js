// Shared resolution for metadata-driven docking contracts.
const MX3_DOCKING_CONTRACT = Object.freeze({
    source_action: "device_input",
    source_port: "frame",
    target_port: "frame",
});

/**
 * Resolve a complete docking contract, including the legacy MX3 fallback.
 *
 * @param {string} operationId Target operation identifier.
 * @param {object|null} docking Docking metadata from the operation config.
 * @param {(id:string) => string} normalizeOperationId Identifier normalizer.
 * @returns {{source_action:string,source_port:string,target_port:string}|null}
 */
export function resolveDockingContract(
    operationId,
    docking,
    normalizeOperationId,
) {
    if (
        docking?.source_action &&
        docking?.source_port &&
        docking?.target_port
    ) {
        return docking;
    }

    return normalizeOperationId(operationId) === "mx3_async_object_detection"
        ? MX3_DOCKING_CONTRACT
        : null;
}
