const CAMERA_MODES = new Set(["localize", "detect", "both"]);

/**
 * Add or replace one camera setup while preserving camera order.
 *
 * @param {Array<object>} setups - Current per-camera setup records.
 * @param {object} nextSetup - Completed setup for one camera.
 * @returns {Array<object>} Updated setup records.
 */
export function upsertCameraSetup(setups, nextSetup) {
    const index = setups.findIndex(
        (setup) => setup.bus_id === nextSetup.bus_id,
    );
    if (index < 0) return [...setups, nextSetup];
    return setups.map((setup, currentIndex) =>
        currentIndex === index ? nextSetup : setup,
    );
}

/**
 * Build and validate the backend generation payload.
 *
 * @param {Array<object>} setups - Completed per-camera setup records.
 * @param {string} networkTableAddress - NetworkTables server address.
 * @returns {{network_table_address: string, cameras: Array<object>}} Request body.
 */
export function buildGenerationPayload(setups, networkTableAddress) {
    const address = networkTableAddress.trim();
    if (!address) throw new Error("NetworkTables address is required.");
    if (!setups.length) throw new Error("Configure at least one camera.");

    const seen = new Set();
    const cameras = setups.map((setup) => {
        if (!setup.bus_id || seen.has(setup.bus_id)) {
            throw new Error("Each camera can be configured once.");
        }
        if (!CAMERA_MODES.has(setup.mode)) {
            throw new Error(`Unsupported camera mode: ${setup.mode}`);
        }
        seen.add(setup.bus_id);
        return {
            bus_id: setup.bus_id,
            mode: setup.mode,
            model_id: setup.mode === "localize" ? "" : setup.model_id || "",
        };
    });
    return { network_table_address: address, cameras };
}
