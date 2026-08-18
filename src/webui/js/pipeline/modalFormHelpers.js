/** Shared form-control styling for pipeline modal dialogs. */
export const INPUT_CLASS =
    "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]";

/**
 * Returns a useful error message from an HTTP response without assuming its
 * response body is JSON or has a particular schema.
 * @param {Response} response Failed HTTP response.
 * @returns {Promise<string>} Backend error message.
 */
export async function responseError(response) {
    try {
        const data = await response.json();
        if (data && typeof data === "object") {
            return (
                data.error ||
                data.detail ||
                data.message ||
                JSON.stringify(data)
            );
        }
        return String(data);
    } catch (_) {
        return response.statusText || `HTTP ${response.status}`;
    }
}
