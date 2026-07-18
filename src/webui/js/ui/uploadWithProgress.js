import { BACKEND_BASE_URL } from "../config.js";

const DEFAULT_UPLOAD_TIMEOUT_MS = 10 * 60 * 1000;

/**
 * Uploads FormData with XMLHttpRequest so upload progress can be reported.
 *
 * @param {{url: string, formData: FormData, onProgress?: (percent: number) => void, timeoutMs?: number}} options
 * @returns {Promise<any>}
 */
export function uploadWithProgress({
    url,
    formData,
    onProgress,
    timeoutMs = DEFAULT_UPLOAD_TIMEOUT_MS,
}) {
    const requestUrl = url.startsWith("http") ? url : `${BACKEND_BASE_URL}${url}`;

    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", requestUrl);
        xhr.timeout = timeoutMs;

        xhr.upload.addEventListener("progress", (event) => {
            if (!event.lengthComputable || typeof onProgress !== "function") {
                return;
            }
            const percent = Math.round((event.loaded / event.total) * 100);
            onProgress(percent);
        });

        xhr.addEventListener("load", () => {
            let payload = {};
            try {
                payload = xhr.responseText ? JSON.parse(xhr.responseText) : {};
            } catch {
                payload = {};
            }

            if (xhr.status >= 200 && xhr.status < 300) {
                if (typeof onProgress === "function") {
                    onProgress(100);
                }
                resolve(payload);
                return;
            }

            const error = new Error(
                payload.error || payload.message || `Request failed: ${xhr.status}`,
            );
            error.payload = payload;
            error.status = xhr.status;
            reject(error);
        });

        xhr.addEventListener("error", () => {
            const error = new Error("Network error during upload");
            error.payload = {};
            error.status = 0;
            reject(error);
        });

        xhr.addEventListener("abort", () => {
            const error = new Error("Upload aborted");
            error.payload = {};
            error.status = 0;
            reject(error);
        });

        xhr.addEventListener("timeout", () => {
            const error = new Error("Upload timed out");
            error.payload = { error: "Upload timed out" };
            error.status = 0;
            reject(error);
        });

        xhr.send(formData);
    });
}
