import { buildBackendUrl } from "../config.js";

/**
 * Handles terminal controls and log display interactions in the settings UI.
 */
let logsLoaded = false;
let terminalPromptText = "$";
let terminalCommandHistory = [];
let terminalHistoryIndex = -1;
let terminalDraftCommand = "";
let terminalCommandInFlight = false;
let terminalInitialized = false;

/**
 * Wire up terminal and log-related UI event handlers.
 */
export function initializeTerminalHandlers() {
    const terminalPanel = document.getElementById("terminalPanel");
    const clearLogsBtn = document.getElementById("clearLogsBtn");
    const clearTerminalBtn = document.getElementById("clearTerminalBtn");
    const sendCommandBtn = document.getElementById("sendCommandBtn");
    const terminalInput = document.getElementById("terminalInput");
    const logsOutput = document.getElementById("logsOutput");
    const downloadLogsBtn = document.getElementById("downloadLogsBtn");

    if (terminalPanel) {
        loadLogMessages();
        initializeTerminalSession();
    }

    if (clearLogsBtn) {
        clearLogsBtn.addEventListener(
            "click",
            /**
             * Clear the visible logs without affecting backend storage.
             */
            function () {
                if (logsOutput) {
                    logsOutput.innerHTML = "";
                    appendToLogs(
                        "Logs cleared (display only - backend logs preserved)",
                        "INFO",
                    );
                }
            },
        );
    }

    if (clearTerminalBtn) {
        clearTerminalBtn.addEventListener(
            "click",
            /**
             * Reset the terminal display while preserving session state.
             */
            function () {
                clearTerminalDisplay();
            },
        );
    }

    if (downloadLogsBtn) {
        downloadLogsBtn.addEventListener(
            "click",
            /**
             * Trigger the log file download action.
             */
            function () {
                downloadLogFile();
            },
        );
    }

    if (sendCommandBtn) {
        sendCommandBtn.addEventListener("click", sendTerminalCommand);
    }

    if (terminalInput) {
        terminalInput.addEventListener(
            "keydown",
            /**
             * Handle Enter submission and arrow-key command history.
             *
             * @param {KeyboardEvent} event - Keydown event.
             */
            function (event) {
                if (event.key === "Enter") {
                    event.preventDefault();
                    sendTerminalCommand();
                    return;
                }

                if (event.key === "ArrowUp") {
                    event.preventDefault();
                    navigateTerminalHistory(-1);
                    return;
                }

                if (event.key === "ArrowDown") {
                    event.preventDefault();
                    navigateTerminalHistory(1);
                }
            },
        );
    }
}

/**
 * Initialize the terminal session prompt and welcome output.
 *
 * @returns {Promise<void>}
 */
async function initializeTerminalSession() {
    if (terminalInitialized) {
        return;
    }

    terminalInitialized = true;
    await refreshTerminalPrompt();
    clearTerminalDisplay();
}

/**
 * Refresh the terminal prompt text from the backend session.
 *
 * @returns {Promise<void>}
 */
async function refreshTerminalPrompt() {
    try {
        const response = await fetch(buildBackendUrl("/terminal/cwd"));
        const data = await response.json();
        applyTerminalSessionState(data);
    } catch (error) {
        terminalPromptText = "$";
        updateTerminalPromptDisplay();
        appendToTerminal(
            getTerminalOutputElement(),
            `Failed to load terminal session: ${error.message}`,
            "error",
        );
    }
}

/**
 * Apply prompt and cwd metadata returned by the terminal API.
 *
 * @param {{prompt?: string}} data - Terminal session payload.
 */
function applyTerminalSessionState(data) {
    if (data && typeof data.prompt === "string" && data.prompt.trim()) {
        terminalPromptText = data.prompt.trim();
    }
    updateTerminalPromptDisplay();
}

/**
 * Update the visible prompt next to the command input.
 */
function updateTerminalPromptDisplay() {
    const terminalPrompt = document.getElementById("terminalPrompt");
    if (terminalPrompt) {
        terminalPrompt.textContent = terminalPromptText;
    }
}

/**
 * Clear the terminal output pane and show the current prompt.
 */
function clearTerminalDisplay() {
    const terminalOutput = getTerminalOutputElement();
    if (!terminalOutput) {
        return;
    }

    terminalOutput.innerHTML = "";
    appendToTerminal(terminalOutput, terminalPromptText, "prompt");
}

/**
 * Return the terminal output container element.
 *
 * @returns {HTMLElement | null} Terminal output element when present.
 */
function getTerminalOutputElement() {
    return document.getElementById("terminalOutput");
}

/**
 * Navigate through previously executed terminal commands.
 *
 * @param {number} direction - History direction (-1 older, 1 newer).
 */
function navigateTerminalHistory(direction) {
    const terminalInput = document.getElementById("terminalInput");
    if (!terminalInput || terminalCommandHistory.length === 0) {
        return;
    }

    if (terminalHistoryIndex === -1 && direction === -1) {
        terminalDraftCommand = terminalInput.value;
        terminalHistoryIndex = terminalCommandHistory.length - 1;
        terminalInput.value = terminalCommandHistory[terminalHistoryIndex];
        return;
    }

    if (terminalHistoryIndex === -1) {
        return;
    }

    const nextIndex = terminalHistoryIndex + direction;
    if (nextIndex < 0) {
        return;
    }

    if (nextIndex >= terminalCommandHistory.length) {
        terminalHistoryIndex = -1;
        terminalInput.value = terminalDraftCommand;
        return;
    }

    terminalHistoryIndex = nextIndex;
    terminalInput.value = terminalCommandHistory[terminalHistoryIndex];
}

/**
 * Record a submitted command in the local history buffer.
 *
 * @param {string} command - Command text to store.
 */
function pushTerminalHistory(command) {
    if (
        terminalCommandHistory.length === 0 ||
        terminalCommandHistory[terminalCommandHistory.length - 1] !== command
    ) {
        terminalCommandHistory.push(command);
    }
    terminalHistoryIndex = -1;
    terminalDraftCommand = "";
}

/**
 * Toggle whether the terminal input controls are enabled.
 *
 * @param {boolean} isEnabled - Whether input should accept commands.
 */
function setTerminalInputEnabled(isEnabled) {
    const terminalInput = document.getElementById("terminalInput");
    const sendCommandBtn = document.getElementById("sendCommandBtn");

    if (terminalInput) {
        terminalInput.disabled = !isEnabled;
    }
    if (sendCommandBtn) {
        sendCommandBtn.disabled = !isEnabled;
    }
}

/**
 * Send the current terminal input to the backend for execution.
 *
 * @returns {Promise<void>}
 */
async function sendTerminalCommand() {
    const terminalInput = document.getElementById("terminalInput");
    const terminalOutput = getTerminalOutputElement();

    if (!terminalInput || !terminalOutput || terminalCommandInFlight) {
        return;
    }

    const command = terminalInput.value.trim();
    if (!command) {
        return;
    }

    if (command === "clear" || command === "cls") {
        pushTerminalHistory(command);
        terminalInput.value = "";
        clearTerminalDisplay();
        return;
    }

    pushTerminalHistory(command);
    appendToTerminal(terminalOutput, `${terminalPromptText} ${command}`, "command");
    terminalInput.value = "";
    terminalCommandInFlight = true;
    setTerminalInputEnabled(false);

    try {
        const response = await fetch(buildBackendUrl("/terminal/execute"), {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ command }),
        });
        const data = await response.json();
        applyTerminalSessionState(data);

        if (data.output) {
            appendToTerminal(terminalOutput, data.output, "output");
        }
        if (data.error) {
            appendToTerminal(terminalOutput, data.error, "error");
        }
        if (
            typeof data.exit_code === "number" &&
            data.exit_code !== 0 &&
            !data.error
        ) {
            appendToTerminal(
                terminalOutput,
                `Command exited with code ${data.exit_code}`,
                "error",
            );
        }
    } catch (error) {
        appendToTerminal(
            terminalOutput,
            `Error: ${error.message}`,
            "error",
        );
    } finally {
        terminalCommandInFlight = false;
        setTerminalInputEnabled(true);
        terminalInput.focus();
    }
}

/**
 * Append one or more lines of terminal output with styling for the given type.
 *
 * @param {HTMLElement | null} terminalOutput - Terminal output container element.
 * @param {string} text - Text to append.
 * @param {"command"|"error"|"output"|"prompt"} type - Output styling type.
 */
function appendToTerminal(terminalOutput, text, type) {
    if (!terminalOutput || text === "") {
        return;
    }

    const lines = text.split("\n");
    for (const line of lines) {
        const lineDiv = document.createElement("div");
        if (type === "command" || type === "prompt") {
            lineDiv.className = "text-green-400";
            lineDiv.textContent = line;
        } else if (type === "error") {
            lineDiv.className = "text-red-400";
            lineDiv.textContent = line;
        } else {
            lineDiv.className = "text-gray-300";
            lineDiv.textContent = line;
        }
        lineDiv.style.whiteSpace = "pre-wrap";
        terminalOutput.appendChild(lineDiv);
    }

    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

/**
 * Load log messages from the backend once per page lifecycle.
 *
 * @returns {Promise<void>}
 */
async function loadLogMessages() {
    if (logsLoaded) return;

    const logsOutput = document.getElementById("logsOutput");
    if (!logsOutput) return;

    try {
        const response = await fetch(buildBackendUrl("/get-log-messages"));
        const data = await response.json();

        if (data.messages && Array.isArray(data.messages)) {
            logsOutput.innerHTML = "";
            for (const message of data.messages) {
                appendLogMessage(message);
            }
            logsLoaded = true;
        }
    } catch (error) {
        console.error("Failed to load log messages:", error);
        appendToLogs("Failed to load log history", "ERROR");
    }
}

/**
 * Convert ANSI color escape sequences into styled text segments.
 *
 * @param {string} text - Text potentially containing ANSI escape codes.
 * @returns {{text: string, color: string}[]} Parsed text segments.
 */
function parseAnsiColors(text) {
    const ansiRegex = /\u001b\[([0-9;]+)m/g;
    const colorMap = {
        0: "text-gray-300",
        91: "text-red-400",
        92: "text-green-400",
        93: "text-yellow-400",
        94: "text-blue-400",
        96: "text-cyan-400",
    };

    let currentColor = "text-gray-300";
    const segments = [];
    let lastIndex = 0;
    let match;

    while ((match = ansiRegex.exec(text)) !== null) {
        if (match.index > lastIndex) {
            segments.push({
                text: text.substring(lastIndex, match.index),
                color: currentColor,
            });
        }

        const code = match[1];
        currentColor = colorMap[code] || currentColor;
        if (code === "0") currentColor = "text-gray-300";

        lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
        segments.push({
            text: text.substring(lastIndex),
            color: currentColor,
        });
    }

    return segments.length > 0 ? segments : [{ text, color: "text-gray-300" }];
}

/**
 * Append a single log message to the logs output area.
 *
 * @param {string} message - Log message content.
 */
function appendLogMessage(message) {
    const logsOutput = document.getElementById("logsOutput");
    if (!logsOutput) return;

    const logDiv = document.createElement("div");
    logDiv.className = "flex items-start";
    logDiv.style.whiteSpace = "pre-wrap";

    const segments = parseAnsiColors(message);
    for (const segment of segments) {
        const span = document.createElement("span");
        span.className = segment.color;
        span.textContent = segment.text;
        logDiv.appendChild(span);
    }

    logsOutput.appendChild(logDiv);
    logsOutput.scrollTop = logsOutput.scrollHeight;
}

/**
 * Append incoming log updates to the logs output.
 *
 * @param {{messages?: string[]}} data - Log update payload.
 */
export function handleLogUpdate(data) {
    if (data.messages && Array.isArray(data.messages)) {
        for (const message of data.messages) {
            appendLogMessage(message);
        }
    }
}

/**
 * Clear and reload the current log messages.
 */
export function refreshLogMessages() {
    logsLoaded = false;
    const logsOutput = document.getElementById("logsOutput");
    if (logsOutput) {
        logsOutput.innerHTML = "";
        loadLogMessages();
    }
}

/**
 * Download the backend log file to the user's device.
 */
function downloadLogFile() {
    fetch(buildBackendUrl("/download-log-file"))
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then((logContent) => {
            const blob = new Blob([logContent], { type: "text/plain" });
            const downloadLink = document.createElement("a");
            downloadLink.href = URL.createObjectURL(blob);
            downloadLink.download = "eagleeye_logs.txt";
            document.body.appendChild(downloadLink);
            downloadLink.click();
            downloadLink.remove();
            URL.revokeObjectURL(downloadLink.href);
        })
        .catch((error) => {
            console.error("Failed to download log file:", error);
            appendToLogs("Failed to download log file", "ERROR");
        });
}

/**
 * Append a formatted message to the logs output area.
 *
 * @param {string} message - Log message content.
 * @param {"INFO"|"ERROR"|"WARNING"|"DEBUG"} [level="INFO"] - Log severity level.
 */
export function appendToLogs(message, level = "INFO") {
    const logsOutput = document.getElementById("logsOutput");
    if (logsOutput) {
        const logDiv = document.createElement("div");
        const timestamp = new Date().toLocaleTimeString();

        let colorClass = "text-gray-300";
        if (level === "ERROR") colorClass = "text-red-400";
        else if (level === "WARNING") colorClass = "text-orange-400";
        else if (level === "INFO") colorClass = "text-yellow-400";
        else if (level === "DEBUG") colorClass = "text-gray-400";

        logDiv.className = colorClass;
        logDiv.style.whiteSpace = "pre-wrap";
        logDiv.textContent = `[${level}] ${timestamp} - ${message}`;
        logsOutput.appendChild(logDiv);
        logsOutput.scrollTop = logsOutput.scrollHeight;
    }
}
