let terminalVisible = false;
let logsLoaded = false;

export function initializeTerminalHandlers() {
    const terminalCheckbox = document.getElementById("printTerminalCheckbox");
    const terminalPanel = document.getElementById("terminalPanel");
    const settingsPanel = document.getElementById("settingsPanel");
    const clearLogsBtn = document.getElementById("clearLogsBtn");
    const clearTerminalBtn = document.getElementById("clearTerminalBtn");
    const sendCommandBtn = document.getElementById("sendCommandBtn");
    const terminalInput = document.getElementById("terminalInput");
    const logsOutput = document.getElementById("logsOutput");
    const terminalOutput = document.getElementById("terminalOutput");
    const downloadLogsBtn = document.getElementById("downloadLogsBtn");

    if (terminalCheckbox && settingsPanel) {
        terminalVisible = terminalCheckbox.checked;

        if (terminalVisible === false) {
            const viewSettings = document.getElementById("view-settings");
            if (viewSettings) {
                const containerWidth = viewSettings.clientWidth;
                const settingsPanelWidth = 400;
                const centerOffset =
                    containerWidth / 2 - settingsPanelWidth / 2;
                settingsPanel.style.setProperty(
                    "--center-offset",
                    `${centerOffset}px`,
                );
            }
            settingsPanel.classList.add("centered");
        } else {
            terminalPanel.classList.add("visible");
            loadLogMessages();
        }

        terminalCheckbox.addEventListener("change", function () {
            terminalVisible = this.checked;
            updateTerminalVisibility();
            if (terminalVisible && !logsLoaded) {
                loadLogMessages();
            }
        });
    }

    if (clearLogsBtn) {
        clearLogsBtn.addEventListener("click", function () {
            if (logsOutput) {
                logsOutput.innerHTML = "";
                appendToLogs(
                    "Logs cleared (display only - backend logs preserved)",
                    "INFO",
                );
            }
        });
    }

    if (clearTerminalBtn) {
        clearTerminalBtn.addEventListener("click", function () {
            if (terminalOutput) {
                terminalOutput.innerHTML =
                    '<div class="text-green-400">pi@eagleeye:~$ </div>';
            }
        });
    }

    if (downloadLogsBtn) {
        downloadLogsBtn.addEventListener("click", function () {
            downloadLogFile();
        });
    }

    if (sendCommandBtn) {
        sendCommandBtn.addEventListener("click", sendTerminalCommand);
    }

    if (terminalInput) {
        terminalInput.addEventListener("keypress", function (event) {
            if (event.key === "Enter") {
                sendTerminalCommand();
            }
        });
    }
}

function updateTerminalVisibility() {
    const terminalPanel = document.getElementById("terminalPanel");
    const settingsPanel = document.getElementById("settingsPanel");
    const viewSettings = document.getElementById("view-settings");

    if (terminalPanel && settingsPanel && viewSettings) {
        if (terminalVisible) {
            settingsPanel.classList.remove("centered");
            setTimeout(() => {
                terminalPanel.classList.add("visible");
            }, 800);
        } else {
            terminalPanel.classList.remove("visible");
            setTimeout(() => {
                const containerWidth = viewSettings.clientWidth;
                const settingsPanelWidth = 400;
                const centerOffset =
                    containerWidth / 2 - settingsPanelWidth / 2;
                settingsPanel.style.setProperty(
                    "--center-offset",
                    `${centerOffset}px`,
                );
                settingsPanel.classList.add("centered");
            }, 500);
        }
    }
}

function sendTerminalCommand() {
    const terminalInput = document.getElementById("terminalInput");
    const terminalOutput = document.getElementById("terminalOutput");

    if (terminalInput && terminalOutput) {
        const command = terminalInput.value.trim();
        if (command) {
            appendToTerminal(terminalOutput, command, "command");
            terminalInput.value = "";

            fetch("http://localhost:5001/terminal/execute", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ command: command }),
            })
                .then((response) => response.json())
                .then((data) => {
                    if (data.output) {
                        appendToTerminal(terminalOutput, data.output, "output");
                    }
                    if (data.error) {
                        appendToTerminal(terminalOutput, data.error, "error");
                    }
                })
                .catch((error) => {
                    appendToTerminal(
                        terminalOutput,
                        `Error: ${error.message}`,
                        "error",
                    );
                });
        }
    }
}

function appendToTerminal(terminalOutput, text, type) {
    const lines = text.split("\n");
    for (const line of lines) {
        const lineDiv = document.createElement("div");
        if (type === "command") {
            lineDiv.className = "text-green-400";
            lineDiv.textContent = `pi@eagleeye:~$ ${line}`;
        } else if (type === "error") {
            lineDiv.className = "text-red-400";
            lineDiv.textContent = line;
        } else {
            lineDiv.className = "text-gray-300";
            lineDiv.textContent = line;
        }
        terminalOutput.appendChild(lineDiv);
    }

    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

async function loadLogMessages() {
    if (logsLoaded) return;

    const logsOutput = document.getElementById("logsOutput");
    if (!logsOutput) return;

    try {
        const response = await fetch("http://localhost:5001/get-log-messages");
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

export function handleLogUpdate(data) {
    if (data.messages && Array.isArray(data.messages)) {
        for (const message of data.messages) {
            appendLogMessage(message);
        }
    }
}

export function refreshLogMessages() {
    logsLoaded = false;
    const logsOutput = document.getElementById("logsOutput");
    if (logsOutput) {
        logsOutput.innerHTML = "";
        loadLogMessages();
    }
}

function downloadLogFile() {
    fetch("http://localhost:5001/download-log-file")
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then((logContent) => {
            // Create a blob with the log content
            const blob = new Blob([logContent], { type: "text/plain" });

            // Create a temporary anchor element to trigger download
            const a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "eagleeye_logs.txt";
            document.body.appendChild(a);
            a.click();
            a.remove();

            // Clean up the object URL
            URL.revokeObjectURL(a.href);
        })
        .catch((error) => {
            console.error("Failed to download log file:", error);
            appendToLogs("Failed to download log file", "ERROR");
        });
}

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
