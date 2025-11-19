let terminalVisible = false;

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

    if (terminalCheckbox && settingsPanel) {
        terminalVisible = terminalCheckbox.checked;
        
        if (!terminalVisible) {
            const viewSettings = document.getElementById("view-settings");
            if (viewSettings) {
                const containerWidth = viewSettings.clientWidth;
                const settingsPanelWidth = 400;
                const centerOffset = (containerWidth / 2) - (settingsPanelWidth / 2);
                settingsPanel.style.setProperty('--center-offset', `${centerOffset}px`);
            }
            settingsPanel.classList.add("centered");
        } else {
            terminalPanel.classList.add("visible");
        }

        terminalCheckbox.addEventListener("change", function () {
            terminalVisible = this.checked;
            updateTerminalVisibility();
        });
    }

    if (clearLogsBtn) {
        clearLogsBtn.addEventListener("click", function () {
            if (logsOutput) {
                logsOutput.innerHTML = '<div class="text-gray-400">[INFO] Logs cleared</div>';
            }
        });
    }

    if (clearTerminalBtn) {
        clearTerminalBtn.addEventListener("click", function () {
            if (terminalOutput) {
                terminalOutput.innerHTML = '<div class="text-green-400">pi@eagleeye:~$ </div>';
            }
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
                const centerOffset = (containerWidth / 2) - (settingsPanelWidth / 2);
                settingsPanel.style.setProperty('--center-offset', `${centerOffset}px`);
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
                        "error"
                    );
                });
        }
    }
}

function appendToTerminal(terminalOutput, text, type) {
    const lines = text.split("\n");
    lines.forEach((line) => {
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
    });

    terminalOutput.scrollTop = terminalOutput.scrollHeight;
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
        logDiv.textContent = `[${level}] ${timestamp} - ${message}`;
        logsOutput.appendChild(logDiv);
        logsOutput.scrollTop = logsOutput.scrollHeight;
    }
}
