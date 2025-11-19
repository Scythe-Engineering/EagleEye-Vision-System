export function loadSettings() {
    const settings = fetch("/get-settings", {
        method: "GET",
        headers: {
            "Content-Type": "application/json",
        },
    }).then((response) => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        return response.json();
    });

    settings.then((settings) => {
        document.getElementById("logCheckbox").checked =
            settings["Constants"]["log"];
        const terminalCheckbox = document.getElementById("printTerminalCheckbox");
        if (terminalCheckbox) {
            terminalCheckbox.checked = settings["Constants"]["print_terminal"];
            terminalCheckbox.dispatchEvent(new Event("change"));
        }
        document.getElementById("detectionLoggingCheckbox").checked =
            settings["Constants"]["detection_logging"];
        document.getElementById("simulationModeCheckbox").checked =
            settings["Constants"]["simulation_mode"];
        document.getElementById("serverAddressInput").value =
            settings["NetworkTableConstants"]["server_address"];
        document.getElementById("robotPositionKeyInput").value =
            settings["NetworkTableConstants"]["robot_position_key"];
        document.getElementById("robotRotationKeyInput").value =
            settings["NetworkTableConstants"]["robot_rotation_key"];
        document.getElementById("inputSizeInput").value =
            settings["ObjectDetectionConstants"]["input_size"];
        document.getElementById("confidenceThresholdInput").value =
            settings["ObjectDetectionConstants"]["confidence_threshold"];
        document.getElementById("combinedThresholdInput").value =
            settings["ObjectDetectionConstants"]["combined_threshold"];
        document.getElementById("maxDistanceInput").value =
            settings["ObjectDetectionConstants"]["max_distance"];
    });
}
