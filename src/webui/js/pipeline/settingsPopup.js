import { BACKEND_BASE_URL } from "../config.js";
import { registerFileManagerPopup } from "./fileManager.js";

/**
 * Builds and registers the settings popup UI for pipeline operations.
 */
export function registerSettingsPopup() {
    registerFileManagerPopup();

    if (globalThis.SettingsPopup) {
        return globalThis.SettingsPopup;
    }

    const OVERLAY_ID = "operationSettingsOverlay";
    const MODAL_ID = "operationSettingsModal";
    // Visualization state
    let _visInterval = null;
    let _currentVisObjectUrl = null;
    let _currentVisPipeline = null;
    let _currentVisAction = null;
    let _availableCameras = null;
    let _availableCamerasLoading = null;

    /**
     * Creates a DOM element with the provided attributes and children.
     */
    function createElement(tag, attrs = {}, children = []) {
        const el = document.createElement(tag);
        Object.entries(attrs).forEach(([k, v]) => {
            if (k === "className") {
                el.className = v;
            } else if (k === "text") {
                el.textContent = v;
            } else if (k === "html") {
                el.innerHTML = v;
            } else if (k.startsWith("on") && typeof v === "function") {
                el.addEventListener(k.substring(2).toLowerCase(), v);
            } else if (v !== undefined && v !== null) {
                el.setAttribute(k, String(v));
            }
        });
        (children || []).forEach((c) => el.appendChild(c));
        return el;
    }

    /**
     * Determines whether an operation configuration exposes editable settings.
     */
    function operationHasSettings(config) {
        const params = config?.parameters;
        const hasParameters =
            params &&
            typeof params === "object" &&
            Object.keys(params).length > 0;
        const hasDynamicGroup =
            config?.dynamic_group && typeof config.dynamic_group === "object";
        return hasParameters || hasDynamicGroup;
    }

    /**
     * Renders a summary of static and dynamic pipeline port groups.
     */
    function renderPortGroupsSummary(modalBody, config) {
        const inputNodes = Array.isArray(config?.input_nodes)
            ? config.input_nodes
            : [];
        const outputNodes = Array.isArray(config?.output_nodes)
            ? config.output_nodes
            : [];
        const dynamicGroup =
            config?.dynamic_group && typeof config.dynamic_group === "object"
                ? config.dynamic_group
                : null;

        const staticInputNames = inputNodes.map((node) =>
            typeof node === "object" && node?.name ? node.name : String(node),
        );
        const staticOutputNames = outputNodes.map((node) =>
            typeof node === "object" && node?.name ? node.name : String(node),
        );

        let dynamicInputBase = null;
        let dynamicOutputBase = null;
        let maxInputs = 0;
        let mirrored = false;
        let hasDynamicInput = false;
        let hasDynamicOutput = false;

        if (dynamicGroup) {
            const inputDynDisabled =
                dynamicGroup.input_dynamic_group === false ||
                String(dynamicGroup.input_dynamic_group).toLowerCase() ===
                    "false";
            hasDynamicInput = !inputDynDisabled;

            mirrored =
                dynamicGroup.mirrored_output_group === true ||
                String(dynamicGroup.mirrored_output_group).toLowerCase() ===
                    "true";
            const outputDyn =
                dynamicGroup.output_dynamic_group === true ||
                String(dynamicGroup.output_dynamic_group).toLowerCase() ===
                    "true";
            hasDynamicOutput = mirrored || outputDyn;

            dynamicInputBase =
                dynamicGroup.input_base_name ||
                dynamicGroup.input_node ||
                staticInputNames[staticInputNames.length - 1] ||
                "data";
            dynamicOutputBase =
                dynamicGroup.output_base_name ||
                dynamicGroup.output_node ||
                staticOutputNames[staticOutputNames.length - 1] ||
                dynamicInputBase;
            maxInputs = Math.max(
                1,
                Number.parseInt(dynamicGroup.max_inputs ?? 1, 10) || 1,
            );
        }

        const effectiveStaticInputs =
            dynamicInputBase && hasDynamicInput
                ? staticInputNames.filter((name) => name !== dynamicInputBase)
                : staticInputNames;

        const effectiveStaticOutputs =
            dynamicOutputBase && hasDynamicOutput
                ? staticOutputNames.filter((name) => name !== dynamicOutputBase)
                : staticOutputNames;

        if (
            effectiveStaticInputs.length === 0 &&
            effectiveStaticOutputs.length === 0 &&
            !dynamicGroup
        ) {
            return;
        }

        const summaryContainer = createElement("div", {
            className:
                "mb-5 bg-[#1a1a1a] border border-[#414141] rounded-lg p-3 space-y-3",
        });

        const summaryTitle = createElement("div", {
            className: "text-sm font-semibold text-[#f9c845]",
            text: "Pipeline Ports",
        });
        summaryContainer.appendChild(summaryTitle);

        const blocksContainer = createElement("div", {
            className: "grid grid-cols-1 md:grid-cols-2 gap-3",
        });

        const staticBlock = createElement("div", {
            className: "bg-[#232323] border border-[#3a3a3a] rounded-md p-3",
        });
        staticBlock.appendChild(
            createElement("div", {
                className:
                    "text-xs uppercase tracking-wide text-[#ac8a2f] mb-2",
                text: "Static slots",
            }),
        );

        const staticInputsLine = createElement("div", {
            className: "text-xs text-gray-300 mb-1",
            text: `Inputs: ${effectiveStaticInputs.length > 0 ? effectiveStaticInputs.join(", ") : "None"}`,
        });
        const staticOutputsLine = createElement("div", {
            className: "text-xs text-gray-300",
            text: `Outputs: ${effectiveStaticOutputs.length > 0 ? effectiveStaticOutputs.join(", ") : "None"}`,
        });
        staticBlock.appendChild(staticInputsLine);
        staticBlock.appendChild(staticOutputsLine);
        blocksContainer.appendChild(staticBlock);

        if (dynamicGroup) {
            const dynamicBlock = createElement("div", {
                className:
                    "bg-[#1a2430] border border-[#2f5f89] rounded-md p-3",
            });
            dynamicBlock.appendChild(
                createElement("div", {
                    className:
                        "text-xs uppercase tracking-wide text-[#8dc8ff] mb-2",
                    text: "Dynamic group",
                }),
            );
            if (hasDynamicInput) {
                dynamicBlock.appendChild(
                    createElement("div", {
                        className: "text-xs text-[#d9ecff] mb-1",
                        text: `Input base: ${dynamicInputBase} (starts at 1, max ${maxInputs})`,
                    }),
                );
            }
            if (hasDynamicOutput) {
                const maxOutputs = Math.max(
                    1,
                    Number.parseInt(
                        dynamicGroup.max_outputs ?? maxInputs,
                        10,
                    ) || maxInputs,
                );
                dynamicBlock.appendChild(
                    createElement("div", {
                        className: "text-xs text-[#d9ecff]",
                        text: mirrored
                            ? `Output base: ${dynamicOutputBase} (mirrored to input count)`
                            : `Output base: ${dynamicOutputBase} (starts at 1, max ${maxOutputs})`,
                    }),
                );
            } else {
                dynamicBlock.appendChild(
                    createElement("div", {
                        className: "text-xs text-[#d9ecff]",
                        text: "Output mirroring: disabled",
                    }),
                );
            }
            blocksContainer.appendChild(dynamicBlock);
        }

        summaryContainer.appendChild(blocksContainer);
        modalBody.appendChild(summaryContainer);
    }

    /**
     * Loads and caches the list of available cameras from the backend.
     */
    async function loadAvailableCameras() {
        if (Array.isArray(_availableCameras)) {
            return _availableCameras;
        }
        if (_availableCamerasLoading) {
            return _availableCamerasLoading;
        }
        _availableCamerasLoading = fetch(
            `${BACKEND_BASE_URL}/get-available-cameras`,
        )
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then((data) => {
                const cameras = Object.entries(data || {}).map(
                    ([name, cameraInfo]) => {
                        let id;

                        if (
                            cameraInfo?.bus_id !== undefined &&
                            cameraInfo?.bus_id !== null
                        ) {
                            id = String(cameraInfo.bus_id);
                        } else if (
                            cameraInfo?.id !== undefined &&
                            cameraInfo?.id !== null
                        ) {
                            id = String(cameraInfo.id);
                        } else {
                            id = name;
                        }

                        return {
                            name,
                            id,
                        };
                    },
                );
                _availableCameras = cameras;
                return cameras;
            })
            .catch((error) => {
                console.warn("Failed to fetch available cameras:", error);
                _availableCameras = [];
                return [];
            })
            .finally(() => {
                _availableCamerasLoading = null;
            });
        return _availableCamerasLoading;
    }

    /**
     * Notifies the pipeline creator that the camera list has changed.
     */
    function notifyCameraListUpdated(cameras) {
        if (
            globalThis.pipelineCreator?.getAvailableCameras &&
            cameras.length &&
            !globalThis.pipelineCreator?.refreshAvailableCameras
        ) {
            return;
        }
        if (globalThis.pipelineCreator?.refreshAvailableCameras) {
            globalThis.pipelineCreator
                .refreshAvailableCameras()
                .catch((error) =>
                    console.warn("Failed to refresh pipeline cameras:", error),
                );
        }
    }

    /**
     * Builds an HSV picker field for color parameters.
     */
    function buildHsvPickerField(currentValue, fieldId, label, isEdited) {
        const hsvValue = currentValue || [0, 0, 0];

        const container = createElement("div", { className: "mb-4" });

        const labelRow = createElement(
            "div",
            { className: "flex items-center mb-2" },
            [label],
        );
        container.appendChild(labelRow);

        const inputContainer = createElement("div", {
            className: "relative flex gap-2 items-center",
        });

        const colorPreview = createElement("div", {
            className: "w-10 h-10 rounded border-2 border-[#414141]",
            style: `background-color: hsl(${hsvValue[0] * 2}, ${hsvValue[1] / 2.55}%, ${hsvValue[2] / 2.55}%)`,
        });

        const hsvInputs = createElement("div", {
            className: "flex gap-2 flex-1",
        });

        const hInput = createElement("input", {
            id: `${fieldId}-h`,
            type: "number",
            className:
                "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            min: "0",
            max: "179",
            step: "1",
            value: hsvValue[0],
        });
        const hLabel = createElement("label", {
            for: `${fieldId}-h`,
            className: "text-xs text-[#ac8a2f]",
            text: "H",
        });

        const sInput = createElement("input", {
            id: `${fieldId}-s`,
            type: "number",
            className:
                "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            min: "0",
            max: "255",
            step: "1",
            value: hsvValue[1],
        });
        const sLabel = createElement("label", {
            for: `${fieldId}-s`,
            className: "text-xs text-[#ac8a2f]",
            text: "S",
        });

        const vInput = createElement("input", {
            id: `${fieldId}-v`,
            type: "number",
            className:
                "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            min: "0",
            max: "255",
            step: "1",
            value: hsvValue[2],
        });
        const vLabel = createElement("label", {
            for: `${fieldId}-v`,
            className: "text-xs text-[#ac8a2f]",
            text: "V",
        });

        const updateColor = () => {
            const h = Math.max(
                0,
                Math.min(179, Number.parseInt(hInput.value, 10) || 0),
            );
            const s = Math.max(
                0,
                Math.min(255, Number.parseInt(sInput.value, 10) || 0),
            );
            const v = Math.max(
                0,
                Math.min(255, Number.parseInt(vInput.value, 10) || 0),
            );
            colorPreview.style.backgroundColor = `hsl(${h * 2}, ${s / 2.55}%, ${v / 2.55}%)`;
        };

        hInput.addEventListener("input", updateColor);
        sInput.addEventListener("input", updateColor);
        vInput.addEventListener("input", updateColor);

        const hWrapper = createElement("div", { className: "flex-1" }, [
            hLabel,
            hInput,
        ]);
        const sWrapper = createElement("div", { className: "flex-1" }, [
            sLabel,
            sInput,
        ]);
        const vWrapper = createElement("div", { className: "flex-1" }, [
            vLabel,
            vInput,
        ]);

        hsvInputs.appendChild(hWrapper);
        hsvInputs.appendChild(sWrapper);
        hsvInputs.appendChild(vWrapper);

        inputContainer.appendChild(colorPreview);
        inputContainer.appendChild(hsvInputs);

        const editedIndicator = createElement("div", {
            className:
                "absolute -left-1 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-yellow-400 rounded-full",
            title: "This field has been modified from its default value",
            style: isEdited ? "" : "display: none;",
        });
        inputContainer.appendChild(editedIndicator);

        container.appendChild(inputContainer);

        return {
            wrapper: container,
            getValue: () => [
                Number.parseInt(hInput.value, 10) || 0,
                Number.parseInt(sInput.value, 10) || 0,
                Number.parseInt(vInput.value, 10) || 0,
            ],
        };
    }

    /**
     * Builds a nested object field from a schema definition.
     */
    function buildObjectField(
        name,
        def,
        currentValue,
        originalValue,
        label,
        operationName,
        path,
    ) {
        const container = createElement("div", {
            className:
                "mb-4 bg-[#1a1a1a] border border-[#414141] rounded-lg p-4",
        });

        const header = createElement("div", {
            className: "flex items-center justify-between mb-3",
        });
        const title = createElement("h4", {
            className: "text-sm font-medium text-[#f9c845]",
            text: def.description || name,
        });
        header.appendChild(title);
        container.appendChild(header);

        const body = createElement("div", {
            className: "space-y-3",
        });

        const subFields = [];
        const schema = def.schema || {};

        // Object.keys() preserves insertion order in ES2015+
        // This maintains the field order as defined in the config JSON file
        Object.keys(schema).forEach((subName) => {
            const subDef = schema[subName];
            const subField = buildField(
                subName,
                subDef,
                currentValue || {},
                originalValue || {},
                operationName,
                path,
            );
            subFields.push({ name: subName, ...subField });
            body.appendChild(subField.wrapper);
        });

        container.appendChild(body);

        return {
            wrapper: container,
            getValue: () => {
                const result = {};
                subFields.forEach((f) => {
                    result[f.name] = f.getValue();
                });
                return result;
            },
        };
    }

    /**
     * Builds an editable list field from a list definition.
     */
    function buildListField(
        name,
        def,
        currentValue,
        originalValue,
        label,
        operationName,
        path,
    ) {
        const container = createElement("div", { className: "mb-4" });

        const header = createElement("div", {
            className: "flex items-center justify-between mb-2",
        });
        const title = createElement("h4", {
            className: "text-sm font-medium text-[#f9c845]",
            text: def.description || name,
        });
        header.appendChild(title);
        container.appendChild(header);

        const itemsContainer = createElement("div", {
            className: "space-y-2",
        });

        const itemWrappers = [];
        const itemFields = [];
        const removeButtons = [];

        const updateButtonStates = () => {
            const currentCount = itemWrappers.length;
            const minItems = def.min_items !== undefined ? def.min_items : 0;
            const maxItems = def.max_items;
            const canRemoveItems = currentCount > minItems;

            removeButtons.forEach((btn) => {
                btn.style.display = canRemoveItems ? "" : "none";
            });

            if (addBtn) {
                const canAddItems =
                    maxItems === undefined || currentCount < maxItems;
                addBtn.style.display = canAddItems ? "" : "none";
            }
        };

        const renderItem = (index, itemValue, itemOriginalValue) => {
            const itemContainer = createElement("div", {
                className:
                    "bg-[#1a1a1a] border border-[#414141] rounded-lg p-3",
            });

            const itemHeader = createElement("div", {
                className: "flex items-center justify-between mb-2",
            });

            let itemLabel = `Item ${index + 1}`;
            if (
                def.item_labels &&
                Array.isArray(def.item_labels) &&
                def.item_labels[index]
            ) {
                itemLabel = def.item_labels[index];
            }

            const itemTitle = createElement("span", {
                className: "text-xs text-[#ac8a2f]",
                text: itemLabel,
            });
            itemHeader.appendChild(itemTitle);

            const removeBtn = createElement("button", {
                type: "button",
                className:
                    "px-2 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-xs",
                text: "Remove",
                onclick: () => {
                    const idx = itemWrappers.indexOf(itemContainer);
                    if (idx > -1) {
                        itemWrappers.splice(idx, 1);
                        itemFields.splice(idx, 1);
                        removeButtons.splice(idx, 1);
                        itemContainer.remove();
                        updateButtonStates();
                    }
                },
            });
            removeButtons.push(removeBtn);
            itemHeader.appendChild(removeBtn);
            itemContainer.appendChild(itemHeader);

            const itemBody = createElement("div", {
                className: "space-y-2",
            });

            if (def.item_type === "object" && def.schema) {
                const subFields = [];
                Object.keys(def.schema).forEach((subName) => {
                    const subDef = def.schema[subName];
                    const subField = buildField(
                        subName,
                        subDef,
                        itemValue || {},
                        itemOriginalValue || {},
                        operationName,
                        `${path}-${index}`,
                    );
                    subFields.push({ name: subName, ...subField });
                    itemBody.appendChild(subField.wrapper);
                });

                itemFields.push({
                    getValue: () => {
                        const result = {};
                        for (const field of subFields) {
                            result[field.name] = field.getValue();
                        }
                        return result;
                    },
                });
            } else {
                const itemInput = createElement("input", {
                    id: `${path}-${index}`,
                    type:
                        def.item_type === "int" || def.item_type === "float"
                            ? "number"
                            : "text",
                    className:
                        "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
                    value: itemValue !== undefined ? String(itemValue) : "",
                });

                if (def.item_type === "int") {
                    itemInput.step = "1";
                } else if (def.item_type === "float") {
                    itemInput.step = "any";
                }

                itemBody.appendChild(itemInput);

                itemFields.push({
                    getValue: () => {
                        if (def.item_type === "int") {
                            return Number.parseInt(itemInput.value, 10) || 0;
                        } else if (def.item_type === "float") {
                            return Number.parseFloat(itemInput.value) || 0;
                        }
                        return itemInput.value;
                    },
                });
            }

            itemContainer.appendChild(itemBody);
            itemsContainer.appendChild(itemContainer);
            itemWrappers.push(itemContainer);
        };

        const listValue = Array.isArray(currentValue) ? currentValue : [];
        const listOriginal = Array.isArray(originalValue) ? originalValue : [];

        listValue.forEach((item, index) => {
            renderItem(index, item, listOriginal[index]);
        });

        container.appendChild(itemsContainer);

        let addBtn = null;
        addBtn = createElement("button", {
            type: "button",
            className:
                "w-full mt-2 px-3 py-2 bg-[#f9c845] text-[#232323] rounded-md hover:bg-[#d4a83a] transition-colors text-sm font-medium",
            text: "Add Item",
            onclick: () => {
                const newIndex = itemWrappers.length;
                const defaultValue =
                    def.default || (def.item_type === "object" ? {} : "");
                renderItem(newIndex, defaultValue, defaultValue);
                updateButtonStates();
            },
        });
        container.appendChild(addBtn);

        updateButtonStates();

        return {
            wrapper: container,
            getValue: () => {
                return itemFields.map((f) => f.getValue());
            },
        };
    }

    /**
     * Builds the appropriate form field for a parameter definition.
     */
    function buildField(
        name,
        def,
        currentValues,
        originalValues,
        operationName = null,
        parentPath = "",
    ) {
        const fieldId = `setting-${parentPath}${parentPath ? "-" : ""}${name}`;
        const label = createElement("label", {
            for: fieldId,
            className: "block text-sm font-medium text-[#f9c845] mb-1",
            text: def.description || name,
        });

        let input;
        let secondaryInput = null;
        const currentValue =
            currentValues && name in currentValues
                ? currentValues[name]
                : def.default;

        const originalValue =
            originalValues && name in originalValues
                ? originalValues[name]
                : def.default;

        // Check if field has been edited
        const isEdited =
            JSON.stringify(currentValue) !== JSON.stringify(originalValue);

        const isPathParameter = name.endsWith("_path") && def.type === "str";

        // Handle UI hints for specialized editors
        if (def.ui_hint === "hsv_picker") {
            return buildHsvPickerField(currentValue, fieldId, label, isEdited);
        }

        // Handle object type - recursive rendering
        if (def.type === "object" && def.schema) {
            return buildObjectField(
                name,
                def,
                currentValue,
                originalValue,
                label,
                operationName,
                `${parentPath}${parentPath ? "-" : ""}${name}`,
                false,
            );
        }

        // Handle list type
        if (def.type === "list") {
            return buildListField(
                name,
                def,
                currentValue,
                originalValue,
                label,
                operationName,
                `${parentPath}${parentPath ? "-" : ""}${name}`,
                false,
            );
        }

        const normalizedOperationName = String(operationName || "")
            .replace(/\.py$/i, "")
            .toLowerCase()
            .replaceAll(/\s+/g, "_");
        const isDeviceInputBusId =
            normalizedOperationName === "device_input" &&
            name === "bus_id" &&
            def.type === "str";
        const isCameraBusIdParameter =
            name === "camera_bus_id" && def.type === "str";

        if (isDeviceInputBusId || isCameraBusIdParameter) {
            input = createElement("select", {
                id: fieldId,
                className:
                    "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            });

            const fallbackOption = createElement("option", {
                value: "",
                text: "Loading cameras...",
            });
            input.appendChild(fallbackOption);

            const updateOptions = (cameras) => {
                const selectedValue =
                    input.value ||
                    (currentValue !== undefined && currentValue !== null
                        ? String(currentValue)
                        : "");
                input.innerHTML = "";

                const normalizedCameras = Array.isArray(cameras)
                    ? cameras
                          .map((camera) => {
                              if (typeof camera === "string") {
                                  return { id: camera, name: camera };
                              }
                              if (!camera || typeof camera !== "object") {
                                  return null;
                              }
                              const cameraId =
                                  camera.id !== undefined && camera.id !== null
                                      ? String(camera.id)
                                      : "";
                              if (!cameraId) {
                                  return null;
                              }
                              return {
                                  id: cameraId,
                                  name: String(camera.name || cameraId),
                              };
                          })
                          .filter(Boolean)
                    : [];
                if (normalizedCameras.length > 0) {
                    normalizedCameras.forEach((camera) => {
                        const optionText =
                            camera.name && camera.name !== camera.id
                                ? `${camera.id} (${camera.name})`
                                : camera.id;
                        const optEl = createElement("option", {
                            value: camera.id,
                            text: optionText,
                        });
                        if (camera.id === selectedValue) {
                            optEl.selected = true;
                        }
                        input.appendChild(optEl);
                    });

                    const knownBusIds = normalizedCameras.map(
                        (camera) => camera.id,
                    );

                    if (selectedValue && !knownBusIds.includes(selectedValue)) {
                        const customOption = createElement("option", {
                            value: selectedValue,
                            text: `${selectedValue} (custom)`,
                            selected: true,
                        });
                        input.appendChild(customOption);
                    }
                    return;
                }

                const customOption = createElement("option", {
                    value: selectedValue,
                    text: selectedValue
                        ? `${selectedValue} (custom)`
                        : "No cameras available",
                });
                customOption.selected = true;
                input.appendChild(customOption);
            };

            const pipelineCameras =
                globalThis.pipelineCreator?.getAvailableCameras?.() || [];
            if (pipelineCameras.length > 0) {
                updateOptions(pipelineCameras);
            } else {
                void loadAvailableCameras().then((cameras) => {
                    updateOptions(cameras);
                    notifyCameraListUpdated(cameras);
                });
            }
        } else if (isPathParameter && operationName) {
            input = createElement("select", {
                id: fieldId,
                className:
                    "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            });

            const customOption = createElement("option", {
                value: "",
                text: "Custom path...",
            });
            input.appendChild(customOption);

            let basePath = "";
            const loadFiles = async () => {
                try {
                    let normalizedOpName = operationName.toLowerCase();
                    if (normalizedOpName.endsWith(".py")) {
                        normalizedOpName = normalizedOpName.slice(0, -3);
                    }
                    normalizedOpName = normalizedOpName.replace(/\s+/g, "_");
                    const response = await fetch(
                        `${BACKEND_BASE_URL}/get-operation-files/${encodeURIComponent(normalizedOpName)}/${encodeURIComponent(name)}`,
                    );
                    if (response.ok) {
                        const data = await response.json();
                        basePath = data.base_path || "";

                        const currentSelectedValue = input.value;

                        while (input.options.length > 0) {
                            input.remove(0);
                        }

                        input.appendChild(customOption);

                        data.files.forEach((filename) => {
                            const fullPath = basePath
                                ? `${basePath}/${filename}`
                                : filename;
                            const optEl = createElement("option", {
                                value: fullPath,
                                text: filename,
                            });
                            const currentPathValue = currentValue || "";
                            if (
                                currentPathValue === fullPath ||
                                currentPathValue.endsWith(`/${filename}`) ||
                                currentPathValue === filename ||
                                currentSelectedValue === fullPath
                            ) {
                                optEl.selected = true;
                            }
                            customOption.before(optEl);
                        });

                        if (
                            currentValue &&
                            !data.files.some((f) => {
                                const fullPath = basePath
                                    ? `${basePath}/${f}`
                                    : f;
                                return (
                                    currentValue === fullPath ||
                                    currentValue.endsWith(`/${f}`) ||
                                    currentValue === f
                                );
                            })
                        ) {
                            const customValueOption = createElement("option", {
                                value: currentValue,
                                text: currentValue + " (custom)",
                                selected: currentSelectedValue === currentValue,
                            });
                            customOption.before(customValueOption);
                        } else if (currentValue) {
                            customOption.selected = false;
                        } else if (
                            !currentSelectedValue ||
                            currentSelectedValue === ""
                        ) {
                            customOption.selected = true;
                        }
                    }
                } catch (error) {
                    console.error("Error loading files:", error);
                }
            };

            loadFiles();

            input.addEventListener("change", () => {
                if (input.value === "") {
                    const customPath = prompt(
                        "Enter custom path:",
                        currentValue || "",
                    );
                    if (customPath !== null && customPath !== "") {
                        const existingOption = Array.from(input.options).find(
                            (opt) => opt.value === customPath,
                        );
                        if (existingOption) {
                            input.value = customPath;
                        } else {
                            const newOption = createElement("option", {
                                value: customPath,
                                text: customPath + " (custom)",
                                selected: true,
                            });
                            customOption.before(newOption);
                            input.value = customPath;
                        }
                    } else if (customPath === null || customPath === "") {
                        input.value = currentValue || "";
                    }
                }
            });

            globalThis.refreshPathDropdown = (selectedFilename) => {
                loadFiles().then(() => {
                    if (selectedFilename) {
                        const fullPath = basePath
                            ? `${basePath}/${selectedFilename}`
                            : selectedFilename;
                        const existingOption = Array.from(input.options).find(
                            (opt) => opt.value === fullPath,
                        );
                        if (existingOption) {
                            input.value = fullPath;
                        } else {
                            const newOption = createElement("option", {
                                value: fullPath,
                                text: selectedFilename,
                                selected: true,
                            });
                            customOption.before(newOption);
                            input.value = fullPath;
                        }
                        const event = new Event("change", { bubbles: true });
                        input.dispatchEvent(event);
                    }
                });
            };
        } else if (def.options && Array.isArray(def.options)) {
            input = createElement("select", {
                id: fieldId,
                className:
                    "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            });
            def.options.forEach((opt) => {
                const optEl = createElement("option", {
                    value: String(opt),
                    text: String(opt),
                });
                if (
                    currentValue !== undefined &&
                    String(currentValue) === String(opt)
                )
                    optEl.selected = true;
                input.appendChild(optEl);
            });
        } else {
            const type = def.type;
            const useSlider =
                def.ui_control === "slider" &&
                (type === "int" || type === "float") &&
                typeof def.min === "number" &&
                typeof def.max === "number";
            let inputType = "text";
            if (useSlider) inputType = "range";
            else if (type === "int" || type === "float") inputType = "number";
            if (type === "str") inputType = "text";
            if (type === "bool") inputType = "checkbox";

            const attrs = {
                id: fieldId,
                type: inputType,
                className:
                    inputType === "checkbox"
                        ? "h-4 w-4 text-[#f9c845] focus:ring-[#f9c845] border-[#414141] rounded bg-[#232323]"
                        : inputType === "range"
                          ? "flex-1 accent-[#f9c845] cursor-pointer"
                          : "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            };
            if (inputType === "number" || inputType === "range") {
                if (typeof def.min === "number") attrs.min = String(def.min);
                if (typeof def.max === "number") attrs.max = String(def.max);
                if (typeof def.step === "number") attrs.step = String(def.step);
                else if (type === "int") attrs.step = "1";
                else if (type === "float") attrs.step = inputType === "range" ? "0.001" : "any";
            }
            input = createElement("input", attrs);
            if (inputType === "checkbox") {
                input.checked = Boolean(currentValue);
            } else if (currentValue !== undefined && currentValue !== null) {
                input.value = String(currentValue);
            }

            if (useSlider) {
                secondaryInput = createElement("input", {
                    type: "number",
                    min: String(def.min),
                    max: String(def.max),
                    step: attrs.step || "0.001",
                    value: input.value,
                    className:
                        "w-24 bg-[#232323] border border-[#414141] text-white rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
                    "aria-label": `${name} numeric value`,
                });
                input.addEventListener("input", () => {
                    secondaryInput.value = input.value;
                });
                secondaryInput.addEventListener("input", () => {
                    input.value = secondaryInput.value;
                    input.dispatchEvent(new Event("input", { bubbles: true }));
                });
            }
        }

        const hint = createElement("div", {
            className: "text-xs text-[#ac8a2f] ml-auto",
            text: def.required ? "Required" : "Optional",
        });

        const labelRow = createElement(
            "div",
            {
                className: "flex items-center mb-2",
            },
            [label, hint],
        );

        // Create input container with relative positioning
        const inputContainer = createElement("div", {
            className: "relative flex gap-2",
        });

        // Add input to container
        inputContainer.appendChild(input);
        if (secondaryInput) {
            inputContainer.appendChild(secondaryInput);
        }

        let restartIndicator = null;
        if (def.restart_for_change) {
            restartIndicator = createElement("span", {
                className:
                    "group/restart absolute left-0 top-1/2 z-20 inline-flex w-3 h-3 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-[#f9c845]",
                "aria-label": "Restart required to apply config change",
                style: isEdited ? "" : "display: none;",
            });
            restartIndicator.appendChild(
                createElement("span", {
                    className:
                        "pointer-events-none absolute left-full top-1/2 ml-2 w-max max-w-64 -translate-y-1/2 rounded-md border border-orange-300/60 bg-[#1f1f1f] px-2 py-1 text-xs leading-tight text-orange-100 opacity-0 shadow-lg transition-opacity duration-75 group-hover/restart:opacity-100 group-focus/restart:opacity-100",
                    text: "Restart required to apply config change",
                }),
            );
            inputContainer.appendChild(restartIndicator);
        }

        // Add Manage button for path parameters
        let manageButton = null;
        if (isPathParameter && operationName) {
            manageButton = createElement("button", {
                type: "button",
                className:
                    "px-3 py-2 bg-[#f9c845] text-[#232323] rounded-md hover:bg-[#d4a83a] transition-colors text-sm font-medium whitespace-nowrap",
                text: "Manage",
                onclick: () => {
                    if (globalThis.FileManagerPopup) {
                        let normalizedOpName = operationName.toLowerCase();
                        if (normalizedOpName.endsWith(".py")) {
                            normalizedOpName = normalizedOpName.slice(0, -3);
                        }
                        normalizedOpName = normalizedOpName.replace(
                            /\s+/g,
                            "_",
                        );
                        globalThis.FileManagerPopup.open(
                            normalizedOpName,
                            name,
                            currentValue,
                            (selectedFile) => {
                                if (
                                    selectedFile &&
                                    globalThis.refreshPathDropdown
                                ) {
                                    globalThis.refreshPathDropdown(
                                        selectedFile,
                                    );
                                }
                            },
                        );
                    } else {
                        console.error("FileManagerPopup not available");
                    }
                },
            });
            inputContainer.appendChild(manageButton);
        }

        // Create edited indicator (yellow circle) positioned next to input
        const editedIndicator = createElement("div", {
            className:
                "absolute -left-1 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-yellow-400 rounded-full",
            title: "This field has been modified from its default value",
            style: isEdited && !def.restart_for_change ? "" : "display: none;",
        });
        inputContainer.appendChild(editedIndicator);

        // Add event listener to detect real-time changes
        const updateIndicator = () => {
            let currentVal;
            if (input.tagName.toLowerCase() === "select") {
                currentVal = input.value;
            } else if (input.type === "checkbox") {
                currentVal = input.checked;
            } else if (def.type === "int") {
                currentVal = Number.parseInt(input.value, 10);
            } else if (def.type === "float") {
                currentVal = Number.parseFloat(input.value);
            } else {
                currentVal = input.value;
            }

            const edited =
                JSON.stringify(currentVal) !== JSON.stringify(originalValue);
            editedIndicator.style.display =
                edited && !def.restart_for_change ? "block" : "none";
            if (restartIndicator) {
                restartIndicator.style.display = edited
                    ? "inline-flex"
                    : "none";
            }
        };

        input.addEventListener("input", updateIndicator);
        input.addEventListener("change", updateIndicator);

        const wrapper = createElement("div", { className: "mb-4" }, [
            labelRow,
            inputContainer,
        ]);
        return {
            wrapper,
            getValue: () => {
                if (input.tagName.toLowerCase() === "select")
                    return input.value;
                if (input.type === "checkbox") return input.checked;
                if (def.type === "int") return Number.parseInt(input.value, 10);
                if (def.type === "float") return Number.parseFloat(input.value);
                return input.value;
            },
        };
    }

    /**
     * Renders the full settings form for an operation.
     */
    function renderForm(
        modalBody,
        config,
        initialValues,
        originalValues,
        onSave,
        operationName = null,
    ) {
        modalBody.innerHTML = "";
        const fields = [];

        renderPortGroupsSummary(modalBody, config);

        const params = config?.parameters || {};
        // Object.keys() preserves insertion order in ES2015+
        // This maintains the parameter order as defined in the config JSON file
        Object.keys(params).forEach((key) => {
            const field = buildField(
                key,
                params[key],
                initialValues || {},
                originalValues || {},
                operationName,
                "",
            );
            fields.push({ name: key, ...field });
            modalBody.appendChild(field.wrapper);
        });

        // Set up auto-save event listeners
        setupAutoSaveListeners(
            modalBody,
            fields,
            originalValues,
            onSave,
            config,
        );

        return () => {
            const result = {};
            fields.forEach((f) => {
                result[f.name] = f.getValue();
            });
            return result;
        };
    }

    /**
     * Attaches listeners that keep form state in sync for auto-save.
     */
    function setupAutoSaveListeners(
        modalBody,
        fields,
        originalValues,
        onSave,
        config,
    ) {
        console.log(
            "Setting up auto-save listeners for",
            fields.length,
            "fields",
        );

        // Function to check if restart is required
        const checkIfRestartRequired = (
            currentValues,
            originalValues,
            config,
        ) => {
            if (!config?.parameters) return false;

            for (const field of fields) {
                const paramConfig = config.parameters?.[field.name];
                if (paramConfig?.restart_for_change) {
                    const currentValue = currentValues[field.name];
                    const originalValue = originalValues[field.name];
                    if (
                        JSON.stringify(currentValue) !==
                        JSON.stringify(originalValue)
                    ) {
                        console.log(
                            `Field ${field.name} requires restart (changed from ${originalValue} to ${currentValue})`,
                        );
                        return true;
                    }
                }
            }
            return false;
        };

        // Function to trigger auto-save
        const triggerAutoSave = () => {
            console.log("[SETTINGS] Auto-save triggered", {
                operationName: config?.operationName || "unknown",
                timestamp: new Date().toISOString(),
            });
            // Get current form values and call onSave callback if provided
            const currentValues = {};
            fields.forEach((field) => {
                currentValues[field.name] = field.getValue();
            });
            console.log(
                "[SETTINGS] Current form values during auto-save:",
                currentValues,
            );

            // Check if restart is required
            const requiresRestart = checkIfRestartRequired(
                currentValues,
                originalValues,
                config,
            );
            currentValues._requiresRestart = requiresRestart;

            if (typeof onSave === "function") {
                // Mark this as an auto-save call
                currentValues._isAutoSave = true;
                console.log("Calling onSave with _isAutoSave=true");
                console.log(
                    "globalThis.pipelineCreator exists:",
                    !!globalThis.pipelineCreator,
                );
                if (globalThis.pipelineCreator) {
                    console.log(
                        "globalThis.pipelineCreator.autoSavePipeline exists:",
                        !!globalThis.pipelineCreator.autoSavePipeline,
                    );
                }
                onSave(currentValues);
            } else {
                console.log("onSave function not provided");
            }
        };

        const processedInputs = new WeakSet();

        // Set up event listeners for a single input
        const setupInputListener = (input) => {
            if (processedInputs.has(input)) {
                return;
            }
            processedInputs.add(input);

            if (input.type === "range") {
                input.addEventListener("change", () => {
                    triggerAutoSave();
                });
            } else if (
                input.tagName.toLowerCase() !== "select" &&
                input.type !== "checkbox"
            ) {
                // Text inputs and number inputs: Enter key and blur
                input.addEventListener("keydown", (e) => {
                    if (e.key === "Enter") {
                        e.preventDefault();
                        triggerAutoSave();
                    }
                });

                input.addEventListener("blur", () => {
                    triggerAutoSave();
                });
            } else {
                // Select elements and checkboxes: change event
                input.addEventListener("change", () => {
                    triggerAutoSave();
                });
            }
        };

        // Set up event listeners for all existing inputs
        const allInputs = modalBody.querySelectorAll("input, select");
        allInputs.forEach((input) => {
            setupInputListener(input);
        });

        // Set up MutationObserver to handle dynamically added inputs
        const handleAddedNode = (node) => {
            if (node.nodeType !== Node.ELEMENT_NODE) {
                return;
            }

            const inputs = node.querySelectorAll
                ? node.querySelectorAll("input, select")
                : [];
            for (const input of inputs) {
                setupInputListener(input);
            }
            if (node.tagName === "INPUT" || node.tagName === "SELECT") {
                setupInputListener(node);
            }
        };

        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    handleAddedNode(node);
                }
            }
        });

        observer.observe(modalBody, {
            childList: true,
            subtree: true,
        });

        // Store observer for cleanup
        modalBody._autoSaveObserver = observer;
    }

    /**
     * Locates the popup overlay and modal elements in the DOM.
     */
    function findOverlayElements() {
        const overlay = document.getElementById(OVERLAY_ID);
        const modal = document.getElementById(MODAL_ID);
        const liveViewPanel = document.getElementById("operationLiveViewPanel");
        const liveViewCloseBtn = document.getElementById(
            "operationLiveCloseButton",
        );
        const settingsContent = document.getElementById(
            "operationSettingsContent",
        );
        return {
            overlay,
            modal,
            liveViewPanel,
            liveViewCloseBtn,
            settingsContent,
        };
    }

    /**
     * Shows or hides the settings panel container.
     */
    function setSettingsPanelVisibility(showSettings) {
        const { modal, liveViewPanel, liveViewCloseBtn, settingsContent } =
            findOverlayElements();

        if (modal) {
            modal.style.display = showSettings ? "" : "none";
        }
        if (settingsContent) {
            settingsContent.classList.toggle(
                "single-panel-layout",
                !showSettings,
            );
        }
        if (liveViewPanel) {
            liveViewPanel.classList.toggle(
                "live-view-fullscreen",
                !showSettings,
            );
        }
        if (liveViewCloseBtn) {
            liveViewCloseBtn.classList.toggle("hidden", showSettings);
        }
    }

    /**
     * Applies the popup title to the modal header.
     */
    function applyTitle(modal, title) {
        const titleEl = modal.querySelector("[data-role='modal-title']");
        if (titleEl) titleEl.textContent = title || "Operation Settings";
    }

    /**
     * Refreshes the live visualization for the selected operation.
     */
    function updateLiveView(operationName, isSecondary = false) {
        const { liveViewPanel } = findOverlayElements();
        if (!liveViewPanel) return;

        const liveViewContainer = liveViewPanel.querySelector(
            "[data-role='live-view-container']",
        );
        if (!liveViewContainer) return;

        // Update the live view title to include operation name
        const titleEl = liveViewPanel.querySelector("h3");
        if (titleEl) {
            titleEl.textContent = `${operationName} - Live View`;
        }
    }

    /**
     * Checks whether a live visualization is available for an operation.
     */
    function isVisualizationAvailable(operationName, operationId) {
        const operations = globalThis.pipelineCreator?.getOperations?.() || [];
        if (operations.length === 0) {
            return true;
        }
        if (operationId) {
            const directMatch = operations.find((op) => op.id === operationId);
            if (directMatch) {
                return Boolean(directMatch.hasVisualization);
            }
        }
        if (!operationName) {
            return true;
        }
        const normalizedName = String(operationName || "")
            .replace(/\.py$/i, "")
            .toLowerCase();
        const match = operations.find((op) =>
            String(op.id || "")
                .replace(/\.py$/i, "")
                .toLowerCase()
                .includes(normalizedName),
        );
        return match ? Boolean(match.hasVisualization) : true;
    }

    /**
     * Removes the currently displayed live visualization image.
     */
    function removeCurrentLiveImage() {
        const { liveViewPanel } = findOverlayElements();
        if (!liveViewPanel) return;

        const liveViewContainer = liveViewPanel.querySelector(
            "[data-role='live-view-container']",
        );
        if (!liveViewContainer) return;

        const imgEl = liveViewContainer.querySelector("#operationLiveImage");
        if (imgEl) {
            imgEl.remove();
        }
    }

    /**
     * Displays a visualization error message in the popup.
     */
    function showVisualizationErrorMessage(
        message = "Error getting visualization",
    ) {
        removeCurrentLiveImage();
        const { liveViewPanel } = findOverlayElements();
        if (!liveViewPanel) return;

        const liveViewContainer = liveViewPanel.querySelector(
            "[data-role='live-view-container']",
        );
        if (!liveViewContainer) return;

        // Override parent container's flex centering to allow full width while maintaining vertical centering
        liveViewContainer.style.justifyContent = "stretch";
        liveViewContainer.style.alignItems = "center";

        const contentWrapper = liveViewContainer.querySelector(".text-center");

        // Create or update error message element as a styled box
        let errorMsgEl = liveViewContainer.querySelector(
            "#visualizationErrorMessage",
        );
        if (!errorMsgEl) {
            errorMsgEl = createElement("div", {
                id: "visualizationErrorMessage",
                className:
                    "block w-full bg-red-900/20 border-2 border-red-500/50 rounded-xl shadow-lg p-6 text-center text-red-400 text-lg font-medium my-8",
            });
            if (contentWrapper) {
                contentWrapper.appendChild(errorMsgEl);
            } else {
                liveViewContainer.appendChild(errorMsgEl);
            }
        }
        errorMsgEl.textContent = message;
        errorMsgEl.style.display = "block";
        errorMsgEl.style.width = "100%";
        errorMsgEl.style.position = "relative";
        errorMsgEl.classList.remove("hidden");
    }

    /**
     * Initializes the settings popup singleton and DOM wiring.
     */
    function init() {
        let { overlay } = findOverlayElements();

        const closeButtons = overlay.querySelectorAll("[data-action='close']");
        closeButtons.forEach((btn) =>
            btn.addEventListener("click", () => close()),
        );
        overlay.addEventListener("click", (e) => {
            if (e.target === overlay) close();
        });
    }

    /**
     * Opens the settings popup for the provided operation.
     */
    function open({
        title,
        operationName,
        operationId,
        operationUuid,
        isSecondary,
        initialValues,
        onSave,
    }) {
        console.log("[SETTINGS] Opening settings popup", {
            operationName,
            isSecondary,
            title,
            initialValuesKeys: Object.keys(initialValues || {}),
            timestamp: new Date().toISOString(),
        });

        const { overlay, modal } = findOverlayElements();
        if (!overlay || !modal) return;

        applyTitle(modal, title);
        updateLiveView(operationName, isSecondary);
        const body = modal.querySelector("[data-role='modal-body']");
        setSettingsPanelVisibility(true);

        // Show loading state
        body.innerHTML =
            '<div class="text-center text-[#f9c845] py-8">Loading configuration...</div>';

        // Determine pipeline from pipeline builder dropdown
        let selectedPipelineName = null;
        try {
            const pipelineSelectEl = document.getElementById("pipelineSelect");
            if (pipelineSelectEl?.value) {
                selectedPipelineName = pipelineSelectEl.value;
            }
        } catch (err) {
            console.warn("Could not read pipeline selection:", err);
        }

        // Start visualization on backend if pipeline is available
        const startVisIfReady = async () => {
            removeCurrentLiveImage();
            if (!selectedPipelineName) {
                console.log(
                    "[SETTINGS] Skipping visualization - missing pipeline",
                    {
                        selectedPipelineName,
                    },
                );
                return;
            }
            if (!isVisualizationAvailable(operationName, operationId)) {
                showVisualizationErrorMessage("Operation has no visualization");
                return;
            }
            try {
                console.log("[SETTINGS] Starting visualization", {
                    pipeline: selectedPipelineName,
                    operationUuid,
                    timestamp: new Date().toISOString(),
                });
                const startResponse = await fetch(
                    `${BACKEND_BASE_URL}/start-visualize/${encodeURIComponent(selectedPipelineName)}/${encodeURIComponent(operationUuid)}`,
                    {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                    },
                );

                if (startResponse.status !== 200) {
                    console.warn(
                        "[SETTINGS] start_visualize returned non-200 status:",
                        startResponse.status,
                    );
                    showVisualizationErrorMessage(
                        "Error getting visualization",
                    );
                    return;
                }

                _currentVisPipeline = selectedPipelineName;
                _currentVisAction = operationUuid;

                // Ensure an img element exists and hide placeholder
                const liveViewPanelEl = document.getElementById(
                    "operationLiveViewPanel",
                );
                const liveViewContainer = liveViewPanelEl.querySelector(
                    "[data-role='live-view-container']",
                );

                const contentWrapper =
                    liveViewContainer.querySelector(".text-center");

                let imgEl = liveViewContainer.querySelector(
                    "#operationLiveImage",
                );
                if (!imgEl) {
                    imgEl = document.createElement("img");
                    imgEl.id = "operationLiveImage";
                    imgEl.alt = "Live visualization";
                    imgEl.className =
                        "mx-auto mt-4 rounded-lg max-w-full max-h-[60vh]";
                    if (contentWrapper) {
                        contentWrapper.appendChild(imgEl);
                    } else {
                        liveViewContainer.appendChild(imgEl);
                    }
                }

                const errorMsg = liveViewContainer.querySelector(
                    "#visualizationErrorMessage",
                );
                if (errorMsg) errorMsg.remove();

                imgEl.classList.remove("hidden");
                imgEl.style.display = "block";

                if (_visInterval) {
                    clearInterval(_visInterval);
                    _visInterval = null;
                }
                const streamUrl = `${BACKEND_BASE_URL}/visualize/stream/${encodeURIComponent(_currentVisPipeline)}`;
                imgEl.src = streamUrl;
            } catch (err) {
                console.warn("Failed to start visualization:", err);
                showVisualizationErrorMessage("Error getting visualization");
            }
        };

        // Fetch config data from server
        fetchConfigData(operationName, isSecondary)
            .then(async (config) => {
                if (!config) {
                    body.innerHTML =
                        '<div class="text-center text-red-400 py-8">Failed to load configuration</div>';
                    // Wait for visualization to be set up before returning
                    await startVisIfReady();
                    return;
                }

                const hasSettings = operationHasSettings(config);
                setSettingsPanelVisibility(hasSettings);

                if (!hasSettings) {
                    body.innerHTML =
                        '<div class="text-center text-[#f9c845] py-8">This operation has no configurable settings.</div>';
                } else {
                    // Use the loaded values as baseline for comparison, not defaults
                    const originalValues = { ...initialValues };

                    const getValues = renderForm(
                        body,
                        config,
                        initialValues,
                        originalValues,
                        onSave,
                        operationName,
                    );

                    const saveBtn = modal.querySelector("[data-action='save']");
                    const cancelBtn = modal.querySelector(
                        "[data-action='cancel']",
                    );

                    if (saveBtn) {
                        saveBtn.onclick = () => {
                            const values = getValues();
                            console.log(
                                "[SETTINGS] Saving operation settings",
                                {
                                    operationName,
                                    isSecondary,
                                    savedValues: values,
                                    timestamp: new Date().toISOString(),
                                },
                            );
                            if (typeof onSave === "function") onSave(values);
                            console.log(
                                "[SETTINGS] Settings saved, closing popup",
                                {
                                    operationName,
                                    timestamp: new Date().toISOString(),
                                },
                            );
                            close();
                        };
                    }
                    if (cancelBtn) cancelBtn.onclick = () => close();
                }

                // Start visualization now that modal content is ready - wait for it to complete
                await startVisIfReady();
            })
            .catch(async (error) => {
                console.error("Error loading config:", error);
                body.innerHTML =
                    '<div class="text-center text-red-400 py-8">Error loading configuration</div>';
                // Wait for visualization to be set up
                await startVisIfReady();
            });

        overlay.classList.remove("hidden");
    }

    /**
     * Stops any active live visualization polling.
     */
    function stopVisualizationIfActive() {
        if (!_currentVisPipeline) {
            console.log("[SETTINGS] No active visualization to stop");
            return;
        }
        console.log("[SETTINGS] Stopping active visualization", {
            pipeline: _currentVisPipeline,
            action: _currentVisAction,
            timestamp: new Date().toISOString(),
        });
        try {
            fetch(
                `${BACKEND_BASE_URL}/stop-visualize/${encodeURIComponent(_currentVisPipeline)}`,
                { method: "POST" },
            ).catch((err) =>
                console.warn("Failed to stop visualization:", err),
            );
        } finally {
            _currentVisPipeline = null;
            _currentVisAction = null;
            console.log("[SETTINGS] Visualization state cleared");
        }
    }

    /**
     * Closes the settings popup and clears transient state.
     */
    function close() {
        console.log("[SETTINGS] Closing settings popup", {
            wasVisualizing: !!_visInterval,
            hadVisualization: !!_currentVisObjectUrl,
            timestamp: new Date().toISOString(),
        });

        const { overlay } = findOverlayElements();
        if (!overlay) return;

        // Stop polling and visualization
        if (_visInterval) {
            clearInterval(_visInterval);
            _visInterval = null;
            console.log("[SETTINGS] Stopped visualization polling");
        }
        if (_currentVisObjectUrl) {
            try {
                URL.revokeObjectURL(_currentVisObjectUrl);
                console.log("[SETTINGS] Revoked visualization object URL");
            } catch (e) {
                console.warn("Failed to revoke object URL:", e);
            }
            _currentVisObjectUrl = null;
        }
        const liveViewPanelEl = document.getElementById(
            "operationLiveViewPanel",
        );
        if (liveViewPanelEl) {
            const imgEl = liveViewPanelEl.querySelector("#operationLiveImage");
            if (imgEl) {
                imgEl.removeAttribute("src");
            }
        }
        stopVisualizationIfActive();
        removeCurrentLiveImage();

        // Show placeholder content again and hide image
        if (liveViewPanelEl) {
            const liveViewContainer = liveViewPanelEl.querySelector(
                "[data-role='live-view-container']",
            );
            if (liveViewContainer) {
                // Reset container flex styles to original state
                liveViewContainer.style.justifyContent = "";
                liveViewContainer.style.alignItems = "";

                const imgEl = liveViewContainer.querySelector(
                    "#operationLiveImage",
                );

                if (imgEl) imgEl.classList.add("hidden");

                // Hide error message
                const errorMsg = liveViewContainer.querySelector(
                    "#visualizationErrorMessage",
                );
                if (errorMsg) {
                    errorMsg.style.display = "none";
                    errorMsg.classList.add("hidden");
                }

                // Reset the title
                const titleEl = liveViewPanelEl.querySelector("h3");
                if (titleEl) {
                    titleEl.textContent = "Live View";
                }
            }
        }

        setSettingsPanelVisibility(true);
        overlay.classList.add("hidden");
    }

    /**
     * Fetches configuration data for the requested operation.
     */
    function fetchConfigData(operationName, isSecondary = false) {
        if (!operationName) {
            return Promise.reject(new Error("Operation name is required"));
        }

        // Convert boolean to integer for URL (0 = false, 1 = true)
        const isSecondaryInt = isSecondary ? 1 : 0;

        return fetch(
            `${BACKEND_BASE_URL}/get-operation-config-data/${encodeURIComponent(operationName)}/${isSecondaryInt}`,
        )
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then((data) => {
                // Ensure the data has the expected structure
                if (!data || typeof data !== "object") {
                    throw new Error("Invalid config data structure");
                }
                return data;
            })
            .catch((error) => {
                console.error("Error fetching config data:", error);
                return null;
            });
    }

    const popup = {
        init,
        open,
        close,
    };
    globalThis.SettingsPopup = popup;

    // Auto-init when DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    return popup;
}

registerSettingsPopup();
