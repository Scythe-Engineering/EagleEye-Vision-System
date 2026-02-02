import { BACKEND_BASE_URL } from "../config.js";

(function () {
    const OVERLAY_ID = "operationSettingsOverlay";
    const MODAL_ID = "operationSettingsModal";
    // Visualization state
    let _visInterval = null;
    let _currentVisObjectUrl = null;
    let _currentVisPipeline = null;
    let _currentVisAction = null;
    let _availableCameras = null;
    let _availableCamerasLoading = null;

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
                const cameras = Object.entries(data).map(([name]) => name);
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

    function notifyCameraListUpdated(cameras) {
        if (globalThis.pipelineCreator?.getAvailableCameras && cameras.length) {
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
            const h = Math.max(0, Math.min(179, parseInt(hInput.value) || 0));
            const s = Math.max(0, Math.min(255, parseInt(sInput.value) || 0));
            const v = Math.max(0, Math.min(255, parseInt(vInput.value) || 0));
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
                parseInt(hInput.value) || 0,
                parseInt(sInput.value) || 0,
                parseInt(vInput.value) || 0,
            ],
        };
    }

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

            removeButtons.forEach(btn => {
                if (currentCount <= minItems) {
                    btn.style.display = 'none';
                } else {
                    btn.style.display = '';
                }
            });

            if (addBtn) {
                if (maxItems !== undefined && currentCount >= maxItems) {
                    addBtn.style.display = 'none';
                } else {
                    addBtn.style.display = '';
                }
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
            if (def.item_labels && Array.isArray(def.item_labels) && def.item_labels[index]) {
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
                        subFields.forEach((f) => {
                            result[f.name] = f.getValue();
                        });
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
                            return parseInt(itemInput.value, 10) || 0;
                        } else if (def.item_type === "float") {
                            return parseFloat(itemInput.value) || 0;
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
            );
        }

        const normalizedOperationName = String(operationName || "")
            .replace(/\.py$/i, "")
            .toLowerCase()
            .replace(/\s+/g, "_");
        const isDeviceInputCameraName =
            normalizedOperationName === "device_input" &&
            name === "camera_name" &&
            def.type === "str";

        if (isDeviceInputCameraName) {
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

            const updateOptions = (cameraNames) => {
                const selectedValue =
                    input.value ||
                    (currentValue !== undefined && currentValue !== null
                        ? String(currentValue)
                        : "");
                input.innerHTML = "";

                if (!Array.isArray(cameraNames) || cameraNames.length === 0) {
                    const customOption = createElement("option", {
                        value: selectedValue,
                        text: selectedValue
                            ? `${selectedValue} (custom)`
                            : "No cameras available",
                    });
                    customOption.selected = true;
                    input.appendChild(customOption);
                    return;
                }

                const normalizedNames = cameraNames.filter(Boolean);
                normalizedNames.forEach((cameraName) => {
                    const optEl = createElement("option", {
                        value: cameraName,
                        text: cameraName,
                    });
                    if (cameraName === selectedValue) {
                        optEl.selected = true;
                    }
                    input.appendChild(optEl);
                });

                if (
                    selectedValue &&
                    !normalizedNames.some((name) => name === selectedValue)
                ) {
                    const customOption = createElement("option", {
                        value: selectedValue,
                        text: `${selectedValue} (custom)`,
                        selected: true,
                    });
                    input.appendChild(customOption);
                }
            };

            const pipelineCameras =
                globalThis.pipelineCreator?.getAvailableCameras?.() || [];
            const pipelineNames = pipelineCameras.map((camera) => camera.name);
            if (pipelineNames.length > 0) {
                updateOptions(pipelineNames);
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
            let inputType = "text";
            if (type === "int" || type === "float") inputType = "number";
            if (type === "str") inputType = "text";
            if (type === "bool") inputType = "checkbox";

            const attrs = {
                id: fieldId,
                type: inputType,
                className:
                    inputType === "checkbox"
                        ? "h-4 w-4 text-[#f9c845] focus:ring-[#f9c845] border-[#414141] rounded bg-[#232323]"
                        : "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]",
            };
            if (inputType === "number") {
                if (typeof def.min === "number") attrs.min = String(def.min);
                if (typeof def.max === "number") attrs.max = String(def.max);
                if (type === "int") attrs.step = "1";
                if (type === "float") attrs.step = "any";
            }
            input = createElement("input", attrs);
            if (inputType === "checkbox") {
                input.checked = Boolean(currentValue);
            } else if (currentValue !== undefined && currentValue !== null) {
                input.value = String(currentValue);
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
            style: isEdited ? "" : "display: none;",
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
            editedIndicator.style.display = edited ? "block" : "none";
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

                // Update restart indicator if available
                if (globalThis.pipelineCreator?.updateRestartIndicator) {
                    const requiresRestart = checkIfRestartRequired(
                        currentValues,
                        originalValues,
                        config,
                    );
                    globalThis.pipelineCreator.updateRestartIndicator(
                        requiresRestart,
                    );
                }
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

            if (
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
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        const inputs = node.querySelectorAll
                            ? node.querySelectorAll("input, select")
                            : [];
                        inputs.forEach((input) => {
                            setupInputListener(input);
                        });
                        if (
                            node.tagName === "INPUT" ||
                            node.tagName === "SELECT"
                        ) {
                            setupInputListener(node);
                        }
                    }
                });
            });
        });

        observer.observe(modalBody, {
            childList: true,
            subtree: true,
        });

        // Store observer for cleanup
        modalBody._autoSaveObserver = observer;
    }

    function findOverlayElements() {
        const overlay = document.getElementById(OVERLAY_ID);
        const modal = document.getElementById(MODAL_ID);
        const liveViewPanel = document.getElementById("operationLiveViewPanel");
        return { overlay, modal, liveViewPanel };
    }

    function applyTitle(modal, title) {
        const titleEl = modal.querySelector("[data-role='modal-title']");
        if (titleEl) titleEl.textContent = title || "Operation Settings";
    }

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

        // For now, keep the placeholder content
        // Future enhancement: Add actual live visualization logic here
        const placeholderContent =
            liveViewContainer.querySelector(".text-center");
        if (placeholderContent) {
            const operationText = placeholderContent.querySelector("p.text-lg");
            if (operationText) {
                operationText.textContent = `${operationName} Live Preview`;
            }
        }
    }

    function showVisualizationErrorMessage(
        message = "Error getting visualization",
    ) {
        const { liveViewPanel } = findOverlayElements();
        if (!liveViewPanel) return;

        const liveViewContainer = liveViewPanel.querySelector(
            "[data-role='live-view-container']",
        );
        if (!liveViewContainer) return;

        // Override parent container's flex centering to allow full width while maintaining vertical centering
        liveViewContainer.style.justifyContent = "stretch";
        liveViewContainer.style.alignItems = "center";

        // Find the main content wrapper (the .text-center div)
        const contentWrapper = liveViewContainer.querySelector(".text-center");

        // Store references to elements we need to restore later
        if (!liveViewContainer._storedElements && contentWrapper) {
            liveViewContainer._storedElements = {
                contentWrapper: contentWrapper,
                placeholder: contentWrapper.querySelector(
                    "[data-role='live-view-placeholder']",
                ),
                textElements: Array.from(contentWrapper.querySelectorAll("p")),
                noVisMessage: contentWrapper.querySelector(
                    "#noVisualizationMessage",
                ),
                imgEl: contentWrapper.querySelector("#operationLiveImage"),
                parentContainer: liveViewContainer,
            };
        }

        // Remove content wrapper from DOM to ensure error message takes full width
        if (contentWrapper) {
            contentWrapper.remove();
        }

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
            liveViewContainer.appendChild(errorMsgEl);
        }
        errorMsgEl.textContent = message;
        errorMsgEl.style.display = "block";
        errorMsgEl.style.width = "100%";
        errorMsgEl.style.position = "relative";
        errorMsgEl.classList.remove("hidden");
    }

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

    function open({
        title,
        operationName,
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

        // Compute action name for visualize API (normalize similar to backend expectations)
        const computeActionName = (name) => {
            let result = String(name || "");
            // Remove .py extension (case-insensitive)
            if (result.toLowerCase().endsWith(".py")) {
                result = result.slice(0, -3);
            }
            return result.toLowerCase().replace(/\s+/g, "_");
        };
        const actionNameForApi = computeActionName(operationName || "");

        // Start visualization on backend if pipeline is available
        const startVisIfReady = async () => {
            if (!selectedPipelineName) {
                console.log(
                    "[SETTINGS] Skipping visualization - missing pipeline",
                    {
                        selectedPipelineName,
                    },
                );
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

                // Find or restore content wrapper
                let contentWrapper =
                    liveViewContainer.querySelector(".text-center");
                if (
                    !contentWrapper &&
                    liveViewContainer._storedElements?.contentWrapper
                ) {
                    contentWrapper =
                        liveViewContainer._storedElements.contentWrapper;
                    liveViewContainer.appendChild(contentWrapper);
                }

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

                // Remove placeholder content completely when showing live image
                const placeholderEl = liveViewContainer.querySelector(
                    "[data-role='live-view-placeholder']",
                );
                const textElements = Array.from(
                    liveViewContainer.querySelectorAll("p"),
                );
                const noVisMessage = liveViewContainer.querySelector(
                    "#noVisualizationMessage",
                );
                const errorMsg = liveViewContainer.querySelector(
                    "#visualizationErrorMessage",
                );

                // Store elements for potential restoration if visualization fails
                if (!liveViewContainer._storedElements && contentWrapper) {
                    liveViewContainer._storedElements = {
                        contentWrapper: contentWrapper,
                        placeholder: placeholderEl,
                        textElements: textElements,
                        noVisMessage: noVisMessage,
                        imgEl: imgEl,
                        parentContainer: liveViewContainer,
                    };
                }

                // Remove elements from DOM
                if (placeholderEl) placeholderEl.remove();
                textElements.forEach((p) => p.remove());
                if (noVisMessage) noVisMessage.remove();
                if (errorMsg) errorMsg.remove();

                imgEl.classList.remove("hidden");
                imgEl.style.display = "block";

                // Start polling at 10Hz (every 100ms)
                if (_visInterval) clearInterval(_visInterval);
                let hasError = false;
                _visInterval = setInterval(async () => {
                    if (hasError) {
                        return;
                    }
                    try {
                        const url = `${BACKEND_BASE_URL}/visualize/${encodeURIComponent(_currentVisPipeline)}`;
                        const response = await fetch(url, {
                            cache: "no-store",
                        });

                        if (!response.ok) {
                            hasError = true;
                            if (_visInterval) {
                                clearInterval(_visInterval);
                                _visInterval = null;
                            }
                            showVisualizationErrorMessage(
                                "Error getting visualization",
                            );
                            return;
                        }

                        const blob = await response.blob();
                        const objectUrl = URL.createObjectURL(blob);
                        // Update image src and revoke previous object URL
                        if (_currentVisObjectUrl)
                            URL.revokeObjectURL(_currentVisObjectUrl);
                        _currentVisObjectUrl = objectUrl;
                        imgEl.src = objectUrl;
                    } catch (err) {
                        hasError = true;
                        if (_visInterval) {
                            clearInterval(_visInterval);
                            _visInterval = null;
                        }
                        console.warn(
                            "[SETTINGS] Error processing visualization frame:",
                            err,
                        );
                    }
                }, 100);
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
                const cancelBtn = modal.querySelector("[data-action='cancel']");

                if (saveBtn) {
                    saveBtn.onclick = () => {
                        const values = getValues();
                        console.log("[SETTINGS] Saving operation settings", {
                            operationName,
                            isSecondary,
                            savedValues: values,
                            timestamp: new Date().toISOString(),
                        });
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
        stopVisualizationIfActive();

        // Show placeholder content again and hide image
        const liveViewPanelEl = document.getElementById(
            "operationLiveViewPanel",
        );
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

                // Restore previously removed elements back to DOM
                const storedElements = liveViewContainer._storedElements;
                if (storedElements) {
                    const contentWrapper = storedElements.contentWrapper;
                    if (contentWrapper) {
                        liveViewContainer.appendChild(contentWrapper);

                        // Restore placeholder elements into the content wrapper
                        if (storedElements.placeholder) {
                            contentWrapper.appendChild(
                                storedElements.placeholder,
                            );
                        }
                        if (storedElements.textElements) {
                            storedElements.textElements.forEach((p) => {
                                contentWrapper.appendChild(p);
                            });
                        }
                        if (storedElements.noVisMessage) {
                            contentWrapper.appendChild(
                                storedElements.noVisMessage,
                            );
                            storedElements.noVisMessage.style.display = "none";
                            storedElements.noVisMessage.classList.add("hidden");
                        }
                        if (storedElements.imgEl) {
                            contentWrapper.appendChild(storedElements.imgEl);
                            storedElements.imgEl.classList.add("hidden");
                            storedElements.imgEl.style.display = "none";
                        }
                    }

                    // Clear stored references
                    delete liveViewContainer._storedElements;
                }

                // Hide error message
                const errorMsg = liveViewContainer.querySelector(
                    "#visualizationErrorMessage",
                );
                if (errorMsg) {
                    errorMsg.style.display = "none";
                    errorMsg.classList.add("hidden");
                }

                // Reset the text back to default
                const operationText =
                    liveViewContainer.querySelector("p.text-lg");
                const descriptionText =
                    liveViewContainer.querySelector("p.text-sm");

                if (operationText) {
                    operationText.textContent = "Live Preview";
                }
                if (descriptionText) {
                    descriptionText.textContent =
                        "Visualizer will appear here when operation is active";
                }

                // Reset the title
                const titleEl = liveViewPanelEl.querySelector("h3");
                if (titleEl) {
                    titleEl.textContent = "Live View";
                }
            }
        }

        overlay.classList.add("hidden");
    }

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

    globalThis.SettingsPopup = {
        init,
        open,
        close,
    };

    // Auto-init when DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
