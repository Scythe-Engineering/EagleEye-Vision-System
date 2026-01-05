import { B as BACKEND_BASE_URL } from "./bundle3.js";
(function polyfill() {
  const relList = document.createElement("link").relList;
  if (relList && relList.supports && relList.supports("modulepreload")) {
    return;
  }
  for (const link of document.querySelectorAll('link[rel="modulepreload"]')) {
    processPreload(link);
  }
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type !== "childList") {
        continue;
      }
      for (const node of mutation.addedNodes) {
        if (node.tagName === "LINK" && node.rel === "modulepreload")
          processPreload(node);
      }
    }
  }).observe(document, { childList: true, subtree: true });
  function getFetchOpts(link) {
    const fetchOpts = {};
    if (link.integrity) fetchOpts.integrity = link.integrity;
    if (link.referrerPolicy) fetchOpts.referrerPolicy = link.referrerPolicy;
    if (link.crossOrigin === "use-credentials")
      fetchOpts.credentials = "include";
    else if (link.crossOrigin === "anonymous") fetchOpts.credentials = "omit";
    else fetchOpts.credentials = "same-origin";
    return fetchOpts;
  }
  function processPreload(link) {
    if (link.ep)
      return;
    link.ep = true;
    const fetchOpts = getFetchOpts(link);
    fetch(link.href, fetchOpts);
  }
})();
(function() {
  const OVERLAY_ID = "operationSettingsOverlay";
  const MODAL_ID = "operationSettingsModal";
  let _visInterval = null;
  let _currentVisObjectUrl = null;
  let _currentVisCamera = null;
  let _currentVisPipeline = null;
  let _currentVisAction = null;
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
      } else if (v !== void 0 && v !== null) {
        el.setAttribute(k, String(v));
      }
    });
    (children || []).forEach((c) => el.appendChild(c));
    return el;
  }
  function buildField(name, def, currentValues, originalValues, operationName = null) {
    const fieldId = `setting-${name}`;
    const label = createElement("label", {
      for: fieldId,
      className: "block text-sm font-medium text-[#f9c845] mb-1",
      text: def.description || name
    });
    let input;
    const currentValue = currentValues && name in currentValues ? currentValues[name] : def.default;
    const originalValue = originalValues && name in originalValues ? originalValues[name] : def.default;
    const isEdited = JSON.stringify(currentValue) !== JSON.stringify(originalValue);
    const isPathParameter = name.endsWith("_path") && def.type === "str";
    if (isPathParameter && operationName) {
      input = createElement("select", {
        id: fieldId,
        className: "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]"
      });
      const customOption = createElement("option", {
        value: "",
        text: "Custom path..."
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
            `${BACKEND_BASE_URL}/get-operation-files/${encodeURIComponent(normalizedOpName)}/${encodeURIComponent(name)}`
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
              const fullPath = basePath ? `${basePath}/${filename}` : filename;
              const optEl = createElement("option", {
                value: fullPath,
                text: filename
              });
              const currentPathValue = currentValue || "";
              if (currentPathValue === fullPath || currentPathValue.endsWith(`/${filename}`) || currentPathValue === filename || currentSelectedValue === fullPath) {
                optEl.selected = true;
              }
              customOption.before(optEl);
            });
            if (currentValue && !data.files.some((f) => {
              const fullPath = basePath ? `${basePath}/${f}` : f;
              return currentValue === fullPath || currentValue.endsWith(`/${f}`) || currentValue === f;
            })) {
              const customValueOption = createElement("option", {
                value: currentValue,
                text: currentValue + " (custom)",
                selected: currentSelectedValue === currentValue
              });
              customOption.before(customValueOption);
            } else if (currentValue) {
              customOption.selected = false;
            } else if (!currentSelectedValue || currentSelectedValue === "") {
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
            currentValue || ""
          );
          if (customPath !== null && customPath !== "") {
            const existingOption = Array.from(input.options).find(
              (opt) => opt.value === customPath
            );
            if (existingOption) {
              input.value = customPath;
            } else {
              const newOption = createElement("option", {
                value: customPath,
                text: customPath + " (custom)",
                selected: true
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
            const fullPath = basePath ? `${basePath}/${selectedFilename}` : selectedFilename;
            const existingOption = Array.from(input.options).find(
              (opt) => opt.value === fullPath
            );
            if (existingOption) {
              input.value = fullPath;
            } else {
              const newOption = createElement("option", {
                value: fullPath,
                text: selectedFilename,
                selected: true
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
        className: "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]"
      });
      def.options.forEach((opt) => {
        const optEl = createElement("option", {
          value: String(opt),
          text: String(opt)
        });
        if (currentValue !== void 0 && String(currentValue) === String(opt))
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
        className: inputType === "checkbox" ? "h-4 w-4 text-[#f9c845] focus:ring-[#f9c845] border-[#414141] rounded bg-[#232323]" : "w-full bg-[#232323] border border-[#414141] text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#f9c845]"
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
      } else if (currentValue !== void 0 && currentValue !== null) {
        input.value = String(currentValue);
      }
    }
    const hint = createElement("div", {
      className: "text-xs text-[#ac8a2f] ml-auto",
      text: def.required ? "Required" : "Optional"
    });
    const labelRow = createElement(
      "div",
      {
        className: "flex items-center mb-2"
      },
      [label, hint]
    );
    const inputContainer = createElement("div", {
      className: "relative flex gap-2"
    });
    inputContainer.appendChild(input);
    let manageButton = null;
    if (isPathParameter && operationName) {
      manageButton = createElement("button", {
        type: "button",
        className: "px-3 py-2 bg-[#f9c845] text-[#232323] rounded-md hover:bg-[#d4a83a] transition-colors text-sm font-medium whitespace-nowrap",
        text: "Manage",
        onclick: () => {
          if (globalThis.FileManagerPopup) {
            let normalizedOpName = operationName.toLowerCase();
            if (normalizedOpName.endsWith(".py")) {
              normalizedOpName = normalizedOpName.slice(0, -3);
            }
            normalizedOpName = normalizedOpName.replace(
              /\s+/g,
              "_"
            );
            globalThis.FileManagerPopup.open(
              normalizedOpName,
              name,
              currentValue,
              (selectedFile) => {
                if (selectedFile && globalThis.refreshPathDropdown) {
                  globalThis.refreshPathDropdown(
                    selectedFile
                  );
                }
              }
            );
          } else {
            console.error("FileManagerPopup not available");
          }
        }
      });
      inputContainer.appendChild(manageButton);
    }
    const editedIndicator = createElement("div", {
      className: "absolute -left-1 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-yellow-400 rounded-full",
      title: "This field has been modified from its default value",
      style: isEdited ? "" : "display: none;"
    });
    inputContainer.appendChild(editedIndicator);
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
      const edited = JSON.stringify(currentVal) !== JSON.stringify(originalValue);
      editedIndicator.style.display = edited ? "block" : "none";
    };
    input.addEventListener("input", updateIndicator);
    input.addEventListener("change", updateIndicator);
    const wrapper = createElement("div", { className: "mb-4" }, [
      labelRow,
      inputContainer
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
      }
    };
  }
  function renderForm(modalBody, config, initialValues, originalValues, onSave, operationName = null) {
    modalBody.innerHTML = "";
    const fields = [];
    const params = (config == null ? void 0 : config.parameters) || {};
    Object.keys(params).forEach((key) => {
      const field = buildField(
        key,
        params[key],
        initialValues || {},
        originalValues || {},
        operationName
      );
      fields.push({ name: key, ...field });
      modalBody.appendChild(field.wrapper);
    });
    setupAutoSaveListeners(
      modalBody,
      fields,
      originalValues,
      onSave,
      config
    );
    return () => {
      const result = {};
      fields.forEach((f) => {
        result[f.name] = f.getValue();
      });
      return result;
    };
  }
  function setupAutoSaveListeners(modalBody, fields, originalValues, onSave, config) {
    console.log(
      "Setting up auto-save listeners for",
      fields.length,
      "fields"
    );
    const checkIfRestartRequired = (currentValues, originalValues2, config2) => {
      var _a;
      if (!(config2 == null ? void 0 : config2.parameters)) return false;
      for (const field of fields) {
        const paramConfig = (_a = config2.parameters) == null ? void 0 : _a[field.name];
        if (paramConfig == null ? void 0 : paramConfig.restart_for_change) {
          const currentValue = currentValues[field.name];
          const originalValue = originalValues2[field.name];
          if (JSON.stringify(currentValue) !== JSON.stringify(originalValue)) {
            console.log(
              `Field ${field.name} requires restart (changed from ${originalValue} to ${currentValue})`
            );
            return true;
          }
        }
      }
      return false;
    };
    const triggerAutoSave = () => {
      var _a;
      console.log("[SETTINGS] Auto-save triggered", {
        operationName: (config == null ? void 0 : config.operationName) || "unknown",
        timestamp: (/* @__PURE__ */ new Date()).toISOString()
      });
      const currentValues = {};
      fields.forEach((field) => {
        currentValues[field.name] = field.getValue();
      });
      console.log(
        "[SETTINGS] Current form values during auto-save:",
        currentValues
      );
      const requiresRestart = checkIfRestartRequired(
        currentValues,
        originalValues,
        config
      );
      currentValues._requiresRestart = requiresRestart;
      if (typeof onSave === "function") {
        currentValues._isAutoSave = true;
        console.log("Calling onSave with _isAutoSave=true");
        console.log(
          "globalThis.pipelineCreator exists:",
          !!globalThis.pipelineCreator
        );
        if (globalThis.pipelineCreator) {
          console.log(
            "globalThis.pipelineCreator.autoSavePipeline exists:",
            !!globalThis.pipelineCreator.autoSavePipeline
          );
        }
        onSave(currentValues);
        if ((_a = globalThis.pipelineCreator) == null ? void 0 : _a.updateRestartIndicator) {
          const requiresRestart2 = checkIfRestartRequired(
            currentValues,
            originalValues,
            config
          );
          globalThis.pipelineCreator.updateRestartIndicator(
            requiresRestart2
          );
        }
      } else {
        console.log("onSave function not provided");
      }
    };
    const allInputs = modalBody.querySelectorAll("input, select");
    allInputs.forEach((input) => {
      if (input.tagName.toLowerCase() !== "select" && input.type !== "checkbox") {
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
        input.addEventListener("change", () => {
          triggerAutoSave();
        });
      }
    });
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
      "[data-role='live-view-container']"
    );
    if (!liveViewContainer) return;
    const titleEl = liveViewPanel.querySelector("h3");
    if (titleEl) {
      titleEl.textContent = `${operationName} - Live View`;
    }
    const placeholderContent = liveViewContainer.querySelector(".text-center");
    if (placeholderContent) {
      const operationText = placeholderContent.querySelector("p.text-lg");
      if (operationText) {
        operationText.textContent = `${operationName} Live Preview`;
      }
    }
  }
  function showNoVisualizationMessage(operationName) {
    const { liveViewPanel } = findOverlayElements();
    if (!liveViewPanel) return;
    const liveViewContainer = liveViewPanel.querySelector(
      "[data-role='live-view-container']"
    );
    if (!liveViewContainer) return;
    const imgEl = liveViewContainer.querySelector("#operationLiveImage");
    if (imgEl) {
      imgEl.classList.add("hidden");
    }
    const placeholderEl = liveViewContainer.querySelector(
      "[data-role='live-view-placeholder']"
    );
    const textElements = liveViewContainer.querySelectorAll("p");
    if (placeholderEl) placeholderEl.style.display = "none";
    textElements.forEach((p) => p.style.display = "none");
    const noVisMessage = liveViewContainer.querySelector(
      "#noVisualizationMessage"
    );
    if (noVisMessage) {
      noVisMessage.classList.remove("hidden");
    }
  }
  function init() {
    let { overlay } = findOverlayElements();
    const closeButtons = overlay.querySelectorAll("[data-action='close']");
    closeButtons.forEach(
      (btn) => btn.addEventListener("click", () => close())
    );
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });
  }
  function open({
    title,
    operationName,
    isSecondary,
    initialValues,
    onSave
  }) {
    console.log("[SETTINGS] Opening settings popup", {
      operationName,
      isSecondary,
      title,
      initialValuesKeys: Object.keys(initialValues || {}),
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    });
    const { overlay, modal } = findOverlayElements();
    if (!overlay || !modal) return;
    applyTitle(modal, title);
    updateLiveView(operationName, isSecondary);
    const body = modal.querySelector("[data-role='modal-body']");
    body.innerHTML = '<div class="text-center text-[#f9c845] py-8">Loading configuration...</div>';
    let selectedCameraName = null;
    let selectedPipelineName = null;
    try {
      const cameraSelectEl = document.getElementById("cameraSelect");
      const pipelineSelectEl = document.getElementById("pipelineSelect");
      if (cameraSelectEl && cameraSelectEl.selectedIndex >= 0) {
        selectedCameraName = cameraSelectEl.options[cameraSelectEl.selectedIndex].textContent;
      }
      if (pipelineSelectEl == null ? void 0 : pipelineSelectEl.value) {
        selectedPipelineName = pipelineSelectEl.value;
      }
    } catch (err) {
      console.warn("Could not read camera/pipeline selection:", err);
    }
    const computeActionName = (name) => {
      let result = String(name || "");
      if (result.toLowerCase().endsWith(".py")) {
        result = result.slice(0, -3);
      }
      return result.toLowerCase().replace(/\s+/g, "_");
    };
    const actionNameForApi = computeActionName(operationName || "");
    const startVisIfReady = async () => {
      if (!selectedCameraName || !selectedPipelineName) {
        console.log(
          "[SETTINGS] Skipping visualization - missing camera or pipeline",
          {
            selectedCameraName,
            selectedPipelineName
          }
        );
        return;
      }
      try {
        console.log("[SETTINGS] Starting visualization", {
          camera: selectedCameraName,
          pipeline: selectedPipelineName,
          action: actionNameForApi,
          timestamp: (/* @__PURE__ */ new Date()).toISOString()
        });
        await fetch(
          `${BACKEND_BASE_URL}/start-visualize/${encodeURIComponent(selectedCameraName)}/${encodeURIComponent(selectedPipelineName)}/${encodeURIComponent(actionNameForApi)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" }
          }
        );
        _currentVisCamera = selectedCameraName;
        _currentVisPipeline = selectedPipelineName;
        _currentVisAction = actionNameForApi;
        const liveViewPanelEl = document.getElementById(
          "operationLiveViewPanel"
        );
        const liveViewContainer = liveViewPanelEl.querySelector(
          "[data-role='live-view-container']"
        );
        let imgEl = liveViewContainer.querySelector(
          "#operationLiveImage"
        );
        if (!imgEl) {
          imgEl = document.createElement("img");
          imgEl.id = "operationLiveImage";
          imgEl.alt = "Live visualization";
          imgEl.className = "mx-auto mt-4 rounded-lg max-w-full max-h-[60vh]";
          liveViewContainer.appendChild(imgEl);
        }
        const placeholderEl = liveViewContainer.querySelector(
          "[data-role='live-view-placeholder']"
        );
        const textElements = liveViewContainer.querySelectorAll("p");
        const noVisMessage = liveViewContainer.querySelector(
          "#noVisualizationMessage"
        );
        if (placeholderEl) placeholderEl.style.display = "none";
        textElements.forEach((p) => p.style.display = "none");
        if (noVisMessage) noVisMessage.classList.add("hidden");
        imgEl.classList.remove("hidden");
        if (_visInterval) clearInterval(_visInterval);
        let hasNoVisualization = false;
        _visInterval = setInterval(async () => {
          if (hasNoVisualization) {
            return;
          }
          try {
            const url = `${BACKEND_BASE_URL}/visualize/${encodeURIComponent(_currentVisCamera)}/${encodeURIComponent(_currentVisPipeline)}`;
            const response = await fetch(url, {
              cache: "no-store"
            });
            if (!response.ok) {
              if (response.status === 500) {
                const errorText = await response.text();
                if (errorText.includes(
                  "Function has no visualization"
                )) {
                  hasNoVisualization = true;
                  if (_visInterval) {
                    clearInterval(_visInterval);
                    _visInterval = null;
                  }
                  showNoVisualizationMessage(operationName);
                  return;
                }
              }
              return;
            }
            const blob = await response.blob();
            const objectUrl = URL.createObjectURL(blob);
            if (_currentVisObjectUrl)
              URL.revokeObjectURL(_currentVisObjectUrl);
            _currentVisObjectUrl = objectUrl;
            imgEl.src = objectUrl;
          } catch (err) {
          }
        }, 100);
      } catch (err) {
        console.warn("Failed to start visualization:", err);
      }
    };
    fetchConfigData(operationName, isSecondary).then((config) => {
      if (!config) {
        body.innerHTML = '<div class="text-center text-red-400 py-8">Failed to load configuration</div>';
        startVisIfReady();
        return;
      }
      const originalValues = { ...initialValues };
      const getValues = renderForm(
        body,
        config,
        initialValues,
        originalValues,
        onSave,
        operationName
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
            timestamp: (/* @__PURE__ */ new Date()).toISOString()
          });
          if (typeof onSave === "function") onSave(values);
          console.log(
            "[SETTINGS] Settings saved, closing popup",
            {
              operationName,
              timestamp: (/* @__PURE__ */ new Date()).toISOString()
            }
          );
          close();
        };
      }
      if (cancelBtn) cancelBtn.onclick = () => close();
      startVisIfReady();
    }).catch((error) => {
      console.error("Error loading config:", error);
      body.innerHTML = '<div class="text-center text-red-400 py-8">Error loading configuration</div>';
      startVisIfReady();
    });
    overlay.classList.remove("hidden");
  }
  function stopVisualizationIfActive() {
    if (!_currentVisCamera || !_currentVisPipeline) {
      console.log("[SETTINGS] No active visualization to stop");
      return;
    }
    console.log("[SETTINGS] Stopping active visualization", {
      camera: _currentVisCamera,
      pipeline: _currentVisPipeline,
      action: _currentVisAction,
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    });
    try {
      fetch(
        `${BACKEND_BASE_URL}/stop-visualize/${encodeURIComponent(_currentVisCamera)}/${encodeURIComponent(_currentVisPipeline)}`,
        { method: "POST" }
      ).catch(
        (err) => console.warn("Failed to stop visualization:", err)
      );
    } finally {
      _currentVisCamera = null;
      _currentVisPipeline = null;
      _currentVisAction = null;
      console.log("[SETTINGS] Visualization state cleared");
    }
  }
  function close() {
    console.log("[SETTINGS] Closing settings popup", {
      wasVisualizing: !!_visInterval,
      hadVisualization: !!_currentVisObjectUrl,
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    });
    const { overlay } = findOverlayElements();
    if (!overlay) return;
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
    const liveViewPanelEl = document.getElementById(
      "operationLiveViewPanel"
    );
    if (liveViewPanelEl) {
      const liveViewContainer = liveViewPanelEl.querySelector(
        "[data-role='live-view-container']"
      );
      if (liveViewContainer) {
        const imgEl = liveViewContainer.querySelector(
          "#operationLiveImage"
        );
        const placeholderEl = liveViewContainer.querySelector(
          "[data-role='live-view-placeholder']"
        );
        const textElements = liveViewContainer.querySelectorAll("p");
        if (imgEl) imgEl.classList.add("hidden");
        if (placeholderEl) placeholderEl.style.display = "";
        textElements.forEach((p) => p.style.display = "");
        const noVisMessage = liveViewContainer.querySelector(
          "#noVisualizationMessage"
        );
        if (noVisMessage) {
          noVisMessage.classList.add("hidden");
        }
        const operationText = liveViewContainer.querySelector("p.text-lg");
        const descriptionText = liveViewContainer.querySelector("p.text-sm");
        if (operationText) {
          operationText.textContent = "Live Preview";
        }
        if (descriptionText) {
          descriptionText.textContent = "Visualizer will appear here when operation is active";
        }
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
    const isSecondaryInt = isSecondary ? 1 : 0;
    return fetch(
      `${BACKEND_BASE_URL}/get-operation-config-data/${encodeURIComponent(operationName)}/${isSecondaryInt}`
    ).then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    }).then((data) => {
      if (!data || typeof data !== "object") {
        throw new Error("Invalid config data structure");
      }
      return data;
    }).catch((error) => {
      console.error("Error fetching config data:", error);
      return null;
    });
  }
  globalThis.SettingsPopup = {
    init,
    open,
    close
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
