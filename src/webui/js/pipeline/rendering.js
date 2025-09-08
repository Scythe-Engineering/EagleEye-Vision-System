import { escapeHtml, getIconSVG } from "./utils.js";

// --- Description popup functions

let descriptionPopup = null;

export function createDescriptionPopup() {
    if (descriptionPopup) return; // Already exists

    descriptionPopup = document.createElement("div");
    descriptionPopup.id = "description-popup";
    descriptionPopup.className =
        "fixed z-50 bg-[#232323] border-2 border-[#f9c845] rounded-lg p-3 shadow-lg max-w-xs pointer-events-none opacity-0 transition-opacity duration-200";
    descriptionPopup.style.fontSize = "0.875rem";
    descriptionPopup.style.lineHeight = "1.25rem";

    // Add subtle 3D shadow on right and bottom edges to give depth
    descriptionPopup.style.boxShadow =
        "4px 4px 12px rgba(0,0,0,0.45), 8px 8px 20px rgba(0,0,0,0.25), 2px 2px 6px rgba(249,196,69,0.06)";

    document.body.appendChild(descriptionPopup);
}

export function showDescriptionPopup(name, description, event) {
    if (!descriptionPopup) createDescriptionPopup();

    // Set the popup content with name at the top and description below
    descriptionPopup.innerHTML = `
        <div class="text-[#f9c845] font-semibold text-sm mb-2 border-b border-[#404040] pb-2">${escapeHtml(name)}</div>
        <div class="text-white text-xs">${escapeHtml(description)}</div>
    `;

    // Position the popup near the mouse cursor
    const mouseX = event.clientX;
    const mouseY = event.clientY;

    // Position it slightly offset from cursor
    descriptionPopup.style.left = mouseX + 10 + "px";
    descriptionPopup.style.top = mouseY + 10 + "px";

    // Make it visible
    descriptionPopup.classList.remove("opacity-0");
    descriptionPopup.classList.add("opacity-100");
}

export function hideDescriptionPopup() {
    if (!descriptionPopup) return;
    descriptionPopup.classList.remove("opacity-100");
    descriptionPopup.classList.add("opacity-0");
}

export function addHoverListeners(element, name, description) {
    element.addEventListener("mouseenter", (e) => {
        showDescriptionPopup(name, description, e);
    });

    element.addEventListener("mousemove", (e) => {
        if (
            descriptionPopup &&
            descriptionPopup.classList.contains("opacity-100")
        ) {
            // Update position as mouse moves
            descriptionPopup.style.left = e.clientX + 10 + "px";
            descriptionPopup.style.top = e.clientY + 10 + "px";
        }
    });

    element.addEventListener("mouseleave", () => {
        hideDescriptionPopup();
    });
}

// --- Rendering

export function renderOperations(
    operations,
    operationsList,
    openOperationSettings,
    handleDragStart,
) {
    operationsList.innerHTML = "";
    operations.forEach((op, index) => {
        const el = document.createElement("div");
        el.draggable = true;
        el.className =
            "bg-[#232323] border-2 border-[#404040] rounded-xl p-4 cursor-move hover:border-[#f9c845] transition-all transform hover:scale-105 hover:shadow-lg mb-2 group";
        el.innerHTML = `
        <div class="flex items-center gap-3">
          <div class="bg-[#995e19] text-white text-xs font-semibold px-2 py-1 rounded-md uppercase tracking-wider">${escapeHtml(op.type)}</div>
          <div>
            <h3 class="font-medium text-white truncate max-w-[190px]">${escapeHtml(op.name)}</h3>
            ${index === 0 ? '<p class="text-xs text-gray-500 tracking-wider">Hover for description</p>' : ""}
          </div>
          <div class="ml-auto">
            <button class="op-settings-btn p-2 hover:bg-[#404040] rounded-lg transition-all" title="Settings">
              <img src="../../../assets/settings.svg" alt="Settings" class="w-4 h-4 icon-grayscale" />
            </button>
          </div>
        </div>
      `;

        el.addEventListener("dragstart", (e) =>
            handleDragStart(e, op, null, operations),
        );
        el.addEventListener("dragend", (e) => {
            if (e.currentTarget instanceof HTMLElement) {
                e.currentTarget.classList.remove("dragging");
                e.currentTarget.style.opacity = "";
            }
        });

        // Add hover listeners for description popup
        addHoverListeners(el, op.name, op.description);

        const settingsBtn = el.querySelector(".op-settings-btn");
        if (settingsBtn) {
            settingsBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                e.preventDefault();
                openOperationSettings(op);
            });
        }

        operationsList.appendChild(el);
    });
}

export function renderPipeline(
    pipeline,
    pipelineContainer,
    pipelinePlaceholder,
    openOperationSettings,
    updateRunButton,
    removeFromPipeline,
    handleDragStart,
    handleDragEnd,
) {
    // Reset container
    pipelineContainer.innerHTML = "";

    if (pipeline.length === 0) {
        pipelinePlaceholder.classList.remove("hidden");
        updateRunButton();
        return;
    }

    pipelinePlaceholder.classList.add("hidden");

    pipeline.forEach((item, index) => {
        // wrapper for pipeline item
        const wrapper = document.createElement("div");
        wrapper.dataset.instanceId = item.instanceId;
        wrapper.draggable = true;
        wrapper.className =
            "pipeline-item group relative bg-[#232323] border-2 border-[#404040] rounded-xl p-4 cursor-move hover:border-[#f9c845] transition-all transform hover:scale-105 hover:shadow-lg";

        // Inner content
        wrapper.innerHTML = `
        <div class="flex items-center gap-3">
          <div class="text-gray-600">${getIconSVG("grip")}</div>
          <div class="bg-[#995e19] text-white text-xs font-semibold px-2 py-1 rounded-md uppercase tracking-wider">${escapeHtml(item.type)}</div>
          <div class="flex-1">
            <h3 class="font-semibold text-white truncate max-w-[230px]">${escapeHtml(item.name)}</h3>
            <p class="text-xs text-gray-500 tracking-wider">Hover for description</p>
          </div>
          <div class="flex items-center gap-2">
            <button class="op-settings-btn p-2 hover:bg-[#404040] rounded-lg transition-all" title="Settings">
              <img src="../../../assets/settings.svg" alt="Settings" class="w-4 h-4 icon-grayscale" />
            </button>
            <button class="remove-btn p-2 hover:bg-[#404040] rounded-lg transition-all" title="Remove"><img src="../../../assets/delete.svg" alt="Delete" class="w-4 h-4 icon-grayscale" /></button>
          </div>
        </div>
      `;

        // Events
        wrapper.addEventListener("dragstart", (e) =>
            handleDragStart(e, item, index, pipeline),
        );
        wrapper.addEventListener("dragend", (e) =>
            handleDragEnd(e, pipelineContainer, pipelinePlaceholder, pipeline),
        );

        // Add hover listeners for description popup
        addHoverListeners(wrapper, item.name, item.description);

        const removeBtn = wrapper.querySelector(".remove-btn");
        removeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            removeFromPipeline(item.instanceId);
        });

        const opSettingsBtn = wrapper.querySelector(".op-settings-btn");
        if (opSettingsBtn) {
            opSettingsBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                openOperationSettings(item);
            });
        }

        pipelineContainer.appendChild(wrapper);

        // connector (small vertical line) between items
        if (index < pipeline.length - 1) {
            const connector = document.createElement("div");
            connector.className = "flex justify-center py-1";
            connector.innerHTML = `<div class="w-0.5 h-6 bg-[#f9c845]"></div>`;
            pipelineContainer.appendChild(connector);
        }
    });

    updateRunButton();
}
