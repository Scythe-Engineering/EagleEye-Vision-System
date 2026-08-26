import pipelineTemplates from "../pipelineTemplates.json";
import {
    closeOnBackdropClick,
    closeOnEscape,
    createElement,
    getOrCreateModalElements,
    hideModal,
    showModal,
} from "../../ui/modal.js";

const OVERLAY_ID = "newPipelineDialogOverlay";
const MODAL_ID = "newPipelineDialogModal";
let activeResolve = null;
let initialized = false;

/** Return the shared new-pipeline modal elements. */
function getDialogElements() {
    return getOrCreateModalElements({
        overlayId: OVERLAY_ID,
        modalId: MODAL_ID,
        modalClassName:
            "bg-[#1a1a1a] rounded-xl shadow-2xl w-full max-w-md mx-4 border border-[#414141] overflow-hidden",
    });
}

/** Close the dialog and resolve its current request. */
function resolveDialog(result) {
    if (activeResolve) {
        activeResolve(result);
        activeResolve = null;
    }
    hideModal(getDialogElements().overlay);
}

/** Attach dismissal handlers once. */
function initializeDialog() {
    if (initialized) return;
    const { overlay } = getDialogElements();
    closeOnBackdropClick(overlay, () => resolveDialog(null));
    closeOnEscape(overlay, () => resolveDialog(null));
    initialized = true;
}

/** Ask for a pipeline name and optional template. */
export function newPipelineDialog() {
    initializeDialog();
    if (activeResolve) resolveDialog(null);

    const { overlay, modal } = getDialogElements();
    modal.innerHTML = "";

    const nameInput = createElement("input", {
        id: "newPipelineName",
        name: "pipelineName",
        type: "text",
        required: "",
        autofocus: "",
        autocomplete: "off",
        className:
            "w-full rounded-md border border-[#414141] bg-[#232323] px-3 py-2 text-white outline-none focus:border-[#f9c845]",
        placeholder: "Pipeline name",
    });
    const templateSelect = createElement("select", {
        id: "newPipelineTemplate",
        name: "pipelineTemplate",
        disabled: "",
        className:
            "w-full rounded-md border border-[#414141] bg-[#232323] px-3 py-2 text-white disabled:cursor-not-allowed disabled:opacity-50 focus:border-[#f9c845] focus:outline-none",
    });
    Object.entries(pipelineTemplates).forEach(([id, template]) => {
        templateSelect.appendChild(
            createElement("option", { value: id, text: template.name }),
        );
    });

    const useTemplate = createElement("input", {
        id: "usePipelineTemplate",
        type: "checkbox",
        className: "h-4 w-4 accent-[#f9c845]",
        onchange: () => {
            templateSelect.disabled = !useTemplate.checked;
        },
    });
    const form = createElement(
        "form",
        {
            onsubmit: (event) => {
                event.preventDefault();
                const name = nameInput.value.trim();
                if (!name) return;
                resolveDialog({
                    name,
                    templateId: useTemplate.checked
                        ? templateSelect.value
                        : null,
                });
            },
        },
        [
            createElement("div", { className: "space-y-5 p-5" }, [
                createElement("h3", {
                    className: "text-lg font-bold text-[#f9c845]",
                    text: "New pipeline",
                }),
                createElement("div", { className: "space-y-2" }, [
                    createElement("label", {
                        for: "newPipelineName",
                        className: "block text-sm font-medium text-gray-200",
                        text: "Pipeline name",
                    }),
                    nameInput,
                ]),
                createElement(
                    "label",
                    {
                        for: "usePipelineTemplate",
                        className:
                            "flex cursor-pointer items-center gap-2 text-sm text-gray-200",
                    },
                    [
                        useTemplate,
                        document.createTextNode(
                            "Use a template pipeline as a starting point",
                        ),
                    ],
                ),
                createElement("div", { className: "space-y-2" }, [
                    createElement("label", {
                        for: "newPipelineTemplate",
                        className: "block text-sm font-medium text-gray-200",
                        text: "Template",
                    }),
                    templateSelect,
                ]),
            ]),
            createElement(
                "div",
                {
                    className:
                        "flex justify-end gap-3 border-t border-[#333333] bg-[#171717] px-5 py-3",
                },
                [
                    createElement("button", {
                        type: "button",
                        className:
                            "rounded-md border border-[#414141] bg-[#242424] px-4 py-2 text-sm font-semibold text-[#f9c845] hover:bg-[#303030]",
                        text: "Cancel",
                        onclick: () => resolveDialog(null),
                    }),
                    createElement("button", {
                        type: "submit",
                        className:
                            "rounded-md border border-[#d4a83a] bg-[#f9c845] px-4 py-2 text-sm font-semibold text-[#232323] hover:bg-[#d4a83a]",
                        text: "Create pipeline",
                    }),
                ],
            ),
        ],
    );

    modal.appendChild(form);
    showModal(overlay);
    requestAnimationFrame(() => nameInput.focus());

    return new Promise((resolve) => {
        activeResolve = resolve;
    });
}

/** Return a copy of a bundled pipeline template. */
export function getPipelineTemplate(templateId) {
    const template = pipelineTemplates[templateId];
    return template ? structuredClone(template.nodes) : null;
}
