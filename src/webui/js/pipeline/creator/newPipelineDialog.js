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

    const templateText = createElement("span", {
        className:
            "max-w-[22rem] overflow-hidden whitespace-nowrap opacity-100 transition-all duration-150",
        text: "Use a template pipeline as a starting point",
    });
    const templateSelectWrapper = createElement(
        "div",
        {
            className:
                "invisible grid min-w-0 flex-1 -translate-x-2 grid-rows-[0fr] opacity-0 transition-all delay-150 duration-200",
        },
        [
            createElement("div", { className: "overflow-hidden" }, [
                templateSelect,
            ]),
        ],
    );
    const useTemplate = createElement("input", {
        id: "usePipelineTemplate",
        type: "checkbox",
        className: "h-4 w-4 shrink-0 accent-[#f9c845]",
        onchange: () => {
            const enabled = useTemplate.checked;
            templateSelect.disabled = !enabled;
            templateText.classList.toggle("opacity-0", enabled);
            templateText.classList.toggle("max-w-0", enabled);
            templateText.classList.toggle("max-w-[22rem]", !enabled);
            templateText.classList.toggle("delay-150", !enabled);
            templateSelectWrapper.classList.toggle("invisible", !enabled);
            templateSelectWrapper.classList.toggle("grid-rows-[0fr]", !enabled);
            templateSelectWrapper.classList.toggle("-translate-x-2", !enabled);
            templateSelectWrapper.classList.toggle("opacity-0", !enabled);
            templateSelectWrapper.classList.toggle("grid-rows-[1fr]", enabled);
            templateSelectWrapper.classList.toggle("translate-x-0", enabled);
            templateSelectWrapper.classList.toggle("opacity-100", enabled);
            templateSelectWrapper.classList.toggle("delay-150", enabled);
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
                    "div",
                    { className: "flex min-h-10 items-center gap-2" },
                    [
                        createElement(
                            "label",
                            {
                                for: "usePipelineTemplate",
                                className:
                                    "flex shrink-0 cursor-pointer items-center gap-2 text-sm text-gray-200",
                            },
                            [useTemplate, templateText],
                        ),
                        templateSelectWrapper,
                    ],
                ),
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
