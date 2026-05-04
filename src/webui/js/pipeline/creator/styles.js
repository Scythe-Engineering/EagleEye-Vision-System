// Injects the pipeline creator's shared stylesheet once into the document.
let creatorStylesInjected = false;

/**
 * Ensures the pipeline creator styles are added to the document only once.
 */
export function ensurePipelineCreatorStyles() {
    if (creatorStylesInjected || document.getElementById("pipeline-creator-styles")) {
        creatorStylesInjected = true;
        return;
    }

    const styleEl = document.createElement("style");
    styleEl.id = "pipeline-creator-styles";
    styleEl.textContent = `
.op-settings-btn, .remove-btn { display: none !important; }
#pipelineArea .op-settings-btn, #pipelineArea .remove-btn { display: inline-flex !important; }
.icon-grayscale { filter: grayscale(100%); transition: filter .15s ease-in-out; }
#pipelineArea .group:hover .icon-grayscale, #pipelineArea .group:focus-within .icon-grayscale { filter: none; }
#flowchartCanvas { background-color: #1a1a1a; }
.flowchart-node .node-settings-btn:hover img,
.flowchart-node .node-remove-btn:hover img { filter: none !important; }
.pipeline-error-node { border-color: #ff5c5c !important; }
.pipeline-downstream-disabled { filter: grayscale(100%); opacity: 0.55; }
.pipeline-downstream-disabled .icon-grayscale { filter: grayscale(100%) !important; }
.pipeline-error-icon, .error-info-icon { pointer-events: auto; }
`;
    document.head.appendChild(styleEl);
    creatorStylesInjected = true;
}
