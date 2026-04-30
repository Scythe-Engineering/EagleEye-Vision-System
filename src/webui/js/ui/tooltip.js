const TOOLTIP_DELAY_MS = 500;
const EDGE_OFFSET_PX = 10;
const VIEWPORT_MARGIN_PX = 8;

let tooltipElement = null;
let tooltipArrow = null;
let activeAnchor = null;
let showTimer = null;
let tooltipCleanup = null;

function ensureTooltipElement() {
    if (tooltipElement?.isConnected) {
        return tooltipElement;
    }

    tooltipElement = document.createElement("div");
    tooltipElement.className = "eagle-tooltip";
    tooltipElement.setAttribute("role", "tooltip");

    tooltipArrow = document.createElement("div");
    tooltipArrow.className = "eagle-tooltip__arrow";
    tooltipElement.appendChild(tooltipArrow);

    document.body.appendChild(tooltipElement);
    return tooltipElement;
}

function clearShowTimer() {
    if (showTimer) {
        clearTimeout(showTimer);
        showTimer = null;
    }
}

function normalizeTooltipAnchor(anchor) {
    if (!anchor) {
        return null;
    }

    if (anchor.dataset.tooltip || anchor.dataset.tooltipHtml) {
        return anchor;
    }

    return anchor.closest?.("[data-tooltip], [data-tooltip-html], [title]");
}

function prepareNativeTitleTooltip(element) {
    const title = element.getAttribute("title");
    if (!title) {
        return;
    }

    element.dataset.tooltip = title;
    element.dataset.tooltipOriginalTitle = title;
    element.removeAttribute("title");
}

function getTooltipContent(anchor) {
    prepareNativeTitleTooltip(anchor);

    if (anchor.dataset.tooltipHtml) {
        return {
            html: anchor.dataset.tooltipHtml,
            isHtml: true,
        };
    }

    return {
        text: anchor.dataset.tooltip ?? "",
        isHtml: false,
    };
}

function setTooltipContent(content) {
    const tooltip = ensureTooltipElement();
    tooltip.textContent = "";
    tooltip.appendChild(tooltipArrow);

    const body = document.createElement("div");
    body.className = "eagle-tooltip__body";

    if (content.isHtml) {
        body.innerHTML = content.html;
    } else {
        body.textContent = content.text;
    }

    tooltip.appendChild(body);
}

function choosePlacement(anchorRect, tooltipRect, preferredPlacement) {
    const placements =
        preferredPlacement === "auto"
            ? ["top", "bottom", "right", "left"]
            : [
                  preferredPlacement,
                  ...["top", "bottom", "right", "left"].filter(
                      (placement) => placement !== preferredPlacement,
                  ),
              ];

    return (
        placements.find((placement) => {
            if (placement === "top") {
                return anchorRect.top >= tooltipRect.height + EDGE_OFFSET_PX;
            }
            if (placement === "bottom") {
                return (
                    window.innerHeight - anchorRect.bottom >=
                    tooltipRect.height + EDGE_OFFSET_PX
                );
            }
            if (placement === "right") {
                return (
                    window.innerWidth - anchorRect.right >=
                    tooltipRect.width + EDGE_OFFSET_PX
                );
            }
            return anchorRect.left >= tooltipRect.width + EDGE_OFFSET_PX;
        }) ?? "top"
    );
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function positionTooltip(anchor, preferredPlacement = "auto") {
    const tooltip = ensureTooltipElement();
    const anchorRect = anchor.getBoundingClientRect();

    tooltip.classList.remove("eagle-tooltip--visible");
    tooltip.style.left = "0px";
    tooltip.style.top = "0px";

    const tooltipRect = tooltip.getBoundingClientRect();
    const placement = choosePlacement(
        anchorRect,
        tooltipRect,
        preferredPlacement,
    );
    let left = 0;
    let top = 0;

    if (placement === "top" || placement === "bottom") {
        left = anchorRect.left + anchorRect.width / 2 - tooltipRect.width / 2;
        left = clamp(
            left,
            VIEWPORT_MARGIN_PX,
            window.innerWidth - tooltipRect.width - VIEWPORT_MARGIN_PX,
        );
        top =
            placement === "top"
                ? anchorRect.top - tooltipRect.height - EDGE_OFFSET_PX
                : anchorRect.bottom + EDGE_OFFSET_PX;
    } else {
        top = anchorRect.top + anchorRect.height / 2 - tooltipRect.height / 2;
        top = clamp(
            top,
            VIEWPORT_MARGIN_PX,
            window.innerHeight - tooltipRect.height - VIEWPORT_MARGIN_PX,
        );
        left =
            placement === "right"
                ? anchorRect.right + EDGE_OFFSET_PX
                : anchorRect.left - tooltipRect.width - EDGE_OFFSET_PX;
    }

    tooltip.className = `eagle-tooltip eagle-tooltip--${placement}`;
    tooltip.style.left = `${Math.round(left)}px`;
    tooltip.style.top = `${Math.round(top)}px`;

    const arrowHalfSize = 6;
    if (placement === "top" || placement === "bottom") {
        const arrowLeft = clamp(
            anchorRect.left + anchorRect.width / 2 - left,
            14,
            tooltipRect.width - 14,
        );
        tooltipArrow.style.left = `${Math.round(arrowLeft - arrowHalfSize)}px`;
        tooltipArrow.style.top = "";
    } else {
        const arrowTop = clamp(
            anchorRect.top + anchorRect.height / 2 - top,
            14,
            tooltipRect.height - 14,
        );
        tooltipArrow.style.top = `${Math.round(arrowTop - arrowHalfSize)}px`;
        tooltipArrow.style.left = "";
    }
}

export function hideTooltip(anchor = activeAnchor) {
    if (anchor && activeAnchor && anchor !== activeAnchor) {
        return;
    }

    clearShowTimer();
    tooltipCleanup?.();
    tooltipCleanup = null;
    activeAnchor = null;

    if (tooltipElement) {
        tooltipElement.classList.remove("eagle-tooltip--visible");
    }
}

export function showTooltip(anchor, options = {}) {
    const hasExplicitContent = Boolean(options.text || options.html);
    const normalizedAnchor = hasExplicitContent
        ? anchor
        : normalizeTooltipAnchor(anchor);
    if (!normalizedAnchor) {
        return;
    }

    const delay = options.delay ?? TOOLTIP_DELAY_MS;
    const placement =
        options.placement ??
        normalizedAnchor.dataset.tooltipPlacement ??
        "auto";
    const content = options.html
        ? { html: options.html, isHtml: true }
        : options.text
          ? { text: options.text, isHtml: false }
          : getTooltipContent(normalizedAnchor);

    if (!content.text && !content.html) {
        return;
    }

    clearShowTimer();
    activeAnchor = normalizedAnchor;

    showTimer = setTimeout(() => {
        if (activeAnchor !== normalizedAnchor) {
            return;
        }

        setTooltipContent(content);
        positionTooltip(normalizedAnchor, placement);
        tooltipElement.classList.add("eagle-tooltip--visible");

        const reposition = () => {
            if (activeAnchor === normalizedAnchor) {
                positionTooltip(normalizedAnchor, placement);
                tooltipElement.classList.add("eagle-tooltip--visible");
            }
        };
        window.addEventListener("scroll", reposition, true);
        window.addEventListener("resize", reposition);
        tooltipCleanup = () => {
            window.removeEventListener("scroll", reposition, true);
            window.removeEventListener("resize", reposition);
        };
    }, delay);
}

function handleTooltipEnter(event) {
    const anchor = normalizeTooltipAnchor(event.target);
    if (!anchor) {
        return;
    }
    if (event.relatedTarget && anchor.contains(event.relatedTarget)) {
        return;
    }

    prepareNativeTitleTooltip(anchor);
    showTooltip(anchor);
}

function handleTooltipLeave(event) {
    const anchor = normalizeTooltipAnchor(event.target);
    if (anchor) {
        if (event.relatedTarget && anchor.contains(event.relatedTarget)) {
            return;
        }
        hideTooltip(anchor);
    }
}

function prepareExistingTitleTooltips(root = document) {
    root.querySelectorAll?.("[title]").forEach((element) => {
        prepareNativeTitleTooltip(element);
    });
}

export function initializeTooltips(root = document) {
    prepareExistingTitleTooltips(root);
    const handleEscape = (event) => {
        if (event.key === "Escape") {
            hideTooltip();
        }
    };

    root.addEventListener("mouseover", handleTooltipEnter);
    root.addEventListener("mouseout", handleTooltipLeave);
    root.addEventListener("focusin", handleTooltipEnter);
    root.addEventListener("focusout", handleTooltipLeave);
    root.addEventListener("keydown", handleEscape);

    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    prepareExistingTitleTooltips(node);
                    if (node.hasAttribute("title")) {
                        prepareNativeTitleTooltip(node);
                    }
                }
            });
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true,
    });

    return () => {
        observer.disconnect();
        hideTooltip();
        root.removeEventListener("mouseover", handleTooltipEnter);
        root.removeEventListener("mouseout", handleTooltipLeave);
        root.removeEventListener("focusin", handleTooltipEnter);
        root.removeEventListener("focusout", handleTooltipLeave);
        root.removeEventListener("keydown", handleEscape);
    };
}
