import { CustomOpsController } from "./customOpsController.js";

let _controller = null;

export function initCustomOpsEditor() {
    if (!_controller) {
        _controller = new CustomOpsController();
    }
    _controller.init();
}
