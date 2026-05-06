from __future__ import annotations

from src.config.utils.line_profiling import line_profiling_manager
from src.webui.web_server_utils.constants import PIPELINE_NOT_FOUND_MESSAGE


class LineProfilingMixin:
    def start_line_profiling(
        self, pipeline_name: str, operation_uuid: str
    ) -> tuple[dict, int]:
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
        except KeyError:
            return {"success": False, "error": PIPELINE_NOT_FOUND_MESSAGE}, 404

        operation = pipeline.get_operation_by_uuid(operation_uuid)
        if operation is None:
            return {"success": False, "error": "Operation not found"}, 404

        module_name = getattr(operation.instance.__class__, "__module__", "")
        if not (
            module_name.startswith("src.secondary_operations.")
            or module_name.startswith("src.main_operations.definitions.")
        ):
            return {
                "success": False,
                "error": "Line profiling is only available for main or secondary operations",
            }, 400

        return line_profiling_manager.start_session(pipeline_name, operation)

    def stop_line_profiling(
        self, pipeline_name: str, operation_uuid: str
    ) -> tuple[dict, int]:
        return line_profiling_manager.stop_session(pipeline_name, operation_uuid)

    def get_line_profiling_status(self) -> tuple[dict, int]:
        return {"success": True, **line_profiling_manager.get_status()}, 200

    def get_line_profiling_report(
        self, pipeline_name: str, operation_uuid: str
    ) -> tuple[dict, int]:
        return line_profiling_manager.get_report(pipeline_name, operation_uuid)
