from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from flask import request

from src.webui.web_server_utils.constants import (
    PIPELINE_ERROR_FALLBACK_PUBLISH_INTERVAL_SECONDS,
    PIPELINE_ERROR_PUBLISH_FRAME_INTERVAL,
    PIPELINE_NOT_FOUND_MESSAGE,
    SRC_DIR,
)


class PipelineConfigMixin:
    def get_pipeline_names(self) -> list[str]:
        """
        Get the names of all pipelines.

        Returns:
            list[str]: The names of all pipelines.
        """
        with open(os.path.join(SRC_DIR, "config", "pipeline_config.json"), "r") as f:
            config = json.load(f)
        return list(config.keys())

    def get_pipeline_config_by_name(self, pipeline_name: str) -> list:
        """
        Get the config data for a pipeline by name.

        Args:
            pipeline_name (str): The name of the pipeline.

        Returns:
            list: The config data for the pipeline.
        """
        with open(os.path.join(SRC_DIR, "config", "pipeline_config.json"), "r") as f:
            config = json.load(f)
            if pipeline_name not in config:
                return []
            pipeline_config = config[pipeline_name]

        return self._reorder_pipeline_config(pipeline_config)

    def save_pipeline_config_by_name(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Save the pipeline config by pipeline name.

        Args:
            pipeline_name (str): The name of the pipeline.

        Returns:
            tuple[dict, int]: A success or failure message.
        """
        with open(os.path.join(SRC_DIR, "config", "pipeline_config.json"), "r") as f:
            current_config = json.load(f)
            new_data = request.get_json()

        if pipeline_name not in current_config:
            current_config[pipeline_name] = []

        existing_ops = {op["uuid"]: op for op in current_config[pipeline_name]}
        updated_operations = []
        for operation in new_data:
            operation_uuid = operation["uuid"]
            operation_name = operation["action_name"]
            operation_params = self._reorder_operation_params(
                operation_name, operation["action_params"]
            )

            if operation_uuid in existing_ops:
                merged_op = existing_ops[operation_uuid].copy()
                for key, value in operation.items():
                    if key == "action_params":
                        merged_op["action_params"].update(operation_params)
                    else:
                        merged_op[key] = value
            else:
                merged_op = operation.copy()
                merged_op["action_params"] = operation_params

            updated_operations.append(merged_op)

        current_config[pipeline_name] = updated_operations

        with open(os.path.join(SRC_DIR, "config", "pipeline_config.json"), "w") as f:
            json.dump(current_config, f, indent=4)

        pipeline_objects = self.pipeline_objects_callback()
        if pipeline_name in pipeline_objects:
            pipeline_objects[pipeline_name].update_operations_config(request.get_json())

        return {"message": "Pipeline config saved successfully"}, 200

    def delete_pipeline_by_name(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Delete a pipeline by name.

        Args:
            pipeline_name (str): The name of the pipeline.
        """
        with open(os.path.join(SRC_DIR, "config", "pipeline_config.json"), "r") as f:
            current_config = json.load(f)
            if pipeline_name in current_config:
                del current_config[pipeline_name]
            else:
                return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404
        with open(os.path.join(SRC_DIR, "config", "pipeline_config.json"), "w") as f:
            json.dump(current_config, f, indent=4)
        return {"message": "Pipeline deleted successfully"}, 200

    def _reorder_pipeline_config(
        self, pipeline_config: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Reorder operation parameters for a pipeline config list.

        Args:
            pipeline_config: Configuration list for the pipeline.

        Returns:
            Reordered pipeline config list.
        """
        reordered_pipeline = []
        for operation in pipeline_config:
            operation_name = operation["action_name"]
            action_params = self._reorder_operation_params(
                operation_name, operation["action_params"]
            )

            operation["action_params"] = action_params

            reordered_pipeline.append(operation)

        return reordered_pipeline

    def get_pipeline_active(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Return activity status for a pipeline.

        Args:
            pipeline_name: Name of the pipeline.

        Returns:
            tuple[dict, int]: Dictionary containing active flag.
        """
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
        except KeyError:
            return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404

        try:
            active = bool(pipeline.is_active())
        except Exception as error:
            self.log(
                f"Failed to read active status for pipeline {pipeline_name}: {error}"
            )
            active = False
        return {"pipeline": pipeline_name, "active": active}, 200

    def get_pipeline_thread_info(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Get thread and timestep information for a pipeline.

        Args:
            pipeline_name: Name of pipeline.

        Returns:
            tuple[dict, int]: Dictionary containing total_threads and operations dict
                with thread and timestep for each operation, plus HTTP status code.
        """
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
            return pipeline.get_pipeline_thread_info(), 200
        except KeyError:
            return {"error": PIPELINE_NOT_FOUND_MESSAGE}, 404

    def publish_operation_errors(self, payload: dict[str, Any]) -> None:
        """Publish operation error updates via SSE.

        Args:
            payload: Error payload containing pipeline and operation data.
        """
        try:
            pipeline_name = payload.get("pipeline_name") or "unknown"
            errors = payload.get("errors")
            normalized_payload = {
                "pipeline_name": pipeline_name,
                "errors": errors if isinstance(errors, list) else [],
            }
            with self._pipeline_error_lock:
                self._pipeline_error_cache[pipeline_name] = normalized_payload
                self._pipeline_error_dirty_pipelines.add(pipeline_name)
        except Exception as e:
            self.log(f"Failed to publish pipeline_operation_errors: {e}")

    def _publish_profiling_updates(self) -> None:
        """Publish latest pipeline profiling snapshots over SSE."""
        now = time.time()
        if now - self._last_profiling_publish_ts < self._profiling_publish_interval:
            return
        self._last_profiling_publish_ts = now

        try:
            pipelines = self.pipeline_objects_callback() or {}
        except Exception as error:
            self.log(f"Failed to get pipelines for profiling SSE: {error}")
            return

        for pipeline_name, pipeline in pipelines.items():
            try:
                snapshot = pipeline.get_latest_profile_snapshot()
                if not snapshot:
                    continue
                frame_seq = int(snapshot.get("frame_seq", 0))
                if frame_seq <= 0:
                    continue

                last_sent_seq = self._pipeline_profile_last_seq_sent.get(
                    pipeline_name,
                    0,
                )
                if frame_seq <= last_sent_seq:
                    continue

                self._publish_event("profiling_update", snapshot)
                self._pipeline_profile_last_seq_sent[pipeline_name] = frame_seq
            except Exception as error:
                self.log(
                    f"Failed to publish profiling_update for {pipeline_name}: {error}"
                )

    def _publish_cached_pipeline_errors(self) -> None:
        """Publish cached pipeline operation errors to the active SSE client.

        Error snapshots are published in batches (all operation errors in one
        payload per pipeline) and throttled to reduce event spam.
        """
        with self._pipeline_error_lock:
            cached_payloads = {
                pipeline_name: payload.copy()
                for pipeline_name, payload in self._pipeline_error_cache.items()
            }
            dirty_pipelines = set(self._pipeline_error_dirty_pipelines)
            last_seq_sent = self._pipeline_error_last_seq_sent.copy()
            last_publish_ts = self._pipeline_error_last_publish_ts.copy()

        if not cached_payloads:
            return

        now = time.time()
        frame_seq_by_pipeline: dict[str, int] = {}
        try:
            pipelines = self.pipeline_objects_callback() or {}
            for pipeline_name, pipeline in pipelines.items():
                snapshot = pipeline.get_latest_profile_snapshot()
                if not snapshot:
                    continue
                frame_seq = int(snapshot.get("frame_seq", 0))
                if frame_seq > 0:
                    frame_seq_by_pipeline[pipeline_name] = frame_seq
        except Exception as error:
            self.log(f"Failed to read pipelines for error batching: {error}")

        for pipeline_name, payload in cached_payloads.items():
            if pipeline_name not in dirty_pipelines:
                continue

            current_frame_seq = frame_seq_by_pipeline.get(pipeline_name, 0)
            previously_sent_seq = last_seq_sent.get(pipeline_name, 0)
            previously_sent_ts = last_publish_ts.get(pipeline_name, 0.0)

            frame_gate_open = (
                current_frame_seq > 0
                and current_frame_seq - previously_sent_seq
                >= PIPELINE_ERROR_PUBLISH_FRAME_INTERVAL
            )
            fallback_gate_open = (
                now - previously_sent_ts
                >= PIPELINE_ERROR_FALLBACK_PUBLISH_INTERVAL_SECONDS
            )

            if not frame_gate_open and not fallback_gate_open:
                continue

            errors = payload.get("errors")
            normalized_payload = {
                "pipeline_name": pipeline_name,
                "errors": errors if isinstance(errors, list) else [],
            }
            try:
                self._publish_event("pipeline_operation_errors", normalized_payload)
                with self._pipeline_error_lock:
                    self._pipeline_error_dirty_pipelines.discard(pipeline_name)
                    self._pipeline_error_last_publish_ts[pipeline_name] = now
                    if current_frame_seq > 0:
                        self._pipeline_error_last_seq_sent[pipeline_name] = (
                            current_frame_seq
                        )
            except Exception:
                continue
