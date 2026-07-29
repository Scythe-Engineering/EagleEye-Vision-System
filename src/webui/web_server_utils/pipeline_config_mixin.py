from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from copy import deepcopy
from typing import Any

from flask import request

from src.config.utils.port_validation import validate_pipeline_connections
from src.webui.web_server_utils.constants import (
    PIPELINE_ERROR_FALLBACK_PUBLISH_INTERVAL_SECONDS,
    PIPELINE_ERROR_PUBLISH_FRAME_INTERVAL,
    PIPELINE_NOT_FOUND_MESSAGE,
    SRC_DIR,
)


class PipelineConfigMixin:
    def _pipeline_config_path(self) -> str:
        """Return the absolute path to the persisted pipeline config file."""
        return os.path.join(SRC_DIR, "config", "pipeline_config.json")

    def _load_pipeline_config_file(self) -> dict[str, list[dict[str, Any]]]:
        """Load the persisted pipeline configuration from disk."""
        with open(self._pipeline_config_path(), "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_pipeline_config_text(self, content: str) -> None:
        """Atomically replace the pipeline config with complete text content."""
        config_path = self._pipeline_config_path()
        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=os.path.dirname(config_path),
            prefix=".pipeline_config.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as config_file:
                config_file.write(content)
                config_file.flush()
                os.fsync(config_file.fileno())
            os.replace(temporary_path, config_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _write_pipeline_config_file(
        self, config: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Serialize and atomically write a pipeline configuration."""
        self._write_pipeline_config_text(json.dumps(config, indent=4) + "\n")

    @staticmethod
    def _pipeline_config_revision(content: str) -> str:
        """Return a stable revision token for optimistic editor saves."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_pipeline_config_json(self) -> tuple[dict[str, str], int]:
        """Return the raw pipeline JSON text, including malformed content."""
        with open(self._pipeline_config_path(), "r", encoding="utf-8") as config_file:
            content = config_file.read()
        return {
            "content": content,
            "revision": self._pipeline_config_revision(content),
        }, 200

    def save_pipeline_config_json(self) -> tuple[dict[str, Any], int]:
        """Validate and save raw pipeline JSON submitted by the editor."""
        payload = request.get_json(silent=True)
        content = payload.get("content") if isinstance(payload, dict) else None
        revision = payload.get("revision") if isinstance(payload, dict) else None
        if not isinstance(content, str) or not isinstance(revision, str):
            return {"error": "Expected string fields 'content' and 'revision'"}, 400

        with open(self._pipeline_config_path(), "r", encoding="utf-8") as config_file:
            current_content = config_file.read()
        if revision != self._pipeline_config_revision(current_content):
            return {
                "error": "Pipeline configuration changed while the editor was open. Reload it before saving."
            }, 409

        try:
            parsed_config = json.loads(content)
        except json.JSONDecodeError as error:
            return {
                "error": "Pipeline configuration contains invalid JSON",
                "line": error.lineno,
                "column": error.colno,
                "detail": error.msg,
            }, 400

        if not isinstance(parsed_config, dict):
            return {"error": "Pipeline configuration must be a JSON object"}, 400
        for pipeline_name, pipeline_operations in parsed_config.items():
            try:
                validate_pipeline_connections(pipeline_operations)
            except ValueError as error:
                return {
                    "error": "Pipeline ports or connections are invalid",
                    "detail": f"{pipeline_name}: {error}",
                }, 400

            try:
                for operation in pipeline_operations:
                    self.validate_operation_params(
                        operation.get("action_name", ""),
                        operation.get("action_params", {}),
                    )
            except ValueError as error:
                return {
                    "error": "Pipeline operation parameters are invalid",
                    "detail": f"{pipeline_name}: {error}",
                }, 400

        normalized_content = content.rstrip() + "\n"
        self._write_pipeline_config_text(normalized_content)
        self.restart_required_for_config = True
        return {
            "message": "Pipeline JSON saved successfully",
            "revision": self._pipeline_config_revision(normalized_content),
            "restart_required": True,
            "runtime_id": getattr(self, "runtime_id", ""),
        }, 200

    def _get_runtime_pipeline_config_baseline(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return the pipeline config snapshot this backend process is running."""
        baseline = getattr(self, "_runtime_pipeline_config_baseline", None)
        if baseline is None:
            baseline = self._load_pipeline_config_file()
            self._runtime_pipeline_config_baseline = deepcopy(baseline)
        return baseline

    def get_pipeline_names(self) -> list[str]:
        """
        Get the names of all pipelines.

        Returns:
            list[str]: The names of all pipelines.
        """
        config = self._load_pipeline_config_file()
        return list(config.keys())

    def get_pipeline_config_by_name(self, pipeline_name: str) -> list:
        """
        Get the config data for a pipeline by name.

        Args:
            pipeline_name (str): The name of the pipeline.

        Returns:
            list: The config data for the pipeline.
        """
        config = self._load_pipeline_config_file()
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
        current_config = self._load_pipeline_config_file()
        new_data = request.get_json()

        if pipeline_name not in current_config:
            current_config[pipeline_name] = []

        existing_ops = {op["uuid"]: op for op in current_config[pipeline_name]}
        updated_operations = []
        invalid_operations = []
        for operation in new_data:
            operation_uuid = operation["uuid"]
            operation_name = operation["action_name"]
            try:
                operation_params = self._reorder_operation_params(
                    operation_name, operation["action_params"]
                )
            except ValueError as exc:
                invalid_operations.append(
                    {
                        "uuid": operation_uuid,
                        "name": operation_name,
                        "message": str(exc),
                    }
                )
                continue

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

        if invalid_operations:
            return {
                "message": "Pipeline config contains invalid operation configuration",
                "invalid_operations": invalid_operations,
            }, 400

        try:
            validate_pipeline_connections(updated_operations)
        except ValueError as error:
            return {
                "message": "Pipeline ports or connections are invalid",
                "detail": str(error),
            }, 400

        current_config[pipeline_name] = updated_operations
        restart_state = self._analyze_pipeline_restart_state(current_config)
        self.restart_required_for_config = restart_state["restart_required"]

        self._write_pipeline_config_file(current_config)

        live_update_status = None
        pipeline_objects = self.pipeline_objects_callback()
        if pipeline_name in pipeline_objects:
            live_update_status = pipeline_objects[
                pipeline_name
            ].update_operations_config(request.get_json())

        return {
            "message": "Pipeline config saved successfully",
            "live_update_status": live_update_status,
            **restart_state,
        }, 200

    def delete_pipeline_by_name(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Delete a pipeline by name.

        Args:
            pipeline_name (str): The name of the pipeline.
        """
        current_config = self._load_pipeline_config_file()
        if pipeline_name in current_config:
            del current_config[pipeline_name]
        else:
            return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404
        restart_state = self._analyze_pipeline_restart_state(current_config)
        self.restart_required_for_config = restart_state["restart_required"]
        self._write_pipeline_config_file(current_config)
        remove_settings = getattr(self, "remove_pipeline_settings", None)
        if callable(remove_settings):
            remove_settings(pipeline_name)
        return {"message": "Pipeline deleted successfully", **restart_state}, 200

    def _analyze_pipeline_restart_state(
        self, current_config: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Compare current config to the runtime baseline and summarize restart need."""
        restart_required = self._pipeline_restart_required(
            self._get_runtime_pipeline_config_baseline(),
            current_config,
        )

        return {
            "restart_required": restart_required,
            "runtime_id": getattr(self, "runtime_id", ""),
        }

    def _pipeline_restart_required(
        self,
        baseline: dict[str, list[dict[str, Any]]],
        current_config: dict[str, list[dict[str, Any]]],
    ) -> bool:
        """Return True when current config differs from the running baseline."""
        return self._restart_relevant_config(baseline) != self._restart_relevant_config(
            current_config
        )

    def _restart_relevant_config(
        self, config: dict[str, list[dict[str, Any]]]
    ) -> dict[str, dict[str, Any]]:
        """Return only config fields that affect backend restart requirements."""
        return {
            pipeline_name: {
                uuid: self._restart_relevant_operation(operation)
                for uuid, operation in self._operations_by_uuid(operations).items()
            }
            for pipeline_name, operations in config.items()
        }

    def _operations_by_uuid(
        self, operations: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Return pipeline operations keyed by UUID, skipping malformed entries."""
        return {
            uuid: operation
            for operation in operations
            if isinstance(operation, dict)
            and isinstance((uuid := operation.get("uuid")), str)
            and uuid
        }

    def _restart_relevant_operation(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Return the operation fields that require restart when changed."""
        action_name = operation.get("action_name")
        return {
            "action_name": action_name,
            "connections": self._canonical_connections(operation),
            "restart_params": self._restart_required_params(
                action_name,
                operation.get("action_params", {}),
            ),
        }

    def _canonical_connections(self, operation: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize an operation's connections for deterministic equality checks."""
        connections = operation.get("connections", [])
        if not isinstance(connections, list):
            return []

        canonical = []
        for connection in connections:
            if not isinstance(connection, dict):
                continue
            canonical.append(
                {
                    "from_uuid": connection.get("from_uuid"),
                    "from_port": connection.get("from_port"),
                    "to_uuid": connection.get("to_uuid"),
                    "to_port": connection.get("to_port"),
                    "data_type": connection.get("data_type"),
                    "is_default": bool(connection.get("is_default", False)),
                    "custom_waypoints": connection.get("custom_waypoints"),
                }
            )
        return sorted(
            canonical,
            key=lambda conn: json.dumps(conn, sort_keys=True, default=str),
        )

    def _restart_required_params(
        self,
        action_name: str | None,
        action_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Return action params marked restart_for_change in config data."""
        if not action_name:
            return {}

        config_data = self.get_operation_config_data(action_name, True)
        if not config_data:
            config_data = self.get_operation_config_data(action_name, False)

        parameters = config_data.get("parameters", {})
        if not isinstance(parameters, dict):
            return {}
        if not isinstance(action_params, dict):
            return {}

        return {
            param_name: value
            for param_name, value in action_params.items()
            if isinstance(parameters.get(param_name), dict)
            and parameters[param_name].get("restart_for_change", False)
        }

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
            try:
                self._publish_event("pipeline_operation_errors", normalized_payload)
            except Exception:
                pass
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
