from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List

from flask import request

from src.webui.web_server_utils.constants import SRC_DIR


class OperationConfigMixin:
    def get_available_operations(self) -> dict:
        """
        Get a dict of available operations.

        Returns:
            dict:
                operations: list of dicts with the name and path of the operation file.
        """
        NO_DESCRIPTION_AVAILABLE_MESSAGE = "No description available"
        main_operations = []

        for file in os.listdir(
            os.path.join(SRC_DIR, "main_operations", "definitions")
        ):
            if file.endswith(".py") and not file.startswith("_"):
                config_data_path = os.path.join(
                    SRC_DIR,
                    "main_operations",
                    "definitions",
                    "config_data",
                    file.rstrip(".py") + "_config_def.json",
                )
                try:
                    with open(config_data_path, "r") as f:
                        config_data = json.load(f)
                    description = config_data.get(
                        "description", NO_DESCRIPTION_AVAILABLE_MESSAGE
                    )
                    category = config_data.get("category", "Uncategorized")
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    description = NO_DESCRIPTION_AVAILABLE_MESSAGE
                    category = "Uncategorized"

                main_operations.append(
                    {
                        "name": os.path.basename(file),
                        "path": os.path.join(
                            SRC_DIR, "main_operations", "definitions", file
                        ),
                        "config_data_path": config_data_path,
                        "description": description,
                        "category": category,
                        "is_secondary": False,
                        "has_visualization": self._operation_has_visualization(
                            file,
                            is_secondary=False,
                        ),
                    }
                )

        secondary_operations = []

        for file in os.listdir(os.path.join(SRC_DIR, "secondary_operations")):
            if file.endswith(".py") and not file.startswith("_"):
                config_data_path = os.path.join(
                    SRC_DIR,
                    "secondary_operations",
                    "config_data",
                    file.rstrip(".py") + "_config_def.json",
                )
                try:
                    with open(config_data_path, "r") as f:
                        config_data = json.load(f)
                    description = config_data.get(
                        "description", NO_DESCRIPTION_AVAILABLE_MESSAGE
                    )
                    category = config_data.get("category", "Uncategorized")
                except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    description = NO_DESCRIPTION_AVAILABLE_MESSAGE
                    category = "Uncategorized"

                secondary_operations.append(
                    {
                        "name": os.path.basename(file),
                        "path": os.path.join(SRC_DIR, "secondary_operations", file),
                        "config_data_path": config_data_path,
                        "description": description,
                        "category": category,
                        "is_secondary": True,
                        "has_visualization": self._operation_has_visualization(
                            file,
                            is_secondary=True,
                        ),
                    }
                )

        return {
            "operations": main_operations + secondary_operations,
        }

    def _operation_has_visualization(self, filename: str, is_secondary: bool) -> bool:
        """Check if an operation overrides the base visualize method."""
        from src.main_operations.definitions.base.base_class import OperationInstance

        module_path = (
            f"src.secondary_operations.{filename[:-3]}"
            if is_secondary
            else f"src.main_operations.definitions.{filename[:-3]}"
        )
        try:
            module = __import__(module_path, fromlist=["*"])
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, OperationInstance)
                    and attr is not OperationInstance
                ):
                    return attr.visualize is not OperationInstance.visualize
        except Exception as e:
            self.log(f"Warning: Could not detect visualization for {filename}: {e}")
        return False

    def get_operation_config_data(
        self, operation_name: str, is_secondary: bool = False
    ) -> dict:
        """
        Get the config data for an operation.

        Args:
            operation_name (str): The name of the operation.
            is_secondary (bool): Whether the operation is a secondary operation.

        Returns:
            dict: The config data for the operation.
        """
        config_file_name = (
            operation_name.lower().replace(" ", "_").replace(".py", "")
            + "_config_def.json"
        )

        if is_secondary:
            config_path = os.path.join(
                SRC_DIR,
                "secondary_operations",
                "config_data",
                config_file_name,
            )
        else:
            config_path = os.path.join(
                SRC_DIR,
                "main_operations",
                "definitions",
                "config_data",
                config_file_name,
            )

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f, object_pairs_hook=dict)
                return self._normalize_dynamic_group_config(config_data)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            self.log(f"Error loading config for operation {operation_name}: {e}")
            return {}

    def _normalize_dynamic_group_config(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize optional dynamic group metadata in operation config.

        Args:
            config_data (dict[str, Any]): Raw operation config definition JSON,
                including optional `dynamic_group` metadata that is normalized
                for downstream port handling.

        Returns:
            dict[str, Any]: Config data with normalized `dynamic_group`
            metadata values (for example max counts, boolean flags, and base
            node names).
        """
        if not isinstance(config_data, dict):
            return {}

        dynamic_group = config_data.get("dynamic_group")
        if not isinstance(dynamic_group, dict):
            return config_data

        normalized_group = dict(dynamic_group)
        try:
            normalized_group["max_inputs"] = max(
                1,
                int(dynamic_group.get("max_inputs", 1)),
            )
        except (TypeError, ValueError):
            normalized_group["max_inputs"] = 1

        try:
            normalized_group["max_outputs"] = max(
                1,
                int(dynamic_group.get("max_outputs", normalized_group["max_inputs"])),
            )
        except (TypeError, ValueError):
            normalized_group["max_outputs"] = normalized_group["max_inputs"]

        mirrored_output_group = dynamic_group.get("mirrored_output_group", False)
        if isinstance(mirrored_output_group, str):
            mirrored_output_group = mirrored_output_group.lower() == "true"
        normalized_group["mirrored_output_group"] = bool(mirrored_output_group)

        output_dynamic_group = dynamic_group.get("output_dynamic_group", False)
        if isinstance(output_dynamic_group, str):
            output_dynamic_group = output_dynamic_group.lower() == "true"
        normalized_group["output_dynamic_group"] = bool(output_dynamic_group)

        input_dynamic_group = dynamic_group.get("input_dynamic_group", True)
        if isinstance(input_dynamic_group, str):
            input_dynamic_group = input_dynamic_group.lower() == "true"
        normalized_group["input_dynamic_group"] = bool(input_dynamic_group)

        coupled_groups = dynamic_group.get(
            "coupled_groups",
            normalized_group["mirrored_output_group"],
        )
        if isinstance(coupled_groups, str):
            coupled_groups = coupled_groups.lower() == "true"
        normalized_group["coupled_groups"] = bool(coupled_groups)

        input_nodes = config_data.get("input_nodes") or []
        output_nodes = config_data.get("output_nodes") or []

        input_base_name = normalized_group.get("input_base_name") or normalized_group.get(
            "input_node"
        )
        output_base_name = normalized_group.get("output_base_name") or normalized_group.get(
            "output_node"
        )

        if not input_base_name:
            if input_nodes:
                candidate = input_nodes[-1]
                if isinstance(candidate, dict):
                    input_base_name = candidate.get("name")
                elif isinstance(candidate, str):
                    input_base_name = candidate
            if not input_base_name:
                input_base_name = "data"
        normalized_group["input_base_name"] = input_base_name

        if not output_base_name:
            if output_nodes:
                candidate = output_nodes[-1]
                if isinstance(candidate, dict):
                    output_base_name = candidate.get("name")
                elif isinstance(candidate, str):
                    output_base_name = candidate
            if not output_base_name:
                output_base_name = input_base_name
        normalized_group["output_base_name"] = output_base_name

        config_data["dynamic_group"] = normalized_group
        return config_data

    def _reorder_operation_params(
        self, operation_name: str, action_params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Reorder operation parameters according to their config definition.

        Args:
            operation_name (str): The name of the operation.
            action_params (dict[str, Any]): The current action parameters.

        Returns:
            dict[str, Any]: The reordered action parameters.
        """
        try:
            config_def = self.get_operation_config_data(operation_name, True)
            if not config_def or "parameters" not in config_def:
                config_def = self.get_operation_config_data(operation_name, False)

            if config_def and "parameters" in config_def:
                param_order = list(config_def["parameters"].keys())
                reordered_params = {}
                for param in param_order:
                    if param in action_params:
                        reordered_params[param] = action_params[param]
                for param, value in action_params.items():
                    if param not in reordered_params:
                        reordered_params[param] = value
                return reordered_params
        except Exception as e:
            self.log(f"Warning: Could not reorder parameters for {operation_name}: {e}")

        return action_params

    def _get_parameter_file_extensions(self, parameter_name: str) -> List[str]:
        """
        Get allowed file extensions for a parameter.

        Args:
            parameter_name: Name of the parameter.

        Returns:
            List of allowed file extensions (with dots).
        """
        extension_map = {
            "camera_parameters_path": [".json"],
            "apriltag_map_path": [".fmap", ".json"],
            "model_path": [".onnx", ".dfp", ".pt"],
            "post_processing_model_path": [".onnx"],
        }
        return extension_map.get(parameter_name, [])

    def _ensure_parameter_directory(self, parameter_name: str) -> Path:
        """
        Ensure the parameter-specific file directory exists.

        Args:
            parameter_name: Name of the parameter.

        Returns:
            Path to the parameter-specific directory.
        """
        files_base_dir = Path(SRC_DIR).parent / "files"
        parameter_dir = files_base_dir / parameter_name
        parameter_dir.mkdir(parents=True, exist_ok=True)
        return parameter_dir

    def get_operation_files(
        self, operation_name: str, parameter_name: str
    ) -> tuple[dict, int]:
        """
        Get list of available files for an operation parameter.

        Args:
            operation_name: Name of the operation (for UI context only).
            parameter_name: Name of the parameter.

        Returns:
            Tuple of (response dict, status code).
        """
        try:
            parameter_dir = self._ensure_parameter_directory(parameter_name)
            allowed_extensions = self._get_parameter_file_extensions(parameter_name)

            if not allowed_extensions:
                return {
                    "error": f"No file extensions defined for parameter {parameter_name}"
                }, 400

            files = []
            if parameter_dir.exists():
                for file_path in parameter_dir.iterdir():
                    if (
                        file_path.is_file()
                        and file_path.suffix.lower() in allowed_extensions
                    ):
                        file_stat = file_path.stat()
                        files.append(
                            {
                                "filename": file_path.name,
                                "size": file_stat.st_size,
                                "modified": file_stat.st_mtime,
                            }
                        )

            files.sort(key=lambda x: x["modified"], reverse=True)

            relative_path = parameter_dir.relative_to(Path(SRC_DIR).parent)
            base_path = f"{{project_root}}/{relative_path}"
            return {
                "files": [f["filename"] for f in files],
                "file_details": files,
                "base_path": str(base_path),
            }, 200
        except Exception as e:
            self.log(f"Error getting operation files: {e}")
            return {"error": str(e)}, 500

    def upload_operation_file(
        self, operation_name: str, parameter_name: str
    ) -> tuple[dict, int]:
        """
        Upload a file for an operation parameter.

        Args:
            operation_name: Name of the operation (for UI context only).
            parameter_name: Name of the parameter.

        Returns:
            Tuple of (response dict, status code).
        """
        try:
            if "file" not in request.files:
                return {"error": "No file provided"}, 400

            file = request.files["file"]
            if file.filename == "":
                return {"error": "No file selected"}, 400

            allowed_extensions = self._get_parameter_file_extensions(parameter_name)
            if not allowed_extensions:
                return {
                    "error": f"No file extensions defined for parameter {parameter_name}"
                }, 400

            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                return {
                    "error": f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
                }, 400

            parameter_dir = self._ensure_parameter_directory(parameter_name)
            file_path = parameter_dir / file.filename

            file.save(str(file_path))
            self.log(
                f"Uploaded file {file.filename} for {operation_name}/{parameter_name}"
            )

            relative_path = parameter_dir.relative_to(Path(SRC_DIR).parent)
            full_path = f"{{project_root}}/{relative_path}/{file.filename}"
            return {
                "success": True,
                "filename": file.filename,
                "path": full_path,
            }, 200
        except Exception as e:
            self.log(f"Error uploading operation file: {e}")
            return {"error": str(e)}, 500

    def delete_operation_file(
        self, operation_name: str, parameter_name: str, filename: str
    ) -> tuple[dict, int]:
        """
        Delete a file for an operation parameter.

        Args:
            operation_name: Name of the operation (for UI context only).
            parameter_name: Name of the parameter.
            filename: Name of the file to delete.

        Returns:
            Tuple of (response dict, status code).
        """
        try:
            parameter_dir = self._ensure_parameter_directory(parameter_name)
            file_path = parameter_dir / filename

            if not file_path.exists():
                return {"error": "File not found"}, 404

            if not file_path.is_file():
                return {"error": "Path is not a file"}, 400

            allowed_extensions = self._get_parameter_file_extensions(parameter_name)
            if file_path.suffix.lower() not in allowed_extensions:
                return {"error": "File extension not allowed for this parameter"}, 400

            file_path.unlink()
            self.log(f"Deleted file {filename} for {operation_name}/{parameter_name}")

            return {"success": True}, 200
        except Exception as e:
            self.log(f"Error deleting operation file: {e}")
            return {"error": str(e)}, 500
