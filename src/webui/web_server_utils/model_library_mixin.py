"""HTTP handlers for the startup device registry and managed model library."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from flask import request

from src.utils.device_registry import DeviceNotFoundError
from src.utils.mx3_compiler import (
    Mx3CompileStatus,
    Mx3CompilerBusyError,
    Mx3CompilerError,
    Mx3CompilerService,
)
from src.utils.model_library import (
    ArtifactError,
    ModelLibrary,
    ModelLibraryError,
    ModelMetadata,
    ModelNotFoundError,
    ModelReferencedError,
)


class ModelLibraryMixin:
    """Expose model-library and device-registry services to the WebUI."""

    device_registry: Any
    model_library: ModelLibrary | None
    mx3_compiler: Mx3CompilerService | None
    restart_required_for_config: bool
    _last_mx3_compilation_publish: float
    _publish_event: Callable[[str, object], None]

    def get_device_registry(self) -> tuple[dict[str, Any], int]:
        """Return the immutable startup device inventory."""
        if self.device_registry is None:
            return {"error": "Device registry is not initialized"}, 503
        devices = [
            {
                "id": descriptor.device_id,
                "kind": descriptor.device_type,
                "display_name": descriptor.display_name,
                "physical_index": descriptor.physical_index,
            }
            for descriptor in self.device_registry.descriptors()
        ]
        return {"devices": devices}, 200

    def _require_model_library(self) -> ModelLibrary:
        """Return the configured model library or raise an actionable error."""
        if self.model_library is None:
            raise ModelLibraryError("Model library is not initialized")
        return self.model_library

    def _require_mx3_compiler(self) -> Mx3CompilerService:
        """Return the local compiler service or fail with deployment context."""
        if self.mx3_compiler is None:
            raise Mx3CompilerError(
                "MX3 compiler is unavailable without a model library"
            )
        return self.mx3_compiler

    def _publish_mx3_compilation_progress(self, status: Mx3CompileStatus) -> None:
        """Publish a compiler snapshot and flag referenced pipelines for restart."""
        if status.restart_required:
            self.restart_required_for_config = True
        now = time.monotonic()
        if (
            status.state in {"running", "cancelling"}
            and now - self._last_mx3_compilation_publish < 0.25
        ):
            return
        self._last_mx3_compilation_publish = now
        self._publish_event("mx3_compilation_progress", status.to_dict(log_limit=5))

    @staticmethod
    def _serialize_model(
        model: ModelMetadata, referenced_by: tuple[str, ...]
    ) -> dict[str, Any]:
        """Serialize immutable model metadata for the WebUI."""
        return {
            "id": model.model_id,
            "display_name": model.display_name,
            "class_names": (
                list(model.class_names) if model.class_names is not None else None
            ),
            "artifacts": {
                slot: {
                    "filename": Path(relative_path).name,
                    "path": relative_path,
                }
                for slot, relative_path in model.artifacts.items()
            },
            "mx3_profile": (
                dict(model.mx3_profile) if model.mx3_profile is not None else None
            ),
            "referenced_by": list(referenced_by),
        }

    def get_model_library(self) -> tuple[dict[str, Any], int]:
        """List all models with current pipeline references."""
        try:
            library = self._require_model_library()
            models = library.list_models()
            references_by_model: dict[str, set[str]] = {
                model.model_id: set() for model in models
            }
            if library.pipeline_config_path.exists():
                try:
                    pipeline_config = json.loads(
                        library.pipeline_config_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as error:
                    raise ModelLibraryError(
                        f"Cannot read pipeline configuration: {error}"
                    ) from error
                if isinstance(pipeline_config, dict):
                    for pipeline_name, operations in pipeline_config.items():
                        if not isinstance(operations, list):
                            continue
                        referenced_ids = {
                            model_id
                            for operation in operations
                            if isinstance(operation, dict)
                            and isinstance(operation.get("action_params"), dict)
                            and isinstance(
                                (
                                    model_id := operation["action_params"].get(
                                        "model_id"
                                    )
                                ),
                                str,
                            )
                        }
                        for model_id in referenced_ids & references_by_model.keys():
                            references_by_model[model_id].add(str(pipeline_name))
            references_by_model = {
                model_id: tuple(sorted(pipelines))
                for model_id, pipelines in references_by_model.items()
            }
            serialized_models = [
                self._serialize_model(
                    model, references_by_model.get(model.model_id, ())
                )
                for model in models
            ]
            return {"models": serialized_models}, 200
        except ModelLibraryError as error:
            return {"error": str(error)}, 503

    def create_model_library_entry(self) -> tuple[dict[str, Any], int]:
        """Create a managed model metadata record."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return {"error": "Request body must be an object"}, 400
        allowed_fields = {"display_name", "class_names", "mx3_profile"}
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            return {"error": f"Unknown model fields: {sorted(unknown_fields)}"}, 400
        try:
            library = self._require_model_library()
            model = library.create_model(
                display_name=payload.get("display_name"),
                class_names=payload.get("class_names"),
                mx3_profile=payload.get("mx3_profile"),
            )
            return {
                "id": model.model_id,
                "model": self._serialize_model(model, ()),
            }, 201
        except ModelLibraryError as error:
            return {"error": str(error)}, 400

    def update_model_library_entry(self, model_id: str) -> tuple[dict[str, Any], int]:
        """Rename or update class/profile metadata without changing model ID."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return {"error": "Request body must be an object"}, 400
        allowed_fields = {"display_name", "class_names", "mx3_profile"}
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            return {"error": f"Unknown model fields: {sorted(unknown_fields)}"}, 400
        try:
            library = self._require_model_library()
            update_arguments = {
                field: payload[field] for field in allowed_fields if field in payload
            }
            model = library.update_model(model_id, **update_arguments)
            references = library.references(model_id)
            restart_required = bool(
                references
                and ({"class_names", "mx3_profile"} & update_arguments.keys())
            )
            if restart_required:
                self.restart_required_for_config = True
            return {
                "model": self._serialize_model(model, references),
                "affected_pipelines": list(references) if restart_required else [],
                "restart_required": restart_required,
            }, 200
        except ModelNotFoundError as error:
            return {"error": str(error)}, 404
        except ModelLibraryError as error:
            return {"error": str(error)}, 400

    def delete_model_library_entry(self, model_id: str) -> tuple[dict[str, Any], int]:
        """Delete an unreferenced model and all managed artifacts."""
        try:
            self._require_model_library().delete_model(model_id)
            return {"success": True}, 200
        except ModelNotFoundError as error:
            return {"error": str(error)}, 404
        except ModelReferencedError as error:
            return {"error": str(error)}, 409
        except ModelLibraryError as error:
            return {"error": str(error)}, 400

    def upload_model_artifact(
        self, model_id: str, slot: str
    ) -> tuple[dict[str, Any], int]:
        """Copy an uploaded artifact into a model's managed directory."""
        uploaded_file = request.files.get("file")
        if uploaded_file is None or not uploaded_file.filename:
            return {"error": "No artifact file provided"}, 400
        safe_filename = Path(uploaded_file.filename).name
        if safe_filename != uploaded_file.filename:
            return {"error": "Artifact filename must not contain a path"}, 400
        try:
            library = self._require_model_library()
            model, referenced_by = library.import_artifact(
                model_id,
                slot,
                uploaded_file.stream,
                filename=safe_filename,
            )
            restart_required = bool(referenced_by)
            if restart_required:
                self.restart_required_for_config = True
            return {
                "model": self._serialize_model(model, referenced_by),
                "affected_pipelines": list(referenced_by),
                "restart_required": restart_required,
            }, 200
        except ModelNotFoundError as error:
            return {"error": str(error)}, 404
        except (ArtifactError, ModelLibraryError) as error:
            return {"error": str(error)}, 400

    def delete_model_artifact(
        self, model_id: str, slot: str
    ) -> tuple[dict[str, Any], int]:
        """Remove a managed artifact and mark referenced pipelines for restart."""
        try:
            library = self._require_model_library()
            model, referenced_by = library.remove_artifact(model_id, slot)
            restart_required = bool(referenced_by)
            if restart_required:
                self.restart_required_for_config = True
            return {
                "model": self._serialize_model(model, referenced_by),
                "affected_pipelines": list(referenced_by),
                "restart_required": restart_required,
            }, 200
        except ModelNotFoundError as error:
            return {"error": str(error)}, 404
        except (ArtifactError, ModelLibraryError) as error:
            return {"error": str(error)}, 400

    def get_mx3_compilation(self) -> tuple[dict[str, Any], int]:
        """Return the retained state of the current or most recent compilation."""
        try:
            return {"compilation": self._require_mx3_compiler().status().to_dict()}, 200
        except Mx3CompilerError as error:
            return {"error": str(error)}, 503

    def start_mx3_compilation(self, model_id: str) -> tuple[dict[str, Any], int]:
        """Start one validated local ONNX-to-MX3 compilation job."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return {"error": "Request body must be an object"}, 400
        unknown_fields = set(payload) - {"settings", "profile", "overwrite"}
        if unknown_fields:
            return {
                "error": f"Unknown compilation fields: {sorted(unknown_fields)}"
            }, 400
        settings = payload.get("settings")
        profile = payload.get("profile")
        overwrite = payload.get("overwrite", False)
        if settings is not None and not isinstance(settings, dict):
            return {"error": "settings must be an object"}, 400
        if profile is not None and not isinstance(profile, dict):
            return {"error": "profile must be an object"}, 400
        if not isinstance(overwrite, bool):
            return {"error": "overwrite must be a boolean"}, 400
        try:
            status = self._require_mx3_compiler().start_compile(
                model_id,
                settings,
                profile=profile,
                overwrite=overwrite,
                callback=self._publish_mx3_compilation_progress,
            )
            return {"compilation": status.to_dict()}, 202
        except ModelNotFoundError as error:
            return {"error": str(error)}, 404
        except Mx3CompilerBusyError as error:
            return {"error": str(error)}, 409
        except (Mx3CompilerError, ArtifactError, ModelLibraryError) as error:
            return {"error": str(error)}, 400

    def cancel_mx3_compilation(self, job_id: str) -> tuple[dict[str, Any], int]:
        """Cancel the matching active compilation without affecting newer work."""
        try:
            compiler = self._require_mx3_compiler()
            status = compiler.cancel(job_id)
            if status.job_id != job_id:
                return {"error": "Unknown MX3 compilation job"}, 404
            return {"compilation": status.to_dict()}, 202
        except Mx3CompilerError as error:
            return {"error": str(error)}, 503

    def resolve_model_artifact(self, model_id: str) -> tuple[dict[str, Any], int]:
        """Resolve and expose the deterministic artifact for a selected device."""
        device_id = request.args.get("device_id", "")
        if not device_id:
            return {"error": "device_id query parameter is required"}, 400
        if self.device_registry is None:
            return {"error": "Device registry is not initialized"}, 503
        try:
            self.device_registry.get(device_id)
            artifact = self._require_model_library().resolve_artifact(
                model_id, device_id
            )
            return {
                "artifact": {
                    "slot": artifact.slot,
                    "filename": artifact.path.name,
                }
            }, 200
        except (ModelNotFoundError, DeviceNotFoundError) as error:
            return {"error": str(error)}, 404
        except (ArtifactError, ModelLibraryError, KeyError, RuntimeError) as error:
            return {"error": str(error)}, 400
