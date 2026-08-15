"""Thread-safe managed storage and metadata for inference models."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, Mapping

from src.config.utils.port_validation import PROJECT_ROOT

ARTIFACT_EXTENSIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "pt": frozenset({".pt"}),
        "onnx": frozenset({".onnx"}),
        "engine": frozenset({".engine"}),
        "mx3_dfp": frozenset({".dfp"}),
        "mx3_postprocessor": frozenset({".onnx"}),
    }
)


class _Unset:
    """Sentinel type for omitted optional metadata updates."""


_UNSET = _Unset()


class ModelLibraryError(RuntimeError):
    """Base error raised by the model library."""


class ModelNotFoundError(ModelLibraryError, KeyError):
    """Raised when a requested model ID is not present."""


class ArtifactError(ModelLibraryError):
    """Raised when a model artifact is invalid or incompatible."""


class ModelReferencedError(ModelLibraryError):
    """Raised when attempting to delete a referenced model."""


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Immutable public metadata for one managed model."""

    model_id: str
    display_name: str
    class_names: tuple[str, ...] | None
    artifacts: Mapping[str, str]
    mx3_profile: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """A model artifact selected for one canonical device ID."""

    model_id: str
    device_id: str
    slot: str
    path: Path
    postprocessor_path: Path | None = None
    mx3_profile: Mapping[str, Any] | None = None


class ModelLibrary:
    """Persist model metadata and artifacts below one managed directory."""

    def __init__(
        self,
        root: str | os.PathLike[str] = PROJECT_ROOT / "files" / "models",
        *,
        pipeline_config_path: str | os.PathLike[str] = (
            PROJECT_ROOT / "src" / "config" / "pipeline_config.json"
        ),
    ) -> None:
        """Initialize and validate the managed model library.

        Args:
            root: Directory containing the manifest and per-model directories.
            pipeline_config_path: Pipeline configuration used for reference checks.
        """
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.pipeline_config_path = Path(pipeline_config_path)
        self._lock = threading.RLock()
        if not self.manifest_path.exists():
            self._write_manifest({"version": 1, "models": {}})
        self._load_manifest()
        self._cleanup_stale_generated_artifacts()

    def _cleanup_stale_generated_artifacts(self) -> None:
        """Remove superseded generated MX3 files left from earlier compilations."""
        active_paths = {
            relative_path
            for record in self._load_manifest()["models"].values()
            for relative_path in record["artifacts"].values()
        }
        for pattern in ("*/mx3_dfp-*.dfp", "*/mx3_postprocessor-*.onnx"):
            for path in self.root.glob(pattern):
                if path.relative_to(self.root).as_posix() not in active_paths:
                    path.unlink(missing_ok=True)

    def _load_manifest(self) -> dict[str, Any]:
        """Load and minimally validate the model manifest."""
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelLibraryError(f"Invalid model manifest: {error}") from error
        if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
            raise ModelLibraryError("Invalid model manifest: models must be an object")
        self._validate_manifest_records(data["models"])
        return data

    @classmethod
    def _validate_manifest_records(cls, records: dict[str, Any]) -> None:
        """Validate record shapes and contain all artifact paths in model folders."""
        for model_id, record in records.items():
            if (
                not isinstance(model_id, str)
                or not model_id
                or Path(model_id).name != model_id
                or model_id in {".", ".."}
            ):
                raise ModelLibraryError(f"Invalid model ID in manifest: {model_id!r}")
            if not isinstance(record, dict) or record.get("model_id") != model_id:
                raise ModelLibraryError(
                    f"Invalid manifest record for model {model_id!r}"
                )
            cls._validate_display_name(record.get("display_name"))
            cls._validate_class_names(record.get("class_names"))
            cls._validate_mx3_profile(record.get("mx3_profile"))
            artifacts = record.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ModelLibraryError(
                    f"Artifacts for model {model_id!r} must be an object"
                )
            for slot, relative_path in artifacts.items():
                allowed_extensions = ARTIFACT_EXTENSIONS.get(slot)
                artifact_path = (
                    Path(relative_path) if isinstance(relative_path, str) else None
                )
                if (
                    allowed_extensions is None
                    or artifact_path is None
                    or artifact_path.is_absolute()
                    or ".." in artifact_path.parts
                    or len(artifact_path.parts) != 2
                    or artifact_path.parts[0] != model_id
                    or artifact_path.suffix.lower() not in allowed_extensions
                ):
                    raise ModelLibraryError(
                        f"Invalid {slot!r} artifact path for model {model_id!r}: "
                        f"{relative_path!r}"
                    )

    def _write_manifest(self, data: dict[str, Any]) -> None:
        """Atomically replace the complete model manifest."""
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=".manifest.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.manifest_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _validate_display_name(display_name: object) -> str:
        """Validate and normalize a display name."""
        if not isinstance(display_name, str) or not display_name.strip():
            raise ModelLibraryError("display_name must be a non-empty string")
        return display_name.strip()

    @staticmethod
    def _validate_class_names(class_names: object) -> list[str] | None:
        """Validate an optional ordered class-name list."""
        if class_names is None:
            return None
        if not isinstance(class_names, (list, tuple)) or not all(
            isinstance(class_name, str) and class_name.strip()
            for class_name in class_names
        ):
            raise ModelLibraryError(
                "class_names must be null or an ordered list of non-empty strings"
            )
        return [class_name.strip() for class_name in class_names]

    @staticmethod
    def _validate_mx3_profile(profile: object) -> dict[str, Any] | None:
        """Validate optional MX3 profile metadata without interpreting it."""
        if profile is None:
            return None
        if not isinstance(profile, dict):
            raise ModelLibraryError("mx3_profile must be null or an object")
        return dict(profile)

    @staticmethod
    def _public_metadata(record: dict[str, Any]) -> ModelMetadata:
        """Create an immutable public view of a manifest record."""
        profile = record.get("mx3_profile")
        class_names = record.get("class_names")
        return ModelMetadata(
            model_id=record["model_id"],
            display_name=record["display_name"],
            class_names=tuple(class_names) if class_names is not None else None,
            artifacts=MappingProxyType(dict(record.get("artifacts", {}))),
            mx3_profile=(
                MappingProxyType(dict(profile)) if profile is not None else None
            ),
        )

    def create_model(
        self,
        display_name: str,
        class_names: list[str] | tuple[str, ...] | None = None,
        mx3_profile: dict[str, Any] | None = None,
    ) -> ModelMetadata:
        """Create a model record with a stable random ID."""
        normalized_name = self._validate_display_name(display_name)
        normalized_classes = self._validate_class_names(class_names)
        normalized_profile = self._validate_mx3_profile(mx3_profile)
        with self._lock:
            data = self._load_manifest()
            model_id = str(uuid.uuid4())
            record = {
                "model_id": model_id,
                "display_name": normalized_name,
                "class_names": normalized_classes,
                "artifacts": {},
                "mx3_profile": normalized_profile,
            }
            data["models"][model_id] = record
            self._write_manifest(data)
            return self._public_metadata(record)

    def list_models(self) -> tuple[ModelMetadata, ...]:
        """Return models ordered by display name and stable ID."""
        with self._lock:
            records = self._load_manifest()["models"].values()
            ordered_records = sorted(
                records,
                key=lambda record: (
                    str(record.get("display_name", "")).casefold(),
                    str(record.get("model_id", "")),
                ),
            )
            return tuple(self._public_metadata(record) for record in ordered_records)

    def get_model(self, model_id: str) -> ModelMetadata:
        """Return model metadata by exact stable ID."""
        with self._lock:
            record = self._load_manifest()["models"].get(model_id)
            if record is None:
                raise ModelNotFoundError(f"Unknown model ID: {model_id!r}")
            return self._public_metadata(record)

    def update_model(
        self,
        model_id: str,
        *,
        display_name: str | _Unset = _UNSET,
        class_names: list[str] | tuple[str, ...] | _Unset | None = _UNSET,
        mx3_profile: dict[str, Any] | _Unset | None = _UNSET,
    ) -> ModelMetadata:
        """Update supplied metadata fields while preserving the stable model ID."""
        with self._lock:
            data = self._load_manifest()
            record = data["models"].get(model_id)
            if record is None:
                raise ModelNotFoundError(f"Unknown model ID: {model_id!r}")
            if display_name is not _UNSET:
                record["display_name"] = self._validate_display_name(display_name)
            if class_names is not _UNSET:
                record["class_names"] = self._validate_class_names(class_names)
            if mx3_profile is not _UNSET:
                record["mx3_profile"] = self._validate_mx3_profile(mx3_profile)
            self._write_manifest(data)
            return self._public_metadata(record)

    def references(self, model_id: str) -> tuple[str, ...]:
        """Return pipeline names containing an operation that references a model."""
        if not self.pipeline_config_path.exists():
            return ()
        try:
            config = json.loads(self.pipeline_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelLibraryError(
                f"Cannot read pipeline configuration: {error}"
            ) from error

        referenced_by: set[str] = set()
        if isinstance(config, dict):
            for pipeline_name, operations in config.items():
                if not isinstance(operations, list):
                    continue
                if any(
                    isinstance(operation, dict)
                    and isinstance(operation.get("action_params"), dict)
                    and operation["action_params"].get("model_id") == model_id
                    for operation in operations
                ):
                    referenced_by.add(str(pipeline_name))
        return tuple(sorted(referenced_by))

    @staticmethod
    def _source_filename(
        source: str | os.PathLike[str] | BinaryIO,
        filename: str | None,
    ) -> str:
        """Resolve a safe display filename for an artifact source."""
        candidate = filename
        if candidate is None and isinstance(source, (str, os.PathLike)):
            candidate = Path(source).name
        if not candidate or Path(candidate).name != candidate:
            raise ArtifactError("Artifact filename must be a plain filename")
        return candidate

    def import_artifact(
        self,
        model_id: str,
        slot: str,
        source: str | os.PathLike[str] | BinaryIO,
        *,
        filename: str | None = None,
    ) -> tuple[ModelMetadata, tuple[str, ...]]:
        """Copy an artifact into managed storage and report affected pipelines."""
        allowed_extensions = ARTIFACT_EXTENSIONS.get(slot)
        if allowed_extensions is None:
            raise ArtifactError(f"Unknown artifact slot: {slot!r}")
        source_filename = self._source_filename(source, filename)
        extension = Path(source_filename).suffix.lower()
        if extension not in allowed_extensions:
            allowed = ", ".join(sorted(allowed_extensions))
            raise ArtifactError(
                f"Invalid extension for {slot}; expected one of: {allowed}"
            )

        with self._lock:
            data = self._load_manifest()
            record = data["models"].get(model_id)
            if record is None:
                raise ModelNotFoundError(f"Unknown model ID: {model_id!r}")
            # Resolved before committing so an unreadable pipeline configuration
            # cannot turn a completed import into a reported failure.
            references = self.references(model_id)

            relative_path = f"{model_id}/{slot}{extension}"
            target_path = self.root / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=target_path.parent,
                prefix=f".{slot}.",
                suffix=".tmp",
            )
            replaced_name: str | None = None
            try:
                with os.fdopen(file_descriptor, "wb") as output_stream:
                    if isinstance(source, (str, os.PathLike)):
                        source_path = Path(source)
                        if not source_path.is_file():
                            raise ArtifactError(
                                f"Artifact does not exist: {source_path}"
                            )
                        with source_path.open("rb") as input_stream:
                            shutil.copyfileobj(input_stream, output_stream)
                    else:
                        shutil.copyfileobj(source, output_stream)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                # Keep the currently installed file until the manifest commits so
                # a failed write cannot destroy the artifact it still records.
                if target_path.exists():
                    replaced_descriptor, replaced_name = tempfile.mkstemp(
                        dir=target_path.parent,
                        prefix=f".{slot}.replaced.",
                        suffix=".tmp",
                    )
                    os.close(replaced_descriptor)
                    os.replace(target_path, replaced_name)
                os.replace(temporary_name, target_path)
            except BaseException:
                if replaced_name is not None and os.path.exists(replaced_name):
                    os.replace(replaced_name, target_path)
                    replaced_name = None
                raise
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

            previous_relative_path = record["artifacts"].get(slot)
            record["artifacts"][slot] = relative_path
            try:
                self._write_manifest(data)
            except BaseException:
                if replaced_name is not None:
                    os.replace(replaced_name, target_path)
                else:
                    target_path.unlink(missing_ok=True)
                raise
            finally:
                if replaced_name is not None and os.path.exists(replaced_name):
                    os.unlink(replaced_name)
            if previous_relative_path and previous_relative_path != relative_path:
                (self.root / previous_relative_path).unlink(missing_ok=True)
            return self._public_metadata(record), references

    def install_mx3_bundle(
        self,
        model_id: str,
        dfp_source: str | os.PathLike[str],
        postprocessor_source: str | os.PathLike[str] | None,
        profile: Mapping[str, Any],
        *,
        overwrite: bool = False,
        expected_bundle: (
            tuple[str | None, str | None, Mapping[str, Any] | None] | None
        ) = None,
        expected_onnx: tuple[str, int, int] | None = None,
    ) -> tuple[ModelMetadata, tuple[str, ...]]:
        """Atomically make a compiled DFP, optional post model, and profile current.

        The files receive generation-specific names before a single manifest swap.
        Consequently readers see either the old complete MX3 bundle or the new
        complete bundle, never a partially copied compilation result.  Replacing
        a bundle is deliberately opt-in because active pipelines must be
        restarted to consume the new manifest entry.
        """
        from src.utils.mx3_runtime import Mx3Profile, Mx3RuntimeError

        try:
            normalized_profile = Mx3Profile.from_metadata(profile).to_metadata()
        except Mx3RuntimeError as error:
            raise ArtifactError(f"Invalid MX3 profile: {error}") from error

        dfp_path = Path(dfp_source)
        post_path = (
            Path(postprocessor_source) if postprocessor_source is not None else None
        )
        if (
            not dfp_path.is_file()
            or dfp_path.suffix.lower() != ".dfp"
            or dfp_path.stat().st_size == 0
        ):
            raise ArtifactError(
                "MX3 bundle DFP source must be a non-empty existing .dfp file"
            )
        if post_path is not None and (
            not post_path.is_file()
            or post_path.suffix.lower() != ".onnx"
            or post_path.stat().st_size == 0
        ):
            raise ArtifactError(
                "MX3 bundle postprocessor source must be an existing .onnx file"
            )

        with self._lock:
            data = self._load_manifest()
            record = data["models"].get(model_id)
            if record is None:
                raise ModelNotFoundError(f"Unknown model ID: {model_id!r}")
            # Resolved before committing so an unreadable pipeline configuration
            # cannot turn a completed installation into a reported failure.
            references = self.references(model_id)
            if expected_onnx is not None:
                current_onnx = record["artifacts"].get("onnx")
                if current_onnx is None:
                    raise ArtifactError(
                        "ONNX artifact changed while compilation was running"
                    )
                current_onnx_path = self._managed_artifact_path(current_onnx)
                current_onnx_state = (
                    current_onnx,
                    current_onnx_path.stat().st_size,
                    current_onnx_path.stat().st_mtime_ns,
                )
                if current_onnx_state != expected_onnx:
                    raise ArtifactError(
                        "ONNX artifact changed while compilation was running"
                    )

            current_bundle = (
                record["artifacts"].get("mx3_dfp"),
                record["artifacts"].get("mx3_postprocessor"),
                record.get("mx3_profile"),
            )
            if expected_bundle is not None and current_bundle != expected_bundle:
                raise ArtifactError(
                    "MX3 artifacts or profile changed while compilation was running"
                )
            has_existing_bundle = any(value is not None for value in current_bundle)
            if has_existing_bundle and not overwrite:
                raise ArtifactError(
                    "MX3 artifacts or profile already exist; explicit overwrite is required"
                )

            model_directory = self.root / model_id
            model_directory.mkdir(parents=True, exist_ok=True)
            generation = uuid.uuid4().hex
            new_paths = {
                "mx3_dfp": model_directory / f"mx3_dfp-{generation}.dfp",
            }
            if post_path is not None:
                new_paths["mx3_postprocessor"] = (
                    model_directory / f"mx3_postprocessor-{generation}.onnx"
                )
            installation_sources = {"mx3_dfp": dfp_path}
            if post_path is not None:
                installation_sources["mx3_postprocessor"] = post_path
            temporary_paths: dict[str, Path] = {}
            installed_paths: dict[str, Path] = {}
            manifest_committed = False
            try:
                for slot, source_path in installation_sources.items():
                    descriptor, temporary_name = tempfile.mkstemp(
                        dir=model_directory,
                        prefix=f".{slot}-{generation}-",
                        suffix=".tmp",
                    )
                    temporary_path = Path(temporary_name)
                    temporary_paths[slot] = temporary_path
                    with os.fdopen(descriptor, "wb") as output_stream:
                        with source_path.open("rb") as input_stream:
                            shutil.copyfileobj(input_stream, output_stream)
                        output_stream.flush()
                        os.fsync(output_stream.fileno())
                for slot, target_path in new_paths.items():
                    os.replace(temporary_paths[slot], target_path)
                    installed_paths[slot] = target_path

                record["artifacts"]["mx3_dfp"] = (
                    new_paths["mx3_dfp"].relative_to(self.root).as_posix()
                )
                if post_path is None:
                    record["artifacts"].pop("mx3_postprocessor", None)
                else:
                    record["artifacts"]["mx3_postprocessor"] = (
                        new_paths["mx3_postprocessor"].relative_to(self.root).as_posix()
                    )
                record["mx3_profile"] = normalized_profile
                self._write_manifest(data)
                manifest_committed = True
            except BaseException:
                if not manifest_committed:
                    for path in (*temporary_paths.values(), *installed_paths.values()):
                        path.unlink(missing_ok=True)
                raise

            # Superseded generations are deliberately left on disk: a concurrent
            # resolve_artifact() may still be opening the path it just read from
            # the manifest.  _cleanup_stale_generated_artifacts() removes them at
            # the next library initialization, when no reader can reference them.
            return self._public_metadata(record), references

    def remove_artifact(
        self, model_id: str, slot: str
    ) -> tuple[ModelMetadata, tuple[str, ...]]:
        """Remove an artifact and report pipelines that require restart."""
        if slot not in ARTIFACT_EXTENSIONS:
            raise ArtifactError(f"Unknown artifact slot: {slot!r}")
        with self._lock:
            data = self._load_manifest()
            record = data["models"].get(model_id)
            if record is None:
                raise ModelNotFoundError(f"Unknown model ID: {model_id!r}")
            # Resolved before committing so an unreadable pipeline configuration
            # cannot turn a completed removal into a reported failure.
            references = self.references(model_id)
            relative_path = record["artifacts"].pop(slot, None)
            if relative_path is None:
                raise ArtifactError(f"Model has no {slot} artifact")
            self._write_manifest(data)
            (self.root / relative_path).unlink(missing_ok=True)
            return self._public_metadata(record), references

    def delete_model(self, model_id: str) -> None:
        """Delete an unreferenced model and all of its managed artifacts."""
        with self._lock:
            referenced_by = self.references(model_id)
            if referenced_by:
                raise ModelReferencedError(
                    f"Model {model_id} is referenced by: {', '.join(referenced_by)}"
                )
            data = self._load_manifest()
            record = data["models"].pop(model_id, None)
            if record is None:
                raise ModelNotFoundError(f"Unknown model ID: {model_id!r}")
            self._write_manifest(data)
            shutil.rmtree(self.root / model_id, ignore_errors=True)

    def _managed_artifact_path(self, relative_path: str) -> Path:
        """Resolve and validate one manifest artifact path."""
        artifact_path = (self.root / relative_path).resolve()
        if self.root not in artifact_path.parents or not artifact_path.is_file():
            raise ArtifactError(
                f"Managed artifact is missing or unsafe: {relative_path!r}"
            )
        return artifact_path

    def resolve_artifact(self, model_id: str, device_id: str) -> ResolvedArtifact:
        """Select a compatible artifact using deterministic device priority."""
        model = self.get_model(model_id)
        if device_id == "cpu":
            choices = ("onnx", "pt")
        elif device_id.startswith("cuda:") and device_id[5:].isdigit():
            choices = ("engine", "pt", "onnx")
        elif device_id.startswith("mx3:") and device_id[4:].isdigit():
            if "mx3_dfp" not in model.artifacts:
                raise ArtifactError("MX3 requires an mx3_dfp artifact")
            if model.mx3_profile is None:
                raise ArtifactError("MX3 requires profile metadata")
            choices = ("mx3_dfp",)
        else:
            raise ArtifactError(
                f"Invalid or unsupported canonical device ID: {device_id!r}"
            )

        slot = next(
            (candidate for candidate in choices if candidate in model.artifacts),
            None,
        )
        if slot is None:
            raise ArtifactError(
                f"Model {model_id} has no compatible artifact for {device_id}"
            )
        artifact_path = self._managed_artifact_path(model.artifacts[slot])

        postprocessor_path = None
        if slot == "mx3_dfp" and "mx3_postprocessor" in model.artifacts:
            postprocessor_path = self._managed_artifact_path(
                model.artifacts["mx3_postprocessor"]
            )
        return ResolvedArtifact(
            model_id=model_id,
            device_id=device_id,
            slot=slot,
            path=artifact_path,
            postprocessor_path=postprocessor_path,
            mx3_profile=model.mx3_profile,
        )
