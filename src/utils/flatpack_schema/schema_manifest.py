from __future__ import annotations

from dataclasses import dataclass
from struct import pack
from typing import Sequence


class SchemaKind:
    """Constants that describe manifest schema kinds."""

    OBJECT = 0
    OBJECT_ARRAY = 1
    FLOAT_ARRAY = 2


@dataclass(frozen=True)
class SchemaDescriptor:
    """Descriptor modeling the metadata needed for manifest encoding."""

    name: str
    kind: int
    component_names: Sequence[str] = ()


MANIFEST_HEADER = b"FPKM"
MANIFEST_VERSION = 1
MAX_NAME_LENGTH = 255

_SCHEMA_DESCRIPTORS: tuple[SchemaDescriptor, ...] = (
    SchemaDescriptor(
        name="pose3d",
        kind=SchemaKind.OBJECT,
        component_names=("x", "y", "z", "roll", "pitch", "yaw"),
    ),
    SchemaDescriptor(
        name="pose2d",
        kind=SchemaKind.OBJECT,
        component_names=("x", "y", "rotation"),
    ),
    SchemaDescriptor(
        name="vector3",
        kind=SchemaKind.OBJECT,
        component_names=("x", "y", "z"),
    ),
    SchemaDescriptor(
        name="vector2",
        kind=SchemaKind.OBJECT,
        component_names=("x", "y"),
    ),
    SchemaDescriptor(
        name="pose3d_array",
        kind=SchemaKind.OBJECT_ARRAY,
        component_names=("x", "y", "z", "roll", "pitch", "yaw"),
    ),
    SchemaDescriptor(
        name="pose2d_array",
        kind=SchemaKind.OBJECT_ARRAY,
        component_names=("x", "y", "rotation"),
    ),
    SchemaDescriptor(
        name="vector3_array",
        kind=SchemaKind.OBJECT_ARRAY,
        component_names=("x", "y", "z"),
    ),
    SchemaDescriptor(
        name="vector2_array",
        kind=SchemaKind.OBJECT_ARRAY,
        component_names=("x", "y"),
    ),
    SchemaDescriptor(
        name="float_array",
        kind=SchemaKind.FLOAT_ARRAY,
        component_names=(),
    ),
)


def _encode_length_prefixed_string(value: str) -> bytes:
    """Encode a string as a single-byte length followed by UTF-8 data."""

    encoded = value.encode("utf-8")
    if len(encoded) > MAX_NAME_LENGTH:
        raise ValueError("Name exceeds manifest length limit")
    return bytes([len(encoded)]) + encoded


def _encode_descriptor(descriptor: SchemaDescriptor) -> bytes:
    """Encode a descriptor entry into its binary manifest form."""

    encoded_components = b""
    for component_name in descriptor.component_names:
        encoded_components += _encode_length_prefixed_string(component_name)
    return (
        _encode_length_prefixed_string(descriptor.name)
        + bytes([descriptor.kind])
        + bytes([len(descriptor.component_names)])
        + encoded_components
    )


def generate_schema_manifest_bytes() -> bytes:
    """Return the Flatpack schema manifest encoded for NetworkTables distribution."""

    payload = bytearray()
    payload.extend(pack("<H", len(_SCHEMA_DESCRIPTORS)))
    for descriptor in _SCHEMA_DESCRIPTORS:
        payload.extend(_encode_descriptor(descriptor))

    manifest = bytearray(MANIFEST_HEADER)
    manifest.append(MANIFEST_VERSION)
    manifest.extend(pack("<I", len(payload)))
    manifest.extend(payload)
    return bytes(manifest)
