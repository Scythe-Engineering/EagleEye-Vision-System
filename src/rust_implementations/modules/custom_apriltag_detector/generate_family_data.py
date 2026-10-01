"""Translate the pinned, licensed family data into Rust without copying detector code."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

MODULE = Path(__file__).resolve().parent
SOURCE_SHA256 = "a188940eda078da769f3302e83fe58f909c689622c1a6e50c9e24ec28bcfa004"


def generate(source_path: Path) -> str:
    """Translate the externally supplied checksum-pinned licensed family tables."""
    data = source_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Frozen family-data checksum does not match its identity")
    source = data.decode("ascii")
    tables: dict[str, tuple[str, list[int]]] = {}
    for kind, name, body in re.findall(
        r"static const (uint64_t|int) (\w+)\[\] = \{(.*?)\};", source, re.DOTALL
    ):
        values = [
            int(value.strip().removesuffix("ULL"), 0)
            for value in body.split(",")
            if value.strip()
        ]
        tables[name] = ("u64" if kind == "uint64_t" else "i32", values)
    families = re.findall(
        r'\{"([^"]+)",(\d+),(\d+),(\d+),(\d+),(\d+),([01]),(\w+),(\w+),(\w+)\}', source
    )
    if len(families) != 8 or len(tables) != 24:
        raise ValueError("Expected exactly eight families and 24 integer tables")
    lines = [
        "//! Generated licensed family data. See LICENSE.family-data and family-data-sha256.json.",
        "//! Regenerate with generate_family_data.py; no detector implementation is imported.",
        "",
        "use crate::families::Family;",
        "",
    ]
    for name, (kind, values) in tables.items():
        lines.append(
            f"#[rustfmt::skip]\nstatic {name.upper()}: [{kind}; {len(values)}] = ["
        )
        for start in range(0, len(values), 12):
            block = values[start : start + 12]
            lines.append(
                "    "
                + ", ".join(hex(v) if kind == "u64" else str(v) for v in block)
                + ","
            )
        lines.append("];\n")
    lines.append("pub(crate) static FAMILIES: [Family; 8] = [")
    for (
        name,
        count,
        bits,
        hamming,
        width,
        total,
        reversed_border,
        codes,
        x,
        y,
    ) in families:
        nbits, ncodes = int(bits), int(count)
        if (
            len(tables[codes][1]) != ncodes
            or len(tables[x][1]) != nbits
            or len(tables[y][1]) != nbits
        ):
            raise ValueError(f"Family lengths disagree for {name}")
        if any(not 0 <= value < 1 << nbits for value in tables[codes][1]):
            raise ValueError(f"Out-of-range codeword in {name}")
        lines.append(
            f'    Family {{ name: "{name}", nbits: {bits}, h: {hamming}, width: {width}, total: {total}, reversed: {"true" if reversed_border == "1" else "false"}, codes: &{codes.upper()}, x: &{x.upper()}, y: &{y.upper()} }},'
        )
    lines.append("];\n")
    return "\n".join(lines)


def main() -> None:
    """Regenerate tables from an explicit input without relying on benchmark archives.

    The checked-in tables already suffice for builds. The original license and
    upstream family source checksums remain alongside this generator.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the pinned families.inc")
    args = parser.parse_args()
    (MODULE / "src/family_data.rs").write_text(generate(args.source))


if __name__ == "__main__":
    main()
