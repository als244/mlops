"""Completeness and isolation checks for the standalone documentation."""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path

import mlops
from mlops import dispatch, explicit, operator_metadata, optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
DOCS = (
    PROJECT_ROOT / "README.md",
    *sorted(DOCS_ROOT.rglob("*.md")),
    *sorted((PROJECT_ROOT / "tests").rglob("*.md")),
)


def _operation_section(reference: str, name: str) -> str:
    match = re.search(
        rf"^### `{re.escape(name)}`\s*$\n(.*?)(?=^#{{2,3}} |\Z)",
        reference,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing API section for {name}"
    return match.group(1)


def _heading_anchors(source: Path) -> set[str]:
    """Return GitHub-style heading anchors, including duplicate suffixes."""
    anchors = set()
    occurrences: dict[str, int] = {}
    for line in source.read_text().splitlines():
        if not re.match(r"^#{1,6} ", line):
            continue
        heading = line.lstrip("#").strip().lower()
        base = re.sub(r"[^a-z0-9 _-]", "", heading).replace(" ", "-")
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        anchors.add(base if occurrence == 0 else f"{base}-{occurrence}")
    return anchors


def test_document_set_is_complete_and_self_contained():
    assert all(path.is_file() for path in DOCS)
    forbidden = ("reference_models", "dataflow", "../operations.py")
    for path in DOCS:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"{path.relative_to(PROJECT_ROOT)}: {token}"


def test_local_markdown_links_resolve_inside_project():
    link_pattern = re.compile(r"\]\(([^)#]+\.md)(?:#[^)]+)?\)")
    for source in DOCS:
        for target in link_pattern.findall(source.read_text()):
            destination = (source.parent / target).resolve()
            assert destination.is_relative_to(PROJECT_ROOT), (source, target)
            assert destination.is_file(), (source, target)


def test_local_markdown_anchors_resolve():
    link_pattern = re.compile(r"\]\((?:(?P<path>[^)#]+\.md))?(?:#(?P<anchor>[^)]+))?\)")
    for source in DOCS:
        for match in link_pattern.finditer(source.read_text()):
            anchor = match.group("anchor")
            if anchor is None:
                continue
            path = match.group("path")
            destination = source if path is None else (source.parent / path).resolve()
            assert anchor in _heading_anchors(destination), (
                source.relative_to(PROJECT_ROOT),
                match.group(0),
            )


def test_public_manuals_are_reachable_from_the_documentation_index():
    manuals = {path.resolve() for path in DOCS_ROOT.rglob("*.md")}
    pending = [DOCS_ROOT / "README.md"]
    reached = set()
    links = re.compile(r"\]\(([^)#]+\.md)(?:#[^)]+)?\)")
    while pending:
        page = pending.pop()
        if page in reached:
            continue
        reached.add(page)
        for raw in links.findall(page.read_text()):
            target = (page.parent / raw).resolve()
            if target in manuals and target not in reached:
                pending.append(target)
    assert not manuals - reached, sorted(
        str(path.relative_to(PROJECT_ROOT)) for path in manuals - reached
    )


def test_python_examples_compile():
    """Shape signatures use text fences; Python examples must have valid syntax."""
    blocks = re.compile(
        r"^(?P<indent> *)```python\n(?P<body>.*?)\n(?P=indent)```",
        re.MULTILINE | re.DOTALL,
    )
    for page in DOCS:
        for index, block in enumerate(blocks.finditer(page.read_text())):
            ast.parse(textwrap.dedent(block["body"]), filename=f"{page}:{index}")


def test_semantic_api_signatures_are_documented():
    reference = (DOCS_ROOT / "OPS.md").read_text()
    for name in mlops.__all__:
        section = _operation_section(reference, name)
        for parameter in inspect.signature(getattr(mlops, name)).parameters:
            assert parameter in section, f"{name}: {parameter}"


def test_explicit_api_signatures_are_documented():
    reference = (DOCS_ROOT / "EXPLICIT_OPS.md").read_text()
    for name in explicit.__all__:
        match = re.search(
            rf"^### `explicit\.{re.escape(name)}`\s*$\n(.*?)"
            rf"(?=^#{{2,3}} |\Z)",
            reference,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match is not None, f"missing explicit API section for {name}"
        section = match.group(1)
        module = getattr(explicit, name)
        for entrypoint in ("forward", "backward"):
            for parameter in inspect.signature(getattr(module, entrypoint)).parameters:
                assert parameter in section, f"{name}.{entrypoint}: {parameter}"


def test_optimizer_api_signatures_are_documented():
    reference = (DOCS_ROOT / "OPTIMIZERS.md").read_text()
    for name in optim.__all__:
        match = re.search(
            rf"^## `{re.escape(name)}`\s*$\n(.*?)(?=^## |\Z)",
            reference,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match is not None, f"missing optimizer API section for {name}"
        for parameter in inspect.signature(getattr(optim, name)).parameters:
            assert parameter in match.group(1), f"{name}: {parameter}"


def test_dispatch_exports_appear_in_dispatch_reference():
    reference = (DOCS_ROOT / "DISPATCH.md").read_text()
    for name in dispatch.__all__:
        assert name in reference, f"undocumented dispatch export: {name}"


def test_compiler_metadata_exports_are_documented():
    reference = (DOCS_ROOT / "COMPILER_METADATA.md").read_text()
    for name in operator_metadata.__all__:
        assert name in reference, f"undocumented operator metadata: {name}"


def test_extension_guide_covers_supported_workflows():
    reference = (DOCS_ROOT / "EXTENDING.md").read_text()
    required_sections = (
        "## Add an implementation for an existing operation",
        "## Add a new semantic operation",
        "## Compiler-facing declarations",
        "## Validation matrix",
    )
    for section in required_sections:
        assert section in reference
    # Examples must lead to real adapters, not proposed registrations presented
    # beside supported implementations. API completeness is checked separately.
    examples = re.findall(r"\]\((\.\./src/mlops/providers/[^)]+\.py)\)", reference)
    assert examples, "link to working provider adapters"
    for example in examples:
        assert (DOCS_ROOT / example).resolve().is_file(), example


def test_markdown_tables_have_consistent_column_counts():
    separator = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+$")
    malformed = []
    for source in DOCS:
        lines = source.read_text().splitlines()
        for index in range(len(lines) - 1):
            if not lines[index].startswith("|") or not separator.match(
                lines[index + 1]
            ):
                continue
            expected = lines[index].count("|")
            row = index + 2
            while row < len(lines) and lines[row].startswith("|"):
                if lines[row].count("|") != expected:
                    malformed.append(f"{source.name}:{row + 1}")
                row += 1
    assert not malformed, malformed
