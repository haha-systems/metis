"""Content fingerprint for experiments created outside a Git checkout."""

import hashlib
from pathlib import Path


def source_fingerprint(root: Path = None) -> str:
    """Hash model, training, benchmark, PRD, and package configuration sources.

    The helper excludes itself so adding this metadata facility does not
    perturb the fingerprint of the implementation being measured.
    """
    project_root = root or Path(__file__).resolve().parents[1]
    sources = []
    for directory in ("metis", "training", "benchmarks"):
        sources.extend((project_root / directory).rglob("*.py"))
    sources.extend((project_root / "pyproject.toml", project_root / "prds" / "01.md"))
    digest = hashlib.sha256()
    for path in sorted(set(sources)):
        if path.resolve() == Path(__file__).resolve() or not path.is_file():
            continue
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
