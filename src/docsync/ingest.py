"""Stage B1 — whole-repo ingest for `docsync bootstrap`.

Where the update pipeline starts from a *diff*, bootstrap starts from a *snapshot*:
walk a service repo (strictly read-only) and distill each source file into a
lightweight :class:`~docsync.models.SourceUnit` (path + kind + top-level symbol
names — never the file body). A whole repo's worth of these is small enough to
hand the planner in one prompt; the actual file text is read per-page, later, only
for the pages the planner decides to author.

Symbol extraction is AST-based for Python (more accurate than the diff module's
hunk regexes, which don't apply to whole files), with a regex fallback for files
that don't parse and a light export scan for TypeScript.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import re
from pathlib import Path

from .models import RepoDigest, SourceUnit

__all__ = ["walk_repo", "walk_repos", "extract_symbols", "read_excerpt"]

# Sensible defaults so a caller can `walk_repo(path)` and get just source code.
DEFAULT_INCLUDE = ("*.py", "*.ts", "*.tsx")
# Directories that never hold documentable source — pruned during the walk so we
# don't read (or even stat) thousands of irrelevant files.
DEFAULT_EXCLUDE_DIRS = frozenset(
    {
        ".git", ".github", "node_modules", ".venv", "venv", "__pycache__",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".next",
        "tests", "test", "migrations", "__tests__", ".docsync",
        # Universally non-product noise — never the surface you document.
        ".tox", "htmlcov", "coverage", "site-packages", "vendor", ".idea", ".vscode",
    }
)


def resolve_exclude_dirs(extra: list[str] | None = None) -> frozenset[str]:
    """Combine :data:`DEFAULT_EXCLUDE_DIRS` with caller/config-supplied names.

    `extra` is typically ``DocsyncConfig.ingest_exclude_dirs`` — repo-specific dirs
    (e.g. ``examples``, ``deploy``, a generated ``docs`` site) that are noise for the
    plan but too project-specific to bake into the defaults. Blank entries are dropped.
    """
    return DEFAULT_EXCLUDE_DIRS | frozenset(d.strip() for d in (extra or []) if d.strip())

# Per-file excerpt budget handed to the author stage (chars). Generous enough to
# show a route module's signatures without blowing the author prompt.
_EXCERPT_MAX_CHARS = 8_000

# TypeScript/JS top-level exports — best-effort (no TS parser in-tree).
_TS_EXPORT_RE = re.compile(
    r"^export\s+(?:default\s+)?(?:async\s+)?"
    r"(?:function|const|let|var|class|interface|type|enum)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
# Reused for the Python regex fallback when ast.parse fails (syntax error / py2).
_PY_DEF_OR_CLASS = re.compile(r"^(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", re.MULTILINE)


def _kind(path: str) -> str:
    if path.endswith(".py"):
        return "python"
    if path.endswith((".ts", ".tsx")):
        return "typescript"
    return "other"


def _python_symbols(text: str) -> list[str]:
    """Top-level functions, classes, and module-level assignment names via AST.

    Only *module-level* defs/classes/assignments are returned — nested helpers and
    methods are noise for anchoring. Falls back to a line regex when the file
    doesn't parse (partial file, syntax error, Python 2), so ingest never crashes
    on one bad file.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(name: str | None) -> None:
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        for m in _PY_DEF_OR_CLASS.finditer(text):
            _add(m.group(1))
        return out

    for node in tree.body:  # module level only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    _add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            _add(node.target.id)
    return out


def extract_symbols(path: str, text: str) -> list[str]:
    """Top-level symbol names for a whole source file (language-dispatched).

    Python → AST (regex fallback); TypeScript → export regex; anything else → [].
    Unlike `diff.extract_changed_symbols`, this works on full file text, not hunks.
    """
    kind = _kind(path)
    if kind == "python":
        return _python_symbols(text)
    if kind == "typescript":
        seen: set[str] = set()
        out: list[str] = []
        for m in _TS_EXPORT_RE.finditer(text):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out
    return []


def _matches_any(name: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in globs)


def walk_repo(
    repo_path: str | Path,
    *,
    repo: str | None = None,
    include_globs: tuple[str, ...] = DEFAULT_INCLUDE,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
    max_files: int = 0,
) -> RepoDigest:
    """Read-only walk of *repo_path* → a lightweight :class:`RepoDigest`.

    NEVER writes to the repo. Directories in *exclude_dirs* are pruned (not
    descended). Files whose basename matches *include_globs* are read once to
    extract symbols. *max_files* > 0 caps how many files are ingested (0 =
    unlimited); units are returned in sorted path order for determinism.

    Args:
        repo_path: Local checkout to walk.
        repo: Identifier to stamp on the digest (defaults to the dir name).
        include_globs: fnmatch patterns over file *basenames* to ingest.
        exclude_dirs: directory names pruned anywhere in the tree.
        max_files: optional cap on ingested files (0 = no cap).
    """
    root = Path(repo_path).resolve()
    repo_id = repo or root.name
    units: list[SourceUnit] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded dirs in place so os.walk doesn't descend into them.
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs)
        for filename in sorted(filenames):
            if not _matches_any(filename, include_globs):
                continue
            abs_path = Path(dirpath) / filename
            rel = abs_path.relative_to(root).as_posix()
            try:
                text = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            units.append(
                SourceUnit(path=rel, kind=_kind(rel), symbols=extract_symbols(rel, text))
            )
            if max_files and len(units) >= max_files:
                return RepoDigest(repo=repo_id, root=str(root), units=units)

    return RepoDigest(repo=repo_id, root=str(root), units=units)


def walk_repos(
    specs: list[tuple[str, str | Path]],
    *,
    include_globs: tuple[str, ...] = DEFAULT_INCLUDE,
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
    max_files: int = 0,
) -> list[RepoDigest]:
    """Read-only walk of several repos → one :class:`RepoDigest` each.

    `specs` is a list of ``(repo_id, path)`` pairs. Used by `docsync bootstrap` to
    ingest a whole platform (e.g. all four Keep services) for a cross-repo doc plan.
    Digests are returned in spec order; each walk is independent and read-only.
    """
    return [
        walk_repo(
            path, repo=repo_id, include_globs=include_globs,
            exclude_dirs=exclude_dirs, max_files=max_files,
        )
        for repo_id, path in specs
    ]


def read_excerpt(root: str | Path, rel_path: str, *, max_chars: int = _EXCERPT_MAX_CHARS) -> str:
    """Read a source file (read-only) for the author stage, truncated to a budget.

    Returns "" if the file is missing/unreadable so one bad path can't sink the
    author of an otherwise-fine page. A truncation marker is appended when capped.
    """
    fp = Path(root) / rel_path
    try:
        text = fp.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n… (truncated)\n"
    return text
