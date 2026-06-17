"""Mintlify doc-framework adapter.

Mintlify pages are `.mdx` (also `.md`) files with YAML frontmatter and inline
MDX components (`<Card>`, `<CardGroup>`, `<Steps>`, ...). This adapter knows how
to read that frontmatter, fingerprint the structural shapes an edit must preserve,
and run Mintlify's own broken-link checker over a patched tree.

It is the only adapter the MVP ships; `base.DocAdapter` is the seam where a
Docusaurus / GitBook adapter would slot in later.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import frontmatter

from docsync.adapters.base import DocAdapter

# MDX component opening tags we count. Restricting to a known set keeps the
# signature stable against stray capitalised prose words while still covering the
# components Mintlify pages actually use.
_KNOWN_COMPONENTS = frozenset(
    {
        "CardGroup",
        "Card",
        "Warning",
        "Note",
        "Tip",
        "Info",
        "Steps",
        "Step",
        "Accordion",
        "AccordionGroup",
        "Tabs",
        "Tab",
        "CodeGroup",
        "Frame",
    }
)

# Opening tag like `<Card ...>` / `<Card>` / `<Card/>`. The capture is the name.
_OPEN_TAG_RE = re.compile(r"<([A-Z][A-Za-z]*)")
# Closing tag like `</Card>`.
_CLOSE_TAG_RE = re.compile(r"</([A-Z][A-Za-z]*)")
# A code-fence marker is a line whose first non-space content is ``` (3+ backticks).
_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
# A mermaid fence opener: ```mermaid (optionally with trailing space).
_MERMAID_RE = re.compile(r"^\s*```mermaid\b", re.MULTILINE)

# Component tag in document order: opening `<Card ...>`, self-closing `<Card/>`, or
# closing `</Card>`. Groups: (1) leading slash, (2) name, (3) trailing slash.
_TAG_RE = re.compile(r"<(/?)([A-Z][A-Za-z]*)\b[^>]*?(/?)>", re.DOTALL)
# Fenced code blocks and inline code spans — component tags inside them are example
# text, not real structure, and must be ignored by the well-formedness scan.
_FENCE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")

# Broken-link check tuning: be defensive, never let the doc pipeline hang on it.
_LINK_CHECK_TIMEOUT_S = 30
_LINK_PROBLEM_RE = re.compile(r"broken|not found|404|missing|unreachable", re.IGNORECASE)
# Output that means the CLI itself never ran (package absent, npx aborted, ...).
# Such lines must NOT be mistaken for broken-link findings.
_CLI_UNAVAILABLE_RE = re.compile(
    r"npm error|npx canceled|command not found|could not determine executable"
    r"|is not recognized|need to install",
    re.IGNORECASE,
)


class MintlifyAdapter(DocAdapter):
    """Adapter for Mintlify-flavoured MDX docs."""

    name = "mintlify"

    # -- ownership ---------------------------------------------------------

    def owns(self, page_path: str) -> bool:
        """True for `.mdx` and `.md` files (case-insensitive)."""
        lowered = page_path.lower()
        return lowered.endswith(".mdx") or lowered.endswith(".md")

    # -- frontmatter -------------------------------------------------------

    def split_frontmatter(self, text: str) -> tuple[dict, str]:
        """Return (metadata, body). Empty dict + original text if no frontmatter.

        Uses the `frontmatter` library; on any parse error we degrade to treating
        the whole text as the body (no frozen keys to compare), so a malformed
        edit surfaces elsewhere rather than crashing the splitter.
        """
        try:
            post = frontmatter.loads(text)
        except Exception:
            return {}, text
        # `metadata` is a dict (possibly empty); `content` is the body without
        # the frontmatter block.
        return dict(post.metadata), post.content

    def frontmatter_keys_to_freeze(self) -> list[str]:
        return ["title", "description"]

    # -- structural fingerprint -------------------------------------------

    def structural_signature(self, text: str) -> dict:
        """A flat {str: int} fingerprint of MDX structure to hold invariant.

        Keys:
          - one entry per known component open tag: `{"Card": n}`
          - one entry per known component close tag: `{"/Card": n}`
          - `fence_count`: number of ``` code-fence markers (must stay even)
          - `mermaid_count`: number of ```mermaid fences

        Flat int values so two signatures compare with a plain `==`.
        """
        signature: dict = {}

        for match in _OPEN_TAG_RE.finditer(text):
            name = match.group(1)
            if name in _KNOWN_COMPONENTS:
                signature[name] = signature.get(name, 0) + 1

        for match in _CLOSE_TAG_RE.finditer(text):
            name = match.group(1)
            if name in _KNOWN_COMPONENTS:
                key = f"/{name}"
                signature[key] = signature.get(key, 0) + 1

        signature["fence_count"] = len(_FENCE_RE.findall(text))
        signature["mermaid_count"] = len(_MERMAID_RE.findall(text))
        return signature

    # -- well-formedness (nesting / balance) ------------------------------

    def structural_problems(self, text: str) -> list[str]:
        """Stack-check known component tags for balance + correct nesting.

        Catches the failure modes the count signature misses: a dropped closing
        tag, a stray close, or a swapped/reordered pair (whose counts still match).
        Tags inside fenced code blocks / inline code are ignored (they're examples),
        and only known components are tracked (so arbitrary `<Thing>` prose can't
        produce false positives). Self-closing `<Card/>` is balanced by definition.
        """
        scan = _INLINE_CODE_RE.sub("", _FENCE_BLOCK_RE.sub("", text))
        stack: list[str] = []
        problems: list[str] = []

        for match in _TAG_RE.finditer(scan):
            closing, name, self_closing = match.group(1), match.group(2), match.group(3)
            if name not in _KNOWN_COMPONENTS:
                continue
            if self_closing:
                continue
            if not closing:
                stack.append(name)
                continue
            # A closing tag.
            if stack and stack[-1] == name:
                stack.pop()
            elif name in stack:
                problems.append(
                    f"mismatched nesting: </{name}> closes <{stack[-1]}>"
                )
                while stack and stack.pop() != name:
                    pass
            else:
                problems.append(f"stray closing </{name}> with no matching open tag")

        problems.extend(f"unclosed <{name}> (no matching close tag)" for name in stack)
        return problems

    # -- link check (soft gate) -------------------------------------------

    def check_links(self, docs_root: Path) -> list[str]:
        """Run Mintlify's broken-link checker over `docs_root`.

        Returns human-readable problem lines (empty == clean / unavailable). This
        must NEVER raise: a missing CLI, a timeout, or a non-zero exit are all
        treated as "no findings" so validation can downgrade it to a warning.
        """
        # Try `npx --no-install mintlify broken-links` first (project-local CLI),
        # then a bare `mintlify broken-links` (globally installed CLI).
        command_variants = (
            ["npx", "--no-install", "mintlify", "broken-links"],
            ["mintlify", "broken-links"],
        )

        for command in command_variants:
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(docs_root),
                    capture_output=True,
                    text=True,
                    timeout=_LINK_CHECK_TIMEOUT_S,
                    check=False,
                )
            except (
                FileNotFoundError,
                subprocess.TimeoutExpired,
                OSError,
                ValueError,
            ):
                # CLI not present / hung / bad invocation — try the next variant.
                continue
            except Exception:
                # Absolutely never let a link check sink the pipeline.
                continue

            output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            # If the CLI never actually ran (package not installed, npx aborted),
            # treat it as unavailable and try the next variant rather than parsing
            # tooling noise as broken-link findings.
            if _CLI_UNAVAILABLE_RE.search(output):
                continue
            return self._parse_link_output(output)

        return []

    @staticmethod
    def _parse_link_output(output: str) -> list[str]:
        """Pull broken-link lines out of the CLI's stdout/stderr, defensively."""
        problems: list[str] = []
        try:
            for raw_line in output.splitlines():
                line = raw_line.strip()
                if line and _LINK_PROBLEM_RE.search(line):
                    problems.append(line)
        except Exception:
            return []
        return problems
