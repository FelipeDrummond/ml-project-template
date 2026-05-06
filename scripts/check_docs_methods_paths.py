"""Pre-commit hook: verify every code path referenced in docs/methods.md exists.

Greps inline-code spans (`backtick text`) that look like file/module paths
(start with `src/` or end in `.py`) and asserts each path resolves on disk.
This catches doc rot when code is moved/renamed without a docs update.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOC = Path("docs/methods.md")
PATH_RE = re.compile(r"`([^`]+)`")

# Inline-code spans that "look like a project path" — paths we want to verify.
PATH_PREFIXES = ("src/", "tests/", "scripts/", "configs/", "docs/")
PATH_SUFFIXES = (".py", ".yaml", ".yml", ".md", ".toml", ".json")


def is_path_like(s: str) -> bool:
    if " " in s or ":" in s:  # exclude URIs and free-form text
        return False
    return s.startswith(PATH_PREFIXES) or s.endswith(PATH_SUFFIXES)


def main() -> int:
    if not DOC.exists():
        print(f"{DOC} missing — skipping methods-doc check.")
        return 0

    text = DOC.read_text()
    missing: list[str] = []
    for token in PATH_RE.findall(text):
        if not is_path_like(token):
            continue
        # Strip trailing punctuation that might leak in
        candidate = token.rstrip(".,);:")
        if not Path(candidate).exists():
            missing.append(candidate)

    if missing:
        print("docs/methods.md references paths that no longer exist:")
        for p in missing:
            print(f"  - {p}")
        print("\nUpdate docs/methods.md or restore the paths.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
