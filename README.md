# docsync

Keep documentation in sync with code changes. docsync ingests a merged PR's diff
from a service repo, maps it to the documentation pages it affects (which may live
in a **different** repo), uses an LLM to make surgical edits to the existing `.mdx`,
validates them, and opens a reviewable PR against the docs repo — as part of CI.

MVP target: the **Keep** platform (4 service repos → `keep-developer-docs`, Mintlify).

> Full docs land in task 7. This is a stub so the package installs cleanly.
