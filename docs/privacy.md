# Public Release Checklist

Before publishing a fork, issue attachment, or experiment branch, verify that it contains only source code and generic documentation.

- Keep raw LKN files and topic maps outside the repository.
- Keep checkpoints, prediction tables, XAI tables, figures, logs, and run directories under `outputs/`.
- Review Git's staged file list with `git status --short` and `git diff --cached --name-only` before every push.
- Search staged text for local paths, personal names, email addresses, access tokens, and dataset identifiers.
- Do not add notebook outputs, manuscript drafts, source PDFs, system specifications, or cached package metadata.

The repository `.gitignore` is a guardrail, not a substitute for a staged-file review.
