# Public release provenance — v1.0.0

This package is a packaging-only public candidate derived from the qualified private Uncertainty Lab v1.0.0 release.

## Changes from the private package

- removed `__pycache__`, `.pyc`, `.pyo`, and `.pytest_cache` artifacts;
- replaced the private-facing README with public-facing documentation;
- added `PRIVACY.md`, `SECURITY.md`, and this public-release provenance file;
- regenerated `validation_receipt.json` for this exact cleaned package;
- added exact benchmark results for this public candidate.

No Python scientific implementation file, test implementation file, plugin manifest, skill file, or requirements file is intentionally modified by the cleanup step.

## Submission checklist outside the code package

Before directory submission, the publisher should supply any account-level listing fields required by the submission UI (for example publisher identity and any support/contact destination). This package does not invent those account-specific details.

## Open-source legal layer

The GitHub-ready distribution additionally includes Apache-2.0 licensing,
project notice/attribution, third-party dependency notices, citation metadata,
and contribution terms. These are packaging/documentation additions only; the
qualified scientific Python source is unchanged.
