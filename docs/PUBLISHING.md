# Publishing Guide

This repository is configured for GitHub Actions based publishing with PyPI Trusted Publishing.

## Recommended release flow

1. Merge release-ready changes to `main`.
2. Update version references and `CHANGELOG.md`.
3. Create a GitHub Release with tag `vX.Y.Z`.
4. GitHub Actions builds the distributions and publishes them to PyPI.

For dry runs or first-time verification, use the manual workflow dispatch path with TestPyPI.

## One-time GitHub repository setup

Create these GitHub environments in `dimgouso/adi-Agent-Decision-Intelligence`:

- `testpypi`
- `pypi`

The workflow file that publishes is:

- `.github/workflows/release.yml`

## One-time PyPI setup

### TestPyPI

Create a Trusted Publisher for:

- owner: `dimgouso`
- repository: `adi-Agent-Decision-Intelligence`
- workflow: `release.yml`
- environment: `testpypi`

### PyPI

Create a Trusted Publisher for:

- owner: `dimgouso`
- repository: `adi-Agent-Decision-Intelligence`
- workflow: `release.yml`
- environment: `pypi`
- project name: `adi-decision`

If `adi-decision` does not yet exist on PyPI, configure it as a pending publisher first and then run the workflow.

## Workflow behavior

- `release.published`: builds distributions and publishes to PyPI
- `workflow_dispatch`: lets you manually choose `testpypi` or `pypi`

## Local verification

```bash
pip install -e ".[dev]"
pytest
python -m build
twine check dist/*
```

## Suggested release checklist

- tests pass locally and in CI
- `CHANGELOG.md` is updated
- version in `pyproject.toml`, `adi/__init__.py`, and API metadata is correct
- TestPyPI publish succeeds before the first production PyPI publish
