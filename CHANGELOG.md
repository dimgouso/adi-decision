# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and the project uses Semantic Versioning.

## [0.1.1] - Unreleased

### Added

- release automation for building distributions and publishing to TestPyPI or PyPI via GitHub Actions
- a dedicated publishing guide for PyPI Trusted Publishing setup
- CI package validation with `python -m build` and `twine check`

### Changed

- refined project metadata for distribution and package index presentation
- improved README positioning, installation guidance, and verification instructions
- cleaned repository ignores for generated build and cache artifacts

## [0.1.0] - 2026-02-28

### Added

- initial public release of the ADI decision engine
- weighted multi-criteria ranking with policy-based scoring
- explainability primitives including contributions, counterfactuals, and constraint reports
- scenario analysis and sensitivity reporting
- profile-based learning from user feedback
- CLI, FastAPI interface, and agent tool adapters for OpenAI and Anthropic
