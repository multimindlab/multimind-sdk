# Releasing multimind-sdk

This document describes how to cut a release of `multimind-sdk` to PyPI using
the `.github/workflows/release.yml` pipeline.

## Overview

The release pipeline has three jobs, run in order:

1. **build** — `python -m build` produces an sdist + wheel, `twine check`
   validates the metadata, and both are uploaded as a `dist` artifact.
2. **test-wheel** — installs the *built wheel* (not an editable checkout) into
   a clean environment and runs a smoke subset (`tests/test_import.py`,
   `tests/test_basic.py`) against it. This catches packaging bugs (missing
   package data, bad `packages.find` excludes, etc.) that an editable install
   would never surface.
3. **publish** — publishes to PyPI using
   [OIDC trusted publishing](https://docs.pypi.org/trusted-publishers/) via
   `pypa/gh-action-pypi-publish`. No PyPI API token is stored as a repo
   secret; the job authenticates using a short-lived OIDC token instead.
   This job only runs for an actual **published GitHub release** — pushing a
   `v*` tag alone (or `workflow_dispatch`) exercises `build`/`test-wheel`
   only, so you can validate packaging without publishing.

## Version bump location

The package version is defined in one place:

```python
# multimind/__init__.py
__version__ = "0.3.0"
```

`pyproject.toml` reads it dynamically (`dynamic = ["version"]`,
`[tool.setuptools.dynamic] version = {attr = "multimind.__version__"}`), so
there is nothing else to bump — no version string in `pyproject.toml`,
`setup.py`, or docs needs manual updating.

## Tag / release flow

1. Bump `__version__` in `multimind/__init__.py` on `develop` (or a release
   branch), open a PR, get it merged.
2. Update `CHANGELOG.md` with the new version's notes.
3. Tag the merge commit and push the tag:
   ```bash
   git tag v0.3.1
   git push origin v0.3.1
   ```
   This alone triggers `build` + `test-wheel` (not `publish`) — a good
   opportunity to confirm the wheel installs and imports cleanly before
   committing to a public release.
4. Create a GitHub Release from that tag (via the GitHub UI, or
   `gh release create v0.3.1 --generate-notes`). Publishing the release is
   what triggers the `publish` job and pushes to PyPI.

## One-time trusted-publisher setup on pypi.org

Trusted publishing must be registered once per PyPI project before the
`publish` job can authenticate. As a maintainer with access to the
`multimind-sdk` project on PyPI:

1. Log in to <https://pypi.org> and go to the project's
   **Publishing** settings (Account settings → Publishing, or directly at
   `https://pypi.org/manage/project/multimind-sdk/settings/publishing/`).
2. Add a new **GitHub** trusted publisher with:
   - **Owner**: `multimindlab`
   - **Repository name**: `multimind-sdk`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi` (must match the `environment:` block in
     `release.yml`'s `publish` job)
3. Save. No API token needs to be generated or copied into GitHub secrets —
   the OIDC exchange is handled entirely by `id-token: write` permission in
   the workflow plus this registration.
4. In the GitHub repo, create an **environment** named `pypi`
   (Settings → Environments) if it doesn't already exist. Optionally add
   required reviewers here for an extra manual gate before publish runs.

### TestPyPI (optional)

For dry-running the whole pipeline against TestPyPI before it's wired to the
real index, repeat the same steps on <https://test.pypi.org> with:

- **Environment name**: `testpypi`

and uncomment the "TestPyPI variant" step block at the bottom of the
`publish` job in `release.yml` (pointed at
`https://test.pypi.org/legacy/`), gated behind its own `testpypi`
environment.

## Manual release checklist

- [ ] `__version__` in `multimind/__init__.py` bumped and merged to `develop`
- [ ] `CHANGELOG.md` updated
- [ ] CI green on the commit being tagged (`.github/workflows/ci.yml`)
- [ ] Tag pushed (`git tag vX.Y.Z && git push origin vX.Y.Z`)
- [ ] `build` + `test-wheel` jobs green for that tag
- [ ] GitHub Release created from the tag (this triggers `publish`)
- [ ] `publish` job green — confirm the new version appears on
      <https://pypi.org/project/multimind-sdk/>
- [ ] Smoke-install from PyPI in a scratch venv:
      `pip install multimind-sdk==X.Y.Z && python -c "import multimind; print(multimind.__version__)"`
