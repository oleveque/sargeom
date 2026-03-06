# GitHub Copilot Instructions for sargeom

## Project Overview

This is the sargeom Python package for Synthetic Aperture Radar (SAR) geometry calculations. The project follows modern Python packaging standards and semantic versioning.

## Version Bumping Guidelines

When bumping the version of the sargeom API, follow these steps **in order**. ALL steps are mandatory unless marked as optional.

### 1. Update CHANGELOG.md

- Move all changes from the `[Unreleased]` section to a new version section
- Add the new version number in the format `[X.Y.Z] - YYYY-MM-DD`
- Use the current date for the release
- Create a new empty `[Unreleased]` section at the top
- Keep the existing structure with subsections: `Added`, `Changed`, `Fixed`, `Deprecated`, `Removed`, `Security`

Example structure:
```markdown
## [Unreleased]

## [0.4.0] - 2025-10-23

### Added
- New feature description

### Changed
- Changed feature description
```

### 2. Update pyproject.toml

- Modify the `version` field under the `[project]` section (typically line 7)
- Follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
  - **MAJOR** version (X.0.0): incompatible API changes
  - **MINOR** version (0.X.0): backward-compatible functionality additions
  - **PATCH** version (0.0.X): backward-compatible bug fixes

Example:
```toml
[project]
version = "0.4.0"
```

### 3. Update README.md

- Update version references in the installation examples
- Look for version tags in `pyproject.toml` and `requirements.txt` sections (typically around lines 24 and 31)
- Update both occurrences from `@vX.Y.Z` to the new version

Example:
```toml
dependencies = [
    "sargeom @ git+https://github.com/oleveque/sargeom@v0.4.0"
]
```

### 4. Update docs/source/conf.py

- Update the `release` and `version` variables (typically lines 12-13)
- These control the version displayed in the Sphinx documentation

Example:
```python
release = '0.4.0'
version = '0.4.0'
```

### 5. Update docs/source/index.rst

- Update version references in the installation examples
- Look for version tags in the `pyproject.toml` and `requirements.txt` code blocks (typically around lines 43 and 49)
- Update both occurrences from `@vX.Y.Z` to the new version

Example:
```toml
[project]
dependencies = [
    "sargeom @ git+https://github.com/oleveque/sargeom@v0.4.0"
]
```

```text
sargeom @ git+https://github.com/oleveque/sargeom@v0.4.0
```

### 6. Update src/sargeom/__init__.py

- Update the `__version__` variable (typically line 3)
- This makes the version accessible programmatically and allows users to check it with `import sargeom; print(sargeom.__version__)`

Example:
```python
__version__ = "0.4.0"
```

## Pre-Release Checklist

Before proceeding with the release steps, verify:

1. **Run all tests locally**:
   ```bash
   python -m unittest discover -s tests
   python -m doctest src/sargeom/**/*.py
   ```

2. **Check that all changes are documented** in CHANGELOG.md under `[Unreleased]`

3. **Verify the version number follows Semantic Versioning**:
   - **MAJOR** (X.0.0): Breaking changes or incompatible API changes
   - **MINOR** (0.X.0): New features, backward-compatible
   - **PATCH** (0.0.X): Bug fixes, backward-compatible

4. **Ensure all modified files are saved** and uncommitted changes are reviewed

## Post-Update Steps (User Responsibility)

After Copilot has updated the files above, the user should:

1. **Review all changes carefully**
   - Verify version numbers are consistent across all files
   - Check CHANGELOG.md formatting and completeness
   - Ensure README.md and index.rst examples reference the correct version

2. **Run tests one final time**:
   ```bash
   python -m unittest discover -s tests
   ```

3. **Commit the changes**:
   ```bash
   git add pyproject.toml CHANGELOG.md README.md docs/source/conf.py docs/source/index.rst src/sargeom/__init__.py
   git commit -m "Bump version to X.Y.Z"
   ```

4. **Create an annotated tag**:
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   ```

5. **Push to remote** (this triggers GitHub Actions workflows):
   ```bash
   git push origin main --tags
   ```

6. **Verify GitHub Actions** complete successfully:
   - **tests.yml**: Unit tests pass
   - **docs.yml**: Documentation builds and deploys to GitHub Pages
   - **release.yml**: Package builds and creates GitHub release

7. **(Optional) Build and publish to PyPI**:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

## GitHub Actions Workflows

The project uses three automated workflows:

### 1. tests.yml
- **Trigger**: On every push and pull request
- **Purpose**: Runs unit tests and doctests
- **Command**: `python -m unittest discover -s tests`

### 2. docs.yml
- **Trigger**: On tag push matching `v[0-9]+.*`
- **Purpose**: Builds Sphinx documentation and deploys to GitHub Pages
- **Output**: https://oleveque.github.io/sargeom/

### 3. release.yml
- **Trigger**: On tag push matching `v[0-9]+.*`
- **Purpose**: Builds wheel package and creates GitHub release
- **Output**: GitHub release with `.whl` file attached

## Rollback Procedure

If a release needs to be rolled back:

1. **Delete the remote tag**:
   ```bash
   git push origin :refs/tags/vX.Y.Z
   ```

2. **Delete the local tag**:
   ```bash
   git tag -d vX.Y.Z
   ```

3. **Delete the GitHub release** manually from the GitHub repository releases page

4. **Revert the version bump commit** if already pushed:
   ```bash
   git revert <commit-hash>
   git push origin main
   ```

5. **Update files** back to the previous version and recommit

## Pre-Release Versions (Advanced)

For alpha, beta, or release candidate versions:

- Use format: `X.Y.Z-alpha.N`, `X.Y.Z-beta.N`, or `X.Y.Z-rc.N`
- Example: `0.4.0-alpha.1`, `0.4.0-beta.2`, `0.4.0-rc.1`
- These should be used for testing before the official release
- Document pre-releases in CHANGELOG.md under the version section
- Tag format: `vX.Y.Z-alpha.N`

**Note**: GitHub Actions workflows will trigger on pre-release tags as well.

## Code Style and Standards

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Maintain comprehensive docstrings for all public APIs
- Keep backward compatibility for MINOR and PATCH versions
- Document all breaking changes in CHANGELOG.md

## Project Structure

- `src/sargeom/`: Main package source code
- `tests/`: Unit tests
- `docs/`: Sphinx documentation
- `pyproject.toml`: Project metadata and dependencies
- `CHANGELOG.md`: Version history following Keep a Changelog format

## Dependencies

- Core: `numpy`, `scipy`
- Documentation: `sphinx`, `furo`, `numpydoc`, etc.
- Examples: `folium`, `pandas`, `shapely`, `simplekml`, `pyproj`
- Development: `build`, `twine`

## Git Workflow

- **Main branch**: `main` - stable, production-ready code
- **Development**: Create feature branches for new work
- **Branch naming**: Use descriptive names (e.g., `feature/pivot-support`, `fix/csv-precision`)
- **Pull requests**: Recommended for major changes to ensure CI passes
- **Commits**: Use clear, descriptive commit messages

## Files to Never Modify During Version Bump

The following auto-generated files should **NOT** be manually edited:

- `src/sargeom.egg-info/*` - Auto-generated during installation
- `src/sargeom/__pycache__/*` - Python bytecode cache
- `tests/__pycache__/*` - Test bytecode cache
- `docs/build/*` - Generated documentation (only `docs/source/` should be edited)
- `dist/*` - Built distributions (regenerated for each release)
- `output.h5` - Example/test output file

## Important Notes

- Always update CHANGELOG.md before bumping the version
- Ensure all tests pass before creating a release
- Use annotated tags for releases (not lightweight tags)
- The project requires Python >= 3.8
- GitHub Actions will automatically deploy docs and create releases when a version tag is pushed
- Version consistency is critical: all 6 files must have matching version numbers
- Test locally before pushing tags to avoid unnecessary CI runs
