# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-09-26

### Added
- Introduced `concatenate()` methods for `Cartesian3`, `Cartographic`, and `Trajectory`
- Added `height_mode` parameter to `Cartographic.save_kml()` method to specify height reference (e.g., ellipsoid, orthometric)
- Added support for legacy PamelaX11 format in `Trajectory.read_pamela_traj()`

### Changed
- Refactored `append()` methods to internally use `concatenate()` for improved consistency and maintainability
- Updated all `save_*` methods to return the path of the saved file
- Replaced `pyproj` dependency with `sargeom.coordinates.ellipsoids` for coordinate transformations

### Fixed
- Fixed longitude/latitude precision in CSV output from `%.12f` to `%.15f` (by @oboisot in #1)
  - Affects `Trajectory.save_csv()` and `Cartographic.save_csv()` methods
  - Ensures full double-precision accuracy for geographic coordinates
- Other minor fixes

## [0.2.0] - 2025-08-04

### Added
- Introduced `Trajectory` class for managing spatio-temporal data
- Created `CHANGELOG.md` following the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format
- Added `LICENSE.md` file

## [0.1.0] - 2025-06-03

### Added
- Initial release of the SAR Geometry package
