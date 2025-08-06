# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Introduced `Cartesian3.concatenate()`, `Cartographic.concatenate()`, and `Trajectory.concatenate()` methods for concatenating multiple objects
  - Methods follow numpy/scipy conventions for consistency
  - Supports concatenating multiple `Cartesian3`, `Cartographic`, or `Trajectory` objects

### Changed
- Updated `append()` methods to use `concatenate()` internally for better consistency and maintainability

## [0.2.0] - 2025-08-04

### Added
- Introduced `Trajectory` class for managing spatio-temporal data
- Created `CHANGELOG.md` following the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format
- Added `LICENSE.md` file

## [0.1.0] - 2025-06-03

### Added
- First version of the SAR Geometry package
