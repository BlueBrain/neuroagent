# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add get morphoelectric (me) model tool

## [0.1.1] - 26.09.2024

### Fixed
- Fixed a bug that prevented AsyncSqlite checkpoint to access the DB in streamed endpoints.
- Fixed a bug that caused some unit tests to fail due to a change in how httpx_mock works in version 0.32

## [0.1.0] - 19.09.2024

### Added
- Update readme

### Removed
- Github action to create the docs.

### Changed
- Migration to pydantic V2.

### Fixed
- Streaming with chat agent.
- Deleted some legacy code.
