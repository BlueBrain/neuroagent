# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Swarm copy POC.


## [0.2.0] - 22.10.2024

### Changed
- Switched from OAUTH2 security on FASTAPI to HTTPBearer.
- Switched to async sqlalchemy.
- Expanded list of etypes.

### Added
- Add get morphoelectric (me) model tool
- BlueNaaS simulation tool.
- Validation of the project ID.
- BlueNaaS tool test.
- Human in the loop for bluenaas.

### Fixed
- Fixed 0% unittest coverage bug.
- Get ME model tool querying logic

## [0.1.1] - 26.09.2024

### Fixed
- Fixed a bug that prevented AsyncSqlite checkpoint to access the DB in streamed endpoints.
- Fixed a bug that caused some unit tests to fail due to a change in how httpx_mock works in version 0.32

## [0.1.0] - 19.09.2024

### Added
- Update readme
- Extra multi agent unit tests
- Extra unit tests for dependencies.py

### Removed
- Github action to create the docs.

### Changed
- Migration to pydantic V2.

### Fixed
- Streaming with chat agent.
- Deleted some legacy code.
