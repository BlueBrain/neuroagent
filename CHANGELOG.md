# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LLM evaluation logic
- Integrated Alembic for managing chat history migrations
- Tool implementations without langchain or langgraph dependencies
- CRUDs.
- BlueNaas CRUD tools

## [0.3.3] - 30.10.2024

### Changed
- Removed release please bot and add automatic on tag pushes to ecr.

## [0.3.2](https://github.com/BlueBrain/neuroagent/compare/v0.3.1...v0.3.2) (2024-10-29)


### Bug Fixes

* Fix ([#39](https://github.com/BlueBrain/neuroagent/issues/39)) ([948b8bf](https://github.com/BlueBrain/neuroagent/commit/948b8bf7b77fa62baddba357c293979b9ba05847))

## [0.3.1](https://github.com/BlueBrain/neuroagent/compare/v0.3.0...v0.3.1) (2024-10-29)


### Bug Fixes

* fix ecr yml ([#37](https://github.com/BlueBrain/neuroagent/issues/37)) ([1983b20](https://github.com/BlueBrain/neuroagent/commit/1983b2083e276ce2991cee6b6c3b0fc1e8268512))

## [0.3.0](https://github.com/BlueBrain/neuroagent/compare/v0.2.0...v0.3.0) (2024-10-29)


### Features

* Added release please ([dd11700](https://github.com/BlueBrain/neuroagent/commit/dd1170095a92b086d264e09d6ba417b506f2d3e4))
* Added release please to automate changelogs and releases. ([5b9d30b](https://github.com/BlueBrain/neuroagent/commit/5b9d30b1d304a4a16761939625db31ed581bc57b))
* Added stream ([#33](https://github.com/BlueBrain/neuroagent/issues/33)) ([3df8463](https://github.com/BlueBrain/neuroagent/commit/3df84637649fce5937688f288d4d03f9c4eab0b6))


### Added
- Swarm copy POC.
- Agent memory.


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
