# Neuroagent

LLM agent made to communicate with different neuroscience related tools. It allows to communicate in a ChatGPT like fashion to get information about brain regions, morphologies, electric traces and the scientific literature.


1. [Funding and Acknowledgement](#funding-and-acknowledgement)

## Funding and Acknowledgement

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2024 Blue Brain Project/EPFL

## Release workflow

Adding to the changelog 

Commits with a special prefix will be added to the CHANGELOG of the latest release PR.
The main prefixes are:
* feat: New feature. Will add this to the "Feature" section of the CHANGELOG.
* fix: Will be added to the "Bug fixes" section.
* add: Will be added to the "Added" section.
* fixed: Will be added to the "Fixed" section.
* changed: Will be added to the "Changed" section.

When a PR is merged into the main branch, a new release PR will be created if there is no open one. Otherwise all changes
from the merged branch will be added to the latest existing release PR.

In conclusion, the workflow is:
1. Merging a PR triggers the creation of a new release PR if there is no current open one. If there is, all changes from the newly merged PR are added to the release PR.
2. When merging the release PR, a new release tag is created on github.


