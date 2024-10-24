# Neuroagent

LLM agent made to communicate with different neuroscience related tools. It allows to communicate in a ChatGPT like fashion to get information about brain regions, morphologies, electric traces and the scientific literature.


1. [Funding and Acknowledgement](#funding-and-acknowledgement)

## Funding and Acknowledgement

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2024 Blue Brain Project/EPFL

## Release workflow

Commits with a special prefix will be added to the CHANGELOG of the latest release PR.
The main prefixes can be found here:
https://www.conventionalcommits.org/en/v1.0.0/#summary

When a PR is merged into the main branch, a new release PR will be created if there is no open one. Otherwise all changes
from the merged branch will be added to the latest existing release PR.

The workflow is:
1. When merging a PR, change the squashed commit message to one that contains one of the above prefixes. This will trigger the creation of a release PR if there isnt one. The commit message will be automatically added to the changelog.
2. When the release PR is merged, a new release tag will be automatically created on github.


