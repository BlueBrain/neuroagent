"""Configuration."""

import os
import pathlib
from typing import Literal, Optional

from dotenv import dotenv_values
from fastapi.openapi.models import OAuthFlowPassword, OAuthFlows
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsAgent(BaseModel):
    """Agent setting."""

    model: Literal["simple", "multi"] = "simple"

    model_config = ConfigDict(frozen=True)


class SettingsDB(BaseModel):
    """DB settings for retrieving history."""

    prefix: str | None = None
    user: str | None = None
    password: SecretStr | None = None
    host: str | None = None
    port: str | None = None
    name: str | None = None

    model_config = ConfigDict(frozen=True)


class SettingsKeycloak(BaseModel):
    """Class retrieving keycloak info for authorization."""

    issuer: str = "https://openbluebrain.com/auth/realms/SBO"
    validate_token: bool = False
    # Useful only for service account (dev)
    client_id: str | None = None
    username: str | None = None
    password: SecretStr | None = None

    model_config = ConfigDict(frozen=True)

    @property
    def token_endpoint(self) -> str | None:
        """Define the token endpoint."""
        if self.validate_token:
            return f"{self.issuer}/protocol/openid-connect/token"
        else:
            return None

    @property
    def user_info_endpoint(self) -> str | None:
        """Define the user_info endpoint."""
        if self.validate_token:
            return f"{self.issuer}/protocol/openid-connect/userinfo"
        else:
            return None

    @property
    def flows(self) -> OAuthFlows:
        """Define the flow to override Fastapi's one."""
        return OAuthFlows(
            password=OAuthFlowPassword(
                tokenUrl=self.token_endpoint,
            ),
        )

    @property
    def server_url(self) -> str:
        """Server url."""
        return self.issuer.split("/auth")[0] + "/auth/"

    @property
    def realm(self) -> str:
        """Realm."""
        return self.issuer.rpartition("/realms/")[-1]


class SettingsLiterature(BaseModel):
    """Literature search API settings."""

    url: str
    retriever_k: int = 500
    use_reranker: bool = True
    reranker_k: int = 8

    model_config = ConfigDict(frozen=True)


class SettingsTrace(BaseModel):
    """Trace tool settings."""

    search_size: int = 10

    model_config = ConfigDict(frozen=True)


class SettingsKGMorpho(BaseModel):
    """KG Morpho settings."""

    search_size: int = 3

    model_config = ConfigDict(frozen=True)


class SettingsGetMorpho(BaseModel):
    """Get Morpho settings."""

    search_size: int = 10

    model_config = ConfigDict(frozen=True)


class SettingsKnowledgeGraph(BaseModel):
    """Knowledge graph API settings."""

    base_url: str
    token: SecretStr | None = None
    use_token: bool = False
    download_hierarchy: bool = False
    br_saving_path: pathlib.Path | str = str(
        pathlib.Path(__file__).parent / "data" / "brainregion_hierarchy.json"
    )
    ct_saving_path: pathlib.Path | str = str(
        pathlib.Path(__file__).parent / "data" / "celltypes_hierarchy.json"
    )
    model_config = ConfigDict(frozen=True)

    @property
    def url(self) -> str:
        """Knowledge graph search url."""
        return f"{self.base_url}/search/query/"

    @property
    def sparql_url(self) -> str:
        """Knowledge graph view for sparql query."""
        return f"{self.base_url}/views/neurosciencegraph/datamodels/https%3A%2F%2Fbluebrain.github.io%2Fnexus%2Fvocabulary%2FdefaultSparqlIndex/sparql"

    @property
    def class_view_url(self) -> str:
        """Knowledge graph view for ES class query."""
        return f"{self.base_url}/views/neurosciencegraph/datamodels/https%3A%2F%2Fbbp.epfl.ch%2Fneurosciencegraph%2Fdata%2Fviews%2Fes%2Fdataset/_search"

    @property
    def hierarchy_url(self) -> str:
        """Knowledge graph url for brainregion/celltype files."""
        return "http://bbp.epfl.ch/neurosciencegraph/ontologies/core"


class SettingsTools(BaseModel):
    """Database settings."""

    literature: SettingsLiterature
    morpho: SettingsGetMorpho = SettingsGetMorpho()
    trace: SettingsTrace = SettingsTrace()
    kg_morpho_features: SettingsKGMorpho = SettingsKGMorpho()

    model_config = ConfigDict(frozen=True)


class SettingsOpenAI(BaseModel):
    """OpenAI settings."""

    token: Optional[SecretStr] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0
    max_tokens: Optional[int] = None

    model_config = ConfigDict(frozen=True)


class SettingsLogging(BaseModel):
    """Metadata settings."""

    level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    external_packages: Literal["debug", "info", "warning", "error", "critical"] = (
        "warning"
    )

    model_config = ConfigDict(frozen=True)


class SettingsMisc(BaseModel):
    """Other settings."""

    application_prefix: str = ""
    # list is not hashable, the cors_origins have to be provided as a string with
    # comma separated entries, i.e. "value_1, value_2, ..."
    cors_origins: str = ""

    model_config = ConfigDict(frozen=True)


class Settings(BaseSettings):
    """All settings."""

    tools: SettingsTools
    knowledge_graph: SettingsKnowledgeGraph
    agent: SettingsAgent = SettingsAgent()  # has no required
    db: SettingsDB = SettingsDB()  # has no required
    openai: SettingsOpenAI = SettingsOpenAI()  # has no required
    logging: SettingsLogging = SettingsLogging()  # has no required
    keycloak: SettingsKeycloak = SettingsKeycloak()  # has no required
    misc: SettingsMisc = SettingsMisc()  # has no required

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NEUROAGENT_",
        env_nested_delimiter="__",
        frozen=True,
    )

    @model_validator(mode="after")
    def check_consistency(self) -> "Settings":
        """Check if consistent.

        ATTENTION: Do not put model validators into the child settings. The
        model validator is run during instantiation.

        """
        if not self.keycloak.password and not self.keycloak.validate_token:
            if not self.knowledge_graph.use_token:
                raise ValueError("if no password is provided, please use token auth.")
            if not self.knowledge_graph.token:
                raise ValueError(
                    "No auth method provided for knowledge graph related queries."
                    " Please set either a password or use a fixed token."
                )

        return self


# Load the remaining variables into the environment
# Necessary for things like SSL_CERT_FILE
config = dotenv_values()
for k, v in config.items():
    if k.lower().startswith("neuroagent_"):
        continue
    if v is None:
        continue
    os.environ[k] = os.environ.get(k, v)  # environment has precedence
