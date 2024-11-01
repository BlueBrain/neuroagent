"""Pydantic Schemas."""

from pydantic import BaseModel


class KGMetadata(BaseModel):
    """Knowledge Graph Metadata."""

    file_extension: str
    brain_region: str
    is_lnmc: bool = False
