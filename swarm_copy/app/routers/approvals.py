"""Approvals CRUDs."""

import json
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from redis.asyncio import Redis

from swarm_copy.app.dependencies import (
    get_redis_client,
    get_user_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/approvals", tags=["Approvals' CRUD"])


class ApprovalContent(BaseModel):
    """Schema for approval content."""

    status: Literal["approved", "declined", "pending"]
    parameters: str
    tool_name: str


class ApprovalOut(BaseModel):
    """Schema for approval output."""

    approval_id: str
    value: ApprovalContent
    ttl: str


class ApprovalUpdate(BaseModel):
    """Schema for updating an approval."""

    status: Literal["approved", "declined"]


@router.get("/")
async def get_approvals(
    redis_client: Annotated[Redis, Depends(get_redis_client)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> list[ApprovalOut]:
    """Get all pending approvals for a user."""
    keys = await redis_client.keys(f"approval:{user_id}:*")
    approvals = []

    pipe = redis_client.pipeline()
    for key in keys:
        pipe.get(key)
        pipe.ttl(key)

    results = await pipe.execute()

    for i in range(0, len(results), 2):
        value = results[i]
        ttl = results[i + 1]
        if value:
            full_key = keys[i // 2].decode("utf-8")
            approval_id = full_key.split(":")[-1]
            value_dict = json.loads(value.decode("utf-8"))
            content = ApprovalContent(**value_dict)

            approvals.append(
                ApprovalOut(
                    approval_id=approval_id,
                    value=content,
                    ttl=str(ttl),
                )
            )

    return approvals


@router.get("/{approval_id}")
async def get_approval(
    approval_id: str,
    redis_client: Annotated[Redis, Depends(get_redis_client)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> ApprovalOut:
    """Get a specific approval."""
    key = f"approval:{user_id}:{approval_id}"

    pipe = redis_client.pipeline()
    pipe.get(key)
    pipe.ttl(key)
    value, ttl = await pipe.execute()

    if not value:
        raise HTTPException(status_code=404, detail="Approval not found")

    value_dict = json.loads(value.decode("utf-8"))
    content = ApprovalContent(**value_dict)

    return ApprovalOut(
        approval_id=approval_id,
        value=content,
        ttl=str(ttl),
    )


@router.patch("/{approval_id}")
async def update_approval(
    approval_id: str,
    update: ApprovalUpdate,
    redis_client: Annotated[Redis, Depends(get_redis_client)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> ApprovalOut:
    """Update an approval status."""
    key = f"approval:{user_id}:{approval_id}"
    exists = await redis_client.exists(key)

    if not exists:
        raise HTTPException(status_code=404, detail="Approval not found")

    # Get current TTL before updating
    current_ttl = await redis_client.ttl(key)

    # Get current value to preserve kwargs
    current_value = await redis_client.get(key)
    current_content = json.loads(current_value.decode("utf-8"))

    # Update status while preserving kwargs and tool_name
    new_content = ApprovalContent(
        status=update.status,
        parameters=current_content["parameters"],
        tool_name=current_content["tool_name"],
    )

    # Save updated content and restore TTL
    pipe = redis_client.pipeline()
    pipe.set(key, json.dumps(new_content.model_dump()))
    pipe.expire(key, current_ttl)
    await pipe.execute()

    return ApprovalOut(approval_id=approval_id, value=new_content, ttl=str(current_ttl))
