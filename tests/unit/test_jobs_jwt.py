import os
import sys
import types
from datetime import datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException
from packages.fog.gateway.api.jobs import JobsAPI


@pytest.fixture
def jobs_api(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "testsecret")
    dummy_module = types.ModuleType("packages.core.security.rbac_system")

    class DummyPermission:
        FOG_JOB_SUBMIT = "fog.job.submit"

    class DummyRBACSystem:
        async def check_permission(self, user_id, permission):
            return True

    dummy_module.Permission = DummyPermission
    dummy_module.RBACSystem = DummyRBACSystem
    monkeypatch.setitem(sys.modules, "packages.core.security.rbac_system", dummy_module)
    return JobsAPI()


@pytest.mark.asyncio
async def test_validate_job_permissions_valid_token(jobs_api):
    token = jwt.encode(
        {"user_id": "user1", "exp": datetime.utcnow() + timedelta(minutes=5)},
        os.getenv("JWT_SECRET"),
        algorithm="HS256",
    )
    await jobs_api._validate_job_permissions("user1", f"Bearer {token}", "org/team")


@pytest.mark.asyncio
async def test_validate_job_permissions_expired_token(jobs_api):
    token = jwt.encode(
        {"user_id": "user1", "exp": datetime.utcnow() - timedelta(minutes=5)},
        os.getenv("JWT_SECRET"),
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        await jobs_api._validate_job_permissions("user1", f"Bearer {token}", "org/team")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_validate_job_permissions_tampered_token(jobs_api):
    token = jwt.encode(
        {"user_id": "user1", "exp": datetime.utcnow() + timedelta(minutes=5)},
        os.getenv("JWT_SECRET"),
        algorithm="HS256",
    )
    tampered = token + "a"
    with pytest.raises(HTTPException) as exc:
        await jobs_api._validate_job_permissions("user1", f"Bearer {tampered}", "org/team")
    assert exc.value.status_code == 403
