# rbac control handling of the users (free,pro,admin)
from typing import Sequence
from fastapi import HTTPException, status
from backend.models.models import User, UserRole

class RoleChecker:
    def __init__(self, allowed_roles: Sequence[UserRole]):
        self.allowed_roles = allowed_roles
    def __call__(self, user: User) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Action requires one of: {[r.value for r in self.allowed_roles]}",
            )
        return user
# Convenience guards:
require_admin = RoleChecker([UserRole.ADMIN])
require_pro = RoleChecker([UserRole.PRO, UserRole.ADMIN])