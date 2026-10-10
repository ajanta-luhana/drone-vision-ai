from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..core.security import decode_token, has_permission
from ..models.db import AuditLog, SessionLocal, User

bearer = HTTPBearer(auto_error=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer), db=Depends(get_db)) -> User:
    if not creds:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = decode_token(creds.credentials)
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
    user = db.query(User).filter(User.username == payload.get("sub")).first()
    if not user or not user.active:
        raise HTTPException(401, "User disabled or unknown")
    return user


def require(perm: str):
    """RBAC enforced server-side on every protected endpoint."""
    def dep(user: User = Depends(current_user)) -> User:
        if not has_permission(user.role, perm):
            raise HTTPException(403, f"Role '{user.role}' lacks permission '{perm}'")
        return user
    return dep


def audit(db, user: str, action: str, resource: str, prev=None, new=None, request: Request | None = None):
    ip = request.client.host if request and request.client else None
    db.add(AuditLog(user=user, action=action, resource=resource, previous_state=prev, new_state=new, ip=ip))
    db.commit()
