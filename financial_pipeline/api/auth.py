"""
JWT Authentication Module

CONCEPT: Production APIs must authenticate every request.
JWT (JSON Web Token) pattern:
1. User sends username + password to /auth/login
2. Server validates credentials, returns a signed JWT
3. Client includes JWT in Authorization: Bearer <token> header
4. Server verifies JWT signature on every request (stateless — no DB lookup needed)

In production (Citi): would integrate with Azure AD / Okta / LDAP.
MFA (multi-factor authentication) is mandatory in banking.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from loguru import logger

from config.settings import get_settings

settings = get_settings()

# ── Password hashing ──────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── OAuth2 scheme — FastAPI reads the Bearer token from the Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Pydantic models ───────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = "viewer"


class User(BaseModel):
    user_id: str
    email: str
    full_name: str
    role: str
    is_active: bool


class UserInDB(User):
    hashed_password: str


# ── Demo user store (replace with real DB in production) ─────────────────────
DEMO_USERS = {
    "analyst@citi.com": UserInDB(
        user_id="usr_001",
        email="analyst@citi.com",
        full_name="Demo Analyst",
        role="analyst",
        is_active=True,
        hashed_password=pwd_context.hash("demo1234"),
    ),
    "admin@citi.com": UserInDB(
        user_id="usr_002",
        email="admin@citi.com",
        full_name="Admin User",
        role="admin",
        is_active=True,
        hashed_password=pwd_context.hash("admin1234"),
    ),
}


# ── Auth helpers ──────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    user = DEMO_USERS.get(email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


# ── FastAPI dependency ────────────────────────────────────────────────────────
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    FastAPI dependency: validates JWT on every protected endpoint.
    Raises 401 if token is invalid/expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(user_id=payload.get("user_id"), email=email, role=payload.get("role"))
    except JWTError:
        logger.warning("[AUTH] Invalid JWT token")
        raise credentials_exception

    user = DEMO_USERS.get(token_data.email)
    if user is None or not user.is_active:
        raise credentials_exception

    return User(**user.dict(exclude={"hashed_password"}))


async def require_analyst(current_user: User = Depends(get_current_user)) -> User:
    """Dependency: requires analyst or admin role."""
    if current_user.role not in ("analyst", "admin"):
        raise HTTPException(status_code=403, detail="Analyst role required")
    return current_user


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency: requires admin role."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return current_user
