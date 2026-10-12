import base64, hashlib, hmac, os, time
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from .config import settings
from .db import get_db
from .models import User

ITERATIONS = 200_000

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2${ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(h).decode()}"

def verify_password(password: str, stored: str) -> bool:
    try:
        _, it, salt, h = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(it))
        return hmac.compare_digest(calc, base64.b64decode(h))
    except Exception:
        return False

def make_token(email: str, role: str) -> str:
    return jwt.encode({"sub": email, "role": role, "exp": int(time.time()) + settings.token_minutes * 60},
                      settings.secret_key, algorithm="HS256")

# --- brute-force protection: 5 wrong passwords lock that email for 60 seconds
_fails: dict = {}
def locked_seconds(email: str) -> int:
    _, until = _fails.get(email, (0, 0))
    return max(0, int(until - time.time()))
def record_failure(email: str):
    n, _ = _fails.get(email, (0, 0))
    n += 1
    _fails[email] = (0, time.time() + 60) if n >= 5 else (n, 0)
def clear_failures(email: str):
    _fails.pop(email, None)

bearer = HTTPBearer(auto_error=False)

def current_user(cred: HTTPAuthorizationCredentials = Depends(bearer), db=Depends(get_db)) -> User:
    if not cred:
        raise HTTPException(401, "Not authenticated")
    try:
        data = jwt.decode(cred.credentials, settings.secret_key, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")
    user = db.scalar(select(User).where(User.email == data["sub"]))
    if not user:
        raise HTTPException(401, "Unknown user")
    return user

def require_role(*roles):
    def dep(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(403, "You do not have permission for this action")
        return user
    return dep
