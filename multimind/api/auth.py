"""
Authentication module for the RAG API.
"""

from typing import Optional, Dict, Union, cast
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from datetime import datetime, timedelta
from jose import jwt, JWTError
import os
import json
from pathlib import Path
from pydantic import BaseModel

# API key header
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT settings
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")  # Change in production
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    scopes: list[str] = []

class UserInDB(User):
    hashed_password: str

class UserDB:
    """User database manager."""
    
    def __init__(self):
        self.users: Dict[str, Dict] = {}
        self.load_users()
    
    def load_users(self) -> None:
        """Load users from configuration."""
        config_file = os.getenv("USER_CONFIG", "users.json")
        config_path = Path(config_file)
        
        if config_path.exists():
            with open(config_path) as f:
                self.users = json.load(f)
        else:
            # Default admin user - should be changed in production
            self.users = {
                "admin": {
                    "username": "admin",
                    "email": "admin@example.com",
                    "full_name": "Administrator",
                    "disabled": False,
                    "hashed_password": "changeme",  # Use proper password hashing in production
                    "scopes": ["rag:read", "rag:write"]
                }
            }
            
    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        user_dict = self.users.get(username)
        if user_dict:
            return UserInDB(**user_dict)
        return None

# Initialize user database
user_db = UserDB()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(api_key_header)) -> User:
    """Get current user from API key or JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # First try API key
    if token in os.getenv("API_KEYS", "").split(","):
        return User(
            username="api_user",
            scopes=["rag:read", "rag:write"]
        )

    # Then try JWT
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, scopes=payload.get("scopes", []))
    except JWTError:
        raise credentials_exception

    user = user_db.get_user(username)
    if user is None:
        raise credentials_exception

    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        scopes=user.scopes
    )

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def check_scope(required_scope: str):
    """Check if user has required scope."""
    async def scope_checker(current_user: User = Depends(get_current_active_user)):
        if required_scope not in current_user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {required_scope}"
            )
        return current_user
    return scope_checker