"""Users, roles, password hashing (scrypt) and login sessions stored in the event database."""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import re
import secrets

from event_store import _database

ROLES = ("admin", "reviewer", "viewer")
# What each role may do. Admin implies every permission.
PERMISSIONS = {
    "viewer": {"view"},
    "reviewer": {"view", "review", "delete", "export", "camera", "video"},
    "admin": {"view", "review", "delete", "export", "camera", "video", "admin"},
}
SESSION_HOURS = 12
USERNAME_RE = re.compile(r"[A-Za-z0-9_.@-]{3,64}\Z")


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                                   dklen=len(bytes.fromhex(digest)))
        return hmac.compare_digest(candidate.hex(), digest)
    except (ValueError, TypeError):
        return False


def _check(username, password, role):
    if not isinstance(username, str) or not USERNAME_RE.fullmatch(username):
        raise ValueError("Username must be 3-64 letters, digits, or _ . @ -")
    if password is not None and (not isinstance(password, str) or len(password) < 10 or len(password) > 200):
        raise ValueError("Password must be at least 10 characters")
    if role is not None and role not in ROLES:
        raise ValueError("Role must be admin, reviewer, or viewer")


def user_count(root):
    with _database(root) as connection:
        return connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def create_user(root, username, password, role="reviewer"):
    _check(username, password, role)
    with _database(root) as connection:
        if connection.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            raise ValueError("That username already exists")
        connection.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (
            username, hash_password(password), role, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return {"username": username, "role": role}


def list_users(root):
    with _database(root) as connection:
        rows = connection.execute("SELECT username, role, created FROM users ORDER BY username").fetchall()
    return [{"username": u, "role": r, "created": c} for u, r, c in rows]


def update_user(root, username, role=None, password=None):
    _check(username, password, role)
    with _database(root) as connection:
        current = connection.execute("SELECT role FROM users WHERE username=?", (username,)).fetchone()
        if not current:
            raise FileNotFoundError(username)
        if role and role != "admin" and current[0] == "admin" and connection.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0] <= 1:
            raise ValueError("Keep at least one admin")
        if role:
            connection.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        if password:
            connection.execute("UPDATE users SET password=? WHERE username=?", (hash_password(password), username))
            connection.execute("DELETE FROM sessions WHERE username=?", (username,))
    return {"username": username, "role": role or current[0]}


def delete_user(root, username):
    with _database(root) as connection:
        current = connection.execute("SELECT role FROM users WHERE username=?", (username,)).fetchone()
        if not current:
            raise FileNotFoundError(username)
        if current[0] == "admin" and connection.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0] <= 1:
            raise ValueError("Keep at least one admin")
        connection.execute("DELETE FROM users WHERE username=?", (username,))
        connection.execute("DELETE FROM sessions WHERE username=?", (username,))
    return True


def authenticate(root, username, password):
    with _database(root) as connection:
        row = connection.execute("SELECT password, role FROM users WHERE username=?", (str(username),)).fetchone()
    if not row:
        verify_password(str(password), hash_password("timing-equaliser"))
        return None
    return {"username": username, "role": row[1]} if verify_password(str(password), row[0]) else None


def _token_key(token):
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(root, username):
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
    with _database(root) as connection:
        connection.execute("DELETE FROM sessions WHERE expires < ?", (datetime.now(timezone.utc).isoformat(),))
        connection.execute("INSERT INTO sessions VALUES (?, ?, ?)", (_token_key(token), username, expires.isoformat()))
    return token


def session_user(root, token):
    if not token or len(token) > 100:
        return None
    with _database(root) as connection:
        row = connection.execute(
            "SELECT users.username, users.role, sessions.expires FROM sessions JOIN users "
            "ON users.username = sessions.username WHERE sessions.token=?", (_token_key(token),)).fetchone()
    if not row or row[2] < datetime.now(timezone.utc).isoformat():
        return None
    return {"username": row[0], "role": row[1]}


def end_session(root, token):
    if token:
        with _database(root) as connection:
            connection.execute("DELETE FROM sessions WHERE token=?", (_token_key(token),))


def allowed(user, permission):
    return bool(user) and permission in PERMISSIONS.get(user["role"], set())
