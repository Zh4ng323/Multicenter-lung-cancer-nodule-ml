# -*- coding: utf-8 -*-
"""
User authentication for the research prototype.

No credentials are baked in. The app reads users from (in priority order):
  1. the file at env var ``LUNGAI_USERS_FILE`` (default ``users.json``), and/or
  2. the env var ``LUNGAI_USERS`` as ``user1:pass1,user2:pass2``.

If neither source provides any user, the app runs in local demo mode with
authentication disabled (every login is accepted). This is intended only for
single-user local testing — see web_app/README.md to configure real users
before any networked deployment.
"""

import json
import os
import hashlib

_demo_warned = False


def _hash_password(password):
    """SHA-256 password hashing."""
    return hashlib.sha256(password.encode()).hexdigest()


def _users_from_env():
    """Parse LUNGAI_USERS='user:pass,user:pass' into a hashed dict."""
    raw = os.environ.get("LUNGAI_USERS", "").strip()
    if not raw:
        return {}
    out = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        user, pw = pair.split(":", 1)
        user, pw = user.strip(), pw.strip()
        if user and pw:
            out[user] = _hash_password(pw)
    return out


def users_file_path():
    return os.environ.get("LUNGAI_USERS_FILE", "users.json")


def load_users(path=None):
    """Load the user→hash map from users.json (if present) plus env overrides."""
    path = path or users_file_path()
    users = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                print(f"[auth] Warning: {path} is not valid JSON; ignoring it.")
                users = {}
    users.update(_users_from_env())
    return users


def auth_handler(username, password):
    """Gradio auth callback. Demo mode (no users configured) accepts all."""
    global _demo_warned
    users = load_users()
    if not users:
        if not _demo_warned:
            print("[auth] No users configured (no users.json, no LUNGAI_USERS). "
                  "Running in LOCAL DEMO mode — authentication disabled. "
                  "Set up users before any networked deployment.")
            _demo_warned = True
        return True
    return users.get(username) == _hash_password(password)


def add_user(username, password, path=None):
    """Add a new user to users.json (creates the file if needed)."""
    path = path or users_file_path()
    users = load_users(path) if os.path.exists(path) else {}
    users[username] = _hash_password(password)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    return path
