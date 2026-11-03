"""Save files to the GitHub repository in one commit (GitHub REST "git data" API).

Needs a fine-grained personal access token with "Contents: Read and write" on the repository,
stored as the secret GITHUB_TOKEN. GITHUB_REPO defaults to DARSHAN120806/neuravol-forecast.
"""
from __future__ import annotations

import base64
from pathlib import Path

import requests

API = "https://api.github.com"


class GitHubStore:
    def __init__(self, token: str, repo: str, branch: str = "main"):
        self.repo, self.branch = repo, branch
        self.h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                  "X-GitHub-Api-Version": "2022-11-28"}

    def _req(self, method, path, **kw):
        r = requests.request(method, f"{API}/repos/{self.repo}{path}", headers=self.h, timeout=60, **kw)
        if r.status_code >= 400:
            msg = r.json().get("message", r.text[:200]) if r.headers.get("content-type", "").startswith("application/json") else r.text[:200]
            raise RuntimeError(f"GitHub {method} {path}: {r.status_code} {msg}")
        return r.json() if r.content else {}

    def check(self):
        """Returns (ok, message) without changing anything."""
        try:
            info = self._req("GET", "")
            perms = info.get("permissions", {})
            if perms and not perms.get("push", False):
                return False, "token can read but not write this repository"
            self._req("GET", f"/git/ref/heads/{self.branch}")
            return True, f"write access to {self.repo}@{self.branch}"
        except Exception as e:
            return False, str(e)

    def commit_files(self, root: Path, rel_paths, message: str):
        """Create one commit containing the given files (paths relative to root). Returns the commit sha."""
        ref = self._req("GET", f"/git/ref/heads/{self.branch}")
        parent = ref["object"]["sha"]
        base_tree = self._req("GET", f"/git/commits/{parent}")["tree"]["sha"]
        tree = []
        for rel in rel_paths:
            p = Path(root) / rel
            if not p.exists():
                continue
            blob = self._req("POST", "/git/blobs",
                             json={"content": base64.b64encode(p.read_bytes()).decode(), "encoding": "base64"})
            tree.append({"path": rel.replace("\\", "/"), "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if not tree:
            raise RuntimeError("no files to save")
        new_tree = self._req("POST", "/git/trees", json={"base_tree": base_tree, "tree": tree})["sha"]
        commit = self._req("POST", "/git/commits", json={"message": message, "tree": new_tree, "parents": [parent]})["sha"]
        self._req("PATCH", f"/git/refs/heads/{self.branch}", json={"sha": commit})
        return commit
