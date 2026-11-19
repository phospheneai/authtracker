"""Reproducibility capture: where the code lives, the environment, dependencies."""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional

_log = logging.getLogger("authtracker")

CODE_MAX_BYTES = 1_000_000          # a patch or a script bigger than this is not stored
UNTRACKED_MAX_BYTES = 200_000       # nor is a new file bigger than this inside the patch
PATCH_NAME = "code_changes.patch"

def _git(*args: str, cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, cwd=cwd)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git_out(args: List[str], cwd: Optional[str], ok=(0,)) -> Optional[str]:
    """stdout of a git command, or None if it failed. `ok` lists the exit codes that count as success."""
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=20, cwd=cwd)
    except Exception:
        return None
    return out.stdout if out.returncode in ok else None


def clean_remote(url: str) -> str:
    """A git remote URL that is safe to store: no user name, password or token in it.

    `https://user:ghp_secret@github.com/org/repo.git` -> `https://github.com/org/repo.git`.
    """
    return re.sub(r"^([a-z][a-z0-9+.-]*://)[^/@\s]+@", r"\1", url.strip(), flags=re.I)


def web_url(remote: str) -> Optional[str]:
    """The repository's web address (https://host/org/repo) for a git remote, if it can be worked out."""
    remote = clean_remote(remote)
    match = re.match(r"^(?:ssh://)?(?:[\w.-]+@)?([\w.-]+)[:/](.+?)(?:\.git)?/?$", remote) \
        if not remote.lower().startswith(("http://", "https://")) else None
    if match:
        return f"https://{match.group(1)}/{match.group(2)}"
    match = re.match(r"^https?://([^/]+)/(.+?)(?:\.git)?/?$", remote, flags=re.I)
    return f"https://{match.group(1)}/{match.group(2)}" if match else None


def commit_url(remote: str, sha: str) -> Optional[str]:
    """A link to the commit on GitHub, GitLab or Bitbucket (None for other hosts)."""
    base = web_url(remote)
    if not base:
        return None
    host = base.split("/")[2].lower()
    if "github" in host or "gitlab" in host:
        return f"{base}/commit/{sha}"
    if "bitbucket" in host:
        return f"{base}/commits/{sha}"
    return None


def _candidate_dirs() -> List[str]:
    dirs = [os.getcwd()]
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)   # the script you ran, if any
    if main_file:
        dirs.append(os.path.dirname(os.path.abspath(main_file)))
    return list(dict.fromkeys(dirs))


def git_info(cwd: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Where the code came from: commit, branch, whether it has uncommitted changes, the remote it
    can be fetched from and a link to the commit. None if the code is not in a git repository.

    Looks in the current folder, then in the folder of the script that is running.
    """
    for folder in ([cwd] if cwd else _candidate_dirs()):
        sha = _git("rev-parse", "HEAD", cwd=folder)
        if not sha:
            continue
        info = {"sha": sha, "dirty": str(bool(_git("status", "--porcelain", cwd=folder))).lower()}
        branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=folder)
        if branch and branch != "HEAD":                      # "HEAD" means a detached checkout: no branch
            info["branch"] = branch
        remote = _git("config", "--get", "remote.origin.url", cwd=folder)
        if not remote:
            names = (_git("remote", cwd=folder) or "").split()
            remote = _git("config", "--get", f"remote.{names[0]}.url", cwd=folder) if names else None
        if remote:
            info["remote"] = clean_remote(remote)
            link = commit_url(remote, sha)
            if link:
                info["url"] = link
        return info
    return None


# Things that should never be copied into a run: the code of a run is readable by everyone who can open it.
_SECRET_PATTERNS = [
    ("an AWS access key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("a GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("a Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("an API key (sk-...)", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("a private key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY")),
    ("a password or secret assigned in the code",
     re.compile(r"""(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\w*\s*[:=]\s*['"][^'"\s]{8,}['"]""")),
]


def find_secret(text: str) -> Optional[str]:
    """What kind of secret `text` seems to contain (a best-effort check, not a guarantee), or None."""
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _is_text(data: bytes) -> bool:
    return b"\0" not in data[:8000]


def code_changes(folder: Optional[str] = None) -> Dict[str, Any]:
    """The changes since the last commit as a patch you can `git apply`, new files included.

    Returns {'patch': text or None, 'status': 'none' | 'saved' | 'skipped: <reason>', 'untracked_skipped': n}.
    """
    root = (_git_out(["rev-parse", "--show-toplevel"], folder) or "").strip()
    if not root:
        return {"patch": None, "status": "not a git repository", "untracked_skipped": 0}
    pieces: List[str] = []
    tracked = _git_out(["diff", "HEAD", "--no-color", "--no-ext-diff"], root)
    if tracked:
        pieces.append(tracked)
    skipped = 0
    for rel in filter(None, (_git_out(["ls-files", "--others", "--exclude-standard", "-z"], root) or "").split("\0")):
        full = os.path.join(root, rel)
        try:
            if os.path.getsize(full) > UNTRACKED_MAX_BYTES or not _is_text(open(full, "rb").read(8000)):
                skipped += 1
                continue
        except OSError:
            continue
        added = _git_out(["diff", "--no-index", "--no-color", "--no-ext-diff", "--", "/dev/null", rel], root, ok=(0, 1))
        if added:
            pieces.append(added)
    patch = "".join(p if p.endswith("\n") else p + "\n" for p in pieces)
    if not patch.strip():
        return {"patch": None, "status": "none", "untracked_skipped": skipped}
    if len(patch.encode("utf-8", "replace")) > CODE_MAX_BYTES:
        return {"patch": None, "status": f"skipped: larger than {CODE_MAX_BYTES // 1000} KB", "untracked_skipped": skipped}
    secret = find_secret(patch)
    if secret:
        return {"patch": None, "status": f"skipped: looks like it contains {secret}", "untracked_skipped": skipped}
    return {"patch": patch, "status": "saved", "untracked_skipped": skipped}


def _scripts_dirs() -> set:
    """Folders where installed console scripts live (`pytest`, `mlflow`, `uvicorn`, ...): never your own script."""
    import sysconfig
    found = {sysconfig.get_path("scripts"), os.path.join(sys.prefix, "bin"), os.path.join(sys.prefix, "Scripts")}
    return {os.path.realpath(d) for d in found if d}


def script_snapshot() -> Dict[str, Any]:
    """The script that is running (not a notebook, not a library): its text and its path in the repo."""
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    if not path or not os.path.isfile(path):
        return {"text": None, "status": "not a script (a notebook or the interactive prompt)"}
    path = os.path.abspath(path)
    stdlib = __import__("sysconfig").get_paths().get("stdlib", "\0")
    if ("site-packages" in path or "dist-packages" in path or path.startswith(stdlib)
            or os.path.realpath(os.path.dirname(path)) in _scripts_dirs()):
        return {"text": None, "status": "not your script (a library's entry point)"}
    if os.path.getsize(path) > CODE_MAX_BYTES:
        return {"text": None, "status": f"skipped: larger than {CODE_MAX_BYTES // 1000} KB"}
    try:
        text = open(path, encoding="utf-8").read()
    except (OSError, UnicodeDecodeError):
        return {"text": None, "status": "skipped: not readable as text"}
    secret = find_secret(text)
    if secret:
        return {"text": None, "status": f"skipped: looks like it contains {secret}"}
    folder = os.path.dirname(path)
    root = (_git_out(["rev-parse", "--show-toplevel"], folder) or "").strip()
    relative = os.path.relpath(path, root) if root else None
    return {"text": text, "status": "saved", "name": os.path.basename(path), "repo_path": relative}


# ---------------------------------------------------------------------------------------------- containers
_exists = os.path.exists                       # (a name tests can replace)
_DIGEST = re.compile(r"@(sha256:[0-9a-f]{64})$")
_warned_image: set = set()


def container_runtime() -> Optional[str]:
    """"kubernetes", "docker", "podman" or "container" when this process runs inside a container, else None."""
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    if _exists("/.dockerenv"):
        return "docker"
    if _exists("/run/.containerenv"):
        return "podman"
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as f:
            if re.search(r"docker|kubepods|containerd|libpod|lxc", f.read()):
                return "container"
    except OSError:
        pass
    return None


def _ecs_image() -> Optional[Dict[str, str]]:
    """On AWS ECS / Fargate the platform tells the container its own image and its digest."""
    uri = os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
    if not uri:
        return None
    try:
        import urllib.request
        with urllib.request.urlopen(uri, timeout=2) as response:
            data = json.load(response)
    except Exception:
        return None
    found = {}
    if data.get("Image"):
        found["image"] = data["Image"]
    if str(data.get("ImageID", "")).startswith("sha256:"):
        found["digest"] = data["ImageID"]
    return found or None


def _repository(reference: str) -> str:
    """`registry.io/team/app:v3` -> `registry.io/team/app` (the part without a tag or digest)."""
    name = reference.split("@")[0]
    last = name.rsplit("/", 1)[-1]
    return name[: -(len(last.split(":", 1)[1]) + 1)] if ":" in last else name


def _docker_digest(reference: str) -> Optional[str]:
    """The digest a local image tag points to right now, if the docker command can tell (usually on the host)."""
    try:
        result = subprocess.run(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", reference],
                                capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    match = _DIGEST.search(result.stdout.strip()) if result.returncode == 0 else None
    return match.group(1) if match else None


def container_info(image: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Which container image this ran in, pinned to a digest when that can be found out.

    `image` is what you tell authtracker (`init(image=...)`). Otherwise it looks at the platform (AWS ECS)
    and at CI variables (GitLab's CI_JOB_IMAGE). A container cannot normally see its own image name, so a
    build or deploy step usually has to pass it in. Returns None outside a container when no image is given.
    """
    runtime = container_runtime()
    found: Dict[str, str] = {}
    if image:
        found["image"] = image
    else:
        found = _ecs_image() or ({"image": os.environ["CI_JOB_IMAGE"]} if os.environ.get("CI_JOB_IMAGE") else {})
    if not runtime and not found:
        return None
    info: Dict[str, str] = {"runtime": runtime or "not detected"}
    reference = found.get("image")
    digest = found.get("digest")
    if reference:
        match = _DIGEST.search(reference)
        digest = digest or (match.group(1) if match else None) or _docker_digest(reference)
        info["image"] = reference
    if digest:
        info["digest"] = digest
    info["pinned"] = "true" if digest else "false"
    if reference and digest and not _DIGEST.search(reference):
        info["image_pinned"] = f"{_repository(reference)}@{digest}"
    key = (reference, info["pinned"])
    if info["pinned"] == "false" and key not in _warned_image:
        _warned_image.add(key)
        if reference:
            _log.warning("The container image %r is not pinned to a digest, and a tag can point to a different "
                         "image later. Pass the digest to at.init(image='name@sha256:...') to pin it.", reference)
        else:
            _log.warning("This runs in a container, but its image is not known. Pass it to "
                         "at.init(image='registry/name@sha256:...') to record it.")
    return info


def capture_env_tags() -> Dict[str, str]:
    tags = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    git = git_info()
    if git:
        tags["git_sha"] = git["sha"]
        tags["git_dirty"] = git["dirty"]
        for key in ("branch", "remote", "url"):
            if key in git:
                tags[f"git_{key}"] = git[key]
    return tags


def requirements_text() -> str:
    from importlib import metadata
    lines = sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in metadata.distributions() if d.metadata["Name"]
    )
    return "\n".join(lines) + "\n"


def python_executable() -> str:
    return sys.executable
