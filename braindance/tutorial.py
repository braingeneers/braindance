"""Verified tutorial downloads installed at one stable path per dataset."""

from __future__ import annotations

import hashlib
import filecmp
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from importlib.resources import files as resource_files
from pathlib import Path, PurePosixPath

from braindance.config import get_data_dir


_MARKER = ".complete.json"
_LOCK_TIMEOUT_SECONDS = 30
_DOWNLOAD_TIMEOUT_SECONDS = 30
_MAX_FILE_BYTES = 1024 * 1024 * 1024


class TutorialDataError(RuntimeError):
    """Raised when tutorial data cannot be resolved, downloaded, or verified."""


def _safe_segment(value, label):
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise TutorialDataError(f"Invalid tutorial dataset {label}: {value!r}")
    if any(character in value for character in "/\\:\0") or Path(value).is_absolute():
        raise TutorialDataError(f"Invalid tutorial dataset {label}: {value!r}")
    return value


def _safe_relative_path(value):
    if not isinstance(value, str) or not value:
        raise TutorialDataError("Manifest file paths must be non-empty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise TutorialDataError(f"Unsafe manifest file path: {value!r}")
    if any(character in value for character in "\\:\0") or value == _MARKER:
        raise TutorialDataError(f"Unsafe manifest file path: {value!r}")
    return path


def _load_registry(manifest_path, manifest):
    if manifest_path is not None and manifest is not None:
        raise TypeError("Pass either manifest_path or manifest, not both")
    if manifest is not None:
        registry = manifest
    elif manifest_path is not None:
        try:
            registry = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TutorialDataError(f"Could not read tutorial manifest: {exc}") from exc
    else:
        try:
            text = resource_files("braindance").joinpath("tutorial_data.json").read_text(
                encoding="utf-8"
            )
            registry = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            raise TutorialDataError(f"Could not read packaged tutorial manifest: {exc}") from exc
    if not isinstance(registry, dict) or not isinstance(registry.get("datasets"), dict):
        raise TutorialDataError("Tutorial manifest must contain a 'datasets' object")
    return registry


def _resolve_dataset(registry, name, version):
    dataset = registry["datasets"].get(name)
    if not isinstance(dataset, dict):
        raise TutorialDataError(f"Unknown tutorial dataset: {name!r}")
    if version is None:
        version = dataset.get("default_version")
    version = _safe_segment(version, "version")
    versions = dataset.get("versions")
    spec = versions.get(version) if isinstance(versions, dict) else None
    if not isinstance(spec, dict):
        raise TutorialDataError(f"Unknown version {version!r} for tutorial dataset {name!r}")
    raw_files = spec.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise TutorialDataError(f"Tutorial dataset {name!r} version {version!r} has no files")

    normalized = []
    seen = set()
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise TutorialDataError("Each tutorial manifest file must be an object")
        relative = _safe_relative_path(entry.get("path"))
        if str(relative) in seen:
            raise TutorialDataError(f"Duplicate manifest file path: {relative}")
        seen.add(str(relative))
        size = entry.get("bytes")
        digest = entry.get("sha256")
        url = entry.get("url")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= _MAX_FILE_BYTES:
            raise TutorialDataError(f"Invalid byte count for {relative}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise TutorialDataError(f"Invalid SHA-256 for {relative}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise TutorialDataError(f"Invalid SHA-256 for {relative}") from exc
        content = entry.get("content")
        if content is not None:
            if not isinstance(content, str) or url is not None:
                raise TutorialDataError(f"Inline tutorial file must contain text and no URL: {relative}")
            encoded = content.encode("utf-8")
            if len(encoded) != size or hashlib.sha256(encoded).hexdigest() != digest.lower():
                raise TutorialDataError(f"Inline tutorial file failed verification: {relative}")
            normalized.append({"path": str(relative), "content": content,
                               "bytes": size, "sha256": digest.lower()})
            continue
        parsed = urllib.parse.urlparse(url) if isinstance(url, str) else None
        loopback_http = (
            parsed is not None
            and parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        )
        if parsed is None or (parsed.scheme != "https" and not loopback_http):
            raise TutorialDataError(f"Tutorial file URL must use HTTPS: {url!r}")
        normalized.append(
            {"path": str(relative), "url": url, "bytes": size, "sha256": digest.lower()}
        )
    return version, normalized


def _regular_file(path, root):
    current = path
    while current != root:
        if current.is_symlink():
            return False
        current = current.parent
    return path.is_file() and not path.is_symlink()


def _verify_dataset(folder, expected_files):
    try:
        if folder.is_symlink():
            return False
        marker = folder / _MARKER
        if not _regular_file(marker, folder):
            return False
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded != {"files": expected_files}:
            return False
        for entry in expected_files:
            path = folder.joinpath(*PurePosixPath(entry["path"]).parts)
            if not _regular_file(path, folder) or path.stat().st_size != entry["bytes"]:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _file_matches(path, root, entry):
    if not _regular_file(path, root) or path.stat().st_size != entry["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == entry["sha256"]


def _download_file(entry, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if "content" in entry:
        payload = entry["content"].encode("utf-8")
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise TutorialDataError(f"Inline tutorial file failed verification: {entry['path']}")
        with destination.open("xb") as output:
            output.write(payload)
        return
    digest = hashlib.sha256()
    total = 0
    interactive = sys.stderr.isatty()
    last_update = 0.0

    def show_progress(verified=False):
        fraction = total / entry["bytes"] if entry["bytes"] else float(verified)
        filled = int(24 * fraction)
        bar = "#" * filled + "-" * (24 - filled)
        prefix = "\r" if interactive else ""
        print(
            f"{prefix}[{bar}] {fraction:6.1%} "
            f"{total / 1024:.1f}/{entry['bytes'] / 1024:.1f} KiB"
            f"{' - verified' if verified else ''}",
            end="" if interactive else "\n", file=sys.stderr, flush=True,
        )

    show_progress()
    request = urllib.request.Request(entry["url"], headers={"User-Agent": "braindance-tutorial/1"})
    try:
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            final_url = urllib.parse.urlparse(response.geturl())
            final_loopback_http = (
                final_url.scheme == "http"
                and final_url.hostname in {"localhost", "127.0.0.1", "::1"}
            )
            if final_url.scheme != "https" and not final_loopback_http:
                raise TutorialDataError(
                    f"Download redirected to a non-HTTPS URL: {response.geturl()!r}"
                )
            with destination.open("xb") as output:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > entry["bytes"]:
                        raise TutorialDataError(
                            f"Downloaded file exceeds expected size: {entry['path']}"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                    now = time.monotonic()
                    if interactive and now - last_update >= 0.1:
                        show_progress()
                        last_update = now
        if total != entry["bytes"]:
            raise TutorialDataError(
                f"Wrong byte count for {entry['path']}: expected {entry['bytes']}, got {total}"
            )
        if digest.hexdigest() != entry["sha256"]:
            raise TutorialDataError(f"SHA-256 mismatch for {entry['path']}")
        show_progress(verified=True)
    except (OSError, urllib.error.URLError) as exc:
        raise TutorialDataError(f"Could not download {entry['path']}: {exc}") from exc
    finally:
        if interactive:
            print(file=sys.stderr, flush=True)


def get_test_data(
    name="closed-loop-small",
    *,
    version=None,
    cache_dir=None,
    offline=False,
    update=False,
    manifest_path=None,
    manifest=None,
) -> Path:
    """Download and verify a tutorial dataset, returning its stable local folder.

    Downloads occur only when this function is called. ``cache_dir`` is the root
    under which ``name`` is stored. ``update=True`` replaces an installed version;
    untracked files (including user results) are preserved. Old version directories
    are migrated automatically, reusing verified files even in offline mode.
    Tests and private mirrors may supply a
    decoded registry with ``manifest`` or a JSON registry with ``manifest_path``.
    """

    name = _safe_segment(name, "name")
    registry = _load_registry(manifest_path, manifest)
    version, expected_files = _resolve_dataset(registry, name, version)
    # Resolving here permits an intentionally symlinked cache root while keeping
    # all dataset-owned directories beneath one concrete location.
    root = (Path(cache_dir) if cache_dir is not None else get_data_dir() / "tutorials").resolve()
    target = root / name
    if _verify_dataset(target, expected_files):
        print(f"Using verified cached tutorial data: {target}", flush=True)
        return target
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f".{name}.lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if _verify_dataset(target, expected_files):
                return target
            if time.monotonic() >= deadline:
                raise TutorialDataError(f"Timed out waiting for tutorial dataset lock: {lock}")
            time.sleep(0.05)

    staging = None
    backup = None
    try:
        if _verify_dataset(target, expected_files):
            return target
        # Read only safe manifest-owned paths. Unknown directories/files remain
        # user-owned and are never removed during migration or updates.
        installations = []
        if target.is_dir() and not target.is_symlink():
            versions = registry["datasets"][name]["versions"]
            candidates = [child for child in sorted(target.iterdir())
                          if child.name in versions or child.name.isdecimal()]
            for folder in [target] + candidates:
                marker = folder / _MARKER
                if not folder.is_dir() or folder.is_symlink() or not _regular_file(marker, folder):
                    continue
                try:
                    entries = json.loads(marker.read_text(encoding="utf-8"))["files"]
                    for entry in entries:
                        _safe_relative_path(entry["path"])
                        if not isinstance(entry["bytes"], int) or not isinstance(entry["sha256"], str):
                            raise ValueError("Invalid cached manifest")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise TutorialDataError(f"Cannot safely migrate cached manifest: {marker}") from exc
                installations.append((folder, entries))
        installed = next((entries for folder, entries in installations if folder == target), None)
        if installed is not None and installed != expected_files and not update:
            raise TutorialDataError("A different tutorial version is installed. Pass update=True (CLI: --update) to replace managed files and preserve user results.")
        legacy = {folder.name for folder, _ in installations if folder != target}
        staging = Path(tempfile.mkdtemp(prefix=f".{name}.staging-", dir=root))
        if target.is_dir() and not target.is_symlink():
            shutil.copytree(target, staging, dirs_exist_ok=True, symlinks=True,
                            ignore=lambda directory, names: legacy if Path(directory) == target else [])
        reusable = {}
        for folder, entries in installations:
            owned = {entry["path"] for entry in entries} | {_MARKER}
            for entry in entries:
                source = folder / entry["path"]
                if ("results" in PurePosixPath(entry["path"]).parts and source.exists()
                        and not _file_matches(source, folder, entry)):
                    raise TutorialDataError(f"Downloaded results have been modified: {source}. Move them to a separate results folder before updating; the existing dataset has been preserved.")
                reusable.setdefault((entry["bytes"], entry["sha256"]), []).append((folder, entry["path"]))
                if folder == target:
                    old = staging / entry["path"]
                    if old.exists() or old.is_symlink():
                        if not _regular_file(old, staging):
                            raise TutorialDataError(f"Managed file is not a regular file: {old}")
                        old.unlink()
                        parent = old.parent
                        while parent != staging:
                            try:
                                parent.rmdir()
                            except OSError:
                                break
                            parent = parent.parent
            if folder != target:
                # Preserve additions to every old version, merging only identical
                # content. Conflicting user files need a deliberate resolution.
                for directory, children, filenames in os.walk(folder, followlinks=False):
                    if any((Path(directory) / child).is_symlink() for child in children):
                        raise TutorialDataError(f"Cannot migrate user directory symlink in {directory}")
                    for filename in filenames:
                        source = Path(directory) / filename
                        relative = source.relative_to(folder).as_posix()
                        if relative in owned:
                            continue
                        destination = staging / relative
                        parent = destination
                        while parent != staging:
                            if parent.is_symlink():
                                raise TutorialDataError(f"Cannot merge user files through symlink: {parent}")
                            parent = parent.parent
                        if not _regular_file(source, folder):
                            raise TutorialDataError(f"Cannot migrate user symlink: {source}")
                        if destination.exists():
                            if not _regular_file(destination, staging) or not filecmp.cmp(source, destination, shallow=False):
                                raise TutorialDataError(f"Conflicting user files during migration: {relative}")
                        else:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, destination)
        for index, entry in enumerate(expected_files, start=1):
            destination = staging.joinpath(*PurePosixPath(entry["path"]).parts)
            parent = destination
            while parent != staging:
                if parent.is_symlink():
                    raise TutorialDataError(f"Tutorial destination must not be a symlink: {parent}")
                parent = parent.parent
            if destination.exists():
                # An untracked file must never be silently overwritten.
                if _file_matches(destination, staging, entry):
                    continue
                raise TutorialDataError(f"Tutorial file conflicts with user data: {entry['path']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            reused = False
            for folder, relative in reusable.get((entry["bytes"], entry["sha256"]), []):
                source = folder / relative
                if _file_matches(source, folder, entry):
                    shutil.copy2(source, destination)
                    reused = True
                    break
            if reused:
                continue
            if offline and "content" not in entry:
                raise TutorialDataError(f"Tutorial dataset {name!r} version {version!r} is not available in a verified cache: {entry['path']}")
            operation = "Writing tutorial metadata" if "content" in entry else "Downloading tutorial data"
            print(f"{operation} [{index}/{len(expected_files)}]: "
                  f"{entry['path']} ({entry['bytes']} bytes)", flush=True)
            _download_file(entry, destination)
        (staging / _MARKER).unlink(missing_ok=True)
        (staging / _MARKER).write_text(
            json.dumps({"files": expected_files}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if not _verify_dataset(staging, expected_files):
            raise TutorialDataError("Downloaded tutorial dataset failed final verification")
        if target.exists() or target.is_symlink():
            backup = root / f".{name}.replaced-{os.getpid()}-{time.time_ns()}"
            os.replace(target, backup)
        try:
            os.replace(staging, target)
            staging = None
        except BaseException:
            if backup is not None:
                os.replace(backup, target)
                backup = None
            raise
        if backup is not None:
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup)
            else:
                backup.unlink()
            backup = None
        return target
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and not target.exists():
            os.replace(backup, target)
        shutil.rmtree(lock, ignore_errors=True)
