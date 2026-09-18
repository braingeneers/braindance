"""Prepare a full tutorial cache and pinned manifest from public original files.

Requires the AWS CLI for object listing. Downloads use anonymous HTTPS; this
script does not publish objects or change their access controls. A previously
downloaded source file may be placed at its bundle destination before running.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse
from urllib.request import urlopen


def main(
    output_dir=None,
    version="3",
    source_prefix="braindance/2026-04-10-closedloop/25245hs5/",
    bucket="braingeneers",
    endpoint="https://s3-west.nrp-nautilus.io",
):
    from braindance.tutorial import (
        _download_file,
        _load_registry,
        _resolve_dataset,
        _safe_relative_path,
        _safe_segment,
    )

    version = _safe_segment(str(version), "version")
    if version == "1":
        raise ValueError("Version 1 is immutable; choose a new version")
    if urlparse(endpoint).scheme != "https":
        raise ValueError("The source endpoint must use HTTPS")
    if output_dir is None:
        from braindance.config import get_output_dir

        output_dir = get_output_dir() / "tutorial_full"
    output_dir = Path(output_dir).resolve()
    folder = output_dir / "closed-loop-small"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / ".complete.json"
    marker.unlink(missing_ok=True)
    registry = copy.deepcopy(_load_registry(None, None))
    dataset = registry["datasets"]["closed-loop-small"]
    _, entries = _resolve_dataset(registry, "closed-loop-small", "1")
    chip_path = "2026-04-10-closedloop/25245hs5"
    recording = f"{chip_path}/closed_loop_plasticity/002_closed_loop"
    for entry in entries:
        entry["path"] = entry["path"].replace(f"{chip_path}/002_closed_loop/", recording + "/")
    paths = {entry["path"] for entry in entries}

    listing = subprocess.run(
        ["aws", "--endpoint", endpoint, "s3api", "list-objects-v2",
         "--bucket", bucket, "--prefix", source_prefix, "--output", "json"],
        check=True, capture_output=True, text=True,
    )
    objects = sorted(json.loads(listing.stdout).get("Contents", []),
                     key=lambda item: item["Key"])
    sources = []
    for obj in objects:
        key = obj["Key"]
        if key.endswith("/") or "rt_sort_inter" in PurePosixPath(key).parts:
            continue
        if not key.startswith(source_prefix):
            raise ValueError(f"Object is outside requested source prefix: {key}")
        relative = key.removeprefix("braindance/")
        if relative in paths and PurePosixPath(relative).name == "mapping.csv":
            relative = str(PurePosixPath(relative).with_name("original_mapping.csv"))
        _safe_relative_path(relative)
        if relative in paths:
            raise ValueError(f"Source files collide at bundle path: {relative}")
        paths.add(relative)
        size = obj["Size"]
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"Invalid source object size: {key}")
        sources.append({"path": relative, "bytes": size,
                        "url": f"{endpoint.rstrip('/')}/{quote(bucket)}/{quote(key, safe='/')}"})
    if not sources:
        raise ValueError("No original source files were found")

    for index, entry in enumerate(entries + sources, start=1):
        destination = folder / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        reuse = destination.is_file() and destination.stat().st_size == entry["bytes"]
        if reuse:
            with destination.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            reuse = "sha256" not in entry or digest.hexdigest() == entry["sha256"]
        print(f"[{index}/{len(entries) + len(sources)}] "
              f"{'Reusing' if reuse else 'Downloading'} {entry['path']} "
              f"({entry['bytes']:,} bytes)", flush=True)
        if not reuse:
            temporary = destination.with_name(destination.name + ".part")
            temporary.unlink(missing_ok=True)
            try:
                if "sha256" in entry:
                    _download_file(entry, temporary)
                else:
                    digest = hashlib.sha256()
                    total = 0
                    with urlopen(entry["url"], timeout=60) as response:
                        if urlparse(response.geturl()).scheme != "https":
                            raise ValueError("Source download redirected away from HTTPS")
                        with temporary.open("xb") as handle:
                            while chunk := response.read(1024 * 1024):
                                total += len(chunk)
                                if total > entry["bytes"]:
                                    raise ValueError(f"Source exceeds listed size: {entry['path']}")
                                digest.update(chunk)
                                handle.write(chunk)
                    if total != entry["bytes"]:
                        raise ValueError(f"Source differs from listed size: {entry['path']}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        if "sha256" not in entry:
            entry["sha256"] = digest.hexdigest()

    # Small, derived metadata ships inline with the pinned manifest; original
    # public data URLs and scientific reference values remain unchanged.
    for entry in entries:
        path = folder / entry["path"]
        if entry["path"] == "catalog.csv":
            rows = list(csv.DictReader(io.StringIO(path.read_text())))
            for row in rows:
                rec_path = PurePosixPath(row["proj"], row["chip"], row["experiment"])
                row.update(data_path=str(rec_path / f"{rec_path.name}_spike_data.pkl"),
                           log_path=str(rec_path / f"{rec_path.name}_log.csv"),
                           mapping_path=str(rec_path.parent / "mapping.csv"),
                           results_path=str(rec_path / "results"),
                           raw_data_path=str(rec_path / "002.raw.h5"))
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            content = output.getvalue()
        elif entry["path"] == "reference.json":
            reference = json.loads(path.read_text())
            reference["dataset_version"] = version
            content = json.dumps(reference, indent=2, sort_keys=True) + "\n"
        else:
            continue
        path.write_text(content, encoding="utf-8")
        entry.pop("url", None)
        entry.update(content=content, bytes=len(content.encode("utf-8")),
                     sha256=hashlib.sha256(content.encode("utf-8")).hexdigest())

    dataset["versions"][version] = {"files": sorted(entries + sources,
                                                    key=lambda entry: entry["path"])}
    dataset["default_version"] = version
    dataset["description"] = (
        "Full closed-loop tutorial: original H5 recordings and supporting files, "
        "excluding rt_sort_inter intermediates; includes verified middle-phase analysis exports."
    )
    _, normalized = _resolve_dataset(registry, "closed-loop-small", version)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    marker.write_text(json.dumps({"files": normalized}, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared cache: {folder}\nManifest: {manifest_path}", flush=True)
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--version", default="3")
    main(**vars(parser.parse_args()))
