# tutorial.py

**Path:** `braindance/tutorial.py`
**Module:** `braindance.tutorial`
**Feature Area:** `Examples and Workshop`
**Entry point:** no — library or imported component

## Overview
Installs manifest-defined tutorial datasets into one stable cache directory with strict path, size, and SHA-256 verification. It reuses verified files across versions, preserves user-owned additions, serializes concurrent installs with a lock directory, and atomically replaces completed datasets.

## Connections
- **Used by:** `braindance.examples.get_tutorial_data` — import consumer hint; not a proven runtime call.
- **Used by:** `braindance.examples.validate_tutorial_data` — import consumer hint; not a proven runtime call.
- **Uses:** `get_data_dir` from `braindance.config` — imports (static evidence).
- **Shared data:** Uses packaged `tutorial_data.json` by default and `braindance.config.get_data_dir()/tutorials` when no cache root is supplied.

## Dependencies
- `braindance.config.get_data_dir` — intra-repo import; source import evidence.

## Classes
### TutorialDataError(RuntimeError)
> Report invalid manifests, unsafe cache layouts, download failures, and verification conflicts.
**Source:** `braindance/tutorial.py:28`
**Kind:** class. **Instantiated by:** braindance/tutorial.py:102 (named-call hint); braindance/tutorial.py:104 (named-call hint); braindance/tutorial.py:108 (named-call hint); braindance/tutorial.py:112 (named-call hint); braindance/tutorial.py:115 (named-call hint); braindance/tutorial.py:126 (named-call hint); braindance/tutorial.py:182 (named-call hint); braindance/tutorial.py:213 (named-call hint); braindance/tutorial.py:223 (named-call hint); braindance/tutorial.py:233 (named-call hint); braindance/tutorial.py:237 (named-call hint); braindance/tutorial.py:240 (named-call hint); braindance/tutorial.py:287 (named-call hint); braindance/tutorial.py:313 (named-call hint); braindance/tutorial.py:317 (named-call hint); braindance/tutorial.py:330 (named-call hint); braindance/tutorial.py:336 (named-call hint); braindance/tutorial.py:34 (named-call hint); braindance/tutorial.py:350 (named-call hint); braindance/tutorial.py:36 (named-call hint); braindance/tutorial.py:360 (named-call hint); braindance/tutorial.py:363 (named-call hint); braindance/tutorial.py:366 (named-call hint); braindance/tutorial.py:375 (named-call hint); braindance/tutorial.py:381 (named-call hint); braindance/tutorial.py:393 (named-call hint); braindance/tutorial.py:404 (named-call hint); braindance/tutorial.py:42 (named-call hint); braindance/tutorial.py:45 (named-call hint); braindance/tutorial.py:47 (named-call hint); braindance/tutorial.py:60 (named-call hint); braindance/tutorial.py:68 (named-call hint); braindance/tutorial.py:70 (named-call hint); braindance/tutorial.py:77 (named-call hint); braindance/tutorial.py:84 (named-call hint); braindance/tutorial.py:87 (named-call hint); braindance/tutorial.py:93 (named-call hint); braindance/tutorial.py:96 (named-call hint)
**Constructor:** inherited / implicit
**Key attributes:**
None
**Methods:**
None

## Functions
### `_safe_segment(value, label)`
> unclear — see source
> **Called by:** braindance/tutorial.py:266 (named-call hint); braindance/tutorial.py:80 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/tutorial.py:32`
### `_safe_relative_path(value)`
> unclear — see source
> **Called by:** braindance/tutorial.py:309 (named-call hint); braindance/tutorial.py:94 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/tutorial.py:40`
### `_load_registry(manifest_path, manifest)`
> unclear — see source
> **Called by:** braindance/tutorial.py:267 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/tutorial.py:51`
### `_resolve_dataset(registry, name, version)`
> unclear — see source
> **Called by:** braindance/tutorial.py:268 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/tutorial.py:74`
### `_regular_file(path, root)`
> unclear — see source
> **Called by:** braindance/tutorial.py:147 (named-call hint); braindance/tutorial.py:154 (named-call hint); braindance/tutorial.py:168 (named-call hint); braindance/tutorial.py:304 (named-call hint); braindance/tutorial.py:335 (named-call hint); braindance/tutorial.py:362 (named-call hint); braindance/tutorial.py:365 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/tutorial.py:133`
### `_verify_dataset(folder, expected_files)`
> unclear — see source
> **Called by:** braindance/tutorial.py:273 (named-call hint); braindance/tutorial.py:284 (named-call hint); braindance/tutorial.py:293 (named-call hint); braindance/tutorial.py:403 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/tutorial.py:142`
### `_file_matches(path, root, entry)`
> unclear — see source
> **Called by:** braindance/tutorial.py:329 (named-call hint); braindance/tutorial.py:379 (named-call hint); braindance/tutorial.py:386 (named-call hint). **Side effects:** unclear — see source.
**Source:** `braindance/tutorial.py:167`
### `_download_file(entry, destination)`
> unclear — see source
> **Called by:** braindance/tutorial.py:397 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/tutorial.py:177`
### `get_test_data(name='closed-loop-small', *, version=None, cache_dir=None, offline=False, update=False, manifest_path=None, manifest=None) -> Path`
> Resolve, verify, download, migrate, and atomically install a versioned tutorial dataset, returning its stable local folder.
> **Called by:** braindance/examples/get_tutorial_data.py:18 (named-call hint); braindance/examples/validate_tutorial_data.py:20 (named-call hint). **Side effects:** write call (static hint).
**Source:** `braindance/tutorial.py:246`

## Config / CLI
| Configuration / CLI and meaning |
| --- |
| `get_test_data(name='closed-loop-small', version=None, cache_dir=None, offline=False, update=False, manifest_path=None, manifest=None)` selects the dataset and cache/update policy. |
| Module limits are a 30-second lock timeout, 30-second download timeout, and 1 GiB maximum file size. |

## Data Shapes
- Registry schema is `{'datasets': {name: {'default_version': ..., 'versions': {version: {'files': [...]}}}}}`; each file entry has path, byte count, SHA-256, and exactly one of HTTPS URL or inline text.
- `.complete.json` stores exactly `{'files': expected_files}` and is part of cache verification.

## Notes
- Only HTTPS is accepted except loopback HTTP for tests; redirects are checked again.
- Symlinks beneath managed dataset folders are rejected during verification and migration.
- Existing untracked files are copied into staging and conflicts abort rather than overwrite user data.
- Updates abort if a manifest-managed file beneath a `results` path has been modified locally, protecting user-generated results from replacement.
