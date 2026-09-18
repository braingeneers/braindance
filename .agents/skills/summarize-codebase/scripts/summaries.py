"""Portable source-summary maintenance. Python 3.10+, stdlib only; never imports BrainDance.

Run from any directory: python path/to/summaries.py plan|build|check [options].
Semantic review is supplied by an agent; this script extracts facts and maintains indexes.
"""

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def digest(path):
    # Normalize checkout line endings so hashes survive Windows/Linux clones.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(root, *args):
    result = subprocess.run(
        [os.environ.get("CODE_SUMMARIES_GIT", "git"), "-C", str(root), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8", errors="replace").strip()


def discover(root, include_tests=False):
    skipped_dirs = {"__pycache__", "vendor", "vendored", "third_party", "generated",
                    "build", "dist", "node_modules", "archive", "manuscript_code",
                    "user_manual", "internal_usage", "stdp_pairs"}
    if not include_tests:
        skipped_dirs |= {"tests", "test", "internal_tests"}
    files, trivial, holes = {}, {}, {}
    for directory, dirs, names in os.walk(root / "braindance"):
        for name in list(dirs):
            if name in skipped_dirs or name.startswith("."):
                holes[(Path(directory) / name).relative_to(root).as_posix() + "/"] = "excluded directory"
                dirs.remove(name)
        for name in sorted(names):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                holes[rel] = "symlink/outside source root"
                continue
            if path.suffix not in {".py", ".js", ".html", ".css"}:
                continue
            if name.startswith("TODO_") or (not include_tests and (name.startswith("test_") or name.endswith("_test.py"))):
                holes[rel] = "test or placeholder"
                continue
            if name == "__init__.py":
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                    simple = all(isinstance(n, (ast.Import, ast.ImportFrom)) or
                                 (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)) or
                                 (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
                                  and isinstance(n.value, (ast.List, ast.Tuple))
                                  and all(isinstance(v, ast.Constant) and isinstance(v.value, str) for v in n.value.elts))
                                 for n in tree.body)
                    if simple:
                        exports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
                        trivial[rel] = "; ".join(exports) or "Package marker."
                        continue
                except SyntaxError:
                    pass
            files[rel] = digest(path)
    return dict(sorted(files.items())), dict(sorted(trivial.items())), dict(sorted(holes.items()))


def module_name(path):
    return path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")


def symbols(info):
    names = {n.name for n in info.get("classes", []) + info.get("functions", [])}
    if info.get("tree"):
        for node in info["tree"].body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    return sorted(names)


def facts(root, rel):
    source = (root / rel).read_text(encoding="utf-8-sig")
    result = {"classes": [], "functions": [], "imports": [], "tree": None, "entry": False}
    if not rel.endswith(".py"):
        return result
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result["parse_error"] = f"SyntaxError at line {exc.lineno}: {exc.msg}"
        return result
    result["tree"] = tree
    result["entry"] = any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test) and "__main__" in ast.unparse(n.test) for n in tree.body) or rel.endswith("/__main__.py")
    result["classes"] = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    result["functions"] = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    result["entry"] = result["entry"] or any(n.name == 'main' for n in result['functions'])
    if rel.startswith(('braindance/examples/', 'braindance/experiments/')):
        result['entry'] = result['entry'] or any(
            isinstance(child, ast.Call)
            for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for child in ast.walk(node)
        )
    package = module_name(rel).split(".") if rel.endswith("/__init__.py") else module_name(rel).split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append((alias.name, alias.asname or alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            prefix = ".".join(package[:len(package) - node.level + 1]) if node.level else ""
            mod = ".".join(x for x in (prefix, node.module) if x)
            for alias in node.names:
                result["imports"].append((f"{mod}.{alias.name}", alias.asname or alias.name, node.lineno))
    return result


def compact(value, limit=240):
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def signature(node):
    value = f"{node.name}({ast.unparse(node.args)})"
    if node.returns:
        value += " -> " + ast.unparse(node.returns)
    return value


def purpose(node, annotations, name):
    if name in annotations.get("purposes", {}):
        return compact(annotations["purposes"][name], 450)
    doc = ast.get_docstring(node)
    return compact(doc.split("\n\n")[0], 450) if doc else "unclear — see source"


def effects(node):
    found = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            if any(ast.unparse(t).startswith("self.") for t in targets):
                found.add("mutates instance state")
        if isinstance(child, ast.Call):
            name = ast.unparse(child.func)
            if name.split(".")[-1] in {"save", "dump", "to_csv", "savefig", "write", "write_text", "write_bytes"}:
                found.add("write call (static hint)")
    return "; ".join(sorted(found)) or "unclear — see source"


def render(root, rel, annotation, info, consumers, callers):
    lines = [f"# {Path(rel).name}", "", f"**Path:** `{rel}`", f"**Module:** `{module_name(rel)}`",
             f"**Feature Area:** `{annotation['feature_area']}`",
             f"**Entry point:** {'yes — main / module execution (static candidate)' if info['entry'] else 'no — library or imported component'}",
             "", "## Overview", annotation["overview"], "", "## Connections"]
    critical = sorted(set(target for target, _, _ in info["imports"] if target.startswith("braindance.")))
    for target in critical:
        mod, _, symbol = target.rpartition(".")
        lines.append(f"- **Uses:** `{symbol}` from `{mod}` — imports (static evidence).")
    for caller in sorted(consumers.get(module_name(rel), [])):
        lines.append(f"- **Used by:** `{caller}` — import consumer hint; not a proven runtime call.")
    lines += [f"- **Shared data:** {item}" for item in annotation.get("connections", [])]
    if lines[-1] == "## Connections":
        lines.append("None")
    lines += ["", "## Dependencies"]
    dependencies = sorted(set(t for t, _, _ in info["imports"] if t.split(".")[0] not in sys.stdlib_module_names))
    lines += [f"- `{target}` — {'intra-repo import' if target.startswith('braindance.') else 'external or unresolved local import'}; source import evidence." for target in dependencies] or ["None"]
    lines += ["", "## Classes"]
    if not info["classes"]:
        lines.append("None")
    for cls in info["classes"]:
        bases = ", ".join(ast.unparse(b) for b in cls.bases)
        lines += [f"### {cls.name}({bases})", f"> {purpose(cls, annotation, cls.name)}",
                  f"**Source:** `{rel}:{cls.lineno}`", "**Kind:** class. **Instantiated by:** " + ("; ".join(sorted(callers.get(module_name(rel) + '.' + cls.name, []))) or "unclear — see source")]
        methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        init = next((n for n in methods if n.name == "__init__"), None)
        lines.append(f"**Constructor:** `{signature(init)}`" if init else "**Constructor:** inherited / implicit")
        attrs = {}
        for method in methods:
            for node in ast.walk(method):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                            attrs.setdefault(target.attr, (ast.unparse(node.annotation) if isinstance(node, ast.AnnAssign) else "inferred at runtime", compact(ast.unparse(node.value), 130) if node.value else "annotation only"))
        lines.append("**Key attributes:**")
        if attrs:
            lines += ["| Name | Type | Assignment / meaning |", "| --- | --- | --- |"]
            lines += [f"| `{name}` | {kind} | `{value}` |" for name, (kind, value) in sorted(attrs.items())]
        else:
            lines.append("None")
        lines.append("**Methods:**")
        for method in methods:
            if method.name == "__init__":
                continue
            lines += [f"#### `{signature(method)}`", f"> {purpose(method, annotation, cls.name + '.' + method.name)}",
                      f"> **Called by:** unresolved static dispatch. **Side effects:** {effects(method)}.",
                      f"**Source:** `{rel}:{method.lineno}`"]
        if not any(n.name != "__init__" for n in methods):
            lines.append("None")
    lines += ["", "## Functions"]
    for node in info["functions"]:
        lines += [f"### `{signature(node)}`", f"> {purpose(node, annotation, node.name)}",
                  f"> **Called by:** {'; '.join(sorted(callers.get(module_name(rel) + '.' + node.name, []))) or 'unclear — see source'}. **Side effects:** {effects(node)}.",
                  f"**Source:** `{rel}:{node.lineno}`"]
    for symbol in annotation.get("symbols", []):
        lines += [f"### `{symbol.get('signature', symbol['name'])}`", f"> {symbol['purpose']}", f"**Source:** `{rel}:{symbol['line']}`"]
    if not info["functions"] and not annotation.get("symbols"):
        lines.append("None")
    for title, key in (("Config / CLI", "config"), ("Data Shapes", "data_shapes"), ("Notes", "notes")):
        lines += ["", f"## {title}"]
        values = list(annotation.get(key, []))
        if key == "notes" and info.get("parse_error"):
            values.append(info["parse_error"])
        if key == 'config' and values:
            lines += ['| Configuration / CLI and meaning |', '| --- |']
            lines += ['| ' + item.replace('|', '\\|').replace('\n', ' ') + ' |' for item in values]
        else:
            lines += [f"- {item}" for item in values] or ["None"]
    return "\n".join(lines) + "\n"


def read_summary(path):
    text = path.read_text(encoding="utf-8")
    def field(name):
        match = re.search(r"^\*\*" + re.escape(name) + r":\*\* `?([^`\n]+)", text, re.M)
        if not match:
            raise ValueError(f"Missing {name}: {path}")
        return match.group(1)
    overview = re.search(r"^## Overview\n(.+?)(?=\n## |\Z)", text, re.M | re.S)
    return {"file": field("Path"), "module": field("Module"), "feature_area": field("Feature Area"),
            "entry": field("Entry point").startswith("yes"),
            "overview": compact(overview.group(1) if overview else "unclear — see source", 300),
            "uses": re.findall(r"^- \*\*Uses:\*\* `([^`]+)` from `([^`]+)`", text, re.M),
            "symbols": re.findall(r"^### `?([\w]+)\(", text, re.M)}


def refresh_reverse(out, consumers, callers):
    """Refresh derived incoming edges without re-reviewing unchanged semantics."""
    for path in (out / "braindance").rglob("*.md"):
        record = read_summary(path)
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^- \*\*Used by:\*\*.*\n", "", text, flags=re.M)
        incoming = [f"- **Used by:** `{caller}` — import consumer hint; not a proven runtime call."
                    for caller in sorted(consumers.get(record['module'], []))]
        if incoming:
            text = text.replace("## Connections\nNone\n", "## Connections\n")
            text = text.replace("## Connections\n", "## Connections\n" + "\n".join(incoming) + "\n", 1)
        section, current = None, None
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith('## '):
                section, current = line[3:], None
            match = re.match(r'^### `?(\w+)\(', line)
            if match:
                current = match.group(1)
            if line.startswith('#### '):
                current = None
            hints = '; '.join(sorted(callers.get(record['module'] + '.' + str(current), []))) or 'unclear — see source'
            if section == 'Classes' and current and line.startswith('**Kind:** class. **Instantiated by:**'):
                lines[index] = '**Kind:** class. **Instantiated by:** ' + hints
            if section == 'Functions' and current and line.startswith('> **Called by:**'):
                lines[index] = re.sub(r'^> \*\*Called by:\*\* .*?\. \*\*Side effects:\*\*', '> **Called by:** ' + hints + '. **Side effects:**', line)
        updated = '\n'.join(lines) + '\n'
        if updated != path.read_text(encoding='utf-8'):
            path.write_text(updated, encoding='utf-8')


def assemble(root, out, meta, trivial, holes, features):
    records = [read_summary(path) for path in sorted((out / "braindance").rglob("*.md"))] if (out / "braindance").exists() else []
    for record in records:
        record['symbols'] = sorted(set(record['symbols']) | set(meta['files'].get(record['file'], {}).get('symbols', [])))
    heading = f"Generated: {meta['generated']} | commit: {meta['commit'] or 'unavailable'} | mode: {meta['mode']}"
    index = ["# Code summaries", "", heading, f"\n{len(records)} file summaries; {len(trivial)} package markers. Scope: `braindance/` only.",
             "", "Use `_FEATURES.md` for behavior, `_GRAPH.md` for import consumers. Check freshness with the helper before relying on summaries.",
             "", "## Directory tree", "```text"]
    dirs = sorted({str(Path(r["file"]).parent).replace("\\", "/") for r in records})
    index += ["  " * (d.count("/")) + d.rsplit("/", 1)[-1] + "/" for d in sorted(set(p.as_posix() for d in dirs for p in [Path(d), *Path(d).parents] if p.as_posix() != "."))]
    index += ["```"]
    grouped = defaultdict(list)
    for record in records:
        grouped[record["feature_area"]].append(record)
    for directory in dirs:
        index += ["", f"## {directory}"]
        index += [f"- [{Path(r['file']).name}]({r['file']}.md) — {r['overview']} Symbols: {', '.join(r['symbols']) or 'module-level code'}." for r in records if str(Path(r["file"]).parent).replace("\\", "/") == directory]
    index += ["", "## Package markers and re-exports (no per-file summary)"]
    index += [f"- `{path}` — {compact(exports, 600)}" for path, exports in trivial.items()]
    index += ["", "## Known holes", "- Outside `braindance/`: `proj/`, root scripts, tests and deployment tooling.",
              "- Notebooks, data, weights, images, and other non-source assets are not summarized."]
    index += [f"- `{path}` — {reason}." for path, reason in holes.items()]
    if meta.get('stale_files'):
        index += ["", "## Missing or stale summaries"]
        index += [f"- `{path}` — refresh before relying on navigation." for path in meta['stale_files']]
    graph = ["# Connections graph", "", heading, "", "Edges below are source-backed **imports**, not runtime calls. Dynamic dispatch, wildcard imports, re-exports, and external consumers can be incomplete. Search package-marker exports in `_INDEX.md` when an edge targets a package."]
    feature_index = ["# Features", "", heading]
    for number, (area, flow) in enumerate(features.items(), 1):
        members = grouped.get(area, [])
        if not members:
            continue
        entry = [r for r in members if r["entry"]]
        graph += ["", f"## {area}", flow, "", "Entry points: " + (", ".join(f"[{r['module']}]({r['file']}.md)" for r in entry) or "imported library components below")]
        for r in members:
            for symbol, mod in r["uses"]:
                graph.append(f"- `{r['module']}` --imports--> `{mod}.{symbol}`")
        feature_index += ["", f"## {number}. {area}", flow, "", "**Entry points:** " + (", ".join(f"[{r['module']}]({r['file']}.md)" for r in entry) or "library API; select supporting files below"), "", "**Supporting files:**"]
        feature_index += [f"- [{Path(r['file']).name}]({r['file']}.md) — {r['overview']}" for r in members]
    for name, lines in (("_INDEX.md", index), ("_FEATURES.md", feature_index), ("_GRAPH.md", graph)):
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "build", "check"])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--all", action="store_true")
    modes.add_argument("--since")
    parser.add_argument("paths", nargs="*", help="Repo-relative source paths/globs; quote wildcards")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--annotations", type=Path, nargs="+", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_intermixed_args()
    if args.paths and (args.all or args.since):
        parser.error("paths cannot be combined with --all or --since")
    root = args.root.resolve()
    out = root / ".code-summaries"
    if not (root / "braindance").is_dir():
        parser.error("--root must contain braindance/")
    features = {
        "Configuration": "Choose portable input/catalog and output locations: environment/defaults → path getters → consumers.",
        "Acquisition": "Acquire neural frames from hardware, simulation, or dummy backends: device/socket → environment step → experiment phases.",
        "Experiment Phases": "Compose closed-loop experiments: phase configuration → experiment runner → acquisition/analysis/stimulation → logs and checkpoints.",
        "Stimulation": "Translate stimulation choices into electrode commands: selected channels/amplitudes → pulse sequences → hardware and experiment protocols.",
        "Replay": "Reconstruct recorded experiments: saved frames/events/checkpoints → replay environment → original phase runner and inspection.",
        "Spike Detection": "Detect spike events: voltage windows → neural detector/training pipeline → event candidates.",
        "Spike Sorting": "Assign spikes to units: recordings/detections → templates and propagation sequences → online or offline unit events.",
        "Artifact Removal": "Suppress stimulation artifacts: voltage stream and filter state → cleaned traces → detection and analysis.",
        "Data Catalog": "Find and load recordings: catalog/metadata → Recording and S3/local loader → cached arrays and per-recording results.",
        "Neural Analysis": "Measure neural responses: spike times/stimuli → binning, latency, bursts, connectivity → result arrays and tables.",
        "Visualization": "Inspect experiments and analysis: recordings/results or launch settings → plots, GUI views, and experiment processes.",
        "Games and Reinforcement": "Drive games with neural activity: encoded state → stimulation → neural readout → action/reward and feedback.",
        "Examples and Workshop": "Run example workflows and interactive workshop: experiment specification → configured runner → streaming UI/logs/results.",
    }
    meta_path = out / "_META.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    files, trivial, holes = discover(root, args.include_tests)
    previous = old.get("files", {})
    trivial_hashes = {p: digest(root / p) for p in trivial}
    index_changed = old.get("trivial_files", {}) != trivial_hashes
    stale = {p for p, sha in files.items() if previous.get(p, {}).get("sha256") != sha or not (out / (p + ".md")).exists()}
    deleted = {p for p in previous if p not in files}
    since = args.since or old.get("commit")
    head, warning = None, None
    try:
        head = git(root, "rev-parse", "HEAD")
        if not since:
            since = git(root, "log", "-1", "--format=%H", "--", ".code-summaries") or None
    except (RuntimeError, OSError) as exc:
        if args.since:
            parser.error(str(exc))
        warning = str(exc)
    mode = "path" if args.paths else "all" if args.all or (not since and not old) else "since" if args.since else "incremental"
    selected = set(files) if mode == "all" else set(stale)
    if args.since:
        # --no-renames includes both old and new names; NUL output handles spaces.
        try:
            changed = git(root, "diff", "--name-only", "--no-renames", "-z", args.since, "--", "braindance/").split("\0")
            selected |= set(changed) & set(files)
        except (RuntimeError, OSError) as exc:
            parser.error(str(exc))
    if mode == "path":
        patterns = [p.replace("\\", "/").removeprefix("./").rstrip("/") for p in args.paths]
        match = lambda p: any(p == pattern or p.startswith(pattern + "/") or fnmatch.fnmatchcase(p, pattern) for pattern in patterns)
        selected = {p for p in files if match(p)}
        deleted = {p for p in deleted if match(p)}
        if not selected and not deleted and not any(match(p) for p in trivial):
            parser.error("no supported files match paths within braindance/")
    # Public symbol removal invalidates consumers, even in a targeted refresh.
    all_facts = {}
    if args.action != "check":
        for rel in sorted(selected):
            all_facts[rel] = facts(root, rel)
        removed_targets = set()
        for rel in selected | deleted:
            current = all_facts.get(rel, {})
            names = set(symbols(current))
            removed_names = {name for name in previous.get(rel, {}).get("symbols", []) if not name.startswith('_') and name not in names}
            removed_targets |= {module_name(rel) + "." + name for name in removed_names}
            if removed_names:
                removed_targets.add(module_name(rel))
            if rel in deleted:
                removed_targets.add(module_name(rel))
        # Include public re-export packages when matching import consumers.
        exports = {}
        for rel in trivial:
            for target, alias, _ in facts(root, rel)["imports"]:
                exports[module_name(rel) + "." + alias] = target
        for _ in range(len(exports) + 1):
            additions = {alias for alias, target in exports.items() if any(target == removed or target.startswith(removed + ".") for removed in removed_targets)} - removed_targets
            if not additions:
                break
            removed_targets |= additions
        for rel, record in previous.items():
            if rel in files and any(target == removed or target.startswith(removed + ".") or removed.startswith(target + ".") for target in record.get("uses", []) for removed in removed_targets):
                selected.add(rel)
    if args.action == "check":
        selected = stale
        deleted = set(previous) - set(files)
    report = {"mode": mode, "since": since, "commit": head, "count": len(selected),
              "files": sorted(selected), "deleted": sorted(deleted), "index_changed": index_changed, "warning": warning}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{mode}; since {since or 'none'}; {len(selected)} files; {len(deleted)} removed")
        for rel in sorted(selected):
            print(rel)
        for rel in sorted(deleted):
            print("DELETE " + rel)
        if warning:
            print("Git unavailable; content hashes remain authoritative: " + warning, file=sys.stderr)
    if args.action == "check":
        return 1 if selected or deleted or index_changed or not all((out / name).exists() for name in ("_INDEX.md", "_FEATURES.md", "_GRAPH.md", "_META.json")) else 0
    if args.action == "plan":
        return 0
    annotations = {}
    for path in args.annotations:
        annotations.update(json.loads(path.read_text(encoding="utf-8")))
    # Validate the entire batch before writing any output.
    for rel in sorted(selected):
        annotation = annotations.get(rel)
        if not annotation:
            parser.error(f"missing reviewed annotation: {rel}; follow summarize-codebase/SKILL.md")
        # Accept raw-byte hashes from reviewers; persisted hashes are LF-normalized.
        if annotation.get("source_sha256") not in {files[rel], hashlib.sha256((root / rel).read_bytes()).hexdigest()}:
            parser.error(f"annotation source hash is stale: {rel}")
        if annotation.get("feature_area") not in features or not annotation.get("overview"):
            parser.error(f"invalid feature_area/overview: {rel}")
        for key in ("config", "data_shapes", "notes", "connections"):
            if not isinstance(annotation.get(key), list):
                parser.error(f"annotation requires {key} list: {rel}")
    # Scan imports for conservative file-level consumer hints. No runtime imports.
    consumers = defaultdict(set)
    callers = defaultdict(set)
    if selected or deleted:
        for rel in files:
            if rel not in all_facts:
                all_facts[rel] = facts(root, rel)
            info = all_facts[rel]
            for target, _, _ in info["imports"]:
                for candidate in (target, target.rpartition(".")[0]):
                    consumers[candidate].add(module_name(rel))
            # Only direct calls to named local/imported functions/classes are hints.
            # Attribute receiver dispatch remains unresolved; never guess its type.
            bindings = {alias: target for target, alias, _ in info['imports'] if target.startswith('braindance.')}
            bindings.update({n.name: module_name(rel) + '.' + n.name for n in info['functions'] + info['classes']})
            if info['tree']:
                for node in ast.walk(info['tree']):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in bindings:
                        callers[bindings[node.func.id]].add(f"{rel}:{node.lineno} (named-call hint)")
    refreshed = dict(previous)
    out.mkdir(parents=True, exist_ok=True)
    for rel in sorted(deleted):
        target = (out / (rel + ".md")).resolve()
        if not target.is_relative_to((out / "braindance").resolve()):
            parser.error(f"unsafe metadata path: {rel}")
        target.unlink(missing_ok=True)
        refreshed.pop(rel, None)
    for rel in sorted(selected):
        info = all_facts[rel]
        target = out / (rel + ".md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(root, rel, annotations[rel], info, consumers, callers), encoding="utf-8")
        refreshed[rel] = {"sha256": files[rel], "symbols": symbols(info),
                          "uses": sorted(set(t for t, _, _ in info["imports"] if t.startswith("braindance.")))}
    meta = {"commit": head, "generated": datetime.now(timezone.utc).date().isoformat(),
            "mode": mode, "files_refreshed": len(selected), "source_roots": ["braindance/"],
            "hash_normalization": "CRLF to LF", "files": dict(sorted(refreshed.items())),
            "stale_files": sorted(p for p in files if refreshed.get(p, {}).get("sha256") != files[p] or not (out / (p + '.md')).exists()),
            "include_tests": args.include_tests, "trivial_files": trivial_hashes}
    if selected or deleted:
        refresh_reverse(out, consumers, callers)
    assemble(root, out, meta, trivial, holes, features)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{'up to date' if not selected and not deleted else 'refreshed'} (since {since or 'none'}); {len(refreshed)} summaries; {len(meta['stale_files'])} stale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
