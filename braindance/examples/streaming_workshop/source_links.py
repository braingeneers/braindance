"""Latest-file GitHub links, without importing scientific implementations."""
import ast
from functools import lru_cache
from pathlib import Path
import re
import subprocess
from urllib.parse import quote


def _git(root, *args):
    return subprocess.check_output(
        ['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL,
        timeout=5).rstrip('\n')


@lru_cache(maxsize=1)
def _repository():
    try:
        root = _git(Path(__file__).parent, 'rev-parse', '--show-toplevel')
        remote = _git(root, 'remote', 'get-url', 'origin')
        match = re.fullmatch(r'(?:git@github\.com:|https://github\.com/|ssh://git@github\.com/)([\w.-]+/[\w.-]+?)(?:\.git)?/?', remote)
        if match:
            return root, 'https://github.com/' + match[1], _git(root, 'rev-parse', 'HEAD')
    except (OSError, subprocess.SubprocessError):
        pass
    return None


@lru_cache(maxsize=128)
def _definitions(root, revision, path):
    try:
        tree = ast.parse(_git(root, 'show', f'{revision}:{path}'))
    except (OSError, subprocess.SubprocessError, SyntaxError):
        return {}
    definitions = {}

    def visit(body, prefix=''):
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                definitions[name] = node.lineno
                visit(node.body, name + '.')

    visit(tree.body)
    return definitions


def source_url(module, symbol):
    """Link to the latest file on the GitHub repository's default branch.

    Local committed definitions identify the file, but the URL uses GitHub's
    HEAD so local, unpushed commits cannot produce invalid revision links.
    Omit line anchors because local line numbers may differ from the latest file.
    Missing Git metadata or a non-GitHub origin simply disables these links.
    """
    repository = _repository()
    if repository is None:
        return None
    root, github, revision = repository
    path = module.replace('.', '/') + '.py'
    if symbol not in _definitions(root, revision, path):
        return None
    return f'{github}/blob/HEAD/{quote(path, safe="/")}'


def analysis_source_urls():
    module = 'braindance.examples.streaming_workshop.'
    return {
        **{kind: source_url(module + 'analysis_workspace', 'AnalysisWorkspace._analyze')
           for kind in ('raw', 'spikes', 'sttc', 'latency')},
        'overlap': source_url(module + 'analysis_evoked', 'analyze_evoked'),
        'sorting': source_url(module + 'analysis_sorting', 'run_sorting'),
    }
