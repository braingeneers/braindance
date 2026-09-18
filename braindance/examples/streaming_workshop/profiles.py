"""Named Python profiles with readable settings and retained revisions."""
import ast
import json
import re
import time
from pathlib import Path
from pprint import pformat


def verify_python(code):
    """Check participant code without executing imports or user functions."""
    if not isinstance(code, str) or len(code) > 100000:
        raise ValueError('Python code must contain at most 100000 characters')
    try:
        tree = ast.parse(code)
        compile(tree, '<profile>', 'exec')
    except SyntaxError as exc:
        return dict(ok=False, message=exc.msg, line=exc.lineno, column=exc.offset)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in ('encode', 'decode') + (('train',) if 'train' in functions else ()):
        node = functions.get(name)
        if node is None:
            return dict(ok=False, message=f'Missing top-level {name} function', line=None)
        args = node.args
        positional = len(args.posonlyargs) + len(args.args)
        if (isinstance(node, ast.AsyncFunctionDef) or node.decorator_list or
                positional - len(args.defaults) > 4 or
                (positional < 4 and args.vararg is None) or
                any(default is None for default in args.kw_defaults)):
            return dict(ok=False, message=f'{name} must be an undecorated synchronous function accepting four positional arguments', line=node.lineno)
    return dict(ok=True, message='Syntax and encode/decode signatures' + (' plus optional train' if 'train' in functions else '') + ' passed. Code was not executed. Start or reload to check output shapes and values.', line=None)


class ProfileStore:
    def __init__(self, directory, template_file):
        self.directory = Path(directory).resolve()
        self.template_file = Path(template_file)

    def path(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
            raise ValueError('Profile name: use letters, numbers, underscores, or hyphens (1–64 characters)')
        path = (self.directory / f'{name}.py').resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('Profile must stay inside the profiles directory')
        return path

    def list(self):
        return sorted(p.stem for p in self.directory.glob('*.py'))

    def template(self):
        return self.template_file.read_text(encoding='utf-8')

    def load(self, name):
        path = self.path(name)
        code = path.read_text(encoding='utf-8')
        tree = ast.parse(code)
        settings = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SETTINGS' for t in node.targets):
                settings = ast.literal_eval(node.value)
        if not isinstance(settings, dict):
            raise ValueError('SETTINGS must be a literal dictionary')
        from .custom_analysis import resolve_analysis_config
        settings = resolve_analysis_config(settings, code)
        return dict(name=name, path=str(path), code=code, settings=settings)

    def save(self, name, code, settings):
        path = self.path(name)
        if not isinstance(code, str) or len(code) > 100000:
            raise ValueError('Profile code must contain at most 100000 characters')
        tree = ast.parse(code)
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        if not {'encode', 'decode'} <= names:
            raise ValueError('Profile needs top-level encode() and decode() functions')
        # Settings come from the form; code is never executed while saving/loading.
        settings = json.loads(json.dumps(settings, allow_nan=False))
        if not isinstance(settings, dict):
            raise ValueError('Settings must be a dictionary')
        from .custom_analysis import resolve_analysis_config
        settings = resolve_analysis_config(settings, code)
        lines = code.splitlines(keepends=True)
        for node in reversed(tree.body):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SETTINGS' for t in node.targets):
                del lines[node.lineno - 1:node.end_lineno]
        text = ''.join(lines).rstrip() + '\n\n# Setup values saved with this profile.\nSETTINGS = ' + pformat(settings, sort_dicts=False) + '\n'
        self.directory.mkdir(parents=True, exist_ok=True)
        history = self.directory / 'revisions' / name
        history.mkdir(parents=True, exist_ok=True)
        stamp = str(time.time_ns())
        if path.exists():
            (history / f'{stamp}_previous.py').write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
        (history / f'{stamp}.py').write_text(text, encoding='utf-8')
        temporary = path.with_suffix('.py.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)
        return self.load(name)
