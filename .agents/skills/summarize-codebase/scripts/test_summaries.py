"""Lifecycle checks in disposable repositories; no BrainDance dependencies."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SummaryLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = Path(__file__).with_name("summaries.py").resolve()
        self.env = dict(os.environ)
        # Tests deliberately exercise operation without Git as well as real Git.
        self.env['CODE_SUMMARIES_GIT'] = str(self.root / 'missing-git')
        self.write('braindance/a.py', 'def event(x=2):\n    """Return event value."""\n    return x\n')
        self.write('braindance/b.py', 'from braindance.a import event\n\ndef consume():\n    return event()\n')

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def run_tool(self, action, *args, ok=0):
        result = subprocess.run([sys.executable, str(self.script), action, '--root', str(self.root), *args],
                                env=self.env, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, ok, result.stdout + result.stderr)
        return result.stdout

    def annotations(self):
        records = {}
        for path in (self.root / 'braindance').rglob('*.py'):
            records[path.relative_to(self.root).as_posix()] = {
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'overview': 'Fixture event processing. Used by the lifecycle test.',
                'feature_area': 'Neural Analysis', 'config': [], 'data_shapes': [],
                'notes': [], 'connections': [],
            }
        path = self.root / 'review.json'
        path.write_text(json.dumps(records), encoding='utf-8')
        return str(path)

    def build(self, *args):
        return self.run_tool('build', *args, '--annotations', self.annotations())

    def plan(self, *args):
        return json.loads(self.run_tool('plan', *args, '--json'))

    def test_full_noop_missing_and_untracked(self):
        self.build('--all')
        self.run_tool('check')
        self.assertEqual(self.plan()['files'], [])
        self.assertIn('up to date', self.run_tool('build'))
        (self.root / '.code-summaries/braindance/a.py.md').unlink()
        self.assertEqual(self.plan()['files'], ['braindance/a.py'])
        self.write('braindance/new space.py', 'value = 3\n')
        self.build()
        self.run_tool('check')

    def test_partial_does_not_hide_other_dirty_file(self):
        self.build('--all')
        self.write('braindance/a.py', 'def event(x=9):\n    return x\n')
        self.write('braindance/b.py', 'from braindance.a import event\n\ndef consume():\n    return event() + 1\n')
        self.build('braindance/a.py')
        report = json.loads(self.run_tool('check', '--json', ok=1))
        self.assertEqual(report['files'], ['braindance/b.py'])
        self.assertIn('b.py.md', (self.root / '.code-summaries/_INDEX.md').read_text())

    def test_symbol_removal_includes_consumers(self):
        self.build('--all')
        self.write('braindance/a.py', 'def renamed():\n    return 2\n')
        self.assertEqual(self.plan('braindance/a.py')['files'], ['braindance/a.py', 'braindance/b.py'])

    def test_reexport_consumers_are_invalidated(self):
        self.write('braindance/__init__.py', 'from .a import event\n')
        self.write('braindance/b.py', 'from braindance import event\n')
        self.build('--all')
        self.write('braindance/a.py', 'def renamed():\n    return 2\n')
        self.assertIn('braindance/b.py', self.plan('braindance/a.py')['files'])

    def test_rename_and_deletion(self):
        self.build('--all')
        (self.root / 'braindance/a.py').rename(self.root / 'braindance/new.py')
        self.build()
        self.assertFalse((self.root / '.code-summaries/braindance/a.py.md').exists())
        self.assertTrue((self.root / '.code-summaries/braindance/new.py.md').exists())
        self.run_tool('check')

    def test_stale_review_refused_before_writes(self):
        review = self.annotations()
        self.write('braindance/a.py', 'changed = True\n')
        self.run_tool('build', '--all', '--annotations', review, ok=2)
        self.assertFalse((self.root / '.code-summaries/_META.json').exists())

    def test_trivial_exports_and_line_endings(self):
        self.write('braindance/__init__.py', 'from .a import event\n')
        self.build('--all')
        path = self.root / 'braindance/a.py'
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
        self.run_tool('check')
        self.write('braindance/__init__.py', 'from .b import consume\n')
        self.run_tool('check', ok=1)
        self.run_tool('build')
        self.run_tool('check')
        self.assertIn('consume', (self.root / '.code-summaries/_INDEX.md').read_text())

    def test_scope_and_invalid_ref(self):
        self.write('braindance/internal_usage/private.py', 'secret = 1\n')
        self.write('braindance/test_event.py', 'def test_event(): pass\n')
        self.write('proj/outside.py', 'outside = True\n')
        self.assertEqual(len(self.plan('--all')['files']), 2)
        self.assertEqual(len(self.plan('--all', '--include-tests')['files']), 3)
        self.run_tool('plan', '--since', 'not-a-ref', ok=2)
        self.run_tool('plan', 'proj/*.py', ok=2)

    def test_incoming_edges_refresh_when_importer_changes(self):
        self.build('--all')
        target = self.root / '.code-summaries/braindance/a.py.md'
        self.assertIn('**Used by:** `braindance.b`', target.read_text(encoding='utf-8'))
        self.write('braindance/b.py', 'def consume():\n    return 1\n')
        self.build()
        self.run_tool('check')
        self.assertNotIn('braindance.b', target.read_text(encoding='utf-8'))
        self.assertNotIn('braindance/b.py:', target.read_text(encoding='utf-8'))

    def test_constants_invalidate_consumers(self):
        self.write('braindance/a.py', 'DEFAULTS = {}\n')
        self.write('braindance/b.py', 'from braindance.a import DEFAULTS\n')
        self.build('--all')
        self.write('braindance/a.py', 'OPTIONS = {}\n')
        self.assertIn('braindance/b.py', self.plan('braindance/a.py')['files'])

    def test_targeted_package_marker_and_executable_initializer(self):
        self.write('braindance/__init__.py', 'from .a import event\n')
        self.build('--all')
        self.write('braindance/__init__.py', 'from .b import consume\n')
        self.run_tool('build', 'braindance/__init__.py')
        self.run_tool('check')
        self.write('braindance/__init__.py', '__all__ = register_plugins()\n')
        self.assertIn('braindance/__init__.py', self.plan()['files'])

    def test_git_ref_staged_and_dirty_changes(self):
        executable = os.environ.get('CODE_SUMMARIES_GIT', 'git')
        self.env['CODE_SUMMARIES_GIT'] = executable
        def git(*args):
            result = subprocess.run([executable, '-C', str(self.root), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip()
        git('init')
        git('add', 'braindance')
        git('-c', 'user.name=Summary Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture')
        ref = git('rev-parse', 'HEAD')
        self.build('--all')
        self.write('braindance/a.py', 'def event():\n    return 3\n')
        git('add', 'braindance/a.py')
        self.write('braindance/b.py', 'from braindance.a import event\nresult = event()\n')
        plan = self.plan('--since', ref)
        self.assertEqual(plan['files'], ['braindance/a.py', 'braindance/b.py'])
        self.build('--since', ref)
        self.run_tool('check')


if __name__ == '__main__':
    unittest.main()
