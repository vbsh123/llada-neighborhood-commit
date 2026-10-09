import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import checkout


class CheckoutTests(unittest.TestCase):
    def test_existing_non_repo_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'keep.txt'
            marker.write_text('keep')
            with self.assertRaisesRegex(ValueError, 'Missing SOAR checkout'):
                checkout.prepare_checkout(folder)
            self.assertEqual(marker.read_text(), 'keep')

    def test_commit_dirty_and_source_hash_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            source = root / 'eval_llada8b' / 'generate.py'
            source.parent.mkdir()
            source.write_text('pinned source')
            subprocess.run(['git', '-C', folder, 'add', '.'], check=True)
            subprocess.run(['git', '-C', folder, '-c', 'user.name=Test',
                            '-c', 'user.email=test@example.com', 'commit', '-qm', 'fixture'], check=True)
            actual_commit = checkout.git(root, 'rev-parse', 'HEAD')
            actual_hash = checkout.hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'must be at'):
                checkout.verify_checkout(root)
            with patch.object(checkout, 'COMMIT', actual_commit), patch.object(checkout, 'SOURCE_SHA256', actual_hash):
                self.assertEqual(checkout.verify_checkout(root), source)
                source.write_text('edited')
                with self.assertRaisesRegex(ValueError, 'local changes'):
                    checkout.verify_checkout(root)
                source.write_text('pinned source')
                (root / 'untracked.txt').write_text('extra')
                with self.assertRaisesRegex(ValueError, 'local changes'):
                    checkout.verify_checkout(root)
                (root / 'untracked.txt').unlink()
                with patch.object(checkout, 'SOURCE_SHA256', 'wrong'):
                    with self.assertRaisesRegex(ValueError, 'hash differs'):
                        checkout.verify_checkout(root)


if __name__ == '__main__':
    unittest.main()
