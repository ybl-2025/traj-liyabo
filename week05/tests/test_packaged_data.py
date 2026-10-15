"""The published source must reproduce the exact classroom CSV without downloads."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_data import ROOT,locate_data,sha256


class PublishedDataChecks(unittest.TestCase):
    def test_packaged_source_and_checksums(self):
        manifest=json.loads((ROOT/'data/package_manifest.json').read_text(encoding='utf-8'))
        source=locate_data()
        self.assertEqual(source.name,manifest['filename'])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),manifest['sha256'])
        self.assertEqual(sha256(source),manifest['uncompressed_sha256'])


if __name__=='__main__':
    unittest.main()
