"""Protect the distinction between historical reports and reproduced experiments."""
import copy
import json
import unittest

from tools.verify_experiment_evidence import MANIFEST, verify


class ExperimentEvidence(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(MANIFEST.read_text())

    def test_committed_record_agrees_with_sources(self):
        verify(record=self.record)

    def test_changed_source_hash_fails(self):
        changed = copy.deepcopy(self.record)
        changed["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Source changed"):
            verify(record=changed)

    def test_completion_count_cannot_be_promoted_to_a_full_run(self):
        changed = copy.deepcopy(self.record)
        changed["reported_run"]["completed_samples"] = 5197
        with self.assertRaisesRegex(ValueError, "outcome counts"):
            verify(record=changed)
