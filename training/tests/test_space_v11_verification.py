"""Persisted audit evidence must replay without weakening JSON value checks."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_space_v11_repaired import json_value


class EvidenceRepresentationTests(unittest.TestCase):
    def test_tuple_evidence_matches_persisted_json_arrays(self):
        replay = {'citations': [('Source', '[ACTION_URL_1]')], 'values': (1, 2)}
        stored = {'citations': [['Source', '[ACTION_URL_1]']], 'values': [1, 2]}
        self.assertEqual(json_value(replay), stored)

    def test_changed_value_or_array_order_still_fails(self):
        replay = {'replacements': [('before', 'after')], 'values': (1, 2)}
        self.assertNotEqual(json_value(replay), {'replacements': [['before', 'wrong']], 'values': [1, 2]})
        self.assertNotEqual(json_value(replay), {'replacements': [['before', 'after']], 'values': [2, 1]})

    def test_non_json_evidence_fails_closed(self):
        with self.assertRaises(ValueError):
            json_value({'value': float('nan')})


if __name__ == '__main__':
    unittest.main()
