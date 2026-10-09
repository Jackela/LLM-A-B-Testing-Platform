"""Exercise the actual isolated legacy cache module, without starting the platform."""
import importlib.util
import sys
import unittest
from pathlib import Path

# The legacy performance package eagerly imports unrelated application services.
# Load this complete module to test its real boundary with its declared Redis/Pydantic dependencies.
path = Path(__file__).resolve().parents[2] / "src/infrastructure/performance/cache_manager.py"
spec = importlib.util.spec_from_file_location("offline_cache_contract", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class CacheSerialization(unittest.TestCase):
    def setUp(self):
        self.cache = module.CacheManager(module.CacheConfig())

    def test_json_round_trip_and_compression(self):
        value = {"answer": "中文" * 1000, "citations": [1, None, True]}
        data = self.cache._serialize_value(value)
        self.assertEqual(
            self.cache._deserialize_value(
                self.cache._decompress_data(self.cache._compress_data(data))
            ),
            value,
        )

    def test_pickle_is_never_executed_or_returned_as_a_valid_value(self):
        # A valid pickle invoking builtins.print is inert input to a JSON parser.
        payload = b"cbuiltins\nprint\n(S'cache-canary'\ntR."
        with self.assertRaises(ValueError):
            self.cache._deserialize_value(payload)

    def test_invalid_and_nonfinite_json_remain_cache_misses(self):
        for payload in [b"broken", b"NaN", b"Infinity", b"\xff"]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.cache._deserialize_value(payload)
        with self.assertRaises(ValueError):
            self.cache._serialize_value(float("inf"))
        with self.assertRaises(TypeError):
            self.cache._serialize_value(object())

    def test_legacy_pickle_setting_is_explicitly_rejected(self):
        legacy = module.CacheManager(module.CacheConfig(serialization_format="pickle"))
        with self.assertRaisesRegex(ValueError, "Only JSON"):
            legacy._serialize_value({"answer": "x"})
