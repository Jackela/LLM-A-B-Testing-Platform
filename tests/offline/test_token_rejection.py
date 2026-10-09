"""Use the declared PyJWT implementation to check malformed-token behavior."""
import unittest

from src.infrastructure.security.auth import EnhancedAuthSystem


class TokenRejection(unittest.TestCase):
    def test_invalid_tokens_return_none_instead_of_an_sdk_attribute_error(self):
        auth = EnhancedAuthSystem.__new__(EnhancedAuthSystem)
        auth.secret_key = "offline-fixture-key-only-012345678901234567890123456789"
        auth.algorithm = "HS256"
        auth.users = {}
        for token in ["not-a-jwt", "a.b.c", ""]:
            with self.subTest(token=token):
                self.assertIsNone(auth.verify_token(token))
