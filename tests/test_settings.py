"""Tests for the in-app settings store, provider/image verification, and the
settings service endpoints. Offline: network checks are monkeypatched.
"""
import os
import tempfile
import unittest
from unittest import mock

from bookwriter import runtime_config as rc
from bookwriter import provider, images


class _RcGuard(unittest.TestCase):
    def setUp(self):
        self._saved_path = rc._path
        self._saved_over = dict(rc._overrides)
        rc._overrides.clear()
        rc._path = None
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PIXIO_API_KEY",
                  "BOOKWRITER_LLM_PROVIDER", "BOOKWRITER_IMAGE_PROVIDER"):
            os.environ.pop(k, None)

    def tearDown(self):
        rc._overrides.clear()
        rc._overrides.update(self._saved_over)
        rc._path = self._saved_path


class TestRuntimeConfig(_RcGuard):
    def test_override_beats_env_and_clears(self):
        os.environ["ANTHROPIC_API_KEY"] = "env-key"
        self.assertEqual(rc.getenv("ANTHROPIC_API_KEY"), "env-key")
        rc.set_values({"ANTHROPIC_API_KEY": "override-key"}, persist=False)
        self.assertEqual(rc.getenv("ANTHROPIC_API_KEY"), "override-key")
        rc.set_values({"ANTHROPIC_API_KEY": ""}, persist=False)  # clear -> back to env
        self.assertEqual(rc.getenv("ANTHROPIC_API_KEY"), "env-key")

    def test_unmanaged_key_ignored(self):
        rc.set_values({"NOT_A_MANAGED_KEY": "x"}, persist=False)
        self.assertIsNone(rc.getenv("NOT_A_MANAGED_KEY"))

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            rc.bind_file(os.path.join(d, "settings.json"))
            rc.set_values({"PIXIO_API_KEY": "pxio_live_abc"})
            rc._overrides.clear()
            rc.load()
            self.assertEqual(rc.getenv("PIXIO_API_KEY"), "pxio_live_abc")

    def test_public_state_masks_secrets(self):
        rc.set_values({"OPENAI_API_KEY": "sk-supersecretvalue123"}, persist=False)
        st = rc.public_state()
        self.assertTrue(st["keys"]["OPENAI_API_KEY"]["set"])
        masked = st["keys"]["OPENAI_API_KEY"]["masked"]
        self.assertNotIn("supersecret", masked)
        self.assertIn("…", masked)
        self.assertFalse(st["keys"]["ANTHROPIC_API_KEY"]["set"])


class TestVerify(_RcGuard):
    def test_anthropic_ok(self):
        rc.set_values({"ANTHROPIC_API_KEY": "sk-ant-x"}, persist=False)
        with mock.patch.object(provider, "_http_status", return_value=(200, "")):
            r = provider.verify("anthropic")
        self.assertTrue(r["ok"])

    def test_anthropic_rejected(self):
        rc.set_values({"ANTHROPIC_API_KEY": "sk-ant-bad"}, persist=False)
        with mock.patch.object(provider, "_http_status", return_value=(401, "nope")):
            r = provider.verify("anthropic")
        self.assertFalse(r["ok"])

    def test_anthropic_no_key(self):
        self.assertFalse(provider.verify("anthropic")["ok"])

    def test_claude_cli_presence(self):
        with mock.patch.object(provider, "_claude_binary", return_value="/usr/bin/claude"):
            self.assertTrue(provider.verify("claude-cli")["ok"])
        with mock.patch.object(provider, "_claude_binary", return_value=None):
            self.assertFalse(provider.verify("claude-cli")["ok"])

    def test_image_pixio_ok(self):
        rc.set_values({"PIXIO_API_KEY": "pxio_live_x"}, persist=False)
        with mock.patch.object(images, "_get_json", return_value={"credits": 123}):
            r = images.verify("pixio")
        self.assertTrue(r["ok"])
        self.assertIn("123", r["detail"])

    def test_image_pixio_no_key(self):
        self.assertFalse(images.verify("pixio")["ok"])


try:
    from bookwriter.server.service import BookService
    _HAS_SERVER = True
except Exception:
    _HAS_SERVER = False


@unittest.skipUnless(_HAS_SERVER, "server extras not installed")
class TestSettingsService(_RcGuard):
    def test_get_save_round_trip_masked(self):
        from bookwriter.server.service import BookService
        with tempfile.TemporaryDirectory() as d:
            rc.bind_file(os.path.join(d, "settings.json"))
            svc = BookService(d)
            out = svc.save_settings({"ANTHROPIC_API_KEY": "sk-ant-secretvalue999",
                                     "BOOKWRITER_LLM_PROVIDER": "anthropic"})
            self.assertTrue(out["keys"]["ANTHROPIC_API_KEY"]["set"])
            self.assertNotIn("secretvalue999", str(out))      # never echoed in full
            self.assertEqual(out["llm"]["selected"], "anthropic")
            # And it actually drives availability now.
            self.assertTrue(provider.live_available("anthropic"))

    def test_verify_routes_to_image(self):
        from bookwriter.server.service import BookService
        with mock.patch.object(images, "verify", return_value={"ok": True, "detail": "x"}) as m:
            r = BookService.verify_provider("image", "pixio")
        self.assertTrue(r["ok"]); m.assert_called_once_with("pixio")


if __name__ == "__main__":
    unittest.main()
