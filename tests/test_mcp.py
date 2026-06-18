import os
import tempfile
import unittest
from importlib import import_module
from importlib.util import find_spec

_HAS_MCP = bool(find_spec("mcp"))


def _load_service():
    """Return the service object the MCP tools call.

    The MCP tools (bookwriter.mcp_server) are thin wrappers over a plain-Python
    service obtained from ``get_service()`` exposing create_book / write_book /
    get_manuscript. We exercise that exact object so the smoke test covers the
    same code path the tools do, without needing an MCP transport.

    Falls back to probing for a module that exposes the trio as functions.
    Returns None if nothing suitable is present yet (skip, don't error)."""
    try:
        mod = import_module("bookwriter.mcp_server")
    except Exception:
        mod = None
    if mod is not None and hasattr(mod, "get_service"):
        try:
            svc = mod.get_service()
        except Exception:
            svc = None
        if svc is not None and all(
            callable(getattr(svc, fn, None))
            for fn in ("create_book", "write_book", "get_manuscript")
        ):
            return svc

    candidates = (
        "bookwriter.server.service",
        "bookwriter.mcp_server",
        "bookwriter.service",
    )
    for name in candidates:
        try:
            cand = import_module(name)
        except Exception:
            continue
        if all(callable(getattr(cand, fn, None))
               for fn in ("create_book", "write_book", "get_manuscript")):
            return cand
    return None


@unittest.skipUnless(_HAS_MCP, "mcp not installed")
class TestMCPService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Probe availability without binding a data dir (the real service may
        # cache a singleton on first call); the per-test setUp resolves a fresh
        # service against the tempdir.
        if _load_service() is None:
            raise unittest.SkipTest("MCP service layer not available")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev_data = os.environ.get("BOOKWRITER_DATA_DIR")
        self._prev_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["BOOKWRITER_DATA_DIR"] = self._tmp.name
        os.environ.pop("ANTHROPIC_API_KEY", None)
        # Reset any cached service singleton so it rebinds to this tempdir.
        try:
            import bookwriter.mcp_server as _m
            if hasattr(_m, "_service_singleton"):
                _m._service_singleton = None
        except Exception:
            pass
        svc = _load_service()
        if svc is None:
            self.skipTest("MCP service layer not available")
        self.svc = svc

    def tearDown(self):
        if self._prev_data is None:
            os.environ.pop("BOOKWRITER_DATA_DIR", None)
        else:
            os.environ["BOOKWRITER_DATA_DIR"] = self._prev_data
        if self._prev_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._prev_key
        self._tmp.cleanup()

    @staticmethod
    def _book_id(result):
        """Service calls may return a BookSummary-ish dict (possibly nested
        under 'book') or an id string. Extract the id robustly."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if "book" in result and isinstance(result["book"], dict):
                return result["book"].get("id")
            return result.get("id")
        return getattr(getattr(result, "book", result), "id", None)

    def test_create_write_get_manuscript(self):
        created = self.svc.create_book(
            premise="A clockmaker discovers time runs backward at midnight.",
            chapters=2,
            words_per_chapter=120,
            title="MCP Smoke",
            genre="fantasy",
            mock=True,
        )
        book_id = self._book_id(created)
        self.assertTrue(book_id, f"no book id from create_book: {created!r}")

        # write_book drives the pipeline to completion synchronously for the
        # service/MCP path (MOCK mode, no network).
        self.svc.write_book(book_id)

        man = self.svc.get_manuscript(book_id)
        markdown = man.get("markdown") if isinstance(man, dict) else str(man)
        words = man.get("words") if isinstance(man, dict) else None
        self.assertTrue(markdown, "empty manuscript")
        if words is not None:
            self.assertGreater(words, 0)

    def test_local_summary_includes_words(self):
        # Directly exercise the local-fallback summary path (_LocalBookService),
        # which must include a `words` field matching the HTTP contract.
        import bookwriter.mcp_server as m
        created = self.svc.create_book(
            premise="A cartographer maps a city that rearranges itself nightly.",
            chapters=2, words_per_chapter=120, title="Words Smoke",
            genre="fantasy", mock=True,
        )
        book_id = self._book_id(created)
        self.svc.write_book(book_id)

        local = m._LocalBookService(self._tmp.name)
        # get_book nests the summary under "book"; list_books returns it flat.
        book = local.get_book(book_id)["book"]
        self.assertIn("words", book)
        self.assertGreater(book["words"], 0)
        listed = {b["id"]: b for b in local.list_books()}
        self.assertIn("words", listed[book_id])
        self.assertEqual(listed[book_id]["words"], book["words"])


if __name__ == "__main__":
    unittest.main()
