"""Production-hardening tests: input bounds, bind-host guard, SSRF scheme guard,
and pipeline data-loss resilience (a chapter is never discarded by a downstream
extract/continuity failure). All offline — no network, no keys."""
import os
import tempfile
import unittest

from bookwriter import images, serve
from bookwriter.config import Settings
from bookwriter.mock import MockLLM
from bookwriter.pipeline import BookPipeline
from bookwriter.store import BookStore


# --------------------------------------------------------------------------- #
# M1 — bind-host guard
# --------------------------------------------------------------------------- #
class TestBindHostGuard(unittest.TestCase):
    def test_local_hosts_allowed(self):
        for h in ("127.0.0.1", "localhost", "::1"):
            self.assertIsNone(serve.remote_bind_error(h, allow_remote=False))

    def test_remote_refused_without_optin(self):
        msg = serve.remote_bind_error("0.0.0.0", allow_remote=False)
        self.assertIsNotNone(msg)
        self.assertIn("NO authentication", msg)

    def test_remote_allowed_with_optin(self):
        self.assertIsNone(serve.remote_bind_error("0.0.0.0", allow_remote=True))


# --------------------------------------------------------------------------- #
# M2 — request input bounds
# --------------------------------------------------------------------------- #
class TestCreateBookBounds(unittest.TestCase):
    def setUp(self):
        try:
            from bookwriter.server.schemas import CreateBookRequest  # noqa: F401
        except Exception:
            self.skipTest("server extras (pydantic) not installed")

    def _make(self, **kw):
        from bookwriter.server.schemas import CreateBookRequest
        base = dict(premise="a premise")
        base.update(kw)
        return CreateBookRequest(**base)

    def test_valid_request_ok(self):
        req = self._make(chapters=10, words_per_chapter=2000)
        self.assertEqual(req.chapters, 10)

    def test_rejects_too_many_chapters(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._make(chapters=100000)

    def test_rejects_zero_chapters(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._make(chapters=0)

    def test_rejects_absurd_word_count(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._make(words_per_chapter=10_000_000)

    def test_rejects_empty_premise(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self._make(premise="")


# --------------------------------------------------------------------------- #
# S4 — image fetch refuses non-http(s) schemes (SSRF / file:// guard)
# --------------------------------------------------------------------------- #
class TestImageSchemeGuard(unittest.TestCase):
    def test_file_scheme_refused(self):
        with self.assertRaises(RuntimeError) as ctx:
            images._request("GET", "file:///etc/passwd", headers={})
        self.assertIn("non-http", str(ctx.exception).lower())

    def test_other_scheme_refused(self):
        with self.assertRaises(RuntimeError):
            images._request("GET", "ftp://example.com/x.png", headers={})


# --------------------------------------------------------------------------- #
# rob#1 — a downstream extract/check failure must NOT discard a written chapter
# --------------------------------------------------------------------------- #
class _FlakyExtractLLM:
    """Delegates to MockLLM but raises on the 'extract' JSON stage."""

    def __init__(self):
        self._inner = MockLLM()

    def complete_json(self, *, stage, **kw):
        if stage == "extract":
            raise ValueError("simulated malformed extractor output")
        return self._inner.complete_json(stage=stage, **kw)

    def complete_text(self, **kw):
        return self._inner.complete_text(**kw)


class TestPipelineExtractResilience(unittest.TestCase):
    def _settings(self, tmp):
        # continuity check off so the only failure point is extraction
        return Settings(project_dir=tmp, run_continuity_check=False).with_profile("balanced")

    def test_chapters_persist_despite_extract_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = BookPipeline(_FlakyExtractLLM(), self._settings(tmp))
            pipe.plan(premise="a resilient test", chapters=2, words_per_chapter=150)
            # Must not raise even though every extract call fails.
            pipe.write_all()

            store = BookStore(tmp)
            # Both chapters generated and persisted (prose saved before extract).
            self.assertTrue(store.has_chapter(1))
            self.assertTrue(store.has_chapter(2))
            # Recorded in the graph so a resumed run would skip them.
            self.assertEqual(len(pipe.graph.chapters), 2)
            # Prose was produced and billed.
            self.assertGreater(pipe.ledger.words_written, 0)
            # A per-chapter cost snapshot was flushed during the run.
            self.assertTrue(os.path.exists(os.path.join(tmp, "cost.json")))


if __name__ == "__main__":
    unittest.main()
