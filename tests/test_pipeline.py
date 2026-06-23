import os
import tempfile
import unittest

from bookwriter.config import Settings
from bookwriter.mock import MockLLM
from bookwriter.pipeline import BookPipeline


class TestPipeline(unittest.TestCase):
    def _settings(self, tmp, **kw):
        return Settings(project_dir=tmp, **kw).with_profile("balanced")

    def test_end_to_end_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = BookPipeline(MockLLM(), self._settings(tmp))
            pipe.plan(premise="a test premise", chapters=3, words_per_chapter=200)
            self.assertEqual(len(pipe.graph.bible.outline), 3)
            pipe.write_all()

            # all chapters written and persisted
            self.assertEqual(len(pipe.graph.chapters), 3)
            self.assertTrue(os.path.exists(os.path.join(tmp, "manuscript.md")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "chapters", "01.md")))

            # cost was tracked and prose was produced
            self.assertGreater(pipe.ledger.total_cost(), 0)
            self.assertGreater(pipe.ledger.words_written, 0)

            # the extractor advanced the graph (hero gained knowledge)
            hero = pipe.graph.bible.character("hero")
            self.assertTrue(hero.knowledge)

            # rolling synopsis has one line per chapter
            self.assertEqual(len([s for s in pipe.graph.synopsis if s]), 3)

    def test_words_per_chapter_is_authoritative(self):
        # The Short/Medium/Long picker must control word_target on every chapter,
        # overriding whatever the planner model proposes.
        with tempfile.TemporaryDirectory() as tmp:
            short = BookPipeline(MockLLM(), self._settings(tmp)).plan(
                premise="x", chapters=3, words_per_chapter=900)
            self.assertTrue(all(p.word_target == 900 for p in short.outline))
        with tempfile.TemporaryDirectory() as tmp:
            long = BookPipeline(MockLLM(), self._settings(tmp)).plan(
                premise="x", chapters=3, words_per_chapter=3500)
            self.assertTrue(all(p.word_target == 3500 for p in long.outline))

    def test_prompt_cache_engages_after_first_chapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = BookPipeline(MockLLM(), self._settings(tmp))
            pipe.plan(premise="t", chapters=3, words_per_chapter=150)
            pipe.write_all()
            t = pipe.ledger.totals()
            # bible prefix is identical across chapters -> cache reads accrue
            self.assertGreater(t["cache_read"], 0)
            self.assertGreater(t["cache_write"], 0)
            self.assertGreater(pipe.ledger.cache_savings(), 0)

    def test_no_cache_uses_slice_and_no_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self._settings(tmp)
            s.use_cache = False
            pipe = BookPipeline(MockLLM(), s)
            pipe.plan(premise="t", chapters=2, words_per_chapter=120)
            pipe.write_all()
            self.assertEqual(pipe.ledger.totals()["cache_read"], 0)
            self.assertEqual(pipe.ledger.totals()["cache_write"], 0)

    def test_resume_skips_written_chapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = BookPipeline(MockLLM(), self._settings(tmp))
            pipe.plan(premise="t", chapters=3, words_per_chapter=120)
            pipe.write_all()

            # fresh pipeline, reload from disk, write again -> nothing re-written
            pipe2 = BookPipeline(MockLLM(), self._settings(tmp))
            self.assertTrue(pipe2.load())
            pipe2.write_all(resume=True)
            write_entries = [u for u in pipe2.ledger.entries if u.stage == "write"]
            self.assertEqual(write_entries, [])
            # manuscript still assembled from loaded chapters
            self.assertTrue(os.path.exists(os.path.join(tmp, "manuscript.md")))

    def test_only_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = BookPipeline(MockLLM(), self._settings(tmp))
            pipe.plan(premise="t", chapters=4, words_per_chapter=120)
            pipe.write_all(only=[2])
            writes = [u for u in pipe.ledger.entries if u.stage == "write"]
            self.assertEqual(len(writes), 1)
            self.assertIn(2, pipe.graph.chapters)


if __name__ == "__main__":
    unittest.main()
