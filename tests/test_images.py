"""Tests for the pluggable chapter-image layer (images.py) and the pipeline hook.

All offline: HTTP is monkeypatched, so no Pixio/OpenAI calls and no network.
"""
import os
import tempfile
import unittest
from unittest import mock

from bookwriter import images
from bookwriter.config import Settings, QUALITY_PROFILES
from bookwriter.mock import MockLLM
from bookwriter.pipeline import BookPipeline


class _EnvGuard(unittest.TestCase):
    _VARS = [
        "BOOKWRITER_IMAGE_PROVIDER", "PIXIO_API_KEY", "OPENAI_API_KEY",
        "BOOKWRITER_IMAGE_URL", "BOOKWRITER_PIXIO_MODEL", "PIXIO_IMAGE_MODEL",
        "BOOKWRITER_IMAGE_BODY", "BOOKWRITER_IMAGE_RESULT_PATH", "BOOKWRITER_IMAGE_AUTH",
    ]

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._VARS}
        for k in self._VARS:
            os.environ.pop(k, None)
        images.PixioImageProvider._discovered_model = None

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestImageSelection(_EnvGuard):
    def test_default_provider_is_pixio(self):
        self.assertEqual(images.image_provider_name(), "pixio")

    def test_aliases(self):
        for raw, want in [("OpenAI", "openai"), ("dall-e", "openai"),
                          ("custom", "http"), ("PIXIO", "pixio")]:
            os.environ["BOOKWRITER_IMAGE_PROVIDER"] = raw
            self.assertEqual(images.image_provider_name(), want)

    def test_availability_gating(self):
        self.assertFalse(images.image_available("pixio"))
        os.environ["PIXIO_API_KEY"] = "pxio_live_x"
        self.assertTrue(images.image_available("pixio"))
        self.assertFalse(images.image_available("http"))
        os.environ["BOOKWRITER_IMAGE_URL"] = "https://x/y"
        self.assertTrue(images.image_available("http"))

    def test_status_shape(self):
        os.environ["PIXIO_API_KEY"] = "pxio_live_x"
        s = images.image_status()
        self.assertEqual(s, {"provider": "pixio", "available": True})

    def test_make_provider_dispatch(self):
        self.assertIsInstance(images.make_image_provider("pixio"), images.PixioImageProvider)
        self.assertIsInstance(images.make_image_provider("openai"), images.OpenAIImageProvider)
        self.assertIsInstance(images.make_image_provider("http"), images.HttpImageProvider)


class TestPixioProvider(_EnvGuard):
    def test_generate_discovers_model_polls_and_downloads(self):
        os.environ["PIXIO_API_KEY"] = "pxio_live_x"
        prov = images.PixioImageProvider()
        models = {"models": [
            {"id": "pixio/nano-banana/edit", "type": "image-to-image"},
            {"id": "pixio/flux/dev", "type": "text-to-image"},
        ]}
        gen_started = {"contentId": "gen-123"}
        polls = [
            {"status": "processing"},
            {"status": "succeeded", "outputUrl": "https://cdn.pixio.ai/out/x.png"},
        ]
        with mock.patch.object(images, "_get_json") as gj, \
             mock.patch.object(images, "_post_json", return_value=gen_started) as pj, \
             mock.patch.object(images, "_download", return_value=b"PNGDATA") as dl, \
             mock.patch.object(images.time, "sleep"):
            # First _get_json is the models list, then generation polls.
            gj.side_effect = [models] + polls
            data, ext = prov.generate("a lighthouse at dusk")
        self.assertEqual(data, b"PNGDATA")
        self.assertEqual(ext, "png")
        # Picked the text-to-image model, not the edit model.
        self.assertEqual(pj.call_args.args[1]["modelId"], "pixio/flux/dev")
        self.assertEqual(pj.call_args.args[1]["params"]["prompt"], "a lighthouse at dusk")
        dl.assert_called_once()

    def test_explicit_model_skips_discovery(self):
        os.environ["PIXIO_API_KEY"] = "pxio_live_x"
        os.environ["BOOKWRITER_PIXIO_MODEL"] = "pixio/flux/schnell"
        prov = images.PixioImageProvider()
        with mock.patch.object(images, "_get_json", return_value={"status": "succeeded", "outputUrl": "https://x/a.jpg"}) as gj, \
             mock.patch.object(images, "_post_json", return_value={"contentId": "c"}), \
             mock.patch.object(images, "_download", return_value=b"JJ"), \
             mock.patch.object(images.time, "sleep"):
            data, ext = prov.generate("x")
        self.assertEqual((data, ext), (b"JJ", "jpg"))
        # Only the poll GET happened — no /models discovery call.
        self.assertEqual(gj.call_count, 1)

    def test_no_key_raises(self):
        with self.assertRaises(RuntimeError):
            images.PixioImageProvider().generate("x")


class TestOpenAIAndHttpProviders(_EnvGuard):
    def test_openai_b64(self):
        os.environ["OPENAI_API_KEY"] = "sk-x"
        prov = images.OpenAIImageProvider()
        import base64
        b64 = base64.b64encode(b"IMGBYTES").decode()
        with mock.patch.object(images, "_post_json", return_value={"data": [{"b64_json": b64}]}):
            data, ext = prov.generate("a castle")
        self.assertEqual((data, ext), (b"IMGBYTES", "png"))

    def test_http_generic_with_result_path(self):
        os.environ["BOOKWRITER_IMAGE_URL"] = "https://api.example.com/img"
        os.environ["BOOKWRITER_IMAGE_BODY"] = '{"prompt": "{prompt}", "n": 1}'
        os.environ["BOOKWRITER_IMAGE_RESULT_PATH"] = "data.0.url"
        prov = images.HttpImageProvider()
        captured = {}

        def fake_post(url, payload, headers, timeout=180.0):
            captured["payload"] = payload
            return {"data": [{"url": "https://cdn/x.webp"}]}

        with mock.patch.object(images, "_post_json", side_effect=fake_post), \
             mock.patch.object(images, "_download", return_value=b"WB"):
            data, ext = prov.generate('a "quoted" prompt')
        self.assertEqual((data, ext), (b"WB", "webp"))
        # Prompt was substituted and JSON stayed valid despite the quotes.
        self.assertEqual(captured["payload"]["prompt"], 'a "quoted" prompt')
        self.assertEqual(captured["payload"]["n"], 1)

    def test_dig_helper(self):
        self.assertEqual(images._dig({"a": [{"b": 5}]}, "a.0.b"), 5)
        self.assertIsNone(images._dig({"a": []}, "a.3.b"))


class TestChapterPrompt(unittest.TestCase):
    def test_prompt_has_no_text_directive(self):
        from bookwriter.models import Bible, ChapterPlan, Character, Location
        bible = Bible(title="T", genre="literary horror", tone="dread",
                      characters=[Character(id="hero", name="Wren", appearance="tall, weathered")],
                      locations=[Location(id="home", name="The Light", description="a dying coast")])
        plan = ChapterPlan(number=1, title="The First Sign", purpose="hook the reader",
                           pov_character="hero", location_ids=["home"])
        p = images.build_chapter_prompt(bible, plan)
        self.assertIn("The First Sign", p)
        self.assertIn("Wren", p)
        self.assertIn("The Light", p)
        self.assertIn("literary horror", p)
        self.assertIn("No text", p)


class _FakeImageProvider:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def generate(self, prompt, **kw):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("boom")
        return b"\x89PNG-fake", "png"


class TestPipelineImageHook(unittest.TestCase):
    def _plan_and_settings(self, tmp, chapter_images):
        s = Settings(project_dir=tmp)
        s.profile = QUALITY_PROFILES["draft"]
        s.run_continuity_check = False
        s.chapter_images = chapter_images
        return s

    def test_image_generated_and_event_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            prov = _FakeImageProvider()
            events = []
            pipe = BookPipeline(MockLLM(), self._plan_and_settings(tmp, True),
                                on_event=events.append, image_provider=prov)
            pipe.plan(premise="a lighthouse mystery", chapters=2)
            pipe.write_all()
            done = [e for e in events if e.get("type") == "chapter_done"]
            self.assertTrue(done and all(e["image"] for e in done))
            self.assertTrue(prov.calls)  # the provider was actually invoked
            # Images persisted to disk for each chapter.
            self.assertTrue(pipe.store.has_image(1))

    def test_disabled_skips_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            prov = _FakeImageProvider()
            pipe = BookPipeline(MockLLM(), self._plan_and_settings(tmp, False),
                                image_provider=prov)
            pipe.plan(premise="x", chapters=1)
            pipe.write_all()
            self.assertEqual(prov.calls, [])
            self.assertFalse(pipe.store.has_image(1))

    def test_image_failure_is_best_effort(self):
        with tempfile.TemporaryDirectory() as tmp:
            prov = _FakeImageProvider(fail=True)
            events = []
            pipe = BookPipeline(MockLLM(), self._plan_and_settings(tmp, True),
                                on_event=events.append, image_provider=prov)
            pipe.plan(premise="x", chapters=1)
            pipe.write_all()  # must not raise
            done = [e for e in events if e.get("type") == "chapter_done"]
            self.assertTrue(done and not any(e["image"] for e in done))
            self.assertFalse(pipe.store.has_image(1))


try:
    from bookwriter.server.service import BookService  # noqa: F401
    _HAS_SERVER = True
except Exception:
    _HAS_SERVER = False


@unittest.skipUnless(_HAS_SERVER, "server extras not installed")
class TestManuscriptImageMarkers(unittest.TestCase):
    def test_chapter_image_marker_injected(self):
        from bookwriter.server.service import BookService
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(project_dir=os.path.join(tmp, "bk"))
            s.profile = QUALITY_PROFILES["draft"]
            s.run_continuity_check = False
            s.chapter_images = True
            pipe = BookPipeline(MockLLM(), s, image_provider=_FakeImageProvider())
            pipe.plan(premise="x", chapters=1)
            pipe.write_all()
            md = "# Book\n\n## Chapter 1: The Start\n\nSome body text."
            out = BookService._with_chapter_images("bk", pipe.store, pipe.graph, md)
            self.assertIn("![Chapter 1 illustration](/api/books/bk/chapters/1/image)", out)
            # The original heading and prose are preserved.
            self.assertIn("## Chapter 1: The Start", out)
            self.assertIn("Some body text.", out)


class TestEpubImageEmbedding(unittest.TestCase):
    def test_epub_embeds_chapter_images(self):
        import io
        import zipfile
        from bookwriter.kdp import build_epub, KdpMetadata
        with tempfile.TemporaryDirectory() as tmp:
            prov = _FakeImageProvider()
            s = Settings(project_dir=tmp)
            s.profile = QUALITY_PROFILES["draft"]
            s.run_continuity_check = False
            s.chapter_images = True
            pipe = BookPipeline(MockLLM(), s, image_provider=prov)
            pipe.plan(premise="a lighthouse mystery", chapters=2)
            pipe.write_all()

            images = pipe.store.collect_images([p.number for p in pipe.graph.bible.outline])
            self.assertTrue(images)  # images were collected from disk
            meta = KdpMetadata(title="T", author_first="A", author_last="B")
            epub = build_epub(pipe.graph, meta, images=images)

            with zipfile.ZipFile(io.BytesIO(epub)) as zf:
                names = zf.namelist()
                # An image file is embedded and referenced by a chapter XHTML.
                imgs = [n for n in names if n.startswith("OEBPS/img-")]
                self.assertTrue(imgs, f"no embedded images in {names}")
                opf = zf.read("OEBPS/content.opf").decode("utf-8")
                self.assertIn('media-type="image/png"', opf)
                chap1 = zf.read("OEBPS/chap-01.xhtml").decode("utf-8")
                self.assertIn("<img", chap1)

    def test_epub_without_images_is_unchanged_shape(self):
        import io
        import zipfile
        from bookwriter.kdp import build_epub, KdpMetadata
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(project_dir=tmp)
            s.profile = QUALITY_PROFILES["draft"]
            s.run_continuity_check = False
            pipe = BookPipeline(MockLLM(), s)
            pipe.plan(premise="x", chapters=1)
            pipe.write_all()
            epub = build_epub(pipe.graph, KdpMetadata(title="T", author_first="A", author_last="B"))
            with zipfile.ZipFile(io.BytesIO(epub)) as zf:
                self.assertFalse([n for n in zf.namelist() if n.startswith("OEBPS/img-")])


if __name__ == "__main__":
    unittest.main()
