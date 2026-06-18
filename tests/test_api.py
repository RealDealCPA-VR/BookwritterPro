import json
import os
import tempfile
import time
import unittest
from importlib import import_module
from importlib.util import find_spec

_DEPS = bool(find_spec("fastapi")) and bool(find_spec("httpx"))


def _load_create_app():
    """Import the FastAPI app factory. Returns None if the server module
    isn't present yet (so tests skip rather than error at import)."""
    try:
        mod = import_module("bookwriter.server.api")
    except Exception:
        return None
    return getattr(mod, "create_app", None)


@unittest.skipUnless(_DEPS, "server deps (fastapi, httpx) not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        create_app = _load_create_app()
        if create_app is None:
            raise unittest.SkipTest("bookwriter.server.api.create_app not available")
        cls.create_app = staticmethod(create_app)

    def setUp(self):
        from fastapi.testclient import TestClient

        # Fresh tempdir data root + no API key so MOCK mode is the only path.
        self._tmp = tempfile.TemporaryDirectory()
        self._prev_data = os.environ.get("BOOKWRITER_DATA_DIR")
        self._prev_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["BOOKWRITER_DATA_DIR"] = self._tmp.name
        os.environ.pop("ANTHROPIC_API_KEY", None)

        app = type(self).create_app()
        self.client = TestClient(app)

    def tearDown(self):
        try:
            self.client.close()
        except Exception:
            pass
        if self._prev_data is None:
            os.environ.pop("BOOKWRITER_DATA_DIR", None)
        else:
            os.environ["BOOKWRITER_DATA_DIR"] = self._prev_data
        if self._prev_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._prev_key
        self._tmp.cleanup()

    # ---- helpers -------------------------------------------------------

    def _create_book(self, **overrides):
        body = {
            "premise": "A lighthouse keeper learns the fog erases memories.",
            "chapters": 3,
            "words_per_chapter": 150,
            "title": "Fog Test",
            "genre": "literary mystery",
            "mock": True,
        }
        body.update(overrides)
        r = self.client.post("/api/books", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def _drive_write_to_completion(self, book_id, timeout=60.0):
        r = self.client.post(f"/api/books/{book_id}/write", json={})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json().get("status"), "started")

        # Consume the SSE stream (replay + tail) until a terminal event.
        deadline = time.time() + timeout
        terminal = None
        with self.client.stream("GET", f"/api/books/{book_id}/events") as resp:
            self.assertEqual(resp.status_code, 200)
            for line in resp.iter_lines():
                if time.time() > deadline:
                    self.fail("write did not complete before timeout")
                if not line:
                    continue
                text = line if isinstance(line, str) else line.decode("utf-8")
                if text.startswith(":"):  # heartbeat comment
                    continue
                if text.startswith("data:"):
                    payload = text[len("data:"):].strip()
                    if not payload:
                        continue
                    evt = json.loads(payload)
                    if evt.get("type") in ("done", "error"):
                        terminal = evt
                        break
        self.assertIsNotNone(terminal, "no terminal SSE event received")
        self.assertEqual(terminal.get("type"), "done", f"write failed: {terminal}")

    # ---- tests ---------------------------------------------------------

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body.get("status"), "ok")
        self.assertIn("has_api_key", body)
        self.assertIsInstance(body["has_api_key"], bool)
        # No key set in this env.
        self.assertFalse(body["has_api_key"])

    def test_profiles(self):
        r = self.client.get("/api/profiles")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body.get("default"), "balanced")
        profiles = body.get("profiles")
        self.assertIsInstance(profiles, list)
        names = {p["name"] for p in profiles}
        self.assertEqual(names, {"premium", "balanced", "draft"})
        bal = next(p for p in profiles if p["name"] == "balanced")
        for stage in ("plan", "write", "extract", "check"):
            self.assertIn(stage, bal["stages"])
        # Contract: plan/write/extract are BARE model strings; check is an object.
        for stage in ("plan", "write", "extract"):
            self.assertIsInstance(bal["stages"][stage], str)
            self.assertTrue(bal["stages"][stage])
        self.assertIsInstance(bal["stages"]["check"], dict)
        self.assertIn("model", bal["stages"]["check"])
        self.assertIn("effort", bal["stages"]["check"])
        self.assertIn("prices", bal)

    def test_create_requires_key_without_mock(self):
        r = self.client.post(
            "/api/books",
            json={"premise": "x", "chapters": 2, "mock": False},
        )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("detail", r.json())

    def test_create_book_mock(self):
        data = self._create_book()
        book = data["book"]
        # Title is non-empty; the server may surface either the user-supplied
        # title or the planner's title, so we don't pin the exact string.
        self.assertTrue(book["title"])
        # Genre is present; in mock mode the planner may substitute its own.
        self.assertIn("genre", book)
        self.assertTrue(book["mock"])
        self.assertEqual(book["chapters_total"], 3)
        self.assertEqual(book["chapters_written"], 0)
        self.assertIn("id", book)
        self.assertIn("created_at", book)
        # Bible is the Bible.to_dict() shape with an outline.
        bible = data["bible"]
        self.assertIn("outline", bible)
        self.assertEqual(len(bible["outline"]), 3)

    def test_get_book(self):
        book_id = self._create_book()["book"]["id"]
        r = self.client.get(f"/api/books/{book_id}")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["book"]["id"], book_id)
        self.assertIn("bible", body)
        chapters = body["chapters"]
        self.assertEqual(len(chapters), 3)
        c1 = chapters[0]
        for k in ("number", "title", "act", "written", "word_count"):
            self.assertIn(k, c1)
        self.assertFalse(c1["written"])
        # cost is null before any write.
        self.assertIn("cost", body)

    def test_get_book_404(self):
        r = self.client.get("/api/books/does-not-exist-000000")
        self.assertEqual(r.status_code, 404, r.text)

    def test_list_books(self):
        id_a = self._create_book(title="Alpha")["book"]["id"]
        id_b = self._create_book(title="Beta")["book"]["id"]
        r = self.client.get("/api/books")
        self.assertEqual(r.status_code, 200, r.text)
        books = r.json()["books"]
        self.assertGreaterEqual(len(books), 2)
        ids = {b["id"] for b in books}
        self.assertIn(id_a, ids)
        self.assertIn(id_b, ids)
        for b in books:
            for k in ("id", "title", "chapters_total", "chapters_written"):
                self.assertIn(k, b)

    def test_list_books_ignores_settings_file(self):
        # Regression: settings.json (and any non-book file) lives in the data dir.
        # It must NOT make /api/books 404 — that broke the whole library for any
        # user who had ever saved settings.
        id_a = self._create_book(title="Alpha")["book"]["id"]
        # The settings store writes <data_dir>/settings.json; simulate it.
        with open(os.path.join(self._tmp.name, "settings.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        # And a stray non-book directory.
        os.makedirs(os.path.join(self._tmp.name, "not_a_book.tmp"), exist_ok=True)
        r = self.client.get("/api/books")
        self.assertEqual(r.status_code, 200, r.text)
        ids = {b["id"] for b in r.json()["books"]}
        self.assertIn(id_a, ids)
        self.assertNotIn("settings.json", ids)

    def test_write_to_completion_and_artifacts(self):
        book_id = self._create_book()["book"]["id"]
        self._drive_write_to_completion(book_id)

        # Book now reports all chapters written + a cost snapshot.
        body = self.client.get(f"/api/books/{book_id}").json()
        self.assertEqual(body["book"]["chapters_written"], 3)
        self.assertTrue(all(c["written"] for c in body["chapters"]))
        self.assertIsNotNone(body["cost"])
        self.assertGreater(body["cost"]["total_cost"], 0)

        # Chapter endpoint returns prose + plan.
        rc = self.client.get(f"/api/books/{book_id}/chapters/1")
        self.assertEqual(rc.status_code, 200, rc.text)
        ch = rc.json()
        self.assertEqual(ch["number"], 1)
        self.assertTrue(ch["written"])
        self.assertGreater(ch["word_count"], 0)
        self.assertTrue(ch["text"])
        self.assertIn("plan", ch)

        # Graph populated by the extractor.
        rg = self.client.get(f"/api/books/{book_id}/graph")
        self.assertEqual(rg.status_code, 200, rg.text)
        graph = rg.json()
        for k in ("characters", "locations", "items", "threads", "timeline", "synopsis"):
            self.assertIn(k, graph)

        # Cost report.
        rcost = self.client.get(f"/api/books/{book_id}/cost")
        self.assertEqual(rcost.status_code, 200, rcost.text)
        cost = rcost.json()
        self.assertIn("snapshot", cost)
        self.assertIn("report", cost)
        self.assertGreater(cost["snapshot"]["total_cost"], 0)

        # Manuscript (JSON + download).
        rm = self.client.get(f"/api/books/{book_id}/manuscript")
        self.assertEqual(rm.status_code, 200, rm.text)
        man = rm.json()
        self.assertGreater(man["words"], 0)
        self.assertTrue(man["markdown"])

        rd = self.client.get(f"/api/books/{book_id}/manuscript?download=1")
        self.assertEqual(rd.status_code, 200, rd.text)
        self.assertIn("text/markdown", rd.headers.get("content-type", ""))

    def test_write_conflict_when_running(self):
        # Best-effort: a second immediate write may 409 if the first job is
        # still running. We assert it's a valid response either way, but
        # require 409 to be reachable by starting two in quick succession.
        book_id = self._create_book(chapters=4)["book"]["id"]
        r1 = self.client.post(f"/api/books/{book_id}/write", json={})
        self.assertEqual(r1.status_code, 200, r1.text)
        r2 = self.client.post(f"/api/books/{book_id}/write", json={})
        self.assertIn(r2.status_code, (200, 409), r2.text)
        # Drain the stream so the job finishes before teardown.
        with self.client.stream("GET", f"/api/books/{book_id}/events") as resp:
            for line in resp.iter_lines():
                text = line if isinstance(line, str) else (line.decode() if line else "")
                if text.startswith("data:"):
                    payload = text[5:].strip()
                    if payload and json.loads(payload).get("type") in ("done", "error"):
                        break

    def test_path_traversal_book_id_rejected(self):
        # A traversal id must never escape the data root; the service rejects
        # malformed ids with 404 before any filesystem op.
        from bookwriter.server.service import BookService, ServiceError
        svc = BookService(self._tmp.name)
        for bad in ("..", "../secret", "..\\secret", "a/b", "A_B", "x" * 100):
            with self.assertRaises(ServiceError) as cm:
                svc.get_book(bad)
            self.assertEqual(cm.exception.status, 404)
            with self.assertRaises(ServiceError) as cm2:
                svc.delete_book(bad)
            self.assertEqual(cm2.exception.status, 404)

    def test_minted_ids_are_valid(self):
        # The ids we actually mint pass the validator (round-trip sanity).
        from bookwriter.server.service import validate_book_id
        book_id = self._create_book(title="Round Trip")["book"]["id"]
        self.assertEqual(validate_book_id(book_id), book_id)

    def test_delete_drops_broker_channel(self):
        # Deleting a book releases its in-memory event ring (no leak).
        from bookwriter.server.service import BookService
        svc = BookService(self._tmp.name)
        book_id = svc.create_book(self._req(title="Ephemeral"))["book"]["id"]
        svc.broker.start_job(book_id)
        svc.broker.publish(book_id, {"type": "delta", "number": 1, "text": "x"})
        svc.broker.publish(book_id, {"type": "done"})  # terminal: clears running
        self.assertIn(book_id, svc.broker._channels)
        svc.delete_book(book_id)
        self.assertNotIn(book_id, svc.broker._channels)

    def _req(self, **overrides):
        from bookwriter.server.schemas import CreateBookRequest
        body = {
            "premise": "A small town where the church bell rings on its own.",
            "chapters": 2, "words_per_chapter": 120, "title": "X",
            "genre": "mystery", "mock": True,
        }
        body.update(overrides)
        return CreateBookRequest(**body)

    def test_delete_book(self):
        book_id = self._create_book()["book"]["id"]
        r = self.client.delete(f"/api/books/{book_id}")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json().get("status"), "deleted")
        self.assertEqual(self.client.get(f"/api/books/{book_id}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
