"""Tests for the pluggable LLM provider layer (provider.py + the two adapters).

All offline: the OpenAI adapter gets a fake client injected, and the claude-cli
adapter gets a fake subprocess runner. No SDKs, no network, no API keys.
"""
import json
import os
import unittest

from bookwriter import provider
from bookwriter.config import StageModel
from bookwriter.costs import CostLedger
from bookwriter.llm_claude_cli import ClaudeCliLLM, _build_prompt
from bookwriter.llm_openai import OpenAICompatLLM, _strip_fences


class _EnvGuard(unittest.TestCase):
    """Save/restore the env vars these tests poke at."""

    _VARS = [
        "BOOKWRITER_LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "OPENROUTER_API_KEY", "BOOKWRITER_MODEL_STRONG", "BOOKWRITER_MODEL_MID",
        "BOOKWRITER_MODEL_CHEAP", "BOOKWRITER_CLAUDE_BIN",
        "BOOKWRITER_CODEX_CMD", "BOOKWRITER_GROK_CMD", "BOOKWRITER_CLI_CMD",
    ]

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._VARS}
        for k in self._VARS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestProviderSelection(_EnvGuard):
    def test_default_is_anthropic(self):
        self.assertEqual(provider.provider_name(), "anthropic")

    def test_aliases(self):
        for raw, want in [("OpenAI", "openai"), ("router", "openrouter"),
                          ("subscription", "claude-cli"), ("cli", "cli"),
                          ("claude", "anthropic"), ("chatgpt", "codex"),
                          ("Codex", "codex"), ("grok", "grok-cli")]:
            os.environ["BOOKWRITER_LLM_PROVIDER"] = raw
            self.assertEqual(provider.provider_name(), want)

    def test_cli_command_resolution(self):
        # codex reads the prompt from stdin via the "-" sentinel; grok takes the
        # prompt as an argument after --prompt (bare `grok` opens a TUI).
        self.assertEqual(provider.cli_command("codex"), ["codex", "exec", "-"])
        self.assertEqual(provider.cli_command("grok-cli"), ["grok", "--prompt"])
        self.assertEqual(provider.cli_command("cli"), [])
        os.environ["BOOKWRITER_CODEX_CMD"] = "codex exec --full-auto"
        self.assertEqual(provider.cli_command("codex"), ["codex", "exec", "--full-auto"])

    def test_subscription_cli_gate_uses_binary_on_path(self):
        # 'cli' with no command configured is never live.
        os.environ["BOOKWRITER_LLM_PROVIDER"] = "cli"
        self.assertFalse(provider.live_available())
        # Point it at a command whose binary certainly exists on PATH.
        exe = "cmd" if os.name == "nt" else "sh"
        os.environ["BOOKWRITER_CLI_CMD"] = exe
        self.assertTrue(provider.live_available())

    def test_tier_mapping(self):
        self.assertEqual(provider.tier_of("claude-opus-4-8"), "strong")
        self.assertEqual(provider.tier_of("claude-sonnet-4-6"), "mid")
        self.assertEqual(provider.tier_of("claude-haiku-4-5"), "cheap")

    def test_target_model_defaults_and_override(self):
        self.assertEqual(provider.target_model("openai", "claude-opus-4-8"), "gpt-4.1")
        self.assertEqual(provider.target_model("openrouter", "claude-haiku-4-5"),
                         "openai/gpt-4.1-mini")
        self.assertEqual(provider.target_model("claude-cli", "claude-sonnet-4-6"), "sonnet")
        os.environ["BOOKWRITER_MODEL_STRONG"] = "my/model"
        self.assertEqual(provider.target_model("openai", "claude-opus-4-8"), "my/model")

    def test_live_available_gates_per_provider(self):
        os.environ["BOOKWRITER_LLM_PROVIDER"] = "openai"
        self.assertFalse(provider.live_available())
        os.environ["OPENAI_API_KEY"] = "sk-test"
        self.assertTrue(provider.live_available())

        os.environ["BOOKWRITER_LLM_PROVIDER"] = "anthropic"
        self.assertFalse(provider.live_available())
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        self.assertTrue(provider.live_available())

    def test_claude_cli_gate_uses_binary(self):
        os.environ["BOOKWRITER_LLM_PROVIDER"] = "claude-cli"
        os.environ["BOOKWRITER_CLAUDE_BIN"] = "C:\\fake\\claude.exe"
        self.assertTrue(provider.live_available())

    def test_missing_credentials_message_mentions_provider(self):
        os.environ["BOOKWRITER_LLM_PROVIDER"] = "openrouter"
        msg = provider.missing_credentials_message()
        self.assertIn("openrouter", msg)
        self.assertIn("demo mode (mock)", msg)

    def test_make_llm_mock(self):
        from bookwriter.mock import MockLLM
        self.assertIsInstance(provider.make_llm(mock=True), MockLLM)


# --------------------------------------------------------------------------- #
# OpenAI-compatible adapter, with a fake client
# --------------------------------------------------------------------------- #

class _FakeUsage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content=None, delta_content=None):
        self.message = _FakeMessage(content) if content is not None else None
        self.delta = type("D", (), {"content": delta_content})()


class _FakeResp:
    def __init__(self, content, usage):
        self.choices = [_FakeChoice(content=content)]
        self.usage = usage


class _FakeChunk:
    def __init__(self, delta_content=None, usage=None):
        self.choices = [_FakeChoice(delta_content=delta_content)] if delta_content is not None else []
        self.usage = usage


class _FakeCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, **kwargs):
        self.parent.calls.append(kwargs)
        if kwargs.get("stream"):
            return self.parent.stream_chunks
        return self.parent.json_resp


class _FakeOpenAIClient:
    def __init__(self, json_resp=None, stream_chunks=None):
        self.calls = []
        self.json_resp = json_resp
        self.stream_chunks = stream_chunks or []
        self.chat = type("C", (), {"completions": _FakeCompletions(self)})()


class TestOpenAIAdapter(unittest.TestCase):
    def _model(self):
        return StageModel("claude-opus-4-8", effort="medium")

    def test_complete_json_parses_and_records(self):
        resp = _FakeResp('{"a": 1, "b": "x"}', _FakeUsage(100, 20))
        client = _FakeOpenAIClient(json_resp=resp)
        llm = OpenAICompatLLM(provider="openai", client=client)
        ledger = CostLedger()
        out = llm.complete_json(
            stage="plan", model=self._model(), system="sys", user="hi",
            schema={"type": "object", "properties": {"a": {}}}, max_tokens=500, ledger=ledger,
        )
        self.assertEqual(out, {"a": 1, "b": "x"})
        # model id was translated to the openai default
        self.assertEqual(client.calls[0]["model"], "gpt-4.1")
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(len(ledger.entries), 1)
        self.assertEqual(ledger.entries[0].input_tokens, 100)
        self.assertEqual(ledger.entries[0].output_tokens, 20)

    def test_complete_text_streams_and_forwards_deltas(self):
        chunks = [
            _FakeChunk(delta_content="Hello "),
            _FakeChunk(delta_content="world"),
            _FakeChunk(usage=_FakeUsage(50, 8)),
        ]
        client = _FakeOpenAIClient(stream_chunks=chunks)
        llm = OpenAICompatLLM(provider="openrouter", client=client)
        ledger = CostLedger()
        seen = []
        text = llm.complete_text(
            stage="write", model=self._model(), system="sys", user="go",
            max_tokens=500, ledger=ledger, cached="BIBLE", on_delta=seen.append,
        )
        self.assertEqual(text, "Hello world")
        self.assertEqual(seen, ["Hello ", "world"])
        self.assertTrue(client.calls[0]["stream"])
        # cached bible folded into the system message
        self.assertIn("BIBLE", client.calls[0]["messages"][0]["content"])
        self.assertEqual(ledger.entries[0].output_tokens, 8)

    def test_strip_fences(self):
        self.assertEqual(_strip_fences("```json\n{\"x\":1}\n```"), '{"x":1}')
        self.assertEqual(_strip_fences('{"x":1}'), '{"x":1}')


# --------------------------------------------------------------------------- #
# claude-cli adapter, with a fake subprocess runner
# --------------------------------------------------------------------------- #

class TestClaudeCliAdapter(unittest.TestCase):
    def _model(self):
        return StageModel("claude-sonnet-4-6", effort="medium")

    def _runner(self, payload, code=0, err=""):
        captured = {}

        def run(args, stdin):
            captured["args"] = args
            captured["stdin"] = stdin
            return code, json.dumps(payload), err

        run.captured = captured
        return run

    def test_complete_text_returns_result_and_records_usage(self):
        runner = self._runner({
            "result": "Once upon a time.",
            "usage": {"input_tokens": 120, "output_tokens": 40,
                      "cache_read_input_tokens": 1000},
        })
        llm = ClaudeCliLLM(binary="claude", runner=runner)
        ledger = CostLedger()
        seen = []
        text = llm.complete_text(
            stage="write", model=self._model(), system="sys", user="write ch1",
            max_tokens=8000, ledger=ledger, cached="BIBLE", on_delta=seen.append,
        )
        self.assertEqual(text, "Once upon a time.")
        self.assertEqual(seen, ["Once upon a time."])
        # model translated to a CLI alias
        self.assertIn("--model", runner.captured["args"])
        self.assertIn("sonnet", runner.captured["args"])
        # prompt fed on stdin, includes bible + task
        self.assertIn("BIBLE", runner.captured["stdin"])
        self.assertIn("write ch1", runner.captured["stdin"])
        # usage recorded against the real anthropic id so MODEL_PRICES matches
        e = ledger.entries[0]
        self.assertEqual(e.model, "claude-sonnet-4-6")
        self.assertEqual(e.input_tokens, 120)
        self.assertEqual(e.cache_read_tokens, 1000)
        self.assertGreater(ledger.total_cost(), 0.0)

    def test_complete_json_parses(self):
        runner = self._runner({"result": '{"issues": []}', "usage": {}})
        llm = ClaudeCliLLM(binary="claude", runner=runner)
        out = llm.complete_json(
            stage="check", model=self._model(), system="s", user="u",
            schema={"type": "object", "properties": {"issues": {}}},
            max_tokens=1000, ledger=CostLedger(),
        )
        self.assertEqual(out, {"issues": []})

    def test_nonzero_exit_raises_helpful_error(self):
        runner = self._runner({}, code=1, err="not logged in")
        llm = ClaudeCliLLM(binary="claude", runner=runner)
        with self.assertRaises(RuntimeError) as ctx:
            llm.complete_text(stage="write", model=self._model(), system="s",
                              user="u", max_tokens=10, ledger=CostLedger())
        self.assertIn("claude login", str(ctx.exception))

    def test_is_error_payload_raises(self):
        runner = self._runner({"is_error": True, "result": "boom"})
        llm = ClaudeCliLLM(binary="claude", runner=runner)
        with self.assertRaises(RuntimeError):
            llm.complete_json(stage="x", model=self._model(), system="s", user="u",
                              schema={"type": "object"}, max_tokens=10, ledger=CostLedger())

    def test_build_prompt_structure(self):
        p = _build_prompt("SYS", "CACHE", "TASK", "EXTRA")
        self.assertIn("SYSTEM INSTRUCTIONS:", p)
        self.assertIn("CACHE", p)
        self.assertIn("EXTRA", p)
        self.assertTrue(p.rstrip().endswith("TASK"))


# --------------------------------------------------------------------------- #
# Generic subscription-CLI adapter (codex / grok-cli / cli)
# --------------------------------------------------------------------------- #

class TestGenericCliAdapter(unittest.TestCase):
    def _model(self):
        return StageModel("claude-opus-4-8", effort="medium")

    def _runner(self, stdout, code=0, err=""):
        captured = {}

        def run(args, stdin):
            captured["args"] = args
            captured["stdin"] = stdin
            return code, stdout, err

        run.captured = captured
        return run

    def test_complete_text_pipes_prompt_on_stdin(self):
        from bookwriter.llm_cli import GenericCliLLM
        runner = self._runner("Chapter one. The end.\n")
        llm = GenericCliLLM(command=["codex", "exec"], provider="codex", runner=runner)
        ledger = CostLedger()
        seen = []
        text = llm.complete_text(
            stage="write", model=self._model(), system="sys", user="write ch1",
            max_tokens=8000, ledger=ledger, cached="BIBLE", on_delta=seen.append,
        )
        self.assertEqual(text, "Chapter one. The end.")
        self.assertEqual(seen, ["Chapter one. The end."])
        self.assertEqual(runner.captured["args"], ["codex", "exec"])
        self.assertIn("BIBLE", runner.captured["stdin"])
        self.assertIn("write ch1", runner.captured["stdin"])
        # estimated usage recorded; subscription model id -> $0 cost
        self.assertEqual(len(ledger.entries), 1)
        self.assertGreater(ledger.entries[0].output_tokens, 0)
        self.assertEqual(ledger.total_cost(), 0.0)

    def test_complete_json_extracts_object_from_noisy_stdout(self):
        from bookwriter.llm_cli import GenericCliLLM
        noisy = 'thinking...\nHere you go:\n{"issues": [], "ok": true}\nDone.'
        runner = self._runner(noisy)
        llm = GenericCliLLM(command=["grok"], provider="grok-cli", runner=runner)
        out = llm.complete_json(
            stage="check", model=self._model(), system="s", user="u",
            schema={"type": "object", "properties": {"issues": {}}},
            max_tokens=1000, ledger=CostLedger(),
        )
        self.assertEqual(out, {"issues": [], "ok": True})

    def test_model_flag_appended_when_configured(self):
        from bookwriter.llm_cli import GenericCliLLM
        os.environ["BOOKWRITER_MODEL_STRONG"] = "gpt-5-codex"
        try:
            runner = self._runner("ok")
            llm = GenericCliLLM(command=["codex", "exec"], provider="codex",
                                model_flag="-m", runner=runner)
            llm.complete_text(stage="write", model=self._model(), system="s",
                              user="u", max_tokens=10, ledger=CostLedger())
            self.assertEqual(runner.captured["args"], ["codex", "exec", "-m", "gpt-5-codex"])
        finally:
            os.environ.pop("BOOKWRITER_MODEL_STRONG", None)

    def test_nonzero_exit_raises(self):
        from bookwriter.llm_cli import GenericCliLLM
        runner = self._runner("", code=1, err="not signed in")
        llm = GenericCliLLM(command=["codex", "exec"], provider="codex", runner=runner)
        with self.assertRaises(RuntimeError) as ctx:
            llm.complete_text(stage="write", model=self._model(), system="s",
                              user="u", max_tokens=10, ledger=CostLedger())
        self.assertIn("codex exec", str(ctx.exception))

    def test_empty_command_rejected(self):
        from bookwriter.llm_cli import GenericCliLLM
        with self.assertRaises(RuntimeError):
            GenericCliLLM(command=[], provider="cli")

    def test_grok_passes_prompt_as_arg_not_stdin(self):
        # Bare `grok` opens an interactive TUI and ignores stdin, so the prompt
        # must be appended as an argument after --prompt (stdin stays empty).
        from bookwriter.llm_cli import GenericCliLLM
        runner = self._runner("A grok answer.")
        llm = GenericCliLLM(command=["grok", "--prompt"], provider="grok-cli",
                            stdin=False, runner=runner)
        text = llm.complete_text(stage="write", model=self._model(), system="sys",
                                 user="write ch1", max_tokens=10, ledger=CostLedger())
        self.assertEqual(text, "A grok answer.")
        self.assertEqual(runner.captured["stdin"], "")
        self.assertEqual(runner.captured["args"][:2], ["grok", "--prompt"])
        # the full prompt is the trailing positional argument
        self.assertIn("write ch1", runner.captured["args"][-1])
        self.assertNotIn("write ch1", runner.captured["stdin"])


class TestMakeLlmCliRouting(_EnvGuard):
    """make_llm wires each subscription CLI with the right argv + stdin mode."""

    def test_grok_routed_with_stdin_false(self):
        os.environ["BOOKWRITER_LLM_PROVIDER"] = "grok"
        llm = provider.make_llm()
        self.assertFalse(llm.stdin)
        self.assertEqual(llm.command, ["grok", "--prompt"])

    def test_codex_routed_with_stdin_true_and_sentinel(self):
        os.environ["BOOKWRITER_LLM_PROVIDER"] = "codex"
        llm = provider.make_llm()
        self.assertTrue(llm.stdin)
        self.assertEqual(llm.command, ["codex", "exec", "-"])


class TestAnthropicThinkingParam(unittest.TestCase):
    """_thinking_param: adaptive when on; disabled off — except Fable 5, which
    400s on an explicit disabled and must omit the param (returns None)."""

    def test_adaptive_when_thinking_on(self):
        from bookwriter.llm import _thinking_param
        sm = StageModel("claude-opus-4-8", thinking=True)
        self.assertEqual(_thinking_param(sm), {"type": "adaptive"})

    def test_disabled_for_standard_models(self):
        from bookwriter.llm import _thinking_param
        sm = StageModel("claude-haiku-4-5", thinking=False)
        self.assertEqual(_thinking_param(sm), {"type": "disabled"})

    def test_omitted_for_fable5_when_disabled(self):
        from bookwriter.llm import _thinking_param
        sm = StageModel("claude-fable-5", thinking=False)
        self.assertIsNone(_thinking_param(sm))


if __name__ == "__main__":
    unittest.main()
