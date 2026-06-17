"""Generic subprocess LLM backend for *subscription* coding CLIs.

Same idea as ``llm_claude_cli.py`` but vendor-agnostic: it shells out to any
configured command, feeds the whole prompt on **stdin**, and reads the model's
reply from **stdout**. That's how you ride a monthly subscription instead of an
API key — the CLI carries the auth:

    codex     OpenAI Codex CLI   (`codex exec`)   — ChatGPT Plus/Pro login
    grok-cli  a Grok CLI                          — X Premium / SuperGrok login
    cli       anything you set in BOOKWRITER_CLI_CMD

Because these CLIs rarely report token usage in a stable, parseable way, usage
is *estimated* from text length (so the cost meter isn't blank) and recorded
against the provider/model id — which is normally absent from MODEL_PRICES, so
the reported cost is $0. That's correct: a subscription run is not billed per
token. (Contrast ``ClaudeCliLLM``, which parses Claude's real usage JSON.)

Structured (JSON) stages append a "return only JSON" instruction and the schema
to the prompt, then extract the first JSON object from stdout.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Callable, List, Optional

from .costs import CostLedger, Usage
from .llm_claude_cli import _build_prompt, _strip_fences
from .provider import resolve_model, target_model


class GenericCliLLM:
    """Implements the pipeline ``LLM`` protocol via an arbitrary subprocess.

    ``runner`` may be injected for testing: a callable
    ``(args: list[str], stdin: str) -> (returncode, stdout, stderr)``.
    """

    def __init__(
        self,
        *,
        command: List[str],
        provider: str = "cli",
        model_flag: Optional[str] = None,
        model_override: Optional[str] = None,
        stdin: bool = True,
        runner: Optional[Callable] = None,
    ):
        if not command:
            raise RuntimeError(
                f"No command configured for the '{provider}' provider. "
                f"Set the appropriate BOOKWRITER_*_CMD environment variable."
            )
        self.command = command
        self.provider = provider
        self.model_flag = model_flag
        self.model_override = model_override
        self.stdin = stdin
        self._runner = runner or _subprocess_runner

    # ------------------------------------------------------------------
    def complete_json(self, *, stage, model, system, user, schema, max_tokens,
                      ledger, cached=None, use_cache=True, cache_ttl="1h"):
        instruction = (
            "Return ONLY a single JSON object — no prose, no code fences — "
            "conforming to this JSON Schema:\n" + json.dumps(schema)
        )
        prompt = _build_prompt(system, cached, user, instruction)
        out = self._run(stage, model, prompt)
        self._record(ledger, stage, model, prompt, out, cache_ttl)
        text = _extract_json(out)
        return json.loads(text) if text else {}

    # ------------------------------------------------------------------
    def complete_text(self, *, stage, model, system, user, max_tokens,
                      ledger, cached=None, use_cache=True, cache_ttl="1h", on_delta=None):
        prompt = _build_prompt(system, cached, user, "")
        out = self._run(stage, model, prompt).strip()
        self._record(ledger, stage, model, prompt, out, cache_ttl)
        if on_delta is not None and out:
            on_delta(out)
        return out

    # ------------------------------------------------------------------
    def _target(self, stage, model) -> str:
        return resolve_model(stage, self.model_override,
                             target_model(self.provider, model.model))

    def _run(self, stage, model, prompt: str) -> str:
        args = list(self.command)
        if self.model_flag:
            args += [self.model_flag, self._target(stage, model)]
        stdin = prompt
        if not self.stdin:
            args.append(prompt)
            stdin = ""
        code, out, err = self._runner(args, stdin)
        if code != 0:
            raise RuntimeError(
                f"`{' '.join(self.command)}` (provider '{self.provider}') exited "
                f"{code}. Is the CLI installed and signed in to your subscription? "
                f"stderr: {(err or '').strip()[:500]}"
            )
        return out or ""

    def _record(self, ledger: CostLedger, stage: str, model, prompt: str,
                out: str, cache_ttl: str) -> None:
        # No reliable token counts from these CLIs -> estimate (~4 chars/token).
        # Tag the model id with the provider so it is NOT priced like the Anthropic
        # tier it stands in for: a subscription run is $0 per token, and the cost
        # meter should say so.
        label = f"{self.provider}:{self._target(stage, model)}"
        ledger.add(Usage(
            model=label,
            stage=stage,
            input_tokens=len(prompt) // 4,
            output_tokens=len(out) // 4,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            cache_ttl=cache_ttl,
        ))


def _subprocess_runner(args, stdin):
    proc = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr


def _extract_json(text: str) -> str:
    """Pull a JSON object out of arbitrary CLI stdout (logs, banners, fences)."""
    t = _strip_fences(text)
    try:
        json.loads(t)
        return t
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return t
