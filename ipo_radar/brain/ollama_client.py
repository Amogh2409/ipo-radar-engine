"""
Async Ollama client with enforced structured output.

Ollama's `format` parameter accepts a JSON Schema and constrains decoding to
it, which is the difference between a usable local analyst and a model that
returns prose you then have to regex. Every call here goes through a schema.

Also handles the practical annoyances: reasoning models emitting <think>
blocks, models not being pulled yet, and repeated identical prompts (cached
in SQLite so a restart does not re-burn GPU time).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

import httpx

from ..config import LLMConfig
from ..store import Store

log = logging.getLogger("ipo_radar.llm")

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


class OllamaClient:
    def __init__(self, cfg: LLMConfig, store: Store | None = None) -> None:
        self.cfg = cfg
        self.store = store
        self._client = httpx.AsyncClient(timeout=cfg.timeout)
        self._available: set[str] = set()
        self._ready = False

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------- setup
    async def tags(self) -> list[str]:
        try:
            r = await self._client.get(f"{self.cfg.host}/api/tags", timeout=10.0)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception as exc:
            log.warning("ollama unreachable at %s: %s", self.cfg.host, exc)
            return []

    async def ensure_ready(self) -> bool:
        """
        Make sure a usable model exists. Order of preference:
          1. the configured custom model (ipo-analyst)
          2. build it from the Modelfile if the base is present
          3. pull the base model
          4. fall back to any installed model
        """
        if self._ready:
            return True
        names = await self.tags()
        if not names and not self.cfg.auto_pull:
            return False
        self._available = set(names)

        def have(n: str) -> bool:
            return any(x == n or x.startswith(n + ":") for x in self._available)

        if have(self.cfg.model):
            self._ready = True
            return True

        if not have(self.cfg.base_model):
            if not self.cfg.auto_pull:
                log.warning("base model %s missing and auto_pull is off",
                            self.cfg.base_model)
            else:
                log.info("pulling %s (this runs once)", self.cfg.base_model)
                if not await self.pull(self.cfg.base_model):
                    # last resort: use whatever is installed
                    if self._available:
                        fallback = sorted(self._available)[0]
                        log.warning("falling back to installed model %s", fallback)
                        self.cfg.model = self.cfg.fast_model = fallback
                        self._ready = True
                        return True
                    return False
                self._available.add(self.cfg.base_model)

        if self.cfg.auto_build:
            from .modelfile import build_ipo_analyst
            if await build_ipo_analyst(self, self.cfg):
                self._ready = True
                return True

        log.info("using base model %s directly", self.cfg.base_model)
        self.cfg.model = self.cfg.base_model
        self._ready = True
        return True

    async def pull(self, model: str) -> bool:
        try:
            async with self._client.stream(
                    "POST", f"{self.cfg.host}/api/pull",
                    json={"model": model}, timeout=None) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("error"):
                        log.error("pull failed: %s", msg["error"])
                        return False
                    if msg.get("status") == "success":
                        log.info("pulled %s", model)
                        return True
            return True
        except Exception as exc:
            log.error("pull %s failed: %s", model, exc)
            return False

    async def create(self, name: str, modelfile: str) -> bool:
        """Build a derived model from a Modelfile body."""
        for payload in ({"model": name, "from": self.cfg.base_model,
                         "system": _system_of(modelfile),
                         "parameters": _params_of(modelfile)},
                        {"name": name, "modelfile": modelfile}):
            try:
                async with self._client.stream(
                        "POST", f"{self.cfg.host}/api/create",
                        json=payload, timeout=600.0) as r:
                    if r.status_code >= 400:
                        await r.aread()
                        continue
                    async for line in r.aiter_lines():
                        if line and '"error"' in line:
                            log.debug("create: %s", line[:200])
                log.info("built model %s", name)
                return True
            except Exception as exc:
                log.debug("create attempt failed: %s", exc)
        return False

    # -------------------------------------------------------------- call
    async def structured(self, prompt: str, schema: dict[str, Any],
                         system: str | None = None, model: str | None = None,
                         temperature: float | None = None,
                         cache_key: str | None = None,
                         cache_ttl: float | None = 3600.0) -> dict[str, Any] | None:
        """One constrained-JSON call. Returns the parsed object, or None."""
        model = model or self.cfg.model
        key = None
        if cache_key and self.store:
            raw = f"{model}|{cache_key}|{hashlib.sha256(prompt.encode()).hexdigest()}"
            key = hashlib.sha256(raw.encode()).hexdigest()
            hit = self.store.get_llm_cache(key, cache_ttl)
            if hit:
                try:
                    return json.loads(hit)
                except json.JSONDecodeError:
                    pass

        body: dict[str, Any] = {
            "model": model,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema,
            "think": False,          # ignored by non-reasoning models
            "options": {
                "temperature": (self.cfg.temperature if temperature is None
                                else temperature),
                "num_ctx": self.cfg.num_ctx,
            },
        }
        for attempt in range(2):
            try:
                r = await self._client.post(f"{self.cfg.host}/api/chat", json=body)
                if r.status_code == 400 and "think" in body:
                    body.pop("think")       # older Ollama builds reject it
                    continue
                r.raise_for_status()
                content = (r.json().get("message") or {}).get("content", "")
            except Exception as exc:
                log.warning("llm call failed (attempt %d): %s", attempt + 1, exc)
                continue
            obj = _parse_json(content)
            if obj is not None:
                if key and self.store:
                    self.store.cache_llm(key, model, json.dumps(obj))
                return obj
            log.debug("unparseable llm output: %s", content[:300])
        return None


def _parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = THINK_RE.sub("", text).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"value": obj}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)      # salvage an embedded object
    if m:
        try:
            obj = json.loads(m.group())
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _system_of(modelfile: str) -> str:
    m = re.search(r'SYSTEM\s+"""(.*?)"""', modelfile, re.S)
    return m.group(1).strip() if m else ""


def _params_of(modelfile: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in re.findall(r"PARAMETER\s+(\S+)\s+(\S+)", modelfile):
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out
