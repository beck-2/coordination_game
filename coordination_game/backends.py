"""Model backends. Each returns a Completion with text, any native reasoning trace, and usage."""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field


@dataclass
class Completion:
    text: str
    native_reasoning: str | None = None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    raw: dict | None = None


class Backend:
    name = "base"
    model = ""

    async def complete(self, system: str, user: str, state: dict | None = None) -> Completion:  # pragma: no cover
        raise NotImplementedError


class MockBackend(Backend):
    """Picks from the board with a pull toward the previous round's plurality. Costs nothing."""
    name = "mock"

    def __init__(self, model: str = "mock", conformity: float = 0.6, seed: int = 0):
        self.model, self.conformity, self.r = model, conformity, random.Random(seed)

    async def complete(self, system: str, user: str, state: dict | None = None) -> Completion:
        kind = state.get("kind", "pick")
        # Exercise the chat-archive lookup sometimes (never on the re-ask, which contains the requested logs).
        # Only reachable in chat sessions, so default runs keep the same RNG stream.
        if state.get("lookup_available") and "Chat logs you requested" not in user and self.r.random() < 0.3:
            return Completion(text=json.dumps({"read_chat": [1]}))
        if kind in ("chat", "debrief"):
            out = {"reasoning": "mock", "message": "" if self.r.random() < 0.2 else f"mock {kind} message",
                   "notes": f"mock note ({kind})"}
            if state.get("final"):
                out["roles"] = {f"A{a}": self.r.choice(state["roles_menu"]) for a in state["agents"]}
            return Completion(text=json.dumps(out))
        board, plur = state["board"], state.get("plurality")
        if state.get("target") is not None:  # confederate: hold the target
            pick = state["target"]
        else:
            pick = plur if (plur and self.r.random() < self.conformity) else self.r.choice(board)
        out = {"reasoning": "mock", "pick": pick}
        if "private notepad" in system:  # only when the notepad is on (keeps the RNG stream unchanged otherwise)
            out["notes"] = f"mock note: last pick {pick}"
        return Completion(text=json.dumps(out))


class AdaptionBackend(Backend):
    name = "adaption"

    def __init__(self, model: str, temperature: float | None = 1.2, max_output_tokens: int = 16000):
        from adaption import AsyncAdaption
        self.client = AsyncAdaption(max_retries=4, timeout=900)
        self.model, self.temperature, self.max_output_tokens = model, temperature, max_output_tokens

    async def complete(self, system: str, user: str, state: dict | None = None) -> Completion:
        t = time.time()
        kw = dict(model=self.model, instructions=system, input=user, tools=[],
                  max_output_tokens=self.max_output_tokens)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        r = await self.client.responses.create(**kw)
        reasoning = []
        for item in r.output:
            if item.type == "reasoning":
                for s in (item.summary or []):
                    reasoning.append(str(s.get("text", s)))
                for c in (item.content or []):
                    reasoning.append(getattr(c, "text", None) or str(c))
        return Completion(text=r.output_text or "",
                          native_reasoning="\n".join(reasoning) or None,
                          usage=r.usage.model_dump() if r.usage else {},
                          latency_s=time.time() - t, raw={"id": r.id, "status": r.status})


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str = "gpt-oss:20b", think: str | bool = "low", num_ctx: int = 32768):
        import httpx
        self.client = httpx.AsyncClient(base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"), timeout=600)
        self.model, self.think, self.num_ctx = model, think, num_ctx

    async def complete(self, system: str, user: str, state: dict | None = None) -> Completion:
        t = time.time()
        r = await self.client.post("/api/chat", json={
            "model": self.model, "stream": False, "think": self.think, "format": "json",
            "options": {"num_ctx": self.num_ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        return Completion(text=d["message"].get("content", ""),
                          native_reasoning=d["message"].get("thinking"),
                          usage={"input_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count")},
                          latency_s=time.time() - t)


def make_backend(kind: str, model: str, seed: int = 0, **kw) -> Backend:
    if kind == "mock":
        return MockBackend(seed=seed, **kw)
    if kind == "adaption":
        return AdaptionBackend(model, **kw)
    if kind == "ollama":
        return OllamaBackend(model, **kw)
    raise ValueError(kind)
