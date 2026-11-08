"""Local results viewer: learning charts for all models + transcript browser by experiment, game, round and agent.

    python -m coordination_game.viewer            # then open http://127.0.0.1:8765
    python -m coordination_game.viewer --port 9000

Reads runs/ directly (nothing is copied). Session files are loaded on demand and cached; restart to pick up new runs,
or press "Reload data" in the page. Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path("runs").resolve()
HTML = Path(__file__).with_name("viewer.html")


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a session still being written
            pass
    return out


def _safe(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError("path outside runs/")
    return p


def _holders(cfg: dict) -> list[int]:
    c = cfg.get("config", {})
    return list(c.get("bad_apple_ids", [])) + list(c.get("confederate_ids", []))


# ---------------- session index ----------------

@lru_cache(maxsize=1)
def sessions() -> list[dict]:
    out = []
    for cfgp in sorted(ROOT.rglob("config.json")):
        d = cfgp.parent
        if d.name.startswith("post_discussion") or not (d / "calls.jsonl").exists():
            continue
        try:
            cfg = json.load(open(cfgp))
        except json.JSONDecodeError:
            continue
        rel = d.relative_to(ROOT).as_posix()
        parts = rel.split("/")
        models = sorted({a["model"] for a in cfg.get("agents", {}).values()
                         if a.get("role") != "scripted_bad_apple"})
        n_done = sum(1 for _ in open(d / "games.jsonl")) if (d / "games.jsonl").exists() else 0
        batch = len(parts) == 4  # plan/arm/model/sN
        out.append({
            "path": rel,
            "experiment": parts[0],
            "arm": cfg.get("arm") or (parts[1] if batch else ("badapple" if _holders(cfg) else "control")),
            "model": cfg.get("model") or (parts[2] if batch else (models[0] if len(models) == 1 else "mixed")),
            "seed": cfg.get("config", {}).get("seed"),
            "models": models,
            "holders": _holders(cfg),
            "n_games": cfg.get("config", {}).get("n_games"),
            "games_done": n_done,
            "temperature": cfg.get("temperature", cfg.get("argv", {}).get("temperature")),
            "discussions": sorted(x.name for x in d.iterdir()
                                  if x.is_dir() and x.name.startswith("post_discussion")
                                  and (x / "chat.jsonl").exists()) + (["in-session"] if (d / "chat.jsonl").exists() else []),
        })
    return out


# ---------------- learning curves ----------------

@lru_cache(maxsize=1)
def learning() -> dict:
    """Per experiment/arm/model: by game index, mean rounds, converged share, and (with a holder) the share of regular
    agents already on the holder's item in round 2. Single-model sessions only; mixed-model runs are listed as 'mixed'."""
    groups = defaultdict(list)
    for s in sessions():
        if s["games_done"] < (s["n_games"] or 10):
            continue
        groups[(s["experiment"], s["arm"], s["model"])].append(s)
    out = []
    for (exp, arm, model), ss in sorted(groups.items()):
        by_game = defaultdict(lambda: {"rounds": [], "conv": [], "r2": []})
        for s in ss:
            d = ROOT / s["path"]
            games = _jsonl(d / "games.jsonl")
            rounds = defaultdict(dict)
            for r in _jsonl(d / "rounds.jsonl"):
                rounds[r["game"]][r["round"]] = r["picks"]
            hs = {str(h) for h in s["holders"]}
            for g in games:
                G = by_game[g["game"]]
                G["rounds"].append(g["rounds"])
                G["conv"].append(1.0 if g["converged"] else 0.0)
                item = next(iter(g.get("bad_apple_picks", {}).values()), None)
                r2 = rounds[g["game"]].get(2)
                if item is not None and r2:
                    reg = [v for a, v in r2.items() if a not in hs]
                    G["r2"].append(sum(v == item for v in reg) / max(len(reg), 1))
                elif item is not None:  # converged in round 1 (never happens with a holdout) or no round 2
                    G["r2"].append(1.0)

        def agg(xs):
            if not xs:
                return None, None
            return st.mean(xs), (st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)
        games_idx = sorted(by_game)
        row = {"experiment": exp, "arm": arm, "model": model, "n_sessions": len(ss), "games": games_idx}
        for k, name in (("rounds", "rounds"), ("conv", "converged"), ("r2", "holder_r2")):
            m = [agg(by_game[g][k]) for g in games_idx]
            row[name] = [x[0] for x in m]
            row[name + "_se"] = [x[1] for x in m]
        out.append(row)
    return {"groups": out}


# ---------------- one session ----------------

@lru_cache(maxsize=16)
def session(rel: str) -> dict:
    d = _safe(rel)
    cfg = json.load(open(d / "config.json"))
    rounds = defaultdict(list)
    for r in _jsonl(d / "rounds.jsonl"):
        rounds[r["game"]].append({"round": r["round"], "picks": r["picks"], "plurality": r.get("plurality")})
    games = [{k: g.get(k) for k in ("game", "converged", "converged_on", "rounds", "bad_apple_picks", "board")}
             for g in _jsonl(d / "games.jsonl")]
    discussions = {}
    for sub in [x for x in sorted(d.iterdir()) if x.is_dir() and x.name.startswith("post_discussion")] + [d]:
        msgs = [m for m in _jsonl(sub / "chat.jsonl") if sub != d or m.get("kind") == "debrief"]
        if not msgs:
            continue
        db = _jsonl(sub / "debriefs.jsonl")
        discussions[sub.name if sub != d else "in-session"] = {
            "messages": [{k: m.get(k) for k in ("turn", "agent", "model", "message", "reasoning", "leader_vote", "silent")}
                         | {"native": next((a.get("native_reasoning") for a in reversed(m.get("attempts") or [])
                                            if a.get("native_reasoning")), None)} for m in msgs],
            "tally": db[-1].get("tally") if db else None,
            "display_order": db[-1].get("display_order") if db else None}
    chat = [{k: m.get(k) for k in ("game", "round", "turn", "agent", "message", "reasoning")}
            for m in _jsonl(d / "chat.jsonl") if m.get("kind") == "chat"]
    return {"path": rel, "config": cfg.get("config"), "agents": cfg.get("agents"), "holders": _holders(cfg),
            "games": games, "rounds": {str(k): v for k, v in rounds.items()}, "discussions": discussions, "chat": chat}


@lru_cache(maxsize=8)
def _calls(rel: str) -> list[dict]:
    d = _safe(rel)
    out = []
    for c in _jsonl(d / "calls.jsonl"):
        nat = next((a.get("native_reasoning") for a in reversed(c.get("attempts") or []) if a.get("native_reasoning")), None)
        out.append({"game": c["game"], "round": c["round"], "agent": c["agent"], "model": c.get("model"),
                    "pick": c.get("pick"), "reasoning": c.get("reasoning"), "native": nat,
                    "n_attempts": len(c.get("attempts") or []), "error": c.get("error"),
                    "user": c.get("user"), "system": c.get("system")})
    return out


def calls(rel: str, game: str | None, rnd: str | None, agent: str | None, prompts: bool) -> list[dict]:
    rows = _calls(rel)
    if game:
        rows = [c for c in rows if c["game"] == int(game)]
    if rnd:
        rows = [c for c in rows if c["round"] == int(rnd)]
    if agent:
        rows = [c for c in rows if c["agent"] == int(agent)]
    rows = sorted(rows, key=lambda c: (c["game"], c["round"], c["agent"]))
    if not prompts:
        rows = [{k: v for k, v in c.items() if k not in ("user", "system")} for c in rows]
    return rows[:2000]


# ---------------- server ----------------

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/sessions":
                data = sessions()
            elif u.path == "/api/learning":
                data = learning()
            elif u.path == "/api/session":
                data = session(q["path"])
            elif u.path == "/api/calls":
                data = calls(q["path"], q.get("game"), q.get("round"), q.get("agent"), q.get("prompts") == "1")
            elif u.path == "/api/reload":
                for f in (sessions, learning, session, _calls):
                    f.cache_clear()
                data = {"ok": True}
            else:
                return self._send(404, b"not found", "text/plain")
            self._send(200, json.dumps(data, ensure_ascii=False).encode(), "application/json")
        except (KeyError, ValueError, FileNotFoundError) as e:
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    print(f"Results viewer: http://127.0.0.1:{a.port}  (reading {ROOT})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
