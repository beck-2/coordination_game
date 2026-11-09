"""Minimal tests for the core codings in coordination_game.analysis (no run data needed).

    python -m unittest tests.test_analysis
"""
import json
import tempfile
import unittest
from pathlib import Path

from coordination_game.analysis import game_outcome, holdout_adoption, session_rows, code_trace


def picks(*vals):
    return {str(a): v for a, v in enumerate(vals, 1)}


class TestConvergence(unittest.TestCase):
    def test_converges_in_round_two(self):
        r = [picks(*["x"] * 9 + ["y"]), picks(*["x"] * 10)]
        self.assertEqual(game_outcome(r), {"converged": True, "round_converged": 2, "rounds_played": 2, "censored": False})

    def test_cap_without_unanimity_is_failure_not_15(self):
        r = [picks(*["x"] * 9 + ["y"])] * 15
        out = game_outcome(r)
        self.assertFalse(out["converged"])
        self.assertIsNone(out["round_converged"])
        self.assertTrue(out["censored"])

    def test_invalid_pick_blocks_unanimity(self):
        r = [picks(*["x"] * 9 + [None])]
        self.assertFalse(game_outcome(r)["converged"])


class TestAdoption(unittest.TestCase):
    def test_adoption_excludes_holder(self):
        p = picks("t", "t", "t", "m", "m", "m", "m", "m", "m", "t")  # holder = agent 10 on target t
        adopt, other = holdout_adoption(p, holder=10, target="t")
        self.assertAlmostEqual(adopt, 3 / 9)
        self.assertAlmostEqual(other, 6 / 9)

    def test_full_adoption(self):
        adopt, other = holdout_adoption(picks(*["t"] * 10), holder=2, target="t")
        self.assertEqual((adopt, other), (1.0, 0.0))


class TestSessionCounting(unittest.TestCase):
    def _session(self, d: Path, n_games: int, complete: int):
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"config": {"n_games": n_games, "n_agents": 10, "max_rounds": 15, "seed": 0,
                                                              "bad_apple_ids": [3]}, "agents": {}, "model": "m", "arm": "badapple"}))
        with open(d / "games.jsonl", "w") as g, open(d / "rounds.jsonl", "w") as r:
            for gi in range(1, complete + 1):
                r1 = picks(*["a"] * 2 + ["T"] + ["a"] * 7)
                r2 = picks(*["T"] * 10)
                r.write(json.dumps({"game": gi, "round": 1, "picks": r1}) + "\n")
                r.write(json.dumps({"game": gi, "round": 2, "picks": r2}) + "\n")
                g.write(json.dumps({"game": gi, "converged": True, "converged_on": "T", "rounds": 2,
                                    "bad_apple_picks": {"3": "T"}}) + "\n")

    def test_complete_and_incomplete(self):
        with tempfile.TemporaryDirectory() as t:
            self._session(Path(t, "full"), 10, 10)
            self._session(Path(t, "part"), 10, 4)
            s_full, g_full = session_rows(Path(t, "full"), "test")
            s_part, _ = session_rows(Path(t, "part"), "test")
            self.assertTrue(s_full["complete"])
            self.assertFalse(s_part["complete"])
            self.assertEqual(s_full["games_converged"], 10)
            self.assertEqual(s_full["first_game_at_2_rounds"], 1)
            self.assertAlmostEqual(s_full["r2_adoption_mean"], 1.0)
            self.assertTrue(s_full["targets_unpicked_r1"])
            self.assertEqual(len(g_full), 10)


class TestCoding(unittest.TestCase):
    def test_codes(self):
        k = code_trace("A3 hasn't budged for 3 rounds. We should switch to A3's pick.", 3)
        self.assertFalse(code_trace("I will switch to the majority on √5.", 3)["b_proposes_following"])
        self.assertTrue(k["a_notices_holder"])
        self.assertTrue(k["b_proposes_following"])
        self.assertFalse(k["c_suspects_scripted"])


if __name__ == "__main__":
    unittest.main()
