# Handoff (2026-11-07 evening): review findings + next tasks

Context: Apart Swarm Alignment Hackathon, **Track 2 (Science of Agent Networks)**. Deadline **Sun Nov 8, 11:59 PM AoE**
(≈ Mon 4 AM Pacific). Judged on impact/novelty, execution, and clarity (1–5 each). An 8-page report plus a required
"Limitations and Dual-Use" appendix. Suggested framing: *"Stubbornness is a leadership signal: one agent that refuses to move
can take over a swarm's decisions."*

## 0. State of the working tree (read first)

Uncommitted changes from the review session (no git was run, because the shell couldn't release git lock files):
- `coordination_game/game.py`: holder swap (`Config.swap_game`, `Config.swap_bad_apple_ids`, `Session.bad_ids_for(game)`;
  `_draw_targets` now takes `holders`; games.jsonl gets `holder_ids` in swap runs).
- `coordination_game/batch.py`: arm `badapple_swap`, plans `swap1` (real) and `mockswap` (free). `COMM_ARMS` is now
  `startswith("chat_")` (same 3 arms as before; the old definition would have pulled the new arm into the comm plans).
- `coordination_game/analyze_swap.py`: new.
- `handoff.md`: this file.

Verified: mocktest + mockcomm prompts, picks and chat are byte-identical to HEAD (so `chat_trap` and its overnight qwen
launch behave exactly as before); mockswap runs and the holder actually changes at game 6.
**Todo:** branch `claude/holder-swap`, commit these, PR per CONTRIBUTING.md, and add a plan.md entry (§8 + decision log).

## 1. Holder swap (experiment 1): code done, NOT yet run

Why: the core safety claim needs to show that agents learned "follow whoever won't move" (anyone can exploit that), not
"follow agent k" (which is harmless and specific).

Design: plan `swap1` = glm-5.2, T=1.0, seeds 0–9, 10 games, cap 15. The original holder is the same as overnight1 (`bad_apple_id(seed)`),
so **games 1–5 replicate overnight1 badapple glm-5.2 exactly** and give a paired baseline. From **game 6**, a different agent
(seeded draw, `bad-apple-swap-{seed}`) becomes the scripted holder. The old holder becomes an ordinary glm-5.2 agent.

Launch on the Mac (the cloud/VM shell can't reach the Adaption API). This doesn't conflict with chat_trap (Gemma, parallel 2) or with
the midnight qwen launch, but API latency will rise:
```bash
cd ~/gh/coordination_game
nohup caffeinate -i ~/miniforge3/envs/coordgame/bin/python -m coordination_game.batch --plan swap1 --parallel 3 \
  > runs/swap1_nohup.out 2>&1 &
tail -f runs/swap1/batch.log
```
Estimated cost ~12–17 credits (overnight1 glm-5.2 badapple sessions were ~1.2–1.7 each); time ~2–3 h (sessions took 23–90 min
under load). Check `runs/swap1/badapple_swap/glm-5.2/s0` after game 6 before trusting the rest.

Analyze:
```bash
python -m coordination_game.analyze_swap 'runs/swap1/badapple_swap/glm-5.2/s*'
```
Prints per session: rounds vs the same-seed overnight1 baseline, the round-2 fraction on the new holder's item in games 6+,
what the old holder (now an LLM) does, and reasoning mentions of the old vs new ID. Then a per-game summary.

How to read it:
- Games 6–7 back near the 2-round floor, with a high round-2 fraction on the new holder → they follow *behavior*:
  the takeover result (strong).
- Games 6–7 jump back to game-1-like lengths, reasoning keeps naming the old ID → they learned an *identity*.
- **Known wrinkle:** the old holder becomes an LLM that sees "I held in games 1–5" in the history but has no own reasoning
  for those games. It may keep holding its own picks, which creates two competing holders. `analyze_swap` reports
  "old holds" so this is visible. Report it either way: it's a real (and interesting) effect, not a bug.
- Optional: post-session leader vote via `python -m coordination_game.discuss <session dirs> --out-name post_discussion_shuffled`
  (old vs new holder vote share). Check that discuss.py rebuilds memory correctly with `swap_bad_apple_ids` first.
- `analyze_badapple`, `transcript`, `viewer`, `analyze_votes` still read only `config.bad_apple_ids` as the holder. For swap
  sessions they mislabel games 6+. Use `games.jsonl` `holder_ids` / `bad_apple_picks` keys per game if you patch them.

## 2. Integrity issues to fix or disclose (from the review)

1. **Board order = game order (main threat to the learning-curve claim).** Every session plays boards 0..9 in the same
   order, so "game-1 standoffs are long, then they learn" is confounded with "board 0 is hard". Fix: per-seed board
   permutation (e.g. `random.Random(f"board-order-{seed}").shuffle`) for 3–5 sessions of 2–3 conformer models
   (deepseek ≈ 0.1 credits/session; glm-53-flash cheap). Keep the default arms byte-identical (new opt-in config flag).
2. **chat_trap doesn't isolate chat.** Confederates held 150/150 (100%) in every session, but what they *say* differs:
   gemma-31b's confederate announces "I cannot change my pick... please pivot" (its group converges); gemma-26b's
   confederate just proposes politely (its group never converges). So the supported claim is "a holder who announces it can't move
   gets followed", not "chat breaks the trap". Fix: a **silent-holder chat arm**: a scripted holder in a chat arm that always
   sends an empty message. Silence is now a valid move for everyone (§8g-revised), so a silent agent isn't a dead giveaway
   any more. This needs lifting the `scripted + chat_turns` ValueError for that arm only; the holder's chat turn = silent record.
   Run it on gemma-4-31b / gemma-4-26b (and qwen3.5-397b if budget allows) to compare against chat_trap.
3. **Leader-vote share isn't comparable across arms.** In confederate arms the confederate votes, and votes for itself
   (gemma-31b s0, s1: holder self-vote True), so there are 10 voters including one holder self-vote, vs 9 voters in scripted arms.
   Report holder share **excluding the holder's own vote**. `analyze_votes` also counts a tie at the top as a holder win; report
   ties separately.
4. **Regex-based CoT numbers** ("named in 7,555/7,560", "98% stubborn", "not-agent" counts): validate on ~100 sampled
   traces (by hand or an LLM judge, report agreement) or label them clearly as rough. `\bA1\b`-style patterns can match non-agent text.
5. **Say these plainly in the paper:**
   - The scripted holder picks its target *after* seeing round-1 picks (guarantees ≥2 rounds; a design choice, not rigging).
   - In the bad-apple arm, "converged on the holder's item" is true by definition. Call it "convergence = conformity".
   - qwen3.5-397b stopped at n=6 for cost/outcome reasons (0/60), and partial s6–s8 were excluded.
   - Don't pool the T=1.2 / fixed-A7 / mixed-model pilots with overnight1 in any statistics.
   - Only weekend work is judged: say the project started Sat Nov 7.
6. **Data availability:** `runs/` is gitignored. Publish at least games/rounds/debriefs jsonl (small) or a HF dataset.

## 3. Statistics (free; not computed anywhere yet)

Session = unit. Per model: game-1 vs game-10 rounds (paired Wilcoxon), learning speed (first game at the floor), Kruskal–Wallis
across models + Holm-corrected pairwise tests, bootstrap 95% CIs on holder vote share. Swap: games 5 vs 6 vs 7 paired against the
same-seed baseline.

## 4. Not worth running

- **Confederate without chat:** confederates hold 100%, so its picks would be identical to the scripted bot's. State that in
  the paper instead.
- **chat_confed_bare:** the clause doesn't change holding (100% either way). Only run it to make a disclosure/deception claim.

## 5. Priority with the time left

1. Launch `swap1` now. 2. Silent-holder chat arm on gemma-31b (+26b). 3. Stats + vote-share fix. 4. Board-order shuffle
if credits/time allow. Then write: the report (8 pages) and the dual-use appendix (how a stubborn agent could take over a swarm).
