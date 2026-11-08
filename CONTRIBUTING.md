# Working together

## Git workflow
- `main` is the stable branch. Don't commit to it directly; every change goes through a pull request.
- One branch per piece of work, named `<name>/<topic>` (e.g. `beck/leader-vote-shuffled`). Branch off an up-to-date `main`:
  ```bash
  git switch main && git pull && git switch -c <name>/<topic>
  ```
- Keep branches short-lived. Before opening a PR, bring in the latest `main` (`git pull origin main`) and resolve conflicts.
- Open a PR into `main` and ask the other person to review. Squash-merge, then delete the branch.
- Branch protection isn't available on this private repo's plan, so these rules are by agreement.

## Experiment hygiene
- **Talk before launching paid runs.** We share one pool of Adaption credits. Note planned runs and their estimated cost in the PR or in plan.md first.
- Record decisions and results in `plan.md` (decision log and results sections) in the same PR as the code they relate to.
- Changes to the game prompt or engine must keep existing default-arm prompts byte-identical, unless the change is deliberate and noted in plan.md (check with a mock run against `main`).
- Never commit `.env` or API keys. Each person keeps their own `.env` (see `.env.example`).

## Data
- `runs/` (raw logs, ~1.7 GB) is gitignored and lives only on the machine that ran it. Share run folders out of band, and say in plan.md where the canonical copy is.
- Small derived outputs that are worth keeping (e.g. `results/figures/`) are committed.
