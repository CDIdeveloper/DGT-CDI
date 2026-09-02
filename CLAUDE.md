# Claude Code Instructions

Adapte from andrej-karpathy-skills/claud.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Personal Instructions

## Restrictions

- Do NOT try to access anything on AWS (S3, etc.) directly
- Do NOT run commands that make network requests to AWS services
- Do NOT execute code outside the scope of current conversation
- Do NOT write imports inside functions, methods, or classes - always place imports at the top of the file
- Always briefly highlight main changes and ask for approval before making changes. (especially for git commands)
- Do not use imports inside functions, all imports must be at the top of the file.
- Do not right functions inside a function or methods of a class, unless this function will be used in this function or method specifically.
- Never run pip or conda command
- Always confirm before run command from terminal
- Never check .env file under any folder
- Never touch git or github command
- Donot "AskUserQuestion", just summarize and highlight the questions in response, But still rank the proposed solutions and add "(Recommended)" label for the first. Then wait for answers in prompt from me for next step. 

## Session Management
When the context window (auto-compact) reaches 30% remaining, remind me to update relevant .md files, to make sure smooth transaction between conversation sessions. for example:
- Update current status, completed steps, and next actions
- Note any key decisions made in this session
- Confirm all important context is captured before starting a new session
let me know where the new session shall start from afterwards.
- Do not update session_state.md until user request to.

## Review before moving on
- Read through code and documents, make sure all are consistent with each other, at the end of each phase before next phase in Roadmap of the development plan.

## Project gotchas — read before running anything

Hard-won failures. Each one has bitten this project; each is silent unless you know it.

1. **`main.py` DESTROYS the per-seed results directory on every run.**
   `custom_set_run_dir` calls `makedirs_rm_exist(cfg.run_dir)`, so re-running a config with an
   existing `results/DGT/<config>/<seed>/` wipes it. A 3-epoch smoke test on a live config
   destroyed a completed 50-epoch seed. **Always smoke-test into a throwaway dir:**
   `python main.py --cfg <cfg> --repeat 1 seed 0 optim.max_epoch 3 out_dir /tmp/smoke`.
   Symptom if you forget: `Failed when trying to aggregate multiple runs: Results with
   different seeds must have the save format`.

2. **PyG's processed cache never notices new data.** `InMemoryDataset` re-runs `process()` only
   when the processed file is *absent*, and the filename is keyed on the descriptor selection —
   not on `raw/` contents and not on code version. New parquet under an existing dataset name =
   silently training on the old data with the old normalisation stats.
   `rm -rf datasets/<name>/processed/` after any `prepare_data.py` re-run or loader change.

3. **Never select a config on test.** Use
   `python scripts/rank_configs_by_val.py <run_dirs...> --metric f1 --hide-test`, record the
   verdict, *then* look at test. `agg/test/best.json` is a record, not a decision input. See
   [documents/dgt_porting_guide.md](documents/dgt_porting_guide.md) §2 and
   [documents/projects/paper.md](documents/projects/paper.md) §8 for what happens when this
   rule is broken (a reported +0.0183 ablation gain collapsed to +0.0001).

4. **`Failed when trying to aggregate multiple runs: ... val/stats.json` after a retrain is
   EXPECTED.** `dgt_retrain` writes only the train logger — it has no validation or test phase
   — so `agg_runs` finds nothing to aggregate. The retrain succeeded if the three
   `final_model.*` files were written.

5. **Decision thresholds do not transfer between checkpoints.** `retrain_on_trainval.py` copies
   `best_f1_threshold` from a *training seed's* model into the manifest, but the shipped model
   is retrained on train+val and calibrates differently (observed: 0.375 vs 0.5013). Score the
   deployment bundle on the held-out test split and set the threshold from that.

6. **GPU training is not reproducible across runs**, even with `seed_everything`. Re-running an
   identical config moved one seed's val F1 by 0.013 over 50 epochs. Never re-run to "restore"
   a published number — record numbers when produced, and treat a re-run as new data.

7. **Two biodeg datasets exist and are not comparable.** `biodeg_gwu` (5742/300, InD retained)
   vs `biodeg_gwu_no_ind` (5264/278, canonical). Check `dataset.name` before comparing anything.
