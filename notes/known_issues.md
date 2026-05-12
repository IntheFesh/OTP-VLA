# Known Issues

## Hydra CLI override silently dropped in diagnostic scripts (2026-05-12)

**Affected files**: `scripts/diagnostic/15_gate1_sanity_and_m3.py` (and likely sibling scripts using same `load_config()` pattern)

**Root cause**: `compose(config_name="otp_soft_30e")` does not pass `overrides=sys.argv[1:]`. Hydra's non-decorator API requires explicit overrides; CLI args like `data.suite=libero_goal` are silently dropped. Script always uses yaml default (`libero_spatial`).

**Symptom**: Run completes successfully, produces results for `libero_spatial` instead of intended suite. No error, no warning.

**Secondary lock-in**: `PROBE_TASK_NAMES` is hardcoded LIBERO-Spatial task list (lines 52-63). Even if hydra override worked, task selection wouldn't follow.

**Status**: Not fixed. LIBERO-Goal cross-benchmark replication deferred (paper §VI mechanism-based caveat used instead).

**Fix recipe (if revisited)**:
1. `def load_config(overrides=None): ... compose(config_name="...", overrides=overrides or [])`
2. `cfg = load_config(sys.argv[1:])` in main
3. Move `PROBE_TASK_NAMES` into `cfg.data.probe_tasks` (per-suite yaml field)
4. Adapt `LIBEROOTPDataset` for libero_goal data structure differences (if any)

**Other scripts likely affected**: any `scripts/diagnostic/*.py` using `initialize_config_dir + compose` without explicit overrides. Audit before relying on CLI overrides in those scripts.

---

## Verification protocol for hydra-based scripts (mandatory)

Before running any new diagnostic or training script with CLI overrides — especially before launching GPU work in the 7.5-day ablation plan — verify in single-CPU dry-run that overrides actually propagate:

1. Add `print(OmegaConf.to_yaml(cfg))` at script start (or use `--cfg job --resolve` if `@hydra.main` decorator is used)
2. Run with intended overrides, on CPU, with minimal data load (or `--multirun` dry-run mode if available)
3. Visually confirm cfg fields reflect overrides before launching GPU work
4. If overrides do not propagate, fix the script per "Fix recipe" above; do NOT proceed with GPU launch assuming overrides will work at scale

**Reason**: silent hydra override drops produce results that look successful but answer the wrong question. Catching this at CPU dry-run cost ~30 sec; catching at GPU runtime cost = entire ablation cell (12.5 hr) at worst.

---

