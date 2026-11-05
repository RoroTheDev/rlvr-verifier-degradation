# W4 plan: harness work before the grid (planned, not built)

**Status: plan from the Coding lane, 2026-11-05. Nothing here is built yet.** Each item says what
W3 showed, what to build and how to test it. Order is by how much money or data each protects.

## 1. Runs finish without a controller: upload off the pod, then terminate

**What W3 showed.** Every pilot run finished while the controlling PC was asleep. Each pod stopped
itself, and a stopped pod restarts only on its own host. The hosts had no free GPU for 11 to 20
hours, so collection waited and about $6 of idle time was billed (`docs/w3_pilot_runs.md`). A
108-cell grid cannot depend on one laptop staying awake.

**Build.** At STAGE:DONE, `verl_run.sh` does the following:

1. Checks its own run: exit code, every step present, per-sample record counts. This is the check
   `fleet.py` runs after a fetch.
2. Packs the run directory and uploads it as a W&B artifact. The pod already has `wandb` and
   `WANDB_API_KEY`.
3. Re-downloads the artifact's file list and compares it with the local copy.
4. Only then terminates its own pod with the pod-scoped `RUNPOD_API_KEY`. That deletes the pod, not
   just stops it.

If any step fails, it falls back to today's behaviour: idle self-stop, collected by `fleet.py`.
`fleet.py fetch` then takes a run from its artifact when the pod is gone.

- W&B storage is preferred over a RunPod network volume. A network volume ties every pod to one
  data centre, and stock-outs were the other W3 problem.

**Test.**

- A short run (`STEPS=20`, `MBPP_PASSES=3`) uploads, verifies and self-terminates, and the files
  from the artifact pass `fleet.py`'s fetch check.
- With the upload deliberately broken, the pod falls back to the self-stop.

## 2. Checkpoint-resume

**Why.** Training uses `save_freq=-1`, so a run that dies at step 200 is paid twice. The 15%
relaunch allowance in `docs/w4_grid_cost.md` is that cost, about $45–85 on the larger options.

**Build.**

- Save every 20 steps (the validation cadence) to `/workspace`, keeping only the latest checkpoint.
- Resume with verl's `resume_mode=auto` when a checkpoint exists.
- On resume, cut the per-sample logs (`mixup_logs/`) back to the checkpoint step before training
  restarts. Otherwise the steps after the checkpoint appear twice, and today's "booted once" check
  would void the run.
- The noise stays consistent across a resume: the persistent coin depends only on the task, and the
  resampled coin on (task, encounter), with `encounter` stored in every data row.
- A checkpoint with optimizer state is roughly 20–25 GB for 1.7B parameters (estimate; measure on
  the first run), so `/workspace` must grow from 30 GB to about 60 GB.

**Test.**

- Kill a pod mid-run, resume, and check the per-sample logs have each step exactly once.
- Check that steps after the resume point match a run that never stopped, within noise. They
  cannot match exactly, because vLLM sampling is not bit-reproducible.

## 3. Daily failure report

**Build.** `fleet.py report BATCH...` writes a dated Markdown file:

- every run's stage, step and ETA;
- every failure with its class: no instances, `CUDA_UNAVAILABLE`, OOM, restart, verify failed,
  stuck stopped pod;
- spend against each batch cap and the account balance;
- what needs a human.

Babysit writes one at a fixed time each day and whenever it exits with code 2 or 3.

**Test.** A fake batch state with one run in each failure class produces the matching lines; this
is a unit test in `test_fleet.py`.

## 4. Grid launching

- Launch in waves with a concurrency cap and the batch cost cap.
- Retry stock-outs on the same GPU model only. The pilot rule needs one GPU model per arm, and this
  carries over to W4 arms.
- Keep `cuda_versions` off: `runpod/launch.py` now defaults to no host filter.

## 5. Freeze first

Items 1 and 2 change `verl_run.sh`, so they come before the W4 harness is tagged. Bump
`HARNESS_VERSION` to 3 and re-run the freeze check in `docs/harness_freeze.md` on the result. Grid
runs then all use that one tag.
