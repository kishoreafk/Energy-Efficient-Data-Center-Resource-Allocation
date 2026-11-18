# Reproducing the production run on Kaggle

The results/ and weights/ shipped in this repository were produced by the
verified Kaggle kernel **kishore00afk/dc-dp-rl-run** (CPU runtime; the tabular
sparse-DP workload is CPU-bound, so GPU adds queue time with zero benefit).

## What the kernel does
1. Decodes the embedded (base64) project source zip into /kaggle/working/dc-dp-rl.
2. Installs `gymnasium` (optional — the env falls back to the standalone API).
3. Runs `python main.py`, capturing the full log; the kernel exits with main's
   return code, so kernel status = run status.
4. `results/` + `weights/` land in /kaggle/working -> kernel output.

## Verification protocol used (logs first, artifacts after)
1. `kaggle kernels status kishore00afk/dc-dp-rl-run` until COMPLETE.
2. Pull the kernel LOG (`kaggle kernels output`) and check markers:
   `12/12 checks passed`, `ALL CHECKS PASSED`, `MAIN_EXIT_CODE=0`,
   `max |V_vi - V_pi|_inf = 1.9e-05`, no Traceback.
3. Only then pull the artifact bundle and verify: PNGs non-blank (pixel std > 10),
   summary_metrics.csv identical to the local run, all 5 weight .npz files load
   with the expected shapes, dashboard JSON scientifically identical (only
   timing metadata differs).

## Reproduce yourself
```bash
# 1. place your kaggle.json (API token) at ~/.kaggle/kaggle.json (chmod 600)
cd kaggle
kaggle kernels push -p .          # pushes kernel_script.py + kernel-metadata.json
kaggle kernels status <user>/dc-dp-rl-run
kaggle kernels output <user>/dc-dp-rl-run -p out/
```
Change `"id"` in kernel-metadata.json to your own username. Credentials are
NOT included in this repository.
