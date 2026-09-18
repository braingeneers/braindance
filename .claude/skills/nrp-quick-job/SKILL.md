---
name: nrp-quick-job
description: Run an arbitrary local Python script on the NRP (Nautilus) Kubernetes cluster as a quick job, with no Docker image rebuild. Ships the script via ConfigMap, submits one Job (or one-per-item fan-out from an args file), monitors, and retrieves results. Use for CPU/GPU analysis jobs that are too heavy or too parallel for the local machine — NOT for building/training-container deploys (use deploy-nrp for that).
allowed-tools: Bash(kubectl *) Bash(bash *) Bash(aws *) Bash(chmod *) Bash(envsubst *)
argument-hint: [what to run, e.g. "run analyze.py per-organoid over organoids.txt, 4cpu 16Gi"]
---

# Run a quick job on NRP

Ship a local Python script to the cluster and run it — no image rebuild. Code travels as a
ConfigMap mounted at `/app/data/scripts`; the default image already has numpy/pandas/scipy/
sklearn/boto3/braindance. This skill is for **one-off or fan-out analysis jobs** (per-organoid
sweeps, heavy permutation nulls, anything that saturates the laptop). For building/pushing the
Mamba training container, use `deploy-nrp` instead.

Helper: `.claude/skills/nrp-quick-job/nrp_run.sh` (+ `job_template.yaml`). `chmod +x` it once.

## When to use
- The work is embarrassingly parallel per item (per organoid / per shard) → fan-out.
- A single job is CPU/GPU-heavy and the local machine is contended.
- The script is self-contained OR only needs libs already in the image (`--image` to override).

## Prerequisites (verify before first run)
1. `kubectl config current-context` is `nautilus`; namespace is `braingeneers` (or pass `--ns`).
2. Secret `prp-s3-credentials` exists in the namespace: `kubectl -n braingeneers get secret prp-s3-credentials`.
3. **The data the script reads must be reachable from the cluster** — i.e. on
   `s3://braingeneersdev/hschweig/...` (endpoint `https://s3-west.nrp-nautilus.io`), NOT only on a
   local SSD. If it's local-only, `aws --endpoint https://s3-west.nrp-nautilus.io s3 sync` it up first.
4. The script should **write its outputs to S3** (boto3) so results survive pod teardown; otherwise
   retrieve them with `kubectl cp` before the job's 48h TTL expires.

## Single job
```bash
.claude/skills/nrp-quick-job/nrp_run.sh \
  --script /abs/path/analyze.py --name my-analysis \
  --cpu 4 --mem 16Gi --args "--slot B5_S42_V9 --out s3://braingeneersdev/hschweig/my-analysis/" \
  --watch
```
`--watch` tails the pod logs. Add `--gpu 1` for a GPU job. `--image IMG` to override.

## Fan-out (one Job per line of an args file)
Each non-comment line becomes one Job; that line is passed verbatim as the script's args.
```bash
# organoids.txt
--organoid 20217_24-01-07_data_33 --out s3://braingeneersdev/hschweig/sel/
--organoid 23134_24-05-24_35 --out s3://braingeneersdev/hschweig/sel/
...
.claude/skills/nrp-quick-job/nrp_run.sh \
  --script /abs/path/analyze.py --name sel-geom --args-file organoids.txt --canary --watch
```
**Always `--canary` first**: it submits ONLY the first job so you can confirm it runs clean
(logs, S3 output written) before fanning out the rest. Then re-run the SAME command without
`--canary` to submit all of them.

## Procedure the model should follow
1. Confirm prerequisites (context/namespace/secret; data on S3).
2. If fanning out, generate the args file (e.g. one line per organoid).
3. Submit with `--canary --watch`. Read the canary logs. Confirm: no traceback, expected prints,
   and that it wrote its S3 output (`aws --endpoint https://s3-west.nrp-nautilus.io s3 ls <prefix>`).
4. If clean, re-run without `--canary` to launch the full set. Report job count + prefix.
5. Monitor: `kubectl -n braingeneers get jobs | grep <prefix>` and `... get pods | grep <prefix>`.
6. When done, `aws --endpoint ... s3 sync <prefix> <local>` the results.
7. Teardown: `nrp_run.sh --name <name> --delete` (removes jobs + the script ConfigMap).

## Gotchas (baked into the template / learned here)
- **BLAS thread caps are set** (`OPENBLAS/OMP/MKL/NUMEXPR/VECLIB_NUM_THREADS`, default = `--cpu`).
  Without them, sklearn/numpy oversubscribe threads and many-small-matrix workloads (permutation
  nulls, per-window cross-val) crawl — a single-organoid 200-perm balance decoder took ~8 min
  uncapped. For heavy perm loops pass `--blas-threads 1` or `2` (often *faster* than the CPU count).
- **Node blacklist** (`nodeAffinity NotIn`) for known-bad GPU nodes is already in the template.
- **`imagePullPolicy: Always`** — a freshly-pushed image tag is always pulled (tags get reused).
- **ConfigMap ~1MB cap** — ship a self-contained script; if it imports project modules not baked
  into the image, either inline them or use an image that has them.
- **S3 checksum quirk**: env vars `AWS_*_CHECKSUM_*=when_required` are set; if reads/writes fail
  with a checksum error, that's the knob. `smart_open`+`.zst` is broken on these images → use raw
  boto3 `get_object`/`put_object`.
- **Don't reuse a name across different scripts** without `--delete` first — the ConfigMap is keyed
  on the name and would serve stale code.
- Completed Jobs auto-delete after 48h (`ttlSecondsAfterFinished`); `kubectl get jobs` won't show
  them past that. Pull results before then.
