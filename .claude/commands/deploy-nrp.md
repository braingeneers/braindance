---
name: deploy-nrp
description: Build, push, and test-deploy a Mamba training container to NRP Kubernetes. Bumps docker tag version, builds with cache, updates YAML, submits a test job, monitors logs, and alerts when ready for production jobs.
disable-model-invocation: true
allowed-tools: Bash(docker *) Bash(kubectl *) Bash(python *)
argument-hint: [production-submit-command(s)]
---

# Deploy NRP Mamba Container

Automate the full build-push-test cycle for the Mamba training container.

## Input

`$ARGUMENTS` contains one or more **production submit commands** (the real jobs the user wants to run). If empty, just build and push without submitting a test job.

The skill will automatically create a **test version** of the first submit command by replacing `--max-recordings-per-organoid N` with `--max-recordings-per-organoid 2`. This ensures the test job is identical to production except for data volume. If the command doesn't have `--max-recordings-per-organoid`, append `--max-recordings-per-organoid 2` to the test command. Also append `-test` to the `--job-name` value for the test run.

## Step 1: Determine current version

Read `proj/predictor/nrp/yaml/template_mamba.yaml` and extract the current image tag version number (e.g., `v38` from `hschweiger15/nrp-predictor2-mamba:v38`).

Increment by 1 to get the new version (e.g., `v39`).

Print clearly:
```
Current version: v38
New version: v39
```

## Step 2: Build the Docker image

**Build from `proj/predictor/`** — Dockerfile.mamba lives there and expects that
directory as its build context.

**Cache sources (important, we're on Mac cross-compiling to linux/amd64):**
- `v2` is the designated base cache on DockerHub.
- The immediately preceding version (e.g. `v46` when bumping to `v47`) is
  usually already in the local Docker image store from the previous deploy —
  that's what actually saves most of the time. Check with `docker images | grep nrp-predictor2-mamba` and include the latest local one as an extra `--cache-from`.
- If neither is locally present, `docker pull hschweiger15/nrp-predictor2-mamba:v2` first so `--cache-from` has something to match against.

```bash
cd proj/predictor
DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from hschweiger15/nrp-predictor2-mamba:v2 \
  --cache-from hschweiger15/nrp-predictor2-mamba:vPREV \
  -f Dockerfile.mamba \
  -t hschweiger15/nrp-predictor2-mamba:vNEW .
```

Replace `vNEW` with the new version tag and `vPREV` with the most recent
locally available version. This build can take 5-15 minutes with cache, much
longer without.

If the build fails, diagnose the error, attempt a fix, and retry. Common issues:
- Python syntax errors in recently edited files
- Missing dependencies in requirements.txt
- Running from the wrong directory (Dockerfile.mamba must be visible)

## Step 3: Push to DockerHub

```bash
docker push hschweiger15/nrp-predictor2-mamba:vNEW
```

## Step 4: Update YAML template

Edit `proj/predictor/nrp/yaml/template_mamba.yaml` to update the image tag to the new version.

Also update the build command in `proj/predictor/nrp/README.md` (the `-t` line) to reflect the new version for future reference.

## Step 5: Submit test job (if command provided)

If `$ARGUMENTS` was provided:

1. Take the **first** submit command from `$ARGUMENTS`
2. Create a test version by:
   - Replacing `--max-recordings-per-organoid N` with `--max-recordings-per-organoid 2` (if present), or appending `--max-recordings-per-organoid 2` (if not present)
   - Appending `-test` to the `--job-name` value (e.g., `mamba-v8-ff-sampling` becomes `mamba-v8-ff-sampling-test`)
3. Run the test command from `proj/predictor/nrp/`

Print the exact test command before running it so the user can verify.

If no submit command was provided, stop here and report success.

## Step 6: Monitor test job

After submitting, wait ~90 seconds then check pod logs:

```bash
kubectl logs <job-pod-name>
```

The job name comes from the submit command's `--job-name` flag, prefixed with `hsch-`. Get the exact pod name with:

```bash
kubectl get pods | grep <job-name>
```

### What to look for

**Success indicators** (the job started training correctly):
- `=== SPIKE-TOKEN MAMBA (v8) TRAINING ===` config printout appeared
- `Epoch 1/N` started
- No Python errors/tracebacks

**Failure indicators:**
- `Traceback` or `Error` in logs
- Pod in `CrashLoopBackOff` or `Error` state
- `argparse.ArgumentError` (arg conflict)
- `ModuleNotFoundError` (missing dependency)

### If the test fails

1. Read the error carefully
2. Fix the issue in the source code
3. Go back to Step 1 (bump version again, rebuild, redeploy)
4. Maximum 2 retry cycles before escalating to the user

### If the test succeeds

1. Delete the test job: `kubectl delete job hsch-<test-job-name>`
2. Report to the user:
   - Confirm the test job ran successfully
   - Show the new docker tag version
   - Print the exact production submit commands from `$ARGUMENTS` (unchanged, with original `--max-recordings-per-organoid` values) so the user can review and approve submission
3. Ask the user if they want to submit the production jobs now

## Notes

- The `--cache-from v2` is intentional -- v2 is the base cache layer, not the previous version
- Always build with `--platform linux/amd64` since NRP runs x86 nodes
- The build runs on Mac (ARM) so cross-compilation is expected to be slower
- Docker must be running (Docker Desktop) before starting
