#!/usr/bin/env bash
# nrp_run.sh — ship a local python script to NRP (Nautilus k8s) and run it as a Job.
# No image rebuild: the script is shipped as a ConfigMap and mounted at /app/data/scripts.
#
# Single job:
#   nrp_run.sh --script analyze.py --name my-analysis --cpu 4 --mem 16Gi --args "--slot B5_S42"
#
# Fan-out (one Job per line of an args file; each line is that job's args):
#   nrp_run.sh --script analyze.py --name my-analysis --args-file organoids.txt --canary --watch
#     organoids.txt:
#       --organoid 20217_24-01-07_data_33
#       --organoid 23134_24-05-24_35
#       ...
#
# Teardown:  nrp_run.sh --name my-analysis --delete
#
# Notes:
#   * Data must be reachable from the cluster (read from s3://braingeneersdev/hschweig/...,
#     endpoint https://s3-west.nrp-nautilus.io). The pod mounts prp-s3-credentials.
#   * Have your script WRITE results to S3 (boto3) or the skill will kubectl-cp them before TTL.
#   * ConfigMap cap ~1MB: ship a self-contained script; heavy deps must be in --image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$HERE/job_template.yaml"

# defaults
NS="${NRP_NS:-braingeneers}"
IMAGE="${NRP_IMAGE:-hschweiger15/nrp-predictor-val-metrics:v31}"   # has numpy/pandas/sklearn/scipy/boto3/braindance
CPU=4; MEM="16Gi"; GPU=0; EPHEMERAL="20Gi"; BACKOFF=1; AUTO_UPLOAD=0; BLAS_THREADS=""
SCRIPT=""; NAME=""; ARGS=""; ARGS_FILE=""; CANARY=0; WATCH=0; DELETE=0

while [[ $# -gt 0 ]]; do case "$1" in
  --script) SCRIPT="$2"; shift 2;;
  --name) NAME="$2"; shift 2;;
  --args) ARGS="$2"; shift 2;;
  --args-file) ARGS_FILE="$2"; shift 2;;
  --image) IMAGE="$2"; shift 2;;
  --cpu) CPU="$2"; shift 2;;
  --blas-threads) BLAS_THREADS="$2"; shift 2;;
  --mem) MEM="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;
  --ephemeral) EPHEMERAL="$2"; shift 2;;
  --backoff) BACKOFF="$2"; shift 2;;
  --ns) NS="$2"; shift 2;;
  --auto-upload) AUTO_UPLOAD=1; shift;;
  --canary) CANARY=1; shift;;
  --watch) WATCH=1; shift;;
  --delete) DELETE=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done

# DNS-1123-safe prefix
slug() { echo "$1" | tr '[:upper:]_ /.' '[:lower:]-----' | sed 's/[^a-z0-9-]//g; s/-\{2,\}/-/g; s/^-//; s/-$//' | cut -c1-40; }
PREFIX="hsch-$(slug "$NAME")"

if [[ "$DELETE" == 1 ]]; then
  echo "[nrp] deleting jobs with prefix $PREFIX in ns $NS"
  kubectl -n "$NS" get jobs -o name 2>/dev/null | grep "/$PREFIX" | xargs -r kubectl -n "$NS" delete
  kubectl -n "$NS" delete configmap "${PREFIX}-script" --ignore-not-found
  exit 0
fi

[[ -z "$SCRIPT" || -z "$NAME" ]] && { echo "need --script and --name" >&2; exit 2; }
[[ -f "$SCRIPT" ]] || { echo "script not found: $SCRIPT" >&2; exit 2; }

SCRIPT_NAME="$(basename "$SCRIPT")"
SCRIPT_CM="${PREFIX}-script"
echo "[nrp] shipping $SCRIPT -> configmap $SCRIPT_CM (ns $NS)"
kubectl -n "$NS" create configmap "$SCRIPT_CM" --from-file="$SCRIPT_NAME=$SCRIPT" \
  --dry-run=client -o yaml | kubectl -n "$NS" apply -f -

# GPU resource line: only add nvidia.com/gpu when >0
if [[ "$GPU" -gt 0 ]]; then GPU_LINE=", nvidia.com/gpu: ${GPU}"; else GPU_LINE=""; fi
: "${BLAS_THREADS:=$CPU}"   # default BLAS thread cap = requested CPUs (lower for heavy perm loops)
export IMAGE BACKOFF SCRIPT_NAME AUTO_UPLOAD CPU MEM EPHEMERAL GPU_LINE SCRIPT_CM BLAS_THREADS
SUBST='${JOB_NAME} ${IMAGE} ${BACKOFF} ${SCRIPT_NAME} ${SCRIPT_ARGS} ${AUTO_UPLOAD} ${CPU} ${MEM} ${EPHEMERAL} ${GPU_LINE} ${SCRIPT_CM} ${BLAS_THREADS}'

render_and_apply() {   # $1=job_name  $2=args
  export JOB_NAME="$1" SCRIPT_ARGS="$2"
  envsubst "$SUBST" < "$TEMPLATE" | kubectl -n "$NS" apply -f -
}

FIRST_JOB=""
if [[ -n "$ARGS_FILE" ]]; then
  [[ -f "$ARGS_FILE" ]] || { echo "args-file not found" >&2; exit 2; }
  i=0
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    JN="${PREFIX}-$(printf '%03d' "$i")"
    [[ -z "$FIRST_JOB" ]] && FIRST_JOB="$JN"
    render_and_apply "$JN" "$line"
    echo "[nrp] submitted $JN  args: $line"
    i=$((i+1))
    if [[ "$CANARY" == 1 && "$i" == 1 ]]; then
      echo "[nrp] CANARY: submitted only the first job; inspect it, then re-run without --canary for the rest."
      break
    fi
  done < "$ARGS_FILE"
  echo "[nrp] $i job(s) submitted under prefix $PREFIX"
else
  FIRST_JOB="$PREFIX"
  render_and_apply "$PREFIX" "$ARGS"
  echo "[nrp] submitted $PREFIX  args: $ARGS"
fi

if [[ "$WATCH" == 1 && -n "$FIRST_JOB" ]]; then
  echo "[nrp] waiting for pod of $FIRST_JOB ..."
  for _ in $(seq 1 30); do
    POD="$(kubectl -n "$NS" get pods -l job-name="$FIRST_JOB" -o name 2>/dev/null | head -1)"
    [[ -n "$POD" ]] && break; sleep 4
  done
  [[ -n "${POD:-}" ]] && kubectl -n "$NS" logs -f "$POD" || echo "[nrp] no pod yet; check: kubectl -n $NS get pods | grep $FIRST_JOB"
fi
