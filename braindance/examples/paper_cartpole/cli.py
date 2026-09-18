"""Paper CartPole: recording → causal screening → ranked pairs → game."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("recording", "causal", "rank", "run"))
    parser.add_argument("--json", "-j", required=True, help="Working experiment JSON")
    parser.add_argument("--derived-dir", help="Explicit directory of connectivity .npy inputs")
    parser.add_argument("--order", choices=("first", "multi"), default="multi")
    parser.add_argument("--select-rank", type=int, help="Save this 1-based rank into the working JSON")
    parser.add_argument("--run-index", type=int, default=0,
                        help="0-based session index for C1/C2 trainer cycling")
    args = parser.parse_args()
    path = Path(args.json)
    config = json.loads(path.read_text())
    if args.stage in ("recording", "causal"):
        subprocess.run([sys.executable, "-m", "braindance.examples.paper_cartpole." + args.stage,
                        "--json", str(path.resolve())], check=True)
        return
    derived = args.derived_dir or config.get("derived_dir")
    if not derived:
        parser.error("Provide --derived-dir or derived_dir in JSON; historical paths differ by run")
    config["derived_dir"] = str(Path(derived).resolve())
    if args.stage == "run":
        from braindance.examples.paper_cartpole.game import main as run_game
        run_game(config, run_index=args.run_index)
        return

    import numpy as np
    from braindance.examples.paper_cartpole.ranking import find_connectivity_patterns
    import hashlib

    matrix_path = Path(derived) / f"causal_connectivity_{args.order}.npy"
    matrix = np.load(matrix_path, allow_pickle=False)
    electrodes = config["valid_stim_electrodes"]
    if (matrix.shape != (len(electrodes), len(electrodes)) or len(electrodes) < 4
            or len(set(electrodes)) != len(electrodes) or not np.isfinite(matrix).all()):
        raise ValueError("Need a finite square matrix matching at least four unique electrode IDs")
    if np.any(np.std(matrix, axis=0) == 0):
        raise ValueError("Ranking metric is undefined for zero-variance columns")
    patterns = find_connectivity_patterns(matrix)
    for rank, (a, b, c, d, score, similarity) in enumerate(patterns[:5], 1):
        print(f"{rank}: sensory={[electrodes[a], electrodes[c]]} "
              f"motor={[electrodes[b], electrodes[d]]} score={score:.6g}")
    if args.select_rank is not None:
        if not 1 <= args.select_rank <= len(patterns):
            raise ValueError(f"select-rank must be between 1 and {len(patterns)}")
        a, b, c, d, score, similarity = patterns[args.select_rank - 1]
        config.update(sensory_electrodes=[electrodes[a], electrodes[c]],
                      motor_electrodes=[electrodes[b], electrodes[d]])
        config["pair_selection"] = {
            "metric": "ranked_pairs.find_connectivity_patterns", "order": args.order,
            "rank": args.select_rank, "score": float(score),
            "matrix_path": str(matrix_path.resolve()),
            "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "source_commit": "5f733a8821d63d502042acc66e59abd303e6dae3",
        }
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Saved selection and provenance to {path}")


if __name__ == "__main__":
    main()
