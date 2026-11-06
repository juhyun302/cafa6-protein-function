"""Submit a versioned Azure ML job. --dry-run never contacts Azure."""

from pathlib import Path
import argparse
import json
import shlex


def build_command(stage, workers, threads, foldseek_bin):
    data, output = "$" + "{{inputs.data}}", "$" + "{{outputs.result}}"
    base = ["python", "scripts/cafa.py"]
    if stage == "download":
        return base + ["download", "--accessions", data + "/accessions.txt",
                       "--output-dir", output + "/structures", "--log-dir", output + "/logs",
                       "--workers", str(workers)]
    if stage == "search":
        return base + ["search", "--structures-dir", data, "--output-dir", output,
                       "--threads", str(threads), "--foldseek-bin", foldseek_bin]
    if stage == "propagate":
        return base + ["propagate", "--input-dir", data, "--output-dir", output,
                       "--mapping-pkl", data + "/go_term_mappings.pkl",
                       "--ancestor-npz", data + "/go_ancestor_matrix.npz", "--workers", str(workers)]
    return base + ["make-submission", "--input-dir", data,
                   "--output", output + "/structure-submission.tsv"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["download", "search", "propagate", "make-submission"], required=True)
    parser.add_argument("--data", required=True, help="Registered folder asset URI or local input folder")
    parser.add_argument("--compute", required=True)
    parser.add_argument("--environment", required=True, help="azureml:<environment-name>:<version>")
    parser.add_argument("--config", type=Path, default=Path("configs/azureml.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.threads) < 1 or "latest" in args.environment.lower():
        parser.error("Use positive resource counts and a versioned environment instead of latest")
    if not args.environment.startswith("azureml:") or len(args.environment.split(":")) != 3:
        parser.error("--environment must be azureml:<name>:<version>")
    specification = {
        "stage": args.stage, "data": args.data, "compute": args.compute,
        "environment": args.environment,
        "command": shlex.join(build_command(args.stage, args.workers, args.threads, args.foldseek_bin)),
    }
    if args.dry_run:
        print(json.dumps(specification, ensure_ascii=False, indent=2))
        return
    if not args.config.exists():
        parser.error("Copy configs/azureml.example.json to configs/azureml.json and fill the workspace values")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not all(config.get(key) for key in ("subscription_id", "resource_group", "workspace_name")):
        parser.error("Azure workspace configuration is incomplete")
    try:
        from azure.ai.ml import MLClient, command, Input, Output
        from azure.identity import DefaultAzureCredential
    except ImportError:
        parser.error("Install Azure dependencies: python -m pip install -e .[azure]")
    client = MLClient(credential=DefaultAzureCredential(), **config)
    job = command(
        code=str(Path(__file__).resolve().parents[1]),
        command=specification["command"],
        inputs={"data": Input(type="uri_folder", path=args.data, mode="download")},
        outputs={"result": Output(type="uri_folder", mode="upload")},
        environment=args.environment, compute=args.compute,
        display_name=f"cafa-{args.stage}", experiment_name="cafa-structure-pipeline",
    )
    created = client.jobs.create_or_update(job)
    print(created.studio_url)


if __name__ == "__main__":
    main()
