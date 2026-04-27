import argparse
import importlib.util
import json
import os
from pathlib import Path

from mcsu_eval_summary import (
    build_comparison,
    flatten_for_wandb,
    write_projection_csv,
    write_task_csv,
)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a compact MCSU vs baseline report.")
    parser.add_argument("--mcsu_eval", required=True)
    parser.add_argument("--baseline_eval", required=True)
    parser.add_argument("--mcsu_projection", default=None)
    parser.add_argument("--baseline_projection", default=None)
    parser.add_argument("--output_dir", default="./runs/qwen_mcsu_report")
    parser.add_argument("--wandb_enabled", default="false")
    parser.add_argument("--wandb_entity", default="changwoolabs")
    parser.add_argument("--wandb_project", default="multilingual-amnesia")
    parser.add_argument("--wandb_name", default="qwen_mcsu_baseline_summary")
    parser.add_argument("--wandb_group", default="qwen-npo-comparison")
    parser.add_argument("--wandb_mode", default="online")
    return parser.parse_args()


def log_to_wandb(args, comparison, summary_path, task_csv_path, projection_csv_path):
    if not _parse_bool(args.wandb_enabled):
        os.environ["WANDB_DISABLED"] = "true"
        return None
    if importlib.util.find_spec("wandb") is None:
        raise ImportError("--wandb_enabled true requires wandb. Install it with `pip install wandb`.")

    import wandb

    os.environ.pop("WANDB_DISABLED", None)
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        config=vars(args),
    )

    metrics = {}
    for model_name, model_summary in comparison["models"].items():
        metrics.update(flatten_for_wandb(model_name, model_summary))
    wandb.log(metrics)

    task_columns = [
        "model", "task", "probability", "rougeL_recall", "truth_ratio",
        "avg_gt_loss", "avg_perturb_loss", "chrf", "bleu", "rougeL", "num_examples",
    ]
    task_table = wandb.Table(columns=task_columns)
    for model_name, model_summary in comparison["models"].items():
        for task_summary in model_summary["tasks"].values():
            task_table.add_data(*[model_name if col == "model" else task_summary.get(col) for col in task_columns])
    wandb.log({"tables/eval_summary": task_table})

    projection_columns = ["model", "language", "layer", "mean_projection_energy", "std_projection_energy", "num_examples"]
    projection_table = wandb.Table(columns=projection_columns)
    for model_name, model_summary in comparison["models"].items():
        projection = model_summary.get("projection")
        if not projection:
            continue
        for row in projection["rows"]:
            projection_table.add_data(model_name, row["language"], row["layer"], row["mean_projection_energy"], row["std_projection_energy"], row["num_examples"])
    wandb.log({"tables/projection_summary": projection_table})

    artifact = wandb.Artifact(f"{args.wandb_name}-summary-files", type="evaluation_summary")
    for path in [summary_path, task_csv_path, projection_csv_path]:
        if path is not None and Path(path).exists():
            artifact.add_file(str(path))
    wandb.log_artifact(artifact)
    wandb.finish()
    return run


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison(
        {
            "mcsu": args.mcsu_eval,
            "baseline": args.baseline_eval,
        },
        {
            "mcsu": args.mcsu_projection,
            "baseline": args.baseline_projection,
        },
    )

    summary_path = output_dir / "summary.json"
    task_csv_path = output_dir / "eval_summary.csv"
    projection_csv_path = output_dir / "projection_summary.csv"

    with open(summary_path, "w") as f:
        json.dump(comparison, f, indent=2)
    write_task_csv(comparison, task_csv_path)
    write_projection_csv(comparison, projection_csv_path)

    log_to_wandb(args, comparison, summary_path, task_csv_path, projection_csv_path)
    print(f"Saved compact report to {summary_path}")
    print(f"Saved eval table to {task_csv_path}")
    print(f"Saved projection table to {projection_csv_path}")


if __name__ == "__main__":
    main()
