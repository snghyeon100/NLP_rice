import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import hmean


TASK_LABELS = {
    "eval_log.json": "Retain",
    "eval_real_author_wo_options.json": "Real Authors",
    "eval_real_world_wo_options.json": "Real World",
    "eval_log_forget.json": "Forget",
}


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _numbers(value):
    if isinstance(value, dict):
        out = []
        for nested in value.values():
            out.extend(_numbers(nested))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for nested in value:
            out.extend(_numbers(nested))
        return out
    number = _safe_float(value)
    return [] if number is None else [number]


def _mean(value):
    values = _numbers(value)
    if not values:
        return None
    return float(np.asarray(values, dtype=np.float64).mean())


def _std(value):
    values = _numbers(value)
    if not values:
        return None
    return float(np.asarray(values, dtype=np.float64).std())


def _truth_ratio(task_name, logs):
    data_indices = list(logs["avg_paraphrased_loss"].keys())
    paraphrase = np.asarray([logs["avg_paraphrased_loss"][idx] for idx in data_indices], dtype=np.float64)
    perturb = np.asarray([logs["average_perturb_loss"][idx] for idx in data_indices], dtype=np.float64)
    paraphrase_prob = np.exp(-paraphrase)
    perturb_prob = np.exp(-perturb).mean(axis=-1)
    ratio = perturb_prob / paraphrase_prob
    if "forget" in task_name:
        return float(np.minimum(ratio, 1.0 / ratio).mean())
    return float(np.maximum(0.0, 1.0 - ratio).mean())


def _probability(task_name, logs):
    if "eval_log" in task_name:
        return float(np.exp(-np.asarray(list(logs["avg_gt_loss"].values()), dtype=np.float64)).mean())
    true_prob = np.exp(-np.asarray(list(logs["avg_gt_loss"].values()), dtype=np.float64))
    false_prob = np.exp(-np.asarray(list(logs["average_perturb_loss"].values()), dtype=np.float64))
    all_prob = np.concatenate([true_prob[:, None], false_prob], axis=1).sum(axis=-1)
    return float((true_prob / all_prob).mean())


def summarize_task(task_name, logs):
    chrf = logs.get("chrf", {})
    bleu = logs.get("bleu", {})
    rouge = logs.get("rouge", {})
    return {
        "task": TASK_LABELS.get(task_name, task_name.replace(".json", "")),
        "probability": _probability(task_name, logs),
        "rougeL_recall": _mean(logs.get("rougeL_recall")),
        "truth_ratio": _truth_ratio(task_name, logs),
        "avg_gt_loss": _mean(logs.get("avg_gt_loss")),
        "avg_perturb_loss": _mean(logs.get("average_perturb_loss")),
        "chrf": _safe_float(chrf.get("score")) if isinstance(chrf, dict) else None,
        "bleu": _safe_float(bleu.get("bleu")) if isinstance(bleu, dict) else None,
        "rougeL": _safe_float(rouge.get("rougeL")) if isinstance(rouge, dict) else None,
        "num_examples": len(logs.get("avg_gt_loss", {})),
    }


def summarize_eval_file(path):
    with open(path) as f:
        logs_by_task = json.load(f)
    tasks = {}
    for task_name in TASK_LABELS:
        if task_name in logs_by_task:
            tasks[task_name] = summarize_task(task_name, logs_by_task[task_name])

    utility_values = []
    for task_name, task_summary in tasks.items():
        if task_summary["task"] == "Forget":
            continue
        utility_values.extend([
            task_summary["probability"],
            task_summary["rougeL_recall"],
            task_summary["truth_ratio"],
        ])
    model_utility = float(hmean([value for value in utility_values if value is not None and value > 0]))

    return {
        "path": str(path),
        "model_utility": model_utility,
        "tasks": tasks,
    }


def summarize_projection_file(path):
    with open(path) as f:
        projection = json.load(f)
    rows = []
    for language, layers in projection["projection_energy"].items():
        for layer, stats in layers.items():
            rows.append({
                "language": language,
                "layer": int(layer),
                "mean_projection_energy": float(stats["mean_projection_energy"]),
                "std_projection_energy": float(stats["std_projection_energy"]),
                "num_examples": int(stats["num_examples"]),
            })

    overall = float(np.mean([row["mean_projection_energy"] for row in rows])) if rows else None
    by_language = {}
    for language in sorted({row["language"] for row in rows}):
        values = [row["mean_projection_energy"] for row in rows if row["language"] == language]
        by_language[language] = float(np.mean(values))
    by_layer = {}
    for layer in sorted({row["layer"] for row in rows}):
        values = [row["mean_projection_energy"] for row in rows if row["layer"] == layer]
        by_layer[str(layer)] = float(np.mean(values))

    return {
        "path": str(path),
        "overall_mean_projection_energy": overall,
        "by_language": by_language,
        "by_layer": by_layer,
        "rows": rows,
    }


def build_comparison(eval_paths, projection_paths=None):
    comparison = {"models": {}}
    projection_paths = projection_paths or {}
    for model_name, eval_path in eval_paths.items():
        model_summary = summarize_eval_file(eval_path)
        if model_name in projection_paths and projection_paths[model_name] is not None:
            model_summary["projection"] = summarize_projection_file(projection_paths[model_name])
        comparison["models"][model_name] = model_summary
    return comparison


def flatten_for_wandb(model_name, summary):
    metrics = {f"{model_name}/model_utility": summary["model_utility"]}
    for task_summary in summary["tasks"].values():
        task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_summary["task"].lower())
        for key, value in task_summary.items():
            if key == "task" or value is None:
                continue
            metrics[f"{model_name}/{task}/{key}"] = value
    projection = summary.get("projection")
    if projection:
        metrics[f"{model_name}/projection/overall_mean_energy"] = projection["overall_mean_projection_energy"]
        for language, value in projection["by_language"].items():
            metrics[f"{model_name}/projection/language/{language}"] = value
        for layer, value in projection["by_layer"].items():
            metrics[f"{model_name}/projection/layer/{layer}"] = value
    return metrics


def write_task_csv(comparison, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "task", "probability", "rougeL_recall", "truth_ratio",
        "avg_gt_loss", "avg_perturb_loss", "chrf", "bleu", "rougeL", "num_examples",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model_name, model_summary in comparison["models"].items():
            for task_summary in model_summary["tasks"].values():
                row = {"model": model_name, **task_summary}
                writer.writerow(row)


def write_projection_csv(comparison, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "language", "layer", "mean_projection_energy", "std_projection_energy", "num_examples"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model_name, model_summary in comparison["models"].items():
            projection = model_summary.get("projection")
            if not projection:
                continue
            for row in projection["rows"]:
                writer.writerow({"model": model_name, **row})
