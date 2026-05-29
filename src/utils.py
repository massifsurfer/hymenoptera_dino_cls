from pathlib import Path

import matplotlib.pyplot as plt
from faker import Faker
from git import Repo
from mlflow.client import MlflowClient


def get_git_commit_id() -> str:
    """Retrieves the SHA-1 hash of the current Git commit.

    Parses the internal Git directory structure directly to find the active
    HEAD commit without invoking external shell commands or CLI tools.

    Returns:
        str: The 40-character commit hash if successful, or "unknown".
    """
    try:
        repo = Repo(".", search_parent_directories=True)
        return repo.head.commit.hexsha
    except Exception:
        return "unknown"


def gen_fancy_name() -> str:
    fake = Faker()
    words = fake.words(nb=2)
    name = "_".join(words)
    return name


def save_mlflow_plots(run_id: str, output_dir: str = "plots"):
    """Fetches all metric histories from MLflow for a run and saves them as PNG plots."""

    plots_path = Path(output_dir)
    plots_path.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()
    run = client.get_run(run_id)
    metric_keys = run.data.metrics.keys()

    if not metric_keys:
        print("⚠️ No metrics found in MLflow to plot.")
        return

    print(f"Exporting metrics to '{plots_path}' folder...")
    for key in metric_keys:
        metric_history = client.get_metric_history(run_id, key)

        steps = [m.step for m in metric_history]
        values = [m.value for m in metric_history]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, values, marker="o", linestyle="-", label=key)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xlabel("Step / Epoch")
        plt.ylabel("Value")
        plt.title(f"Training History: {key}")
        plt.legend()

        safe_key = key.replace("/", "_")
        plot_file_path = plots_path / f"{safe_key}.png"

        plt.savefig(plot_file_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved plot: {plot_file_path}")
