import os
import subprocess
import sys
from pathlib import Path

import hydra


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
)
def run_mlflow_server(cfg):
    """Initializes and hosts a local MLflow REST server to serve a registered ONNX model.

    This function sets up the required environment and spawns an interactive model
    serving subprocess:
    1. Locates the persistent SQLite database file relative to the project root
       directory.
    2. Overrides the `MLFLOW_TRACKING_URI` environment variable to point to the
       resolved SQLite database string.
    3. Validates the existence of the database file and terminates with an error
       code if missing.
    4. Resolves the model deployment URI using the model registry name and the
       "latest" version tag.
    5. Assembles and executes a shell command to spin up the MLflow native serving
       infrastructure with disabled Conda environment instantiation.
    6. Captures user interruption sequences (CTRL+C) and subprocess execution
       faults gracefully.

    Args:
        cfg (DictConfig): A Hydra configuration object containing network host,
            port configurations, and the target registered model token.
    """

    print("=== MLflow Serving initialization ===")

    project_root = Path(__file__).parent.parent.parent
    absolute_db_path = project_root / "mlflow.db"

    target_db_uri = f"sqlite:////{absolute_db_path}"

    os.environ["MLFLOW_TRACKING_URI"] = target_db_uri

    if not absolute_db_path.exists():
        print(f"❌ DB doesn't exist: {absolute_db_path}")
        sys.exit(1)

    model_name = cfg.tracking.onnx_registered_model_name
    model_version = "latest"
    correct_model_uri = f"models:/{model_name}/{model_version}"

    host = cfg.tracking.host
    port = cfg.tracking.port

    cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        correct_model_uri,
        "--host",
        host,
        "--port",
        port,
        "--no-conda",
    ]

    print(f"🔗 Model's name in the registry: {model_name} ({model_version})")
    print(f"🗄️ The db's path: {os.environ['MLFLOW_TRACKING_URI']}")
    print(f"🚀 Request endpoint: http://{host}:{port}/invocations")
    print("Нажмите CTRL+C для остановки сервера...\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server was stopped by the user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server error: {e}")


if __name__ == "__main__":
    run_mlflow_server()
