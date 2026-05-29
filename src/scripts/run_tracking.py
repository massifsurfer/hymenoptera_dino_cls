import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Launches the MLflow tracking server locally with graceful shutdown on Ctrl+C."""
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        cfg.tracking.backend_uri,
        "--host",
        cfg.tracking.host,
        "--port",
        cfg.tracking.port,
    ]

    print(
        f"🚀 Starting MLflow server on http://{cfg.tracking.host}:{cfg.tracking.port}..."
    )

    process = subprocess.Popen(cmd)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected! Performing graceful shutdown...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️ Server did not respond in time. Forcing termination...")
            process.kill()

        print("✅ MLflow server stopped cleanly.")


if __name__ == "__main__":
    main()
