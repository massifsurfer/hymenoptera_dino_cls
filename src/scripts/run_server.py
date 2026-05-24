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
    print("=== MLflow Serving initialization ===")

    # 1. Вычисляем абсолютный путь к корню вашего проекта.
    # resolve().parent берет папку, в которой физически лежит этот файл run_server.py.
    project_root = Path(__file__).parent.parent.parent
    absolute_db_path = project_root / "mlflow.db"

    # 2. Формируем правильный абсолютный URI для SQLite под Linux (с 4 косыми чертами)
    target_db_uri = f"sqlite:////{absolute_db_path}"

    # 3. КРИТИЧЕСКИЙ ШАГ: фиксируем переменную окружения в контексте текущего процесса.
    # Это единственный документированный способ заставить внутренний Uvicorn увидеть базу.
    os.environ["MLFLOW_TRACKING_URI"] = target_db_uri

    # 4. Проверяем, существует ли физически файл базы данных
    if not absolute_db_path.exists():
        print(f"❌ DB doesn't exist: {absolute_db_path}")
        sys.exit(1)

    # 5. Настройки модели и сети
    model_name = cfg.tracking.onnx_registered_model_name
    model_version = "latest"  # Автоматически берет самую свежую версию модели
    correct_model_uri = f"models:/{model_name}/{model_version}"

    host = cfg.tracking.host
    port = cfg.tracking.port

    # 6. Собираем список аргументов строго по спецификации CLI 'mlflow models serve'
    # Никаких лишних параметров вроде --backend-store-uri или --registry-store-uri!
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
        "--no-conda",  # Использует текущее окружение .venv (ускоряет запуск)
    ]

    print(f"🔗 Model's name in the registry: {model_name} ({model_version})")
    print(f"🗄️ The db's path: {os.environ['MLFLOW_TRACKING_URI']}")
    print(f"🚀 Request endpoint: http://{host}:{port}/invocations")
    print("Нажмите CTRL+C для остановки сервера...\n")

    # 7. Запускаем изолированный подпроцесс сервера
    # stdout=None перенаправляет поток логов Uvicorn напрямую в вашу текущую консоль
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server was stopped by the user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server error: {e}")


if __name__ == "__main__":
    run_mlflow_server()
