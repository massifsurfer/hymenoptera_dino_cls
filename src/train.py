import os
from pathlib import Path

import hydra
import mlflow
import onnx
import pytorch_lightning as L
import torch
from faker import Faker
from model.models import HymenopteraClassifier
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from data.datamodules import HymenopteraDataModule


def gen_fancy_name() -> str:
    fake = Faker()
    words = fake.words(nb=2)
    name = "_".join(words)
    return name


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="config",
)
def train(cfg):
    # 1. Настройка воспроизводимости (опционально)
    L.seed_everything(cfg.seed)

    # 2. Инициализация DataModule
    datamodule = HymenopteraDataModule(cfg)

    # 3. Инициализация модели
    model = HymenopteraClassifier(cfg)

    # 4. Настройка MLFlow Logger
    # tracking_uri может быть локальной папкой "./mlruns" или адресом удаленного сервера
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.tracking.experiment_name,
        run_name=cfg.tracking.run_name_prefix + gen_fancy_name(),
        tracking_uri=cfg.tracking.uri,
    )

    # Автоматически логируем весь OmegaConf конфиг как гиперпараметры в MLflow
    # Превращаем в плоский словарь, чтобы MLflow корректно отображал вложенные структуры
    flat_config = OmegaConf.to_container(cfg, resolve=True)

    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    mlflow_logger.log_hyperparams(flatten_dict(flat_config))

    # 5. Настройка коллбэков
    # Сохраняем лучшую модель по val_loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(".", cfg.tracking.checkpoint_dir),
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    # Автоматически логирует изменения Learning Rate от вашего CosineAnnealingLR
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 6. Инициализация Trainer
    trainer = L.Trainer(
        max_epochs=cfg.model.max_epochs,
        accelerator=cfg.model.accelerator,  # Автоматически выберет 'gpu' или 'cpu'
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg.tracking.log_every_n_steps,
        deterministic=True,
    )

    # 7. Запуск обучения
    trainer.fit(model, datamodule=datamodule)

    # 8. Тестирование на лучшем чекпоинте (если есть тестовый датасет)
    if datamodule.test_df_path.exists():
        trainer.test(model, datamodule=datamodule, ckpt_path="best", weights_only=False)

    print("Exporting to ONNX...")

    # Загружаем лучшие веса из чекпоинта, который сохранил ModelCheckpoint
    best_model_path = checkpoint_callback.best_model_path
    best_model = HymenopteraClassifier.load_from_checkpoint(
        best_model_path, cfg=cfg, weights_only=False
    )
    best_model.eval()
    best_model.to("cpu")  # Экспортировать безопаснее на CPU

    # Создаем dummy_input для определения структуры графа
    img_size = cfg.dataset.transforms.img_size
    dummy_input = torch.randn(1, 3, img_size, img_size)

    # Генерируем локальный путь для сохранения
    onnx_dir = os.path.join(".", cfg.tracking.checkpoint_dir)
    os.makedirs(onnx_dir, exist_ok=True)
    local_onnx_path = os.path.join(onnx_dir, cfg.project_name + ".onnx")

    # 9. Экспорт структуры графа PyTorch в ONNX файл (Исправленный вариант)
    torch.onnx.export(
        best_model,
        dummy_input,
        local_onnx_path,
        export_params=True,
        opset_version=17,  # Теперь 17 применится успешно
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,  # <--- КРИТИЧЕСКИЙ ФЛАГ: отключает проблемный Dynamo бэкенд
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"The model was succesfully saved locally: {local_onnx_path}")

    # 10. Логирование ONNX-модели в MLflow (Исправленный вариант)
    if isinstance(trainer.logger, MLFlowLogger):
        with mlflow.start_run(run_id=trainer.logger.run_id):
            with torch.no_grad():
                dummy_output = best_model(dummy_input).numpy()

            signature = mlflow.models.infer_signature(dummy_input.numpy(), dummy_output)
            onnx_model = onnx.load(local_onnx_path)

            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                name=cfg.tracking.onnx_artifact_path,  # <--- Исправлено: заменили artifact_path на name
                signature=signature,
                registered_model_name=cfg.tracking.onnx_registered_model_name,
            )
            print("ONNX model was registered in MLflow Artifacts.")


if __name__ == "__main__":
    train()
