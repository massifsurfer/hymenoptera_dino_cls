import os
from pathlib import Path

import hydra
import mlflow
import onnx
import pytorch_lightning as L
import tensorrt as trt
import torch
from model.models import HymenopteraClassifier
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from utils import gen_fancy_name, get_git_commit_id, save_mlflow_plots

from data.datamodules import HymenopteraDataModule


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="config",
)
def train(cfg):
    """Trains a Hymenoptera image classifier and exports the best model to ONNX.

    This function coordinates the full training pipeline using PyTorch Lightning,
    Hydra, and MLflow. It performs the following steps:
    1. Sets reproducibility seeds.
    2. Initializes the data module, model, and MLflow logger.
    3. Flattens and logs the full Hydra configuration to MLflow.
    4. Trains the model while tracking the learning rate and saving the best
       checkpoint based on validation loss.
    5. Evaluates the best checkpoint on the test dataset if available.
    6. Loads the best model weights and exports the model architecture to a
       local ONNX file with dynamic batching.
    7. Logs and registers the exported ONNX model artifact in MLflow with its
       input/output signature.

    Args:
        cfg (DictConfig): A Hierarchical Hydra configuration object containing
            all parameters for the dataset, model, tracking, and training process.
    """

    L.seed_everything(cfg.seed)

    datamodule = HymenopteraDataModule(cfg)

    model = HymenopteraClassifier(cfg)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.tracking.experiment_name,
        run_name=cfg.tracking.run_name_prefix + gen_fancy_name(),
        tracking_uri=cfg.tracking.uri,
        tags={"mlflow.source.git.commit": get_git_commit_id()},
    )

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

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(".", cfg.tracking.checkpoint_dir),
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=cfg.model.max_epochs,
        accelerator=cfg.model.accelerator,
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg.tracking.log_every_n_steps,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)

    if datamodule.test_df_path.exists():
        trainer.test(model, datamodule=datamodule, ckpt_path="best", weights_only=False)

    print("📉 Saving plots...")
    save_mlflow_plots(trainer.logger.run_id, output_dir=cfg.tracking.plots_dir)

    print("Exporting to ONNX...")

    best_model_path = checkpoint_callback.best_model_path
    best_model = HymenopteraClassifier.load_from_checkpoint(
        best_model_path, cfg=cfg, weights_only=False
    )
    best_model.eval()
    best_model.to(cfg.model.accelerator)

    img_size = cfg.dataset.transforms.img_size
    dummy_input = torch.randn(1, 3, img_size, img_size)

    onnx_dir = os.path.join(".", cfg.tracking.checkpoint_dir)
    os.makedirs(onnx_dir, exist_ok=True)
    local_onnx_path = os.path.join(onnx_dir, cfg.project_name + ".onnx")

    torch.onnx.export(
        best_model,
        dummy_input,
        local_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"The model was succesfully saved locally: {local_onnx_path}")

    if isinstance(trainer.logger, MLFlowLogger):
        with mlflow.start_run(run_id=trainer.logger.run_id):
            with torch.no_grad():
                dummy_output = best_model(dummy_input).numpy()

            signature = mlflow.models.infer_signature(dummy_input.numpy(), dummy_output)
            onnx_model = onnx.load(local_onnx_path)

            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                name=cfg.tracking.onnx_artifact_path,
                signature=signature,
                registered_model_name=cfg.tracking.onnx_registered_model_name,
            )
            print("ONNX model was registered in MLflow Artifacts.")

    trt_dir = os.path.join(".", cfg.tracking.checkpoint_dir)
    local_trt_path = os.path.join(trt_dir, cfg.project_name + ".engine")

    if not torch.cuda.is_available():
        print("\nℹ️ CUDA GPU не обнаружен в системе. Экспорт в TensorRT пропущен.")
        print("Для процессоров (CPU) используйте готовую ONNX модель.")
    else:
        print(f"Compiling ONNX to TensorRT engine via Python API: {local_trt_path}...")
        try:
            torch.cuda.init()
            _ = torch.tensor([1.0]).cuda()

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            config = builder.create_builder_config()

            config.set_flag(trt.BuilderFlag.FP16)

            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            parser = trt.OnnxParser(network, logger)

            with open(local_onnx_path, "rb") as model_file:
                if not parser.parse(model_file.read()):
                    print("\n❌ Ошибка парсинга ONNX файла для TensorRT:")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX file.")

            print("Building TensorRT engine...")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                raise RuntimeError("Engine serialization failed.")

            with open(local_trt_path, "wb") as f:
                f.write(serialized_engine)

            print(f"TensorRT Engine successfully built and saved to: {local_trt_path}")

        except ImportError:
            print(
                "\n❌ Ошибка: Библиотека 'tensorrt' не установлена в Python-окружении."
            )
            print("Выполните: pip install tensorrt tensorrt-cu12")
        except Exception as e:
            print(f"\n❌ Ошибка компиляции TensorRT через Python API: {e}")


if __name__ == "__main__":
    train()
