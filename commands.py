import sys

import fire
import hydra
from dvc.cli import main as dvc_main
from PIL import Image

from src import get_val_transform
from src import infer as run_inference
from src import train as run_train


class ProjectCLI:
    """Unified command-line interface entry point for managing the Hymenoptera Classifier project."""

    def _load_config(self, config_name: str):
        """Helper private method to programmatically compose the Hydra configuration object.

        Args:
            config_name (str): The filename of the target YAML configuration (without extension).

        Returns:
            DictConfig: A composed hierarchical Hydra configuration object.
        """
        config_path = "configs"
        with hydra.initialize(version_base=None, config_path=config_path):
            return hydra.compose(config_name=config_name)

    def download(self):
        """Reproduces the DVC pipeline to automatically download and preprocess raw data.

        Raises:
            RuntimeError: If the 'dvc repro' pipeline execution terminates with a non-zero exit code.
        """
        print("🚀 Запуск команды: dvc repro")

        exit_code = dvc_main(["repro"])
        if exit_code != 0:
            raise RuntimeError(f"DVC repro failed with exit code {exit_code}")
        print("✅ Данные успешно подготовлены через DVC!")

    def train(self, config_name: str = "config", **kwargs):
        """Executes the full model training pipeline.

        This method synchronizes data via DVC, injects dynamic overrides into
        the system arguments list, and hands over control execution flow back to
        the original Hydra-decorated training routine.

        Args:
            config_name (str): The filename of the target YAML configuration. Defaults to "config".
            **kwargs: Arbitrary parameter overrides passed down directly to the Hydra CLI parser
                (e.g., seed=42 or model.max_epochs=10).

        Example:
            python src/commands.py train --seed=42 --"model.max_epochs"=10
        """
        print(f"🚀 Запуск команды: train (config: {config_name})")

        self.download()

        hydra_overrides = [f"{k}={v}" for k, v in kwargs.items()]

        # sys.argv = ["src/train.py", f"--config-name={config_name}"] + hydra_overrides
        sys.argv = ["src/train.py", f"--config-name={config_name}", *hydra_overrides]

        print(f"🔧 Передача управления в Hydra с аргументами: {sys.argv[1:]}")
        run_train()

    def infer(self, image_path: str, config_name: str = "config") -> str:
        """Dispatches a target single local image to the inference pipeline to predict its class category.

        Args:
            image_path (str): File system path to the target verification image.
            config_name (str): The filename of the target YAML configuration. Defaults to "config".

        Returns:
            str: Formatted evaluation string tracking the estimated prediction probability.

        Example:
            python src/commands.py infer --image_path="data/test/ant.jpg"
        """
        print("🚀 Запуск команды: infer")
        cfg = self._load_config(config_name)
        transform = get_val_transform(cfg.dataset.transforms)
        image = Image.open(image_path).convert("RGB")
        result = run_inference(image, cfg.tracking.inference.endpoint, transform)
        return f"Вероятность пчелы: {result}"


if __name__ == "__main__":
    fire.Fire(ProjectCLI)
