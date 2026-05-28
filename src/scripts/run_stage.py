from pathlib import Path

import hydra
from download import download_data
from omegaconf import DictConfig
from preprocess import preprocess_data


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Orchestrates data pipeline stages by dispatching execution to specialized modules.

    This function acts as the central entry point for data workflows using Hydra:
    1. Reads the designated lifecycle stage flag from the configuration object.
    2. Utilizes pattern matching to route execution dynamically based on the string
       literal value of the stage.
    3. Invokes the `download_data` pipeline when the stage is explicitly flagged
       as "download".
    4. Invokes the `preprocess_data` pipeline when the stage is explicitly flagged
       as "preprocess".

    Args:
        cfg (DictConfig): A Hydra configuration object containing the structural
            stage identifiers and specific parameters for downstream data modules.
    """

    match cfg.stage:
        case "download":
            download_data(cfg)
        case "preprocess":
            preprocess_data(cfg)


if __name__ == "__main__":
    main()
