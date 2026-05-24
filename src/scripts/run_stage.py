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
    match cfg.stage:
        case "download":
            download_data(cfg)
        case "preprocess":
            preprocess_data(cfg)


if __name__ == "__main__":
    main()
