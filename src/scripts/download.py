from pathlib import Path

import hydra
import kagglehub


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
)
def download_data(cfg):
    local_download_path = Path(cfg.dataset.raw_dir)
    local_download_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {cfg.dataset.kaggle_id} from Kaggle...")

    download_path = kagglehub.dataset_download(
        cfg.dataset.kaggle_id,
        output_dir=str(local_download_path),
        force_download=True,
    )

    print(f"Dataset downloaded to cache: {download_path}")


# if __name__ == "__main__":
#     download_data()
