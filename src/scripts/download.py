from pathlib import Path

import hydra
import kagglehub


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
)
def download_data(cfg):
    """Downloads a dataset from Kaggle to a local directory using kagglehub.

    This function sets up the raw data environment using Hydra configurations:
    1. Creates the target directory for the raw dataset if it does not exist.
    2. Forces the download of the specified Kaggle dataset ID into the defined
       local storage path.
    3. Prints the final path to the cached or downloaded dataset assets.

    Args:
        cfg (DictConfig): A Hydra configuration object containing the Kaggle
            dataset identifier and local destination directory paths.
    """

    local_download_path = Path(cfg.dataset.raw_dir)
    local_download_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {cfg.dataset.kaggle_id} from Kaggle...")

    download_path = kagglehub.dataset_download(
        cfg.dataset.kaggle_id,
        output_dir=str(local_download_path),
        force_download=True,
    )

    print(f"Dataset downloaded to cache: {download_path}")
