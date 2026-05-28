from pathlib import Path

import hydra
import kagglehub
import polars as pl


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
)
def preprocess_data(cfg):
    """Downloads a dataset from Kaggle, generates stratified splits, and saves them as Parquet files.

    This function automates the complete data preparation pipeline:
    1. Downloads the raw dataset from Kaggle via kagglehub into a local folder.
    2. Parses the subdirectories to build initial train and test Polars DataFrames
       with explicit image paths and integer class labels.
    3. Sets a global random seed for Polars to ensure deterministic sampling.
    4. Extracts a stratified validation set from the initial training data by
       sampling a uniform number of examples for each class.
    5. Filters out the validation samples from the training set using an image path
       exclusion filter.
    6. Shuffles the final training, validation, and testing DataFrames independently
       and exports each subset into serialized Parquet format.

    Args:
        cfg (DictConfig): A hierarchical Hydra configuration object containing Kaggle
            identifiers, dataset schemas, folder structures, split tokens, and metrics.
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

    pl.set_random_seed(cfg.seed)

    preprocessed_data_dir = Path(cfg.dataset.preprocessed_dir)
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)

    raw_data_path = Path(cfg.dataset.raw_dir) / cfg.dataset.name

    schema = {
        "path": pl.String,
        "label": pl.Int8,
    }

    def assemble_df_by_split(split: str) -> pl.DataFrame:
        split_path = raw_data_path / split
        paths = []
        labels = []
        for class_name, label in cfg.dataset.classes.items():
            class_path = split_path / class_name
            class_img_paths = [f for f in class_path.iterdir() if f.is_file()]
            paths.extend(class_img_paths)
            labels.extend([label] * len(class_img_paths))

        return pl.DataFrame(
            {
                "path": list(map(str, paths)),
                "label": labels,
            },
            schema=schema,
        )

    initial_train_df = assemble_df_by_split(cfg.dataset.splits.train)
    test_df = assemble_df_by_split(cfg.dataset.splits.test)

    val_chunks = []
    for label_value in cfg.dataset.classes.values():
        chunk = initial_train_df.filter(pl.col("label") == label_value).sample(
            cfg.dataset.split.val_samples_per_class
        )
        val_chunks.append(chunk)

    val_df = pl.concat(val_chunks)
    train_df = initial_train_df.filter(~pl.col("path").is_in(val_df["path"].to_list()))

    train_df.sample(fraction=1.0, shuffle=True).write_parquet(
        preprocessed_data_dir / cfg.dataset.df.train
    )
    val_df.sample(fraction=1.0, shuffle=True).write_parquet(
        preprocessed_data_dir / cfg.dataset.df.val
    )
    test_df.sample(fraction=1.0, shuffle=True).write_parquet(
        preprocessed_data_dir / cfg.dataset.df.test
    )
