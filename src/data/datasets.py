import polars as pl
from PIL import Image
from torch.utils.data import Dataset


class HymenopteraDataset(Dataset):
    """PyTorch Dataset for loading Hymenoptera image data from Parquet files.

    Reads file paths and class labels using Polars, loads images on the fly
    via PIL, and optionally applies data transformations.
    """

    def __init__(self, df_path: str, transform=None):
        self.df = pl.read_parquet(df_path)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        image = Image.open(row["path"]).convert("RGB")
        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, label
