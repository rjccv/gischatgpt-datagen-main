from torchvision import transforms
from dataclasses import dataclass, field
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from typing import Optional

@dataclass
class im2gps3ktestDataset(Dataset):
    data_dir: str
    csv_file: str
    transform: Optional[callable] = field(default=None)

    
    def __post_init__(self):
        self.data_df = pd.read_csv(self.csv_file)
        
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):

        lat = float(self.data_df.iloc[idx, 2])
        lon = float(self.data_df.iloc[idx, 3])
        img_id = self.data_df.iloc[idx, 0]  # Get the IMG_ID

        return img_id, lat, lon
