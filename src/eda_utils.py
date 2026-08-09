import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any

class UniverseEDA:
   
    """
    Exploratory Data Analysis toolkit for the Universe Cataloger project.
    Encapsulates data loading, auditing, and visualization pipelines.
    """

    def __init__(self, raw_data_dir: str | Path):
          
        """
        Initializes the EDA toolkit with the root raw data directory.
        
        Args:
            raw_data_dir: Path to the 'data/raw' directory.
        """
        # Hint: Convert string to pathlib.Path object here for safer path joining
        self.raw_data_dir = Path(raw_data_dir)
        self.solutions_csv = self.raw_data_dir / "training_solutions_rev1.csv"
        self.train_images_dir = self.raw_data_dir / "images_training_rev1"

    def load_and_audit_csv(self) -> pd.DataFrame:
        """
        Loads the training solutions CSV and prints audit metrics 
        (missing values, basic statistical distribution of probabilities).
        
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        pass

    def analyze_image_properties(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Reads a sample of images to verify uniform dimensions and channels.
        
        Args:
            sample_size: Number of images to check.
            
        Returns:
            Dict containing unique shapes found and color modes.
        """
        pass

    def plot_galaxy_grid(self, df: pd.DataFrame, num_images: int = 9) -> None:
        """
        Plots a square grid of galaxy images with their IDs and a selected 
        probability metric as the title.
        
        Args:
            df: The dataframe containing labels and Galaxy IDs.
            num_images: Number of images to plot (perfect squares work best, e.g., 9, 16).
        """
        pass
