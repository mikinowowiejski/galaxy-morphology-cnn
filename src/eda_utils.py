import pandas as pd
import matplotlib.pyplot as plt
import torchvision.io as io
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any
import itertools
import math

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

        df = pd.read_csv(self.solutions_csv, index_col="GalaxyID")
        
        print("=== CSV Audit ===")
        print("Total Rows:", len(df))

        print("Missing Values:\n", df.isnull().sum().sum())

        out_of_bounds = (df < 0.0) | (df > 1.0).sum().sum()
        print("Values out of bounds (0-1):", {out_of_bounds})

        print("\nProbability Distribution Summary:\n")
        print(df.iloc[:, :5].describe().T)

        return df

    def analyze_image_properties(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Reads a sample of images to verify uniform dimensions and channels.
        
        Args:
            sample_size: Number of images to check.
            
        Returns:
            Dict containing unique shapes found and color modes.
        """

        im_dir = self.train_images_dir

        if not im_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {im_dir}")
        
        image_paths = list(itertools.islice(im_dir.glob("*.jpg"), sample_size))
        
        if not image_paths:
            raise ValueError(f"No images found in directory: {im_dir}")
        

        unique_shapes = set()
        color_modes = set()
        
        for path in image_paths:
            im_tensor = io.read_image(str(path))
            unique_shapes.add(tuple(im_tensor.shape))
            color_modes.add(im_tensor.shape[0])  # Assuming shape is (C, H, W)

        return{
            "unique_shapes": unique_shapes,
            "color_modes": color_modes
        }



    

    def plot_galaxy_grid(self, df: pd.DataFrame, num_images: int = 9) -> None:
        """
        Plots a square grid of galaxy images with their IDs and the 
        probability of 'Class 1.1' (Smooth Galaxy) as the title.
        
        Args:
            df: The dataframe containing labels and Galaxy IDs.
            num_images: Number of images to plot (must be a perfect square like 9).
        """
        
        # Calculate grid dimensions (e.g., square root of 9 is 3)
        grid_size = int(math.sqrt(num_images))
        
        # 1. Sample random rows from the DataFrame
        # Hint: Look up pd.DataFrame.sample()
        sampled_df = df.sample(n=num_images, random_state=42)  # Setting random_state for reproducibility

        # 2. Create the Matplotlib figure and axes
        # Hint: plt.subplots(nrows, ncols, figsize=(10,10))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10,10))
        
        # Flatten the 2D axes array into a 1D list for easy iteration
        axes = axes.flatten()
        
        # 3. Iterate over the rows and the axes simultaneously using zip()
        # df.iterrows() yields a tuple of (index, row_data)
        for (galaxy_id, row_data), ax in zip(sampled_df.iterrows(), axes):
            
            # 4. Construct the path: train_images_dir / "galaxy_id.jpg"
            # Hint: You can use the / operator with pathlib.Path objects and f-strings
            img_path = self.train_images_dir / f"{galaxy_id}.jpg"
            
            # 5. Read the image tensor
            img_tensor = io.read_image(str(img_path))
            
            # 6. Fix the shape mismatch (C, H, W) -> (H, W, C)
            # Hint: Use img_tensor.permute(...)
            img_plot_ready = img_tensor.permute(1, 2, 0).numpy()  # Convert to NumPy array for plotting
            
            # 7. Display the image
            # Hint: ax.imshow(img_plot_ready)
            ax.imshow(img_plot_ready)
            
            # 8. Set the title and remove the tick marks
            # Let's show the GalaxyID and the probability of being 'Class 1.1' 
            # (which is usually column 'Class1.1' in Galaxy Zoo)
            smooth_prob = row_data['Class1.1'] 
            ax.set_title(f"ID: {galaxy_id}\nSmooth Prob: {smooth_prob:.2f}")
            ax.axis("off")
            
        plt.tight_layout()
        plt.show()