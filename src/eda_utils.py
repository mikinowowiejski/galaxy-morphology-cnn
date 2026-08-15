import pandas as pd

#biblioteka do tworzenia wykresów i wizualizacji danych w Pythonie
import matplotlib.pyplot as plt 

from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any

class UniverseEDA:
   
   #Klasa służąca do analizy danych (Exploratory Data Analysis) do naszego projektu. Zawiera metody do wczytywania i audytu danych z pliku CSV, analizowania właściwości obrazów oraz wizualizacji próbek galaktyk w siatce.

    def __init__(self, raw_data_dir: str | Path):

        
        # Zmieniamy raw_data_dir na obiekt Path, aby ułatwić operacje na ścieżkach plików. (tworzenie ściezki za pomocą '/')
        # Następnie definiujemy ścieżki do pliku CSV z rozwiązaniami treningowymi oraz katalogu z obrazami treningowymi.
        self.raw_data_dir = Path(raw_data_dir)
        self.solutions_csv = self.raw_data_dir / "training_solutions_rev1.csv"
        self.train_images_dir = self.raw_data_dir / "images_training_rev1"


    # Metoda wczytuje plik CSV z rozwiązaniami treningowymi i przeprowadza audyt danych, sprawdzając brakujące wartości oraz podstawowe statystyki rozkładu prawdopodobieństw.
    def load_and_audit_csv(self) -> pd.DataFrame:

        if not self.solutions_csv.exists():
            raise FileNotFoundError(f"CSV file not found at {self.solutions_csv}")
        
        print(f"Loading CSV from: {self.solutions_csv}")

        # Wczytujemy plik CSV do obiektu DataFrame z Pandas, ustawiając kolumnę "GalaxyID" jako indeks. Następnie wyświetlamy podstawowe informacje o DataFrame, sprawdzamy brakujące wartości i generujemy statystyki opisowe dla każdej kolumny.
        df = pd.read_csv(self.solutions_csv, index_col="GalaxyID")

        print("CSV loaded successfully. Performing audit...")
        df.info()

        print("\n===Missing Data Check===")
        missing_counts = df.isna().sum()
        print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "No missing values found.")

        print("\n===Statistical Summary===")
        print(df.describe().T) #nakładamy transpozycję, aby lepiej widzieć statystyki dla każdej kolumny w pionie

        return df


    #funkcja analizująca właściwości obrazów w katalogu treningowym. Wczytuje próbkę obrazów i sprawdza, czy mają one jednolite wymiary i tryby kolorów (np. RGB). Zwraca unikalne kształty i tryby kolorów znalezione w próbie.
    def analyze_image_properties(self, sample_size: int = 100) -> Dict[str, Any]:
       
        import itertools
    
        if not self.train_images_dir.exists():
          raise FileNotFoundError(f"Image directory not found at {self.train_images_dir}")

        #itertools.islice pozwala na pobranie określonej liczby elementów z generatora , .glob("*.jpg") zwraca generator wszystkich plików .jpg w katalogu. W ten sposób pobieramy określoną liczbę obrazów do analizy.
        image_paths = list(itertools.islice(self.train_images_dir.glob("*.jpg"), sample_size))

        if not image_paths:
            print("No images found in the specified directory.")
            return {"shapes": set(), "modes": set()}


        # Tworzymy zbiory do przechowywania unikalnych kształtów (rozmiarów) i trybów kolorów obrazów. Dla każdego obrazu w próbie wczytujemy go za pomocą PIL.Image, a następnie dodajemy jego rozmiar (width, height) i tryb kolorów (np. 'RGB', 'L') do odpowiednich zbiorów.
        unique_shapes = set()
        unique_modes = set()

        for path in image_paths:
            with Image.open(path) as img:
                unique_shapes.add(img.size)  # (width, height)
                unique_modes.add(img.mode)    # e.g., 'RGB', 'L', etc.

        print(f"Sampled {len(image_paths)} images.")

        #zbiory muszą być jednolite, aby model mógł je poprawnie przetworzyć.
        if len(unique_shapes) == 1 and len(unique_modes) == 1:
            print(f"All sampled images have uniform shape: {unique_shapes.pop()} and mode: {unique_modes.pop()}.")
        else:
            print(f"❌ Warning! Found multiple shapes: {unique_shapes} and modes: {unique_modes}.") 
    

        return {"shapes": unique_shapes, 
                "modes": unique_modes}

    #funkcja wizualizująca próbkę galaktyk w siatce. Wybiera losowo określoną liczbę obrazów z DataFrame, a następnie wyświetla je w kwadratowej siatce z identyfikatorami galaktyk i wybranym prawdopodobieństwem jako tytułami.
    def plot_galaxy_grid(self, df: pd.DataFrame, num_images: int = 9) -> None:
     
        import math

        sample_df = df.sample(n=num_images, random_state=42)
        grid_size = math.ceil(math.sqrt(num_images))

        # Tworzymy siatkę wykresów (subplots) o wymiarach grid_size x grid_size, aby pomieścić wszystkie wybrane obrazy galaktyk. Używamy matplotlib do tworzenia wykresów i ustawiamy rozmiar figury na 10x10 cali.
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        # Spłaszczamy tablicę osi, aby ułatwić iterację po wszystkich subplotach. (z 2d na 1d)
        axes = axes.flatten()

        galaxy_ids = sample_df.index.astype(str).to_list()
        probabilities = sample_df.iloc[:, 0].to_list()

        # enumerate pozwala nam iterować po parach (indeks, wartość) dla osi i odpowiadających im identyfikatorów galaktyk oraz prawdopodobieństw. Dla każdego obrazu w próbie wczytujemy go z katalogu obrazów treningowych, a następnie wyświetlamy go w odpowiednim subplotie z tytułem zawierającym identyfikator galaktyki i prawdopodobieństwo.
        for i, (ax, gal_id, prob) in enumerate(zip(axes, galaxy_ids, probabilities)):
            if i < num_images:
                img_path = self.train_images_dir / f"{gal_id}.jpg"
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f"ID: {gal_id}\nProb: {prob:.2f}", fontsize=9)
            except FileNotFoundError:
                ax.set_title(f"ID: {gal_id}\nImage not found", fontsize=9, color='red')
                
            ax.axis('off')

        #tight_layout() automatycznie dostosowuje rozmieszczenie subplotów, aby uniknąć nakładania się tytułów i osi.
        plt.tight_layout()
        plt.show()
        
