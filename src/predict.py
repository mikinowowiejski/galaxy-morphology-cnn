import torch
from pathlib import Path
from typing import Dict, Callable
from PIL import Image
import torchvision.transforms as T

#Importujemy wewnętrzne klasy/metody
from src.model_cnn import GalaxyCNN
from src.data_loader import get_transforms

class GalaxyPredictor:
    """
    Inference service for the Universe Cataloger.
    Encapsulates state dictionary loading, strictly reproduces validation preprocessing, 
    and executes single-image forward passes to output morphology probabilities.
    """

    def __init__(self, weights_path: Path, device: torch.device, num_classes: int = 37) -> None:
        """
        Initializes the inference engine, instantiates the model architecture, 
        loads the saved weights, and locks the model into evaluation mode.

        Args:
            weights_path (Path): Path to the saved 'best_galaxy_cnn.pth' state dictionary.
            device (torch.device): Compute target (CPU or CUDA) for inference mapping.
            num_classes (int): Number of target morphology classes (default: 37).
        """
        self.device: torch.device = device
        
        #1. tworzymy szkielet sieci (bez wiedzy) 

        self.model: torch.nn.Module = GalaxyCNN(num_classes=num_classes)
        
        #2. Wczytujemy wagi z dysku, map_location -> brak błedu jeśli model był trenowany
        #   na GPU a teraz uruchamiany na CPU 
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        #3. Przeniosimy gotowy model do pamięci używanego urządzenia  
        self.model.to(self.device)

        # 4. Krytyczny krok: blokujemy wagi. Przełączamy warstwy takie jak BatchNorm 
        # i Dropout w tryb wnioskowania (inference), zapobiegając mutacji modelu.
        self.model.eval()
        
        # 5. Pobieramy DOKŁADNIE ten sam potok transformacji, co podczas walidacji.
        # Zapewnia to, że nowe zdjęcie będzie wykadrowane i znormalizowane identycznie.
        self.transform: Callable = get_transforms()["val"] 


    #Metoda wczytuje 'surowe' zdjęcie z dysku, transformuje je i zamienia na tensor
    def _preprocess_image(self, image_path: Path) -> torch.Tensor:

        #1.wczytujemy plik z dysku i wymuszamy paletę RGB
        image = Image.open(image_path).convert('RGB')

        #2. Wywołujemy transformację na zdjęciu -> skalowanie, kadrowanie 
        # i zamiana na macierz liczb
        tensor = self.transform(image)

        #3. Sztucznie dodajemy wymiar "batch" (paczki) PyTorch oczekuje kształtu
        #   [Batch, Channels, Height, Width]
        #   # Unsqueeze(0) zamienia nasz tensor [3, 224, 224] na [1, 3, 224, 224].
        tensor = tensor.unsqueeze(0)

        #4. Wysyłamyy przygotowany tensor do pamięci obliczeniowej
        tensor = tensor.to(self.device)
        return tensor



    @torch.no_grad()
    def predict(self, image_path: Path) -> Dict[int, float]:

        # 1. Wczytanie i przygotowanie obrazu
        image = self._preprocess_image(image_path)

        # 2. Przepuszczenie tensora przez sieć. 
        # Ponieważ trenowaliśmy z MSELoss na ułamkach, model zwraca już wartości ~ [0, 1]
        outputs = self.model(image)

        # 3. Zamiast sigmoid (który psuł nam wyniki), po prostu przycinamy 
        # ewentualne odchylenia do rygorystycznego przedziału 0.0 - 1.0
        probabilities = torch.clamp(outputs, min=0.0, max=1.0)

        # 4. Usuwamy sztuczny wymiar "batch" i zamieniamy tensor na listę
        probs_list = probabilities.squeeze().tolist()

        # 5. Iterujemy po liście wyników i tworzymy słownik
        result = {}
        for class_idx, prob in enumerate(probs_list):
            result[class_idx] = round(prob, 2)

        # 6. Zwracamy czytelny wynik
        return result