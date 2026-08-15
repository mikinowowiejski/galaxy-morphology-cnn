#biblioteka do deep learningu - zawiera zoptymalizowana strukture tensorów, potrafi wykonywac operacje na pamieci VRAM (CUDA) oraz posiada wbudowany silnik do automatycznego obliczania gradientów (autograd) - fundament wstecznej propagacji w DL
import torch 

#biblioteka z narzedziami do obsługi baz danych oraz strumieni. Błyskawicznie wczytują dane tabelaryczne do obiektu DaraFrame
import pandas as pd

#specjalne rozszerzenie do PyTorcha do Computer Vision - zawiera gotowe modele, transformacje, dataset'y i inne narzędzia do pracy z obrazami
import torchvision.transforms as T

#klasa bazowa dla wszystkich datasetów w PyTorch - wymaga implementacji metod __len__ i __getitem__
from torch.utils.data import Dataset

from pathlib import Path

#narzędzie do fizycznego odczytania plików .jpg i konwersji ich do obiektów typu PIL.Image
from PIL import Image

#dict = mapa, tuple (krotka). optional - nullable, callable = obiekt wywoływalny (funkcja, lambda, metoda klasy). typing - narzędzia do statycznego typowania w Pythonie
from typing import Tuple, Callable, Optional, Dict

#DataLoader autmoatycznie zarządza procesami roboczymi (workerami) aby ładowanie plików I/O nie blokowało GPU. Dodatkowo potrafi grupować dane w batch'e i wiele innych
from torch.utils.data import DataLoader

#Tasuje i dzieli dane na zbiór treningowy i walidacyjny. Stratify = zachowuje proporcje klas w obu zbiorach
from sklearn.model_selection import train_test_split



class GalaxyZooDataset(Dataset):

    #konstruktor klasy 
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    #dunder method - zwraca rozmiar datasetu
    def __len__(self):
        return len(self.df)

    #serce klasy - zwraca pojedynczy element datasetu (obraz i odpowiadające mu etykiety) na podstawie indeksu
    def __getitem__(self, idx):

        galaxy_id = self.df.index[idx]

        image_path = self.image_dir / f"{galaxy_id}.jpg" 
        
        #Wczytujemy obraz z dysku i konwertujemy go do formatu RGB (3 kanały) - Sieci konwulcyjne w PyTorch oczekują obrazów w formacie RGB (3 kanały)
        image = Image.open(image_path).convert('RGB')

        #Jeżeli zdefiniowano transformacje, to je stosujemy do obrazu (np. resize, crop, augmentacje)
        if self.transform:
            image = self.transform(image)

        #Pobieramy etykiety z DataFrame i konwertujemy je do tensora typu float32 (PyTorch wymaga tensorów jako wejścia do modelu)
        row = self.df.iloc[idx]
        raw_labels = row.values        
        labels_tensor = torch.tensor(raw_labels, dtype=torch.float32)

        return image, labels_tensor


#funkcja która buduje i zwraca słownik z transformacjami dla zbioru treningowego i walidacyjnego
def get_transforms(crop_size: int = 256, resize_size: int = 224) -> Dict[str, T.Compose]:
    
    # Standardowe statystyki normalizacyjne dla modeli pre-trenowanych na ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Tworzymy obiekt train_transform, który zawiera sekwencję transformacji dla zbioru treningowego. 
    train_transform = T.Compose([
        T.CenterCrop(crop_size),
        T.Resize((resize_size, resize_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=180),
        T.ToTensor(),
        # Krytyczny krok dla Transfer Learningu:
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    

    # Tak samo jak z train_transform, ale bez augmentacji - obiekt walidacyjny
    val_transform = T.Compose([
        T.CenterCrop(crop_size),
        T.Resize((resize_size, resize_size)),
        T.ToTensor(),
        # Normalizacja musi być identyczna jak w zbiorze treningowym:
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    return {
        "train": train_transform,
        "val": val_transform
    }


#funkcja tworząca DataLoadery dla zbioru treningowego i walidacyjnego. Zwraca słownik z kluczami 'train' i 'val' i wartościami typu DataLoader
def create_dataloaders(
    df: pd.DataFrame, 
    image_dir: Path, 
    batch_size: int = 32, 
    val_split: float = 0.2,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    
    
    # 1. dzielimy nasz dataframe na zbiór treningowy i walidacyjny.
    # Używamy train_test_split z sklearn, aby losowo podzielić dane na dwa zbiory. Parametr test_size określa procent danych przeznaczonych na walidację, a random_state zapewnia powtarzalność podziału (seed).
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=42)  
    
    # 2. Pobieramy transformacje dla obu zbiorów (train i val) z funkcji get_transforms(). Transformacje te będą stosowane do obrazów podczas ładowania danych.
    transforms = get_transforms()
    
    # 3. Wywołujemy klasę GalaxyZooDataset dla zbioru treningowego i walidacyjnego, przekazując odpowiednie transformacje
    train_dataset = GalaxyZooDataset(df=train_df, image_dir=image_dir, transform=transforms["train"])
    val_dataset = GalaxyZooDataset(df=val_df, image_dir=image_dir, transform=transforms["val"])

    # 4. Tworzymy DataLoadery dla obu zbiorów.
    # WAżNE: Ustawienie shuffle=True dla zbioru treningowego jest kluczowe, aby model nie uczył się kolejności danych i generalizował lepiej. Dla zbioru walidacyjnego shuffle=False, ponieważ chcemy ocenić model na stałym zestawie danych.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return {
        "train": train_loader,
        "val": val_loader
    }
