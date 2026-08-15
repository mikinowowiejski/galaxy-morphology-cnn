import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from pathlib import Path
from typing import Dict, Any, Optional

#Klasa GalaxyTrainer zarządza cyklem życia treningu, walidacji i zapisywania punktów kontrolnych dla modelu GalaxyCNN.
class GalaxyTrainer:
    

    # Inicjalizuje trenera z wszystkimi wymaganymi komponentami i zależnościami potoku.
    """
    Argumenty:
        model (nn.Module): Instancja GalaxyCNN.
        dataloaders (Dict[str, DataLoader]): Słownik zawierający ładowarki danych 'train' i 'val'.
        optimizer (Optimizer): Algorytm optymalizacji (np. AdamW).
        criterion (nn.Module): Funkcja straty (np. MSELoss dla miękkych prawdopodobieństw).
        device (torch.device): Cel obliczeniowy (CPU lub CUDA).
        checkpoint_dir (Path): Ścieżka katalogu do zapisywania najlepszych wag modelu (.pth).
        scheduler (Optional[Any]): Harmonogram zmiany współczynnika uczenia (np. ReduceLROnPlateau).
    """
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: Path,
        scheduler: Optional[Any] = None
    ) -> None:
     
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler = scheduler

        #float('inf') reprezentuje nieskończoność dodatnią
        self.best_val_loss = float('inf')

    #metoda train_epoch wykonuje pojedynczą epokę treningową nad zestawem danych treningowych, obliczając średnią stratę treningową.
    def train_epoch(self) -> float:

        
        self.model.train()
        running_loss = 0.0

        for images, targets in self.dataloaders["train"]:

            # 1. Przenieś tensoryna urządzenie (CPU lub GPU)
            images = images.to(self.device)
            targets = targets.to(self.device)

            # 2. zero_grad: resetuje gradienty optymalizatora przed obliczeniem nowych gradientów dla bieżącej partii danych.
            self.optimizer.zero_grad()

            # 3. obliczamy wyjścia modelu dla bieżącej partii danych
            outputs = self.model(images)

            # 4. obliczamy stratę między przewidywaniami modelu a rzeczywistymi etykietami
            loss = self.criterion(outputs, targets)

            # 5. wsteczna propagacja: oblicza gradienty straty względem wag modelu, które są następnie używane przez optymalizator do aktualizacji wag.
            loss.backward()

            #6. aktualizujemy wagi modelu na podstawie obliczonych gradientów
            self.optimizer.step()

            # 7. dodajemy stratę bieżącej partii do całkowitej straty epoki (do późniejszego obliczenia średniej straty)
            running_loss += loss.item()
            
            
        #  zwraca średnią stratę treningową dla całej epoki
        return running_loss / len(self.dataloaders["train"])

    # metoda validate ocenia model na zestawie danych walidacyjnych bez śledzenia gradientów (@torch.no_grad()), zwracając średnią stratę walidacyjną.
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0

        for images, targets in self.dataloaders["val"]:

            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(images)
           
            loss = self.criterion(outputs, targets)

            running_loss += loss.item()

        
        return running_loss / len(self.dataloaders["val"])
    
    # metoda fit odpowiada za wykonanie określonej liczby epok, śledzi postęp, obsługuje learning rate i co najważniejsze przerywa proces przedwcześnie gdy nastąpi stagnacja
    def fit(self, epochs: int, patience: int = 5) -> None:
       
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            print(f"Epoch [{epoch}/{epochs}]")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 1. Scheduler obserwuje i zmienia tempo nauki
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # 2. Checkpoint i ewentualny przedwczesny koniec
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_no_improve = 0
                
                checkpoint_path = self.checkpoint_dir / "best_galaxy_cnn.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"--> Saved new best model checkpoint to {checkpoint_path}")
            else:
                epochs_no_improve += 1
                print(f"Early Stopping counter: {epochs_no_improve}/{patience}")
                
                if epochs_no_improve >= patience:
                    print(f"🛑 Early stopping triggered after {epoch} epochs. Training halted to prevent overfitting.")
                    break