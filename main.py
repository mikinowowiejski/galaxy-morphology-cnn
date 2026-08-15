import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.eda_utils import UniverseEDA
from src.data_loader import create_dataloaders
from src.model_cnn import GalaxyCNN
from src.train import GalaxyTrainer
from wakepy import keep

def main() -> None:
    """
    Main execution pipeline for the Universe Cataloger project.
    Implements a Two-Phase Transfer Learning strategy.
    """
    print("Initializing Universe Cataloger Pipeline...")

    # ==========================================
    # 1. Konfiguracja i hiperparametry bazowe
    # ==========================================
    raw_data_dir = Path("data/raw")
    checkpoint_dir = Path("models")
    
    batch_size = 32
    num_classes = 37
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Device: {device}")

    # ==========================================
    # 2. Przygotowanie danych
    # ==========================================
    eda = UniverseEDA(raw_data_dir)
    df = eda.load_and_audit_csv()

    print("\nConstructing DataLoaders...")
    dataloaders = create_dataloaders(
        df=df,
        image_dir=eda.train_images_dir,
        batch_size=batch_size,
        val_split=0.2,
        num_workers=4
    )

    # ==========================================
    # 3. Instancjacja Modelu (Faza 1 - Zamrożony Rdzeń)
    # ==========================================
    print("\nInstantiating GalaxyCNN Architecture (ResNet18 Backbone)...")
    # Zakładamy, że konstruktor ustawia domyślnie freeze_backbone=True
    model = GalaxyCNN(num_classes=num_classes)
    
    criterion = nn.MSELoss()
    
    # Optimizer tylko dla odblokowanych wag (czyli naszej nowej głowicy)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3, 
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    trainer = GalaxyTrainer(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler
    )

    # Blokada uśpienia systemu dla całego procesu
    with keep.running():
        
        # ==========================================
        # 4. Faza 1: Rozgrzewka (Warm-up)
        # ==========================================
        print(f"\n--- [PHASE 1] Training Regression Head (10 Epochs) ---")
        trainer.fit(epochs=10, patience=3)

        # ==========================================
        # 5. Faza 2: Fine-Tuning (Strojenie całej sieci)
        # ==========================================
        print("\n--- [PHASE 2] Unfreezing Backbone for Fine-Tuning ---")
        
        # Odmrażamy wszystkie warstwy ResNeta
        for param in model.parameters():
            param.requires_grad = True

        # Tworzymy nowy optymalizator z 10-krotnie mniejszym krokiem uczenia (1e-4)
        # by nie zniszczyć pre-trenowanych wag ResNeta
        optimizer_ft = optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4
        )
        
        scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ft, mode='min', factor=0.5, patience=2
        )

        # Wstrzykujemy nowe zależności do naszego trenera
        trainer.optimizer = optimizer_ft
        trainer.scheduler = scheduler_ft
        # Resetujemy licznik Early Stopping z poprzedniej fazy
        trainer.best_val_loss = float('inf') 

        print(f"\n--- Starting Fine-Tuning for 40 Epochs ---")
        trainer.fit(epochs=40, patience=5)
        
    print("--- Universe Cataloger Training Complete ---")

if __name__ == "__main__":
    main()