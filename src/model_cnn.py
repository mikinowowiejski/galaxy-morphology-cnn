import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class GalaxyCNN(nn.Module):
    """
    Wytrenowana wcześniej architektura ResNet18 dostosowana do klasyfikacji morfologii galaktyk 
    oraz regresji "miękkich" prawdopodobieństw.
    """

    def __init__(self, num_classes: int = 37, freeze_backbone: bool = True) -> None:
      
        super().__init__()

        # 1. Wczytanie wytrenowanego rdzenia (backbone) ResNet18 przy użyciu nowoczesnych enumeratorów wag PyTorch
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 2. Zamrożenie parametrów rdzenia, jeśli włączono flagę (wymagane w Fazie 1 Transfer Learningu)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. Pobranie liczby cech wejściowych z oryginalnej warstwy w pełni połączonej (fully connected layer)
        # Dla modelu ResNet18 wynosi ona domyślnie 512.
        in_features: int = self.backbone.fc.in_features

        # 4. Zastąpienie oryginalnej głowicy klasyfikacyjnej naszą własną głowicą regresyjną.
        # Nadpisujemy self.backbone.fc, dzięki czemu natywny proces propagacji w przód 
        # (forward pass) ResNeta sam zarządza przepływem danych przez sieć.
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=num_classes)
            # WAŻNE: Nie stosujemy tu funkcji Sigmoid, ponieważ naturalne wyjście modelu (logits) 
            # jest optymalizowane bezpośrednio za pomocą funkcji błędu MSELoss.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        # Wewnętrzny proces propagacji w przód wbudowany w ResNet automatycznie przepuszcza 
        # tensor wejściowy 'x' przez warstwy konwolucyjne, adaptacyjne operacje pooling (AdaptiveAvgPool2d)
        # oraz naszą niestandardową warstwę self.backbone.fc przypiętą na samym końcu.
        return self.backbone(x)